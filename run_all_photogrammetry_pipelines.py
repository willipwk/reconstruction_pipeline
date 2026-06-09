import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

try:
    import GPUtil
except ImportError:
    GPUtil = None


PRINT_LOCK = threading.Lock()
ACTIVE_PROCESS_LOCK = threading.Lock()
ACTIVE_PROCESSES = {}


def _repo_root():
    return Path(__file__).resolve().parent


def _print(message):
    with PRINT_LOCK:
        print(message, flush=True)


def _split_csv(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _in_slurm_allocation():
    return any(
        os.environ.get(name)
        for name in (
            "SLURM_JOB_ID",
            "SLURM_JOBID",
            "SLURM_STEP_ID",
            "SLURM_PROCID",
            "SLURM_GPUS",
            "SLURM_JOB_GPUS",
        )
    )


def _use_gputil_checks(args):
    if args.no_gputil:
        return False
    if args.force_gputil_check:
        return True
    return not _in_slurm_allocation()


def _detect_gpus_with_gputil():
    if GPUtil is None:
        return None
    gpus = GPUtil.getGPUs()
    return [str(gpu.id) for gpu in gpus]


def _detect_gpus_with_nvidia_smi():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _detect_gpus(args):
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices and visible_devices.strip() not in {"", "-1"}:
        return _split_csv(visible_devices)

    if not args.no_gputil:
        gpus = _detect_gpus_with_gputil()
        if gpus:
            return gpus
        if args.require_gputil:
            raise RuntimeError("GPUtil is required but is not installed or did not report any GPUs.")

    try:
        return _detect_gpus_with_nvidia_smi()
    except (FileNotFoundError, subprocess.CalledProcessError) as err:
        raise RuntimeError(
            "Could not detect available GPUs. Install GPUtil, or pass --gpus 0,1,... or --num-gpus N."
        ) from err


def _resolve_gpus(args):
    if args.gpus is not None:
        gpus = _split_csv(args.gpus)
    elif args.num_gpus is not None:
        if args.num_gpus < 1:
            raise ValueError("--num-gpus must be >= 1.")
        gpus = [str(idx) for idx in range(args.num_gpus)]
    else:
        gpus = _detect_gpus(args)

    if not gpus:
        raise ValueError("At least one GPU is required.")
    return gpus


def _discover_objects(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    objects = []
    for path in sorted(data_dir.iterdir()):
        if path.is_dir() and (path / "keyframes").is_dir():
            objects.append(path.name)
    return objects


def _select_objects(args):
    data_dir = Path(args.data_dir)
    if args.objects:
        objects = args.objects
        missing = [name for name in objects if not (data_dir / name).is_dir()]
        if missing:
            raise FileNotFoundError(f"Object folders not found under {data_dir}: {missing}")
    else:
        objects = _discover_objects(data_dir)

    if args.exclude:
        excluded = set(args.exclude)
        objects = [name for name in objects if name not in excluded]

    if not objects:
        raise ValueError(f"No objects selected under {data_dir}.")
    return objects


def _format_duration(seconds):
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes}m{seconds:02d}s"


def _register_process(object_name, process):
    with ACTIVE_PROCESS_LOCK:
        ACTIVE_PROCESSES[object_name] = process


def _unregister_process(object_name):
    with ACTIVE_PROCESS_LOCK:
        ACTIVE_PROCESSES.pop(object_name, None)


def _terminate_active_processes():
    with ACTIVE_PROCESS_LOCK:
        processes = list(ACTIVE_PROCESSES.items())

    for object_name, process in processes:
        if process.poll() is None:
            _print(f"[terminate] object={object_name} pid={process.pid}")
            process.terminate()

    deadline = time.time() + 30.0
    for object_name, process in processes:
        if process.poll() is not None:
            continue
        timeout = max(0.0, deadline - time.time())
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _print(f"[kill] object={object_name} pid={process.pid}")
            process.kill()
            process.wait()


def _pipeline_command(args, object_name, pipeline_overrides):
    root = _repo_root()
    pipeline_script = Path(args.pipeline_script)
    if not pipeline_script.is_absolute():
        pipeline_script = root / pipeline_script
    return [sys.executable, str(pipeline_script), f"object={object_name}", *pipeline_overrides]


def _get_gputil_gpu(gpu_id):
    if GPUtil is None:
        return None
    for gpu in GPUtil.getGPUs():
        if str(gpu.id) == str(gpu_id):
            return gpu
    return None


def _gpu_is_available(gpu_id, args):
    if not _use_gputil_checks(args):
        return True, "GPUtil checks disabled"
    if GPUtil is None:
        if args.require_gputil:
            raise RuntimeError("GPUtil is required but is not installed.")
        return True, "GPUtil unavailable"

    gpu = _get_gputil_gpu(gpu_id)
    if gpu is None:
        if args.require_gputil:
            raise RuntimeError(f"GPUtil did not report GPU {gpu_id}.")
        return True, f"GPU {gpu_id} not reported by GPUtil"

    load_ok = gpu.load <= args.max_gpu_load
    memory_ok = gpu.memoryUtil <= args.max_gpu_memory
    reason = f"load={gpu.load:.2f}/{args.max_gpu_load:.2f}, memory={gpu.memoryUtil:.2f}/{args.max_gpu_memory:.2f}"
    return load_ok and memory_ok, reason


def _wait_until_gpu_available(gpu_id, args, stop_event):
    if not _use_gputil_checks(args):
        return

    last_message_at = 0.0
    while not stop_event.is_set():
        available, reason = _gpu_is_available(gpu_id, args)
        if available:
            return

        now = time.time()
        if now - last_message_at >= args.status_interval:
            _print(f"[wait] gpu={gpu_id} unavailable ({reason}); checking again in {args.gpu_wait_interval}s")
            last_message_at = now
        time.sleep(args.gpu_wait_interval)


def _run_object(object_name, gpu_queue, args, pipeline_overrides, log_dir, stop_event):
    if stop_event.is_set():
        return {
            "object": object_name,
            "gpu": None,
            "returncode": None,
            "duration": "0m00s",
            "log_path": None,
            "status": "skipped",
        }

    gpu = gpu_queue.get()
    try:
        if stop_event.is_set():
            return {
                "object": object_name,
                "gpu": gpu,
                "returncode": None,
                "duration": "0m00s",
                "log_path": None,
                "status": "skipped",
            }

        _wait_until_gpu_available(gpu, args, stop_event)
        if stop_event.is_set():
            return {
                "object": object_name,
                "gpu": gpu,
                "returncode": None,
                "duration": "0m00s",
                "log_path": None,
                "status": "skipped",
            }

        root = _repo_root()
        command = _pipeline_command(args, object_name, pipeline_overrides)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        log_path = log_dir / f"{object_name}.log"
        started_at = time.time()
        with open(log_path, "w") as log_file:
            log_file.write(f"started_at: {datetime.now().isoformat(timespec='seconds')}\n")
            log_file.write(f"object: {object_name}\n")
            log_file.write(f"gpu: {gpu}\n")
            log_file.write(f"cwd: {root}\n")
            log_file.write(f"command: {shlex.join(command)}\n\n")
            log_file.flush()

            _print(f"[start] object={object_name} gpu={gpu} log={log_path}")
            process = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            _register_process(object_name, process)
            try:
                returncode = process.wait()
            finally:
                _unregister_process(object_name)

            duration = _format_duration(time.time() - started_at)
            log_file.write(f"\nfinished_at: {datetime.now().isoformat(timespec='seconds')}\n")
            log_file.write(f"returncode: {returncode}\n")
            log_file.write(f"duration: {duration}\n")

        status = "ok" if returncode == 0 else "failed"
        _print(
            f"[{status}] object={object_name} gpu={gpu} "
            f"returncode={returncode} duration={duration} log={log_path}"
        )
        return {
            "object": object_name,
            "gpu": gpu,
            "returncode": returncode,
            "duration": duration,
            "log_path": log_path,
            "status": status,
        }
    finally:
        gpu_queue.put(gpu)


def run_batch(args):
    gpus = _resolve_gpus(args)
    objects = _select_objects(args)
    pipeline_overrides = list(OmegaConf.to_container(args.pipeline_overrides, resolve=True) or [])

    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = _repo_root() / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    _print(f"Selected objects ({len(objects)}): {', '.join(objects)}")
    _print(f"GPU slots ({len(gpus)}): {', '.join(gpus)}")
    _print(f"Worker count: {min(len(objects), len(gpus))}")
    if _in_slurm_allocation():
        _print("Slurm allocation detected.")
    if _in_slurm_allocation() and not args.force_gputil_check and not args.no_gputil:
        _print("GPUtil load/memory waiting is disabled under Slurm; use --force-gputil-check to enable it.")
    if GPUtil is None and not args.no_gputil:
        _print("GPUtil is not installed; using exclusive GPU slots without load/memory availability checks.")
    if pipeline_overrides:
        _print(f"Pipeline overrides: {shlex.join(pipeline_overrides)}")
    _print(f"Log directory: {log_dir}")

    if args.dry_run:
        for idx, object_name in enumerate(objects):
            gpu = gpus[idx % len(gpus)]
            command = _pipeline_command(args, object_name, pipeline_overrides)
            _print(f"[dry-run] gpu={gpu} CUDA_VISIBLE_DEVICES={gpu} {shlex.join(command)}")
        return 0

    gpu_queue = queue.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)

    stop_event = threading.Event()
    max_workers = min(len(objects), len(gpus))
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_object = {
            executor.submit(_run_object, object_name, gpu_queue, args, pipeline_overrides, log_dir, stop_event): object_name
            for object_name in objects
        }

        try:
            for future in as_completed(future_to_object):
                result = future.result()
                results.append(result)
                if result["status"] == "failed" and args.stop_on_failure:
                    stop_event.set()
        except KeyboardInterrupt:
            stop_event.set()
            _print("Interrupted. Terminating active reconstruction subprocesses.")
            _terminate_active_processes()
            raise

    failed = [result for result in results if result["status"] == "failed"]
    succeeded = [result for result in results if result["status"] == "ok"]
    skipped = [result for result in results if result["status"] == "skipped"]

    _print("Batch summary:")
    _print(f"  succeeded: {len(succeeded)}")
    _print(f"  failed   : {len(failed)}")
    _print(f"  skipped  : {len(skipped)}")
    for result in failed:
        _print(f"  failed object={result['object']} log={result['log_path']}")
    if skipped:
        _print(f"  skipped objects: {', '.join(result['object'] for result in skipped)}")

    return 1 if failed else 0


@hydra.main(version_base=None, config_path="configs", config_name="batch")
def main(cfg: DictConfig):
    raise SystemExit(run_batch(cfg))


if __name__ == "__main__":
    main()
