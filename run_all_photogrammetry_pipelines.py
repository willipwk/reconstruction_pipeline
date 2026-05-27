import argparse
import os
import shlex
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path


def _repo_root():
    return Path(__file__).resolve().parent


def _split_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _detect_gpus():
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices and visible_devices.strip() not in {"", "-1"}:
        return _split_csv(visible_devices)

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as err:
        raise RuntimeError(
            "Could not detect available GPUs. Pass --gpus 0,1,... or --num-gpus N explicitly."
        ) from err

    gpus = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not gpus:
        raise RuntimeError("No GPUs were reported by nvidia-smi.")
    return gpus


def _resolve_gpus(args):
    if args.gpus:
        gpus = _split_csv(args.gpus)
    elif args.num_gpus is not None:
        if args.num_gpus < 1:
            raise ValueError("--num-gpus must be >= 1.")
        gpus = [str(idx) for idx in range(args.num_gpus)]
    else:
        gpus = _detect_gpus()

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


def _strip_remainder_separator(args):
    if args and args[0] == "--":
        return args[1:]
    return args


def _format_duration(seconds):
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes}m{seconds:02d}s"


def _start_job(*, object_name, gpu, args, pipeline_args, log_dir):
    root = _repo_root()
    pipeline_script = Path(args.pipeline_script)
    if not pipeline_script.is_absolute():
        pipeline_script = root / pipeline_script

    command = [sys.executable, str(pipeline_script), object_name, *pipeline_args]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    log_path = log_dir / f"{object_name}.log"
    log_file = open(log_path, "w")
    started_at = time.time()
    log_file.write(f"started_at: {datetime.now().isoformat(timespec='seconds')}\n")
    log_file.write(f"object: {object_name}\n")
    log_file.write(f"gpu: {gpu}\n")
    log_file.write(f"cwd: {root}\n")
    log_file.write(f"command: {shlex.join(command)}\n\n")
    log_file.flush()

    process = subprocess.Popen(
        command,
        cwd=root,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print(f"[start] object={object_name} gpu={gpu} pid={process.pid} log={log_path}")
    return {
        "object": object_name,
        "gpu": gpu,
        "process": process,
        "log_file": log_file,
        "log_path": log_path,
        "started_at": started_at,
    }


def _finish_job(job):
    returncode = job["process"].returncode
    duration = _format_duration(time.time() - job["started_at"])
    job["log_file"].write(f"\nfinished_at: {datetime.now().isoformat(timespec='seconds')}\n")
    job["log_file"].write(f"returncode: {returncode}\n")
    job["log_file"].write(f"duration: {duration}\n")
    job["log_file"].close()

    status = "ok" if returncode == 0 else "failed"
    print(
        f"[{status}] object={job['object']} gpu={job['gpu']} "
        f"returncode={returncode} duration={duration} log={job['log_path']}"
    )
    return {
        "object": job["object"],
        "gpu": job["gpu"],
        "returncode": returncode,
        "duration": duration,
        "log_path": job["log_path"],
    }


def run_batch(args):
    gpus = _resolve_gpus(args)
    objects = _select_objects(args)
    pipeline_args = _strip_remainder_separator(args.pipeline_args)

    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = _repo_root() / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selected objects ({len(objects)}): {', '.join(objects)}")
    print(f"GPU slots ({len(gpus)}): {', '.join(gpus)}")
    if pipeline_args:
        print(f"Pipeline args: {shlex.join(pipeline_args)}")
    print(f"Log directory: {log_dir}")

    if args.dry_run:
        root = _repo_root()
        pipeline_script = Path(args.pipeline_script)
        if not pipeline_script.is_absolute():
            pipeline_script = root / pipeline_script
        for idx, object_name in enumerate(objects):
            gpu = gpus[idx % len(gpus)]
            command = [sys.executable, str(pipeline_script), object_name, *pipeline_args]
            print(f"[dry-run] gpu={gpu} CUDA_VISIBLE_DEVICES={gpu} {shlex.join(command)}")
        return 0

    pending = deque(objects)
    available_gpus = deque(gpus)
    active = []
    results = []
    stop_scheduling = False

    try:
        while pending or active:
            while pending and available_gpus and not stop_scheduling:
                gpu = available_gpus.popleft()
                object_name = pending.popleft()
                active.append(
                    _start_job(
                        object_name=object_name,
                        gpu=gpu,
                        args=args,
                        pipeline_args=pipeline_args,
                        log_dir=log_dir,
                    )
                )

            time.sleep(args.poll_interval)

            still_active = []
            for job in active:
                returncode = job["process"].poll()
                if returncode is None:
                    still_active.append(job)
                    continue

                results.append(_finish_job(job))
                available_gpus.append(job["gpu"])
                if returncode != 0 and args.stop_on_failure:
                    stop_scheduling = True

            active = still_active

            if stop_scheduling and not active:
                break
    except KeyboardInterrupt:
        print("Interrupted; terminating active jobs.")
        for job in active:
            job["process"].terminate()
        for job in active:
            try:
                job["process"].wait(timeout=30)
            except subprocess.TimeoutExpired:
                job["process"].kill()
            results.append(_finish_job(job))
        raise

    skipped = list(pending)
    failed = [result for result in results if result["returncode"] != 0]
    succeeded = [result for result in results if result["returncode"] == 0]

    print("Batch summary:")
    print(f"  succeeded: {len(succeeded)}")
    print(f"  failed   : {len(failed)}")
    print(f"  skipped  : {len(skipped)}")
    if failed:
        for result in failed:
            print(f"  failed object={result['object']} log={result['log_path']}")
    if skipped:
        print(f"  skipped objects: {', '.join(skipped)}")

    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run run_photogrammetry_pipeline.py on multiple data/<object> folders, "
            "assigning one GPU to one object at a time."
        )
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing object folders. Defaults to data/.",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        help="Specific object names to process. Defaults to every data/* folder containing keyframes/.",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        help="Object names to skip when auto-discovering or using --objects.",
    )
    parser.add_argument(
        "--gpus",
        help="Comma-separated GPU IDs to use, for example 0,1,2,3.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        help="Use GPU IDs 0..N-1. Ignored when --gpus is provided.",
    )
    parser.add_argument(
        "--pipeline-script",
        default="run_photogrammetry_pipeline.py",
        help="Pipeline script path. Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/photogrammetry",
        help="Directory for per-object logs.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between worker status checks.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Do not start new objects after the first failure. Already-running jobs are allowed to finish.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print scheduled commands without running them.",
    )
    parser.add_argument(
        "pipeline_args",
        nargs=argparse.REMAINDER,
        help="Arguments after '--' are forwarded to run_photogrammetry_pipeline.py.",
    )

    args = parser.parse_args()
    raise SystemExit(run_batch(args))


if __name__ == "__main__":
    main()
