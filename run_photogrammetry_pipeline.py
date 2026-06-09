import json
import os
import shutil
import subprocess
from pathlib import Path

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from preprocess_polycam import preprocess_polycam


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
DEPTH_ANYTHING_ENCODER_CONFIGS = {
    "vits": "depth-anything/Depth-Anything-V2-Small-hf",
    "vitb": "depth-anything/Depth-Anything-V2-Base-hf",
    "vitl": "depth-anything/Depth-Anything-V2-Large-hf",
}


def _repo_root():
    return Path(__file__).resolve().parent


def _run(command, cwd):
    command = _normalize_subprocess_command(command)
    print(f"[run] cwd={cwd}", flush=True)
    print("[run] " + " ".join(str(part) for part in command), flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(command, cwd=cwd, check=True, env=env)


def _normalize_subprocess_command(command):
    command = [str(part) for part in command]
    if (
        len(command) >= 2
        and command[0] == "conda"
        and command[1] == "run"
        and "--no-capture-output" not in command
        and "--live-stream" not in command
    ):
        return [command[0], command[1], "--no-capture-output", *command[2:]]
    return command


def _safe_symlink(target, link, force=False):
    target = target.resolve()
    link = Path(link)
    if link.is_symlink():
        current = link.resolve()
        if current == target:
            return
        if not force:
            raise FileExistsError(f"{link} already points to {current}; set force_links=true to replace it.")
        link.unlink()
    elif link.exists():
        if not force:
            raise FileExistsError(f"{link} already exists and is not a symlink; set force_links=true to replace it.")
        if link.is_dir():
            raise IsADirectoryError(f"Refusing to replace real directory: {link}")
        link.unlink()

    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=target.is_dir())


def _select_image_camera_dirs(keyframes_dir):
    corrected_images = keyframes_dir / "corrected_images"
    corrected_cameras = keyframes_dir / "corrected_cameras"
    if corrected_images.exists() and corrected_cameras.exists():
        return corrected_images, corrected_cameras
    return keyframes_dir / "images", keyframes_dir / "cameras"


def _files_by_stem(directory, extensions):
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in extensions
    }


def _write_mpsfm_intrinsics(image_dir, camera_dir, output_path):
    images = _files_by_stem(image_dir, IMAGE_EXTENSIONS)
    cameras = _files_by_stem(camera_dir, (".json",))
    frame_ids = sorted(set(images) & set(cameras))
    if not frame_ids:
        raise ValueError(f"No matching image/camera pairs found in {image_dir} and {camera_dir}")

    intrinsics = {}
    for idx, frame_id in enumerate(frame_ids, start=1):
        with open(cameras[frame_id], "r") as f:
            camera = json.load(f)
        intrinsics[idx] = {
            "params": [
                float(camera["fx"]),
                float(camera["fy"]),
                float(camera["cx"]),
                float(camera["cy"]),
            ],
            "images": [images[frame_id].name],
        }

    with open(output_path, "w") as f:
        yaml.safe_dump(intrinsics, f, sort_keys=False)

    return output_path


def _mpsfm_reconstruction_exists(reconstruction_dir):
    reconstruction_dir = Path(reconstruction_dir)
    binary_files = [reconstruction_dir / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
    text_files = [reconstruction_dir / name for name in ("cameras.txt", "images.txt", "points3D.txt")]
    return all(path.exists() for path in binary_files) or all(path.exists() for path in text_files)


def _find_mpsfm_reconstruction_dir(keyframes_dir):
    return _find_colmap_reconstruction_dir(Path(keyframes_dir) / "sfm_outputs")


def _find_colmap_reconstruction_dir(root_dir):
    root_dir = Path(root_dir)
    if not root_dir.exists():
        return None

    candidates = [root_dir / "gluemap_aba", root_dir / "rec", root_dir]
    candidates.extend(path for path in sorted(root_dir.iterdir()) if path.is_dir())
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if _mpsfm_reconstruction_exists(candidate):
            return candidate
    return None


def _run_gluemap_reconstruction(
    *,
    gluemap_dir,
    image_dir,
    output_dir,
    config_path,
    intrinsics_mode,
    command,
    extra_args=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = gluemap_dir / config_path

    _run(
        [
            *command,
            "--config",
            str(config_path),
            "--images_path",
            str(image_dir),
            "--intrinsics_mode",
            intrinsics_mode,
            "--write_path",
            str(output_dir),
            *(extra_args or []),
        ],
        cwd=gluemap_dir,
    )

    reconstruction_dir = _find_colmap_reconstruction_dir(output_dir)
    if reconstruction_dir is None:
        raise FileNotFoundError(f"Expected GlueMap COLMAP reconstruction under: {output_dir}")
    return reconstruction_dir


def _resolve_gluemap_command(command):
    if command:
        return command
    if shutil.which("gluemap-demo"):
        return ["gluemap-demo"]
    return ["python", "demo.py"]


def _resolve_mpsfm_command(command):
    if command:
        return command
    return ["python"]


def _ensure_geosvr_sparse_layout(keyframes_dir, reconstruction_dir=None, force=False):
    sfm_rec = reconstruction_dir or _find_mpsfm_reconstruction_dir(keyframes_dir)
    if sfm_rec is None:
        raise FileNotFoundError(f"Expected MPSfM reconstruction under: {Path(keyframes_dir) / 'sfm_outputs'}")

    sparse_dir = keyframes_dir / "sparse"
    sparse_zero = sparse_dir / "0"
    _safe_symlink(sfm_rec, sparse_zero, force=force)
    return sparse_zero


def _ensure_pgsr_data_layout(pgsr_data_dir, image_dir, reconstruction_dir, force=False):
    pgsr_data_dir = Path(pgsr_data_dir)
    if pgsr_data_dir.exists() and not pgsr_data_dir.is_dir():
        raise NotADirectoryError(f"Expected PGSR data path to be a directory: {pgsr_data_dir}")

    pgsr_data_dir.mkdir(parents=True, exist_ok=True)
    _safe_symlink(image_dir, pgsr_data_dir / image_dir.name, force=force)
    _safe_symlink(reconstruction_dir, pgsr_data_dir / "sparse", force=force)
    (pgsr_data_dir / "depth").mkdir(parents=True, exist_ok=True)
    return pgsr_data_dir


def _depth_maps_complete(image_dir, depth_dir):
    if not Path(depth_dir).exists():
        return False
    images = _files_by_stem(image_dir, IMAGE_EXTENSIONS)
    depths = _files_by_stem(depth_dir, (".png",))
    return bool(images) and set(images).issubset(depths)


def _ensure_pgsr_depth_maps(
    *,
    image_dir,
    depth_dir,
    encoder,
    force=False,
):
    depth_dir = Path(depth_dir)
    if depth_dir.exists() and _depth_maps_complete(image_dir, depth_dir) and not force:
        print(f"[run] PGSR depth maps already exist at {depth_dir}; skipping depth estimation.")
        return depth_dir

    if encoder not in DEPTH_ANYTHING_ENCODER_CONFIGS:
        raise ValueError(f"Unknown Depth Anything encoder: {encoder}")

    depth_dir.mkdir(parents=True, exist_ok=True)
    images = _files_by_stem(image_dir, IMAGE_EXTENSIONS)
    todo_images = [
        image_path
        for _, image_path in sorted(images.items())
        if force or not (depth_dir / f"{image_path.stem}.png").exists()
    ]
    if not todo_images:
        return depth_dir

    print(f"[run] infer PGSR depth for {len(todo_images)} image(s). Saved to {depth_dir}.")

    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = DEPTH_ANYTHING_ENCODER_CONFIGS[encoder]
    image_processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()

    with torch.no_grad():
        for image_path in todo_images:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            inputs = image_processor(images=image, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            depth = outputs["predicted_depth"].unsqueeze(1)
            depth = torch.nn.functional.interpolate(
                depth,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            depth = depth.detach().cpu().numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth = (depth * 255.0).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(depth_dir / f"{image_path.stem}.png"), depth)

    return depth_dir


def _cleanup_mpsfm_cache(keyframes_dir):
    cache_dir = Path(keyframes_dir) / "cache_dir"
    if not cache_dir.exists() and not cache_dir.is_symlink():
        return None

    if cache_dir.name != "cache_dir" or cache_dir.parent.resolve() != Path(keyframes_dir).resolve():
        raise ValueError(f"Refusing to delete unexpected cache path: {cache_dir}")

    print(f"[run] delete MPSfM cache: {cache_dir}")
    if cache_dir.is_symlink():
        cache_dir.unlink()
    else:
        shutil.rmtree(cache_dir)
    return cache_dir


def run_pipeline(
    object_name,
    *,
    sparse_reconstructor="mpsfm",
    renderer="geosvr",
    cfg_path,
    output_path,
    mpsfm_conf,
    mpsfm_command=None,
    gluemap_config="configs/example.yaml",
    gluemap_command=None,
    gluemap_intrinsics_mode="SHARED",
    gluemap_extra_args=None,
    force_preprocess=False,
    force_sfm=False,
    force_links=False,
    force_depth=False,
    cleanup_mpsfm_cache=True,
    skip_preprocess=False,
    skip_sfm=False,
    skip_geosvr=False,
    skip_reconstruction=False,
    simplify_mesh=True,
    simplification_target_reduction=0.9,
    transform_to_polycam=True,
    visualize_alignment=False,
    polycam_mesh_path=None,
    camera_scale=0.05,
    camera_stride=1,
    geosvr_extra_args=None,
    pgsr_extra_args=None,
    pgsr_render_extra_args=None,
    pgsr_max_depth=2.0,
    pgsr_voxel_size=0.005,
    pgsr_num_cluster=1,
    pgsr_use_depth_filter=False,
    depth_anything_encoder="vitl",
    skip_depth_estimation=False,
):
    if sparse_reconstructor not in ("mpsfm", "gluemap"):
        raise ValueError("sparse_reconstructor must be either 'mpsfm' or 'gluemap'.")
    if renderer not in ("geosvr", "pgsr"):
        raise ValueError("renderer must be either 'geosvr' or 'pgsr'.")
    if not 0.0 <= simplification_target_reduction < 1.0:
        raise ValueError("simplification_target_reduction must be in [0.0, 1.0).")
    if camera_stride < 1:
        raise ValueError("camera_stride must be >= 1.")

    root = _repo_root()
    object_dir = root / "data" / object_name
    rotated_keyframes = object_dir / "keyframes_rot"
    mpsfm_dir = root / "third_party" / "mpsfm"
    geosvr_dir = root / "third_party" / "GeoSVR"
    pgsr_dir = root / "third_party" / "pgsr"
    gluemap_dir = root / "third_party" / "gluemap"

    if not object_dir.exists():
        raise FileNotFoundError(f"Polycam object folder not found: {object_dir}")

    if not skip_preprocess:
        preprocess_polycam(
            object_dir,
            output_keyframes_subdir="keyframes_rot",
            force=force_preprocess,
        )

    image_dir, camera_dir = _select_image_camera_dirs(rotated_keyframes)
    if not image_dir.exists():
        raise FileNotFoundError(f"Rotated image directory not found: {image_dir}")
    if not camera_dir.exists():
        raise FileNotFoundError(f"Rotated camera directory not found: {camera_dir}")

    intrinsics_path = rotated_keyframes / "intrinsics.yaml"
    _write_mpsfm_intrinsics(image_dir, camera_dir, intrinsics_path)

    image_subdir = image_dir.name
    mpsfm_link = None
    if sparse_reconstructor == "mpsfm":
        mpsfm_link = mpsfm_dir / "data" / object_name
        _safe_symlink(rotated_keyframes, mpsfm_link, force=force_links)

    mpsfm_reconstruction_dir = _find_mpsfm_reconstruction_dir(rotated_keyframes)
    gluemap_output_dir = rotated_keyframes / "gluemap_outputs"
    gluemap_reconstruction_dir = _find_colmap_reconstruction_dir(gluemap_output_dir)
    sparse_reconstruction_dir = mpsfm_reconstruction_dir if sparse_reconstructor == "mpsfm" else gluemap_reconstruction_dir
    if not skip_sfm:
        if sparse_reconstruction_dir is not None and not force_sfm:
            print(
                f"[run] {sparse_reconstructor} sparse reconstruction already exists at "
                f"{sparse_reconstruction_dir}; skipping."
            )
        elif sparse_reconstructor == "mpsfm":
            if mpsfm_link is None:
                raise RuntimeError("Internal error: MPSfM data link was not prepared.")
            if not mpsfm_dir.exists():
                raise FileNotFoundError(f"MPSfM checkout not found: {mpsfm_dir}")
            _run(
                [
                    *_resolve_mpsfm_command(mpsfm_command),
                    "reconstruct.py",
                    "--data_dir",
                    str(Path("data") / object_name),
                    "--images_dir",
                    str(Path("data") / object_name / image_subdir),
                    "--intrinsics_pth",
                    str(Path("data") / object_name / "intrinsics.yaml"),
                    "--conf",
                    mpsfm_conf,
                ],
                cwd=mpsfm_dir,
            )
            sparse_reconstruction_dir = _find_mpsfm_reconstruction_dir(rotated_keyframes)
        else:
            if not gluemap_dir.exists():
                raise FileNotFoundError(f"GlueMap checkout not found: {gluemap_dir}")
            sparse_reconstruction_dir = _run_gluemap_reconstruction(
                gluemap_dir=gluemap_dir,
                image_dir=image_dir,
                output_dir=gluemap_output_dir,
                config_path=gluemap_config,
                intrinsics_mode=gluemap_intrinsics_mode,
                command=_resolve_gluemap_command(gluemap_command),
                extra_args=gluemap_extra_args,
            )
        if sparse_reconstructor == "mpsfm" and cleanup_mpsfm_cache and sparse_reconstruction_dir is not None:
            _cleanup_mpsfm_cache(rotated_keyframes)
    elif sparse_reconstruction_dir is None:
        expected_dir = rotated_keyframes / ("sfm_outputs" if sparse_reconstructor == "mpsfm" else "gluemap_outputs")
        raise FileNotFoundError(f"Expected {sparse_reconstructor} reconstruction under: {expected_dir}")

    geosvr_link = None
    pgsr_data_dir = None
    if renderer == "geosvr":
        _ensure_geosvr_sparse_layout(rotated_keyframes, reconstruction_dir=sparse_reconstruction_dir, force=force_links)
        geosvr_link = geosvr_dir / "data" / "custom" / object_name
        _safe_symlink(rotated_keyframes, geosvr_link, force=force_links)
    else:
        if not pgsr_dir.exists():
            raise FileNotFoundError(f"PGSR checkout not found: {pgsr_dir}")
        pgsr_data_dir = pgsr_dir / "data" / "custom" / object_name
        _ensure_pgsr_data_layout(
            pgsr_data_dir,
            image_dir=image_dir,
            reconstruction_dir=sparse_reconstruction_dir,
            force=force_links,
        )

    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if renderer == "geosvr":
        mesh_dir = output_path / "mesh" / "tsdf"
        mesh_path = mesh_dir / "tsdf_fusion_post.ply"
    else:
        mesh_dir = output_path / "mesh"
        mesh_path = mesh_dir / "tsdf_fusion_post.ply"

    if skip_geosvr or skip_reconstruction:
        return {
            "rotated_keyframes": rotated_keyframes,
            "mpsfm_data": mpsfm_link,
            "gluemap_output": gluemap_output_dir if sparse_reconstructor == "gluemap" else None,
            "sparse_reconstruction": sparse_reconstruction_dir,
            "geosvr_data": geosvr_link,
            "pgsr_data": pgsr_data_dir,
            "output_path": output_path,
        }

    if mesh_path.exists():
        print(f"[run] {renderer} mesh already exists at {mesh_path}; skipping reconstruction.")
    elif renderer == "geosvr":
        cfg_path = Path(cfg_path)
        if not cfg_path.is_absolute():
            cfg_path = geosvr_dir / cfg_path

        data_path = Path("data") / "custom" / object_name
        extra_args = geosvr_extra_args or []

        _run(
            [
                "python",
                "train.py",
                "--cfg_files",
                str(cfg_path),
                "--source_path",
                str(data_path),
                "--model_path",
                str(output_path),
                "--images",
                image_subdir,
                *extra_args,
            ],
            cwd=geosvr_dir,
        )
        _run(["python", "render.py", str(output_path)], cwd=geosvr_dir)
        env = os.environ.copy()
        env["PYTHONPATH"] = "./" if not env.get("PYTHONPATH") else f"./:{env['PYTHONPATH']}"
        print(f"[run] cwd={geosvr_dir}")
        print(f"[run] PYTHONPATH={env['PYTHONPATH']} python mesh_extract/tsdf_mesh.py {output_path}")
        subprocess.run(
            ["python", "mesh_extract/tsdf_mesh.py", "--voxel_size", "0.005", str(output_path)],
            cwd=geosvr_dir,
            check=True,
            env=env,
        )
    else:
        if not skip_depth_estimation:
            _ensure_pgsr_depth_maps(
                image_dir=image_dir,
                depth_dir=pgsr_data_dir / "depth",
                encoder=depth_anything_encoder,
                force=force_depth,
            )
        elif not _depth_maps_complete(image_dir, pgsr_data_dir / "depth"):
            raise FileNotFoundError(
                f"Expected PGSR depth maps under {pgsr_data_dir / 'depth'} when skip_depth_estimation=true is used."
            )

        pgsr_train_args = pgsr_extra_args or []
        pgsr_render_args = pgsr_render_extra_args or []
        _run(
            [
                "python",
                "train.py",
                "-s",
                str(pgsr_data_dir),
                "-m",
                str(output_path),
                "--images",
                image_subdir,
                *pgsr_train_args,
            ],
            cwd=pgsr_dir,
        )
        render_command = [
            "python",
            "render.py",
            "-m",
            str(output_path),
            "--max_depth",
            str(pgsr_max_depth),
            "--voxel_size",
            str(pgsr_voxel_size),
            "--num_cluster",
            str(pgsr_num_cluster),
        ]
        if pgsr_use_depth_filter:
            render_command.append("--use_depth_filter")
        render_command.extend(pgsr_render_args)
        _run(render_command, cwd=pgsr_dir)

    if not mesh_path.exists():
        raise FileNotFoundError(f"Expected {renderer} mesh not found: {mesh_path}")

    simplified_mesh_path = None
    polycam_alignment = None
    if simplify_mesh:
        simplified_mesh_path = mesh_dir / "tsdf_fusion_post_simplified.ply"

        print(
            "[run] simplify mesh "
            f"{mesh_path} -> {simplified_mesh_path} "
            f"(target_reduction={simplification_target_reduction})"
        )
        from mesh_simplification import simplify_mesh as simplify_mesh_file

        simplify_mesh_file(
            str(mesh_path),
            str(simplified_mesh_path),
            simplification_target_reduction,
        )

    if transform_to_polycam:
        original_keyframes = object_dir / "keyframes"
        _, original_camera_dir = _select_image_camera_dirs(original_keyframes)
        mesh_info_path = object_dir / "mesh_info.json"
        if not original_camera_dir.exists():
            raise FileNotFoundError(f"Original Polycam camera directory not found: {original_camera_dir}")
        if not mesh_info_path.exists():
            raise FileNotFoundError(f"Polycam mesh_info.json not found: {mesh_info_path}")

        mesh_for_alignment = simplified_mesh_path
        if mesh_for_alignment is None:
            mesh_for_alignment = mesh_path

        transformed_mesh_path = mesh_for_alignment.with_name(f"{mesh_for_alignment.stem}_polycam.ply")
        transformed_camera_path = mesh_dir / "camera_pose_polycam.json"
        alignment_path = mesh_dir / "mpsfm_to_polycam_alignment.json"

        print(
            "[run] transform mesh/cameras to Polycam mesh coordinates "
            f"{mesh_for_alignment} -> {transformed_mesh_path}"
        )
        from polycam_alignment import align_mesh_and_cameras_to_polycam

        polycam_alignment = align_mesh_and_cameras_to_polycam(
            reconstruction_dir=sparse_reconstruction_dir,
            polycam_camera_dir=original_camera_dir,
            mesh_info_path=mesh_info_path,
            input_mesh_path=mesh_for_alignment,
            output_mesh_path=transformed_mesh_path,
            output_camera_path=transformed_camera_path,
            output_alignment_path=alignment_path,
        )

        if visualize_alignment:
            print("[run] visualize transformed mesh/cameras and raw Polycam cameras")
            from polycam_alignment import visualize_polycam_alignment

            visualize_polycam_alignment(
                mesh_path=transformed_mesh_path,
                transformed_camera_path=transformed_camera_path,
                polycam_camera_dir=original_camera_dir,
                mesh_info_path=mesh_info_path,
                polycam_mesh_path=polycam_mesh_path,
                camera_scale=camera_scale,
                camera_stride=camera_stride,
            )

    return {
        "rotated_keyframes": rotated_keyframes,
        "mpsfm_data": mpsfm_link,
        "gluemap_output": gluemap_output_dir if sparse_reconstructor == "gluemap" else None,
        "sparse_reconstruction": sparse_reconstruction_dir,
        "geosvr_data": geosvr_link,
        "pgsr_data": pgsr_data_dir,
        "output_path": output_path,
        "mesh": mesh_path,
        "simplified_mesh": simplified_mesh_path,
        "polycam_alignment": polycam_alignment,
    }


def _list_config(value):
    if value is None:
        return []
    return list(OmegaConf.to_container(value, resolve=True))


@hydra.main(version_base=None, config_path="configs", config_name="pipeline")
def main(cfg: DictConfig):
    if cfg.object is None:
        raise ValueError("Set object=<name>, for example: python run_photogrammetry_pipeline.py object=microwave")

    output_path = cfg.output_path
    if output_path is None:
        renderer_dir = "GeoSVR" if cfg.renderer == "geosvr" else "pgsr"
        output_path = Path("third_party") / renderer_dir / "output" / "custom" / cfg.object

    results = run_pipeline(
        cfg.object,
        sparse_reconstructor=cfg.sparse_reconstructor,
        renderer=cfg.renderer,
        cfg_path=cfg.cfg_path,
        output_path=output_path,
        mpsfm_conf=cfg.mpsfm_conf,
        mpsfm_command=_list_config(cfg.mpsfm_command) or None,
        gluemap_config=cfg.gluemap_config,
        gluemap_command=_list_config(cfg.gluemap_command) or None,
        gluemap_intrinsics_mode=cfg.gluemap_intrinsics_mode,
        gluemap_extra_args=_list_config(cfg.gluemap_extra_args),
        force_preprocess=cfg.force_preprocess,
        force_sfm=cfg.force_sfm,
        force_links=cfg.force_links,
        force_depth=cfg.force_depth,
        cleanup_mpsfm_cache=cfg.cleanup_mpsfm_cache,
        skip_preprocess=cfg.skip_preprocess,
        skip_sfm=cfg.skip_sfm,
        skip_geosvr=cfg.skip_geosvr,
        skip_reconstruction=cfg.skip_reconstruction,
        simplify_mesh=cfg.simplify_mesh,
        simplification_target_reduction=cfg.simplification_target_reduction,
        transform_to_polycam=cfg.transform_to_polycam,
        visualize_alignment=cfg.visualize_alignment,
        polycam_mesh_path=cfg.polycam_mesh_path,
        camera_scale=cfg.camera_scale,
        camera_stride=cfg.camera_stride,
        geosvr_extra_args=_list_config(cfg.geosvr_extra_args),
        pgsr_extra_args=_list_config(cfg.pgsr_extra_args),
        pgsr_render_extra_args=_list_config(cfg.pgsr_render_extra_args),
        pgsr_max_depth=cfg.pgsr_max_depth,
        pgsr_voxel_size=cfg.pgsr_voxel_size,
        pgsr_num_cluster=cfg.pgsr_num_cluster,
        pgsr_use_depth_filter=cfg.pgsr_use_depth_filter,
        depth_anything_encoder=cfg.depth_anything_encoder,
        skip_depth_estimation=cfg.skip_depth_estimation,
    )

    print("Pipeline paths:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
