import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml

from preprocess_polycam import preprocess_polycam


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _repo_root():
    return Path(__file__).resolve().parent


def _run(command, cwd):
    print(f"[run] cwd={cwd}")
    print("[run] " + " ".join(str(part) for part in command))
    subprocess.run(command, cwd=cwd, check=True)


def _safe_symlink(target, link, force=False):
    target = target.resolve()
    link = Path(link)
    if link.is_symlink():
        current = link.resolve()
        if current == target:
            return
        if not force:
            raise FileExistsError(f"{link} already points to {current}; pass --force-links to replace it.")
        link.unlink()
    elif link.exists():
        if not force:
            raise FileExistsError(f"{link} already exists and is not a symlink; pass --force-links to replace it.")
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


def _ensure_geosvr_sparse_layout(keyframes_dir, force=False):
    sfm_rec = keyframes_dir / "sfm_outputs" / "rec"
    if not _mpsfm_reconstruction_exists(sfm_rec):
        raise FileNotFoundError(f"Expected MPSfM reconstruction not found: {sfm_rec}")

    sparse_dir = keyframes_dir / "sparse"
    sparse_zero = sparse_dir / "0"
    _safe_symlink(sfm_rec, sparse_zero, force=force)
    return sparse_zero


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
    cfg_path,
    output_path,
    mpsfm_conf,
    force_preprocess=False,
    force_sfm=False,
    force_links=False,
    cleanup_mpsfm_cache=True,
    skip_preprocess=False,
    skip_sfm=False,
    skip_geosvr=False,
    simplify_mesh=True,
    simplification_target_reduction=0.9,
    transform_to_polycam=True,
    visualize_alignment=False,
    polycam_mesh_path=None,
    camera_scale=0.05,
    camera_stride=1,
    geosvr_extra_args=None,
):
    if not 0.0 <= simplification_target_reduction < 1.0:
        raise ValueError("simplification_target_reduction must be in [0.0, 1.0).")
    if camera_stride < 1:
        raise ValueError("camera_stride must be >= 1.")

    root = _repo_root()
    object_dir = root / "data" / object_name
    rotated_keyframes = object_dir / "keyframes_rot"
    mpsfm_dir = root / "mpsfm"
    geosvr_dir = root / "GeoSVR"

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

    mpsfm_link = mpsfm_dir / "data" / object_name
    _safe_symlink(rotated_keyframes, mpsfm_link, force=force_links)

    image_subdir = image_dir.name
    if not skip_sfm:
        sfm_rec = rotated_keyframes / "sfm_outputs" / "rec"
        if _mpsfm_reconstruction_exists(sfm_rec) and not force_sfm:
            print(f"[run] MPSfM sparse reconstruction already exists at {sfm_rec}; skipping.")
        else:
            _run(
                [
                    "python",
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
        if cleanup_mpsfm_cache and _mpsfm_reconstruction_exists(sfm_rec):
            _cleanup_mpsfm_cache(rotated_keyframes)

    _ensure_geosvr_sparse_layout(rotated_keyframes, force=force_links)

    geosvr_link = geosvr_dir / "data" / "custom" / object_name
    _safe_symlink(rotated_keyframes, geosvr_link, force=force_links)

    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    geosvr_mesh_path = output_path / "mesh" / "tsdf" / "tsdf_fusion_post.ply"

    if skip_geosvr:
        return {
            "rotated_keyframes": rotated_keyframes,
            "mpsfm_data": mpsfm_link,
            "geosvr_data": geosvr_link,
            "output_path": output_path,
        }

    cfg_path = Path(cfg_path)
    if not cfg_path.is_absolute():
        cfg_path = geosvr_dir / cfg_path

    data_path = Path("data") / "custom" / object_name
    extra_args = geosvr_extra_args or []

    if geosvr_mesh_path.exists():
        print(f"[run] GeoSVR mesh already exists at {geosvr_mesh_path}; skipping GeoSVR reconstruction.")
    else:
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

    simplified_mesh_path = None
    polycam_alignment = None
    if simplify_mesh:
        mesh_path = geosvr_mesh_path
        simplified_mesh_path = output_path / "mesh" / "tsdf" / "tsdf_fusion_post_simplified.ply"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Expected GeoSVR mesh not found: {mesh_path}")

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
            mesh_for_alignment = output_path / "mesh" / "tsdf" / "tsdf_fusion_post.ply"

        transformed_mesh_path = mesh_for_alignment.with_name(f"{mesh_for_alignment.stem}_polycam.ply")
        transformed_camera_path = output_path / "mesh" / "tsdf" / "camera_pose_polycam.json"
        alignment_path = output_path / "mesh" / "tsdf" / "mpsfm_to_polycam_alignment.json"

        print(
            "[run] transform mesh/cameras to Polycam mesh coordinates "
            f"{mesh_for_alignment} -> {transformed_mesh_path}"
        )
        from polycam_alignment import align_mesh_and_cameras_to_polycam

        polycam_alignment = align_mesh_and_cameras_to_polycam(
            reconstruction_dir=rotated_keyframes / "sfm_outputs" / "rec",
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
        "geosvr_data": geosvr_link,
        "output_path": output_path,
        "simplified_mesh": simplified_mesh_path,
        "polycam_alignment": polycam_alignment,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Polycam -> MPSfM -> GeoSVR photogrammetry pipeline.")
    parser.add_argument("object", help="Object name under data/{object}.")
    parser.add_argument(
        "--cfg-path",
        default="cfg/mipnerf360_mesh.yaml",
        help="GeoSVR config path. Relative paths are resolved from GeoSVR/.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="GeoSVR model output path. Defaults to GeoSVR/output/custom/{object}.",
    )
    parser.add_argument("--mpsfm-conf", default="sp-lg_mogev2", help="MPSfM config name.")
    parser.add_argument("--force-preprocess", action="store_true", help="Reprocess existing rotated Polycam outputs.")
    parser.add_argument("--force-sfm", action="store_true", help="Re-run MPSfM even when sparse reconstruction exists.")
    parser.add_argument("--force-links", action="store_true", help="Replace existing symlinks.")
    parser.add_argument(
        "--keep-mpsfm-cache",
        action="store_true",
        help="Keep keyframes_rot/cache_dir after MPSfM finishes.",
    )
    parser.add_argument("--skip-preprocess", action="store_true", help="Use existing data/{object}/keyframes_rot.")
    parser.add_argument("--skip-sfm", action="store_true", help="Use existing keyframes_rot/sfm_outputs/rec.")
    parser.add_argument("--skip-geosvr", action="store_true", help="Stop after preparing GeoSVR data/sparse layout.")
    parser.add_argument("--no-simplify-mesh", action="store_true", help="Skip mesh simplification after TSDF extraction.")
    parser.add_argument(
        "--no-transform-to-polycam",
        action="store_true",
        help="Skip transforming the final mesh and camera poses back to Polycam mesh coordinates.",
    )
    parser.add_argument(
        "--visualize-polycam-alignment",
        action="store_true",
        help="Open an Open3D view of the transformed mesh, transformed cameras, and raw Polycam cameras.",
    )
    parser.add_argument(
        "--polycam-mesh-path",
        default=None,
        help="Optional raw Polycam mesh path to overlay in the alignment visualization.",
    )
    parser.add_argument(
        "--camera-scale",
        type=float,
        default=0.05,
        help="Camera frustum scale for the alignment visualization.",
    )
    parser.add_argument(
        "--camera-stride",
        type=int,
        default=1,
        help="Draw every Nth camera in the alignment visualization.",
    )
    parser.add_argument(
        "--simplification-target-reduction",
        type=float,
        default=0.5,
        help="Fraction of mesh triangles to remove during simplification.",
    )
    parser.add_argument(
        "--geosvr-arg",
        action="append",
        default=[],
        help="Extra argument token appended to GeoSVR train.py. Repeat for multiple tokens.",
    )

    args = parser.parse_args()
    output_path = args.output_path
    if output_path is None:
        output_path = Path("GeoSVR") / "output" / "custom" / args.object

    results = run_pipeline(
        args.object,
        cfg_path=args.cfg_path,
        output_path=output_path,
        mpsfm_conf=args.mpsfm_conf,
        force_preprocess=args.force_preprocess,
        force_sfm=args.force_sfm,
        force_links=args.force_links,
        cleanup_mpsfm_cache=not args.keep_mpsfm_cache,
        skip_preprocess=args.skip_preprocess,
        skip_sfm=args.skip_sfm,
        skip_geosvr=args.skip_geosvr,
        simplify_mesh=not args.no_simplify_mesh,
        simplification_target_reduction=args.simplification_target_reduction,
        transform_to_polycam=not args.no_transform_to_polycam,
        visualize_alignment=args.visualize_polycam_alignment,
        polycam_mesh_path=args.polycam_mesh_path,
        camera_scale=args.camera_scale,
        camera_stride=args.camera_stride,
        geosvr_extra_args=args.geosvr_arg,
    )

    print("Pipeline paths:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
