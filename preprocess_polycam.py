import argparse
import json
from pathlib import Path

import cv2
import yaml


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
DEPTH_EXTENSIONS = (".png", ".tif", ".tiff", ".exr")
MARKER_NAME = "rotation_preprocess.yaml"


def _select_polycam_image_camera_subdirs(parent_folder, image_subdir, camera_subdir):
    if image_subdir is None and camera_subdir is None:
        corrected_images = "keyframes/corrected_images"
        corrected_cameras = "keyframes/corrected_cameras"
        if (parent_folder / corrected_images).exists() and (parent_folder / corrected_cameras).exists():
            return corrected_images, corrected_cameras
        return "keyframes/images", "keyframes/cameras"

    if image_subdir is None:
        image_subdir = "keyframes/images"
    if camera_subdir is None:
        camera_subdir = "keyframes/cameras"
    return image_subdir, camera_subdir


def _files_with_extensions(directory, extensions):
    return sorted(
        path
        for path in Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def _rotate_raster_clockwise(input_path, output_path):
    raster = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if raster is None:
        raise ValueError(f"Failed to read raster file: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rotated = cv2.rotate(raster, cv2.ROTATE_90_CLOCKWISE)
    if not cv2.imwrite(str(output_path), rotated):
        raise ValueError(f"Failed to write rotated raster file: {output_path}")


def _rotate_camera_json_clockwise(input_path, output_path):
    with open(input_path, "r") as f:
        camera = json.load(f)

    required = ("width", "height", "fx", "fy", "cx", "cy")
    missing = [key for key in required if key not in camera]
    if missing:
        raise KeyError(f"Camera JSON {input_path} is missing keys: {missing}")

    width = camera["width"]
    height = camera["height"]
    fx = camera["fx"]
    fy = camera["fy"]
    cx = camera["cx"]
    cy = camera["cy"]

    camera["width"] = height
    camera["height"] = width
    camera["fx"] = fy
    camera["fy"] = fx
    camera["cx"] = height - 1 - cy
    camera["cy"] = cx

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(camera, f, indent=2)
        f.write("\n")


def _missing_outputs(input_files, output_dir):
    return [output_dir / input_path.name for input_path in input_files if not (output_dir / input_path.name).exists()]


def _rotate_raster_clockwise_if_needed(input_path, output_path, force=False):
    if output_path.exists() and not force and input_path.resolve() != output_path.resolve():
        return False
    _rotate_raster_clockwise(input_path, output_path)
    return True


def _rotate_camera_json_clockwise_if_needed(input_path, output_path, force=False):
    if output_path.exists() and not force and input_path.resolve() != output_path.resolve():
        return False
    _rotate_camera_json_clockwise(input_path, output_path)
    return True


def _resolve_output_dir(parent_folder, input_subdir, output_keyframes_subdir):
    input_dir = parent_folder / input_subdir
    if output_keyframes_subdir is None:
        return input_dir

    relative_parts = Path(input_subdir).parts
    if len(relative_parts) < 2:
        raise ValueError(
            "Input subdirectories must include the keyframes directory, "
            f"got: {input_subdir}"
        )
    return parent_folder / output_keyframes_subdir / relative_parts[-1]


def preprocess_polycam(
    parent_folder,
    *,
    image_subdir=None,
    depth_subdir="keyframes/depth",
    camera_subdir=None,
    output_keyframes_subdir=None,
    force=False,
):
    """
    Rotate Polycam keyframe images, depth maps, and intrinsics 90 degrees clockwise.

    By default this preprocesses keyframes in place. Pass output_keyframes_subdir
    to write a rotated copy to a different keyframes-like folder.

    Args:
        parent_folder (str | Path): Dataset folder containing keyframes.
        image_subdir (str | None): Image directory relative to parent_folder.
            If omitted, corrected_images is used when both corrected_images and
            corrected_cameras exist; otherwise images is used.
        depth_subdir (str): Depth directory relative to parent_folder.
        camera_subdir (str | None): Camera JSON directory relative to
            parent_folder. If omitted, corrected_cameras is used when both
            corrected_images and corrected_cameras exist; otherwise cameras is used.
        output_keyframes_subdir (str | None): Optional output keyframes directory
            name relative to parent_folder. If omitted, inputs are overwritten.
        force (bool): Reprocess existing outputs instead of reusing them.

    Returns:
        Path: The keyframes directory containing the rotated outputs.
    """
    parent_folder = Path(parent_folder)
    image_subdir, camera_subdir = _select_polycam_image_camera_subdirs(
        parent_folder,
        image_subdir,
        camera_subdir,
    )
    image_dir = parent_folder / image_subdir
    depth_dir = parent_folder / depth_subdir
    camera_dir = parent_folder / camera_subdir

    for directory in (image_dir, depth_dir, camera_dir):
        if not directory.exists():
            raise FileNotFoundError(f"Required input directory not found: {directory}")

    output_image_dir = _resolve_output_dir(parent_folder, image_subdir, output_keyframes_subdir)
    output_depth_dir = _resolve_output_dir(parent_folder, depth_subdir, output_keyframes_subdir)
    output_camera_dir = _resolve_output_dir(parent_folder, camera_subdir, output_keyframes_subdir)
    output_keyframes_dir = output_image_dir.parent
    marker_path = output_keyframes_dir / MARKER_NAME

    image_files = _files_with_extensions(image_dir, IMAGE_EXTENSIONS)
    depth_files = _files_with_extensions(depth_dir, DEPTH_EXTENSIONS)
    camera_files = _files_with_extensions(camera_dir, (".json",))

    missing_outputs = [
        *_missing_outputs(image_files, output_image_dir),
        *_missing_outputs(depth_files, output_depth_dir),
        *_missing_outputs(camera_files, output_camera_dir),
    ]
    if marker_path.exists() and not force and not missing_outputs:
        print(f"Rotated Polycam keyframes already exist at {output_keyframes_dir}; skipping.")
        return output_keyframes_dir

    if output_keyframes_subdir is not None and not force and not missing_outputs:
        print(f"Required rotated outputs already exist at {output_keyframes_dir}; skipping.")
        return output_keyframes_dir

    rotated_images = 0
    for image_path in image_files:
        rotated_images += int(
            _rotate_raster_clockwise_if_needed(
                image_path,
                output_image_dir / image_path.name,
                force=force,
            )
        )

    rotated_depths = 0
    for depth_path in depth_files:
        rotated_depths += int(
            _rotate_raster_clockwise_if_needed(
                depth_path,
                output_depth_dir / depth_path.name,
                force=force,
            )
        )

    rotated_cameras = 0
    for camera_path in camera_files:
        rotated_cameras += int(
            _rotate_camera_json_clockwise_if_needed(
                camera_path,
                output_camera_dir / camera_path.name,
                force=force,
            )
        )

    marker = {
        "operation": "rotate_90_clockwise",
        "images": len(image_files),
        "depths": len(depth_files),
        "cameras": len(camera_files),
        "rotated_images": rotated_images,
        "rotated_depths": rotated_depths,
        "rotated_cameras": rotated_cameras,
        "image_dir": str(output_image_dir),
        "depth_dir": str(output_depth_dir),
        "camera_dir": str(output_camera_dir),
        "source_image_dir": str(image_dir),
        "source_camera_dir": str(camera_dir),
    }
    output_keyframes_dir.mkdir(parents=True, exist_ok=True)
    with open(marker_path, "w") as f:
        yaml.safe_dump(marker, f, sort_keys=False)

    return output_keyframes_dir


def main():
    parser = argparse.ArgumentParser(
        description="Rotate Polycam keyframes 90 degrees clockwise before depth refinement.",
    )
    parser.add_argument(
        "parent_folder",
        nargs="?",
        default="data/test_stove",
        help=(
            "Dataset folder containing keyframes. Uses corrected_images/corrected_cameras "
            "when both exist, otherwise images/cameras."
        ),
    )
    parser.add_argument(
        "--output-keyframes-subdir",
        default=None,
        help="Optional output keyframes folder name. Defaults to in-place preprocessing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess existing rotated outputs instead of skipping them.",
    )

    args = parser.parse_args()
    output_dir = preprocess_polycam(
        args.parent_folder,
        output_keyframes_subdir=args.output_keyframes_subdir,
        force=args.force,
    )
    print(f"Rotated Polycam keyframes saved to {output_dir}")


if __name__ == "__main__":
    main()
