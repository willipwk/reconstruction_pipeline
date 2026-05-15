import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pycolmap


# Basis change between a Polycam camera (x-right, y-up, z-out-of-screen) and an
# OpenCV camera (x-right, y-down, z-into-screen): flip y and z. A point with
# coordinates p_opencv in opencv-camera basis equals diag(1, -1, -1) @ p_opencv
# in polycam-camera basis. The matrix is its own inverse.
# Matches the `xyz * [1, -1, -1]` step in align_ego_camera.py:scan2mesh.
POLYCAM_FROM_OPENCV_CAMERA = np.diag([1.0, -1.0, -1.0])
OPENCV_FROM_POLYCAM_CAMERA = POLYCAM_FROM_OPENCV_CAMERA  # self-inverse

# The pipeline rotates Polycam images 90 degrees clockwise before running MPSfM.
# COLMAP therefore estimates poses in the rotated-image OpenCV camera basis:
#   x_rot = -y_orig, y_rot = x_orig, z_rot = z_orig.
# On the pose side, converting rotated OpenCV camera coordinates back into the
# original Polycam camera basis is:
#   polycam = POLYCAM_FROM_OPENCV_CAMERA @ ORIGINAL_OPENCV_FROM_ROTATED_CW_OPENCV @ rotated_opencv
ORIGINAL_OPENCV_FROM_ROTATED_CW_OPENCV = np.array([
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)
POLYCAM_FROM_ROTATED_CW_OPENCV_CAMERA = (
    POLYCAM_FROM_OPENCV_CAMERA @ ORIGINAL_OPENCV_FROM_ROTATED_CW_OPENCV
)
ROTATED_CW_OPENCV_FROM_POLYCAM_CAMERA = POLYCAM_FROM_ROTATED_CW_OPENCV_CAMERA.T

# Polycam's exported .glb mesh lives in a frame that is rotated +90 degrees
# about the X axis relative to the frame produced by mesh_info.alignmentTransform.
# See align_ego_camera.py:scan2mesh, which applies this rotation after
# alignmentTransform to land scan points in the .glb mesh frame.
POLYCAM_MESH_FROM_ALIGNED = np.array([
    [1, 0,  0, 0],
    [0, 0, -1, 0],
    [0, 1,  0, 0],
    [0, 0,  0, 1],
], dtype=np.float64)


def _files_by_stem(directory, suffix):
    return {
        path.stem: path
        for path in sorted(Path(directory).iterdir())
        if path.is_file() and path.suffix.lower() == suffix
    }


def _polycam_camera_to_world(camera_path):
    with open(camera_path, "r") as f:
        camera = json.load(f)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :4] = np.array(
        [
            [camera["t_00"], camera["t_01"], camera["t_02"], camera["t_03"]],
            [camera["t_10"], camera["t_11"], camera["t_12"], camera["t_13"]],
            [camera["t_20"], camera["t_21"], camera["t_22"], camera["t_23"]],
        ],
        dtype=np.float64,
    )
    return transform


def _load_mesh_info(mesh_info_path):
    if mesh_info_path is None:
        return np.eye(4, dtype=np.float64), None

    with open(mesh_info_path, "r") as f:
        mesh_info = json.load(f)

    values = mesh_info.get("alignmentTransform")
    if values is None:
        alignment_transform = np.eye(4, dtype=np.float64)
    else:
        if len(values) != 16:
            raise ValueError(f"alignmentTransform must have 16 values: {mesh_info_path}")

        # Polycam/SceneKit-style transforms are stored column-major, with translation
        # at indices 12:15. We use column-vector math everywhere else in this file.
        alignment_transform = np.array(values, dtype=np.float64).reshape(4, 4).T

    # The .glb mesh lives in the +90deg-about-X rotation of the alignmentTransform
    # frame (matches align_ego_camera.py:scan2mesh).
    transform = POLYCAM_MESH_FROM_ALIGNED @ alignment_transform

    # bboxCenter is reported by Polycam in the alignmentTransform-aligned frame,
    # which is also the frame the final aligned mesh ends up in (after the
    # post-rotation in align_mesh_and_cameras_to_polycam undoes POLYCAM_MESH_FROM_ALIGNED).
    bbox_center = mesh_info.get("bboxCenter")
    if bbox_center is not None:
        bbox_center = np.array(bbox_center, dtype=np.float64)
    return transform, bbox_center


def _load_mesh_info_transform(mesh_info_path):
    transform, _ = _load_mesh_info(mesh_info_path)
    return transform


def _mesh_bbox_center(mesh_path, transform=None):
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise ValueError(f"Input mesh is empty or unreadable: {mesh_path}")

    vertices = np.asarray(mesh.vertices)
    if transform is not None:
        vertices = _apply_transform(vertices, transform)
    return 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))


def _similarity_umeyama(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both have shape (N, 3).")
    if len(source) < 3:
        raise ValueError("At least three matched camera centers are required for alignment.")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vh = np.linalg.svd(covariance)

    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vh) < 0:
        sign[-1] = -1.0

    rotation = u @ np.diag(sign) @ vh
    source_variance = np.sum(source_centered**2) / len(source)
    scale = float(np.sum(singular_values * sign) / source_variance)
    translation = target_mean - scale * rotation @ source_mean

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform, scale, rotation, translation


def _average_rotation_from_pose_pairs(source_rotations, target_rotations):
    accumulator = np.zeros((3, 3), dtype=np.float64)
    for source_rotation, target_rotation in zip(source_rotations, target_rotations):
        accumulator += target_rotation @ source_rotation.T

    u, _, vh = np.linalg.svd(accumulator)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return rotation


def _similarity_from_pose_pairs(source_poses, target_poses):
    source_centers = np.stack([pose[:3, 3] for pose in source_poses])
    target_centers = np.stack([pose[:3, 3] for pose in target_poses])
    source_rotations = [pose[:3, :3] for pose in source_poses]
    target_rotations = [pose[:3, :3] for pose in target_poses]

    rotation = _average_rotation_from_pose_pairs(source_rotations, target_rotations)

    source_mean = source_centers.mean(axis=0)
    target_mean = target_centers.mean(axis=0)
    source_centered = source_centers - source_mean
    target_centered = target_centers - target_mean
    rotated_source = source_centered @ rotation.T

    scale = float(np.sum(target_centered * rotated_source) / np.sum(source_centered**2))
    translation = target_mean - scale * rotation @ source_mean

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform, scale, rotation, translation


def _apply_transform(points, transform):
    points_h = np.concatenate([points, np.ones((len(points), 1), dtype=points.dtype)], axis=1)
    return (transform @ points_h.T).T[:, :3]


def _read_colmap_camera_poses(reconstruction_dir):
    reconstruction_dir = Path(reconstruction_dir)
    binary_files = [reconstruction_dir / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
    text_files = [reconstruction_dir / name for name in ("cameras.txt", "images.txt", "points3D.txt")]
    if not all(path.exists() for path in binary_files) and not all(path.exists() for path in text_files):
        raise FileNotFoundError(
            "Expected COLMAP reconstruction files under "
            f"{reconstruction_dir}: cameras/images/points3D as .bin or .txt"
        )

    reconstruction = pycolmap.Reconstruction(reconstruction_dir)
    poses = {}
    centers = {}
    for image in reconstruction.images.values():
        frame_id = Path(image.name).stem
        cam_from_world = np.asarray(image.cam_from_world.matrix(), dtype=np.float64)
        camera_to_world = np.eye(4, dtype=np.float64)
        camera_to_world[:3, :4] = np.linalg.inv(
            np.vstack([cam_from_world, [0.0, 0.0, 0.0, 1.0]])
        )[:3, :4]
        poses[frame_id] = camera_to_world
        centers[frame_id] = np.asarray(image.projection_center(), dtype=np.float64)
    return poses, centers


def _write_transformed_mesh(input_mesh_path, output_mesh_path, transform):
    mesh = o3d.io.read_triangle_mesh(str(input_mesh_path))
    if mesh.is_empty():
        raise ValueError(f"Input mesh is empty or unreadable: {input_mesh_path}")

    vertices = np.asarray(mesh.vertices)
    mesh.vertices = o3d.utility.Vector3dVector(_apply_transform(vertices, transform))

    linear = transform[:3, :3]
    normal_transform = np.linalg.inv(linear).T
    if mesh.has_vertex_normals():
        normals = np.asarray(mesh.vertex_normals)
        normals = normals @ normal_transform.T
        normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-12)
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    if mesh.has_triangle_normals():
        normals = np.asarray(mesh.triangle_normals)
        normals = normals @ normal_transform.T
        normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-12)
        mesh.triangle_normals = o3d.utility.Vector3dVector(normals)

    output_mesh_path = Path(output_mesh_path)
    output_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_mesh_path), mesh)


def _camera_frustum_lines(poses_by_frame, scale, color, stride=1):
    local_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [-0.6 * scale, -0.4 * scale, -scale],
            [0.6 * scale, -0.4 * scale, -scale],
            [0.6 * scale, 0.4 * scale, -scale],
            [-0.6 * scale, 0.4 * scale, -scale],
            [0.0, 0.0, -1.4 * scale],
        ],
        dtype=np.float64,
    )
    local_lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
        [0, 5],
    ]

    points = []
    lines = []
    colors = []
    for idx, (_, pose) in enumerate(sorted(poses_by_frame.items())):
        if idx % stride != 0:
            continue
        base = len(points)
        world_points = _apply_transform(local_points, pose)
        points.extend(world_points.tolist())
        lines.extend([[base + i, base + j] for i, j in local_lines])
        colors.extend([color for _ in local_lines])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return line_set


def _camera_center_links(source_poses, target_poses, color=(1.0, 0.85, 0.0), stride=1):
    points = []
    lines = []
    colors = []
    for idx, frame_id in enumerate(sorted(set(source_poses) & set(target_poses))):
        if idx % stride != 0:
            continue
        base = len(points)
        points.append(source_poses[frame_id][:3, 3])
        points.append(target_poses[frame_id][:3, 3])
        lines.append([base, base + 1])
        colors.append(color)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return line_set


def _load_polycam_mesh(mesh_path):
    mesh_path = Path(mesh_path)
    suffix = mesh_path.suffix.lower()

    if suffix in {".glb", ".gltf"}:
        try:
            import trimesh

            loaded = trimesh.load(str(mesh_path), force="scene", process=False)
            if isinstance(loaded, trimesh.Scene):
                merged = loaded.dump(concatenate=True)
            else:
                merged = loaded
            vertices = np.asarray(merged.vertices, dtype=np.float64)
            faces = np.asarray(merged.faces, dtype=np.int32)
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            if not mesh.is_empty():
                return mesh, "trimesh(scene)"
        except ImportError:
            pass
        except Exception as err:
            print(f"trimesh load failed for {mesh_path}: {err}")

        try:
            model = o3d.io.read_triangle_model(str(mesh_path))
            if model.meshes:
                merged = o3d.geometry.TriangleMesh()
                for entry in model.meshes:
                    merged += o3d.geometry.TriangleMesh(entry.mesh)
                if not merged.is_empty():
                    return merged, "o3d.read_triangle_model"
        except Exception as err:
            print(f"o3d.read_triangle_model failed for {mesh_path}: {err}")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise ValueError(f"Polycam mesh is empty or unreadable: {mesh_path}")
    return mesh, "o3d.read_triangle_mesh"


def _load_transformed_camera_poses(camera_pose_path):
    with open(camera_pose_path, "r") as f:
        camera_data = json.load(f)

    poses = {}
    for frame_id, record in camera_data["cameras"].items():
        poses[frame_id] = np.array(record["mesh_camera_to_world_polycam_camera"], dtype=np.float64)
    return poses


def _load_raw_polycam_camera_poses_in_mesh(polycam_camera_dir, mesh_info_path, frame_ids=None):
    # Match the post-rotation applied in align_mesh_and_cameras_to_polycam so
    # raw Polycam cameras live in the same aligned frame as the saved
    # transformed cameras and mesh.
    mesh_from_polycam_world = _load_mesh_info_transform(mesh_info_path)
    aligned_from_mesh = POLYCAM_MESH_FROM_ALIGNED.T
    polycam_cameras = _files_by_stem(polycam_camera_dir, ".json")
    if frame_ids is not None:
        polycam_cameras = {frame_id: polycam_cameras[frame_id] for frame_id in frame_ids if frame_id in polycam_cameras}

    poses = {}
    for frame_id, camera_path in polycam_cameras.items():
        poses[frame_id] = aligned_from_mesh @ mesh_from_polycam_world @ _polycam_camera_to_world(camera_path)
    return poses


def visualize_polycam_alignment(
    *,
    mesh_path,
    transformed_camera_path,
    polycam_camera_dir,
    mesh_info_path,
    polycam_mesh_path=None,
    camera_scale=0.05,
    camera_stride=1,
):
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise ValueError(f"Input mesh is empty or unreadable: {mesh_path}")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.72, 0.72, 0.72])

    transformed_poses = _load_transformed_camera_poses(transformed_camera_path)
    raw_poses = _load_raw_polycam_camera_poses_in_mesh(
        polycam_camera_dir,
        mesh_info_path,
        frame_ids=set(transformed_poses),
    )

    polycam_cameras = _files_by_stem(polycam_camera_dir, ".json")
    untransformed_polycam_poses = {
        frame_id: _polycam_camera_to_world(polycam_cameras[frame_id])
        for frame_id in transformed_poses
        if frame_id in polycam_cameras
    }

    transformed_frustums = _camera_frustum_lines(
        transformed_poses,
        scale=camera_scale,
        color=(0.0, 0.8, 0.15),
        stride=camera_stride,
    )
    raw_frustums = _camera_frustum_lines(
        raw_poses,
        scale=camera_scale,
        color=(0.95, 0.05, 0.05),
        stride=camera_stride,
    )
    untransformed_frustums = _camera_frustum_lines(
        untransformed_polycam_poses,
        scale=camera_scale,
        color=(1.0, 0.5, 0.0),
        stride=camera_stride,
    )
    center_links = _camera_center_links(
        transformed_poses,
        raw_poses,
        color=(1.0, 0.85, 0.0),
        stride=camera_stride,
    )
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=camera_scale * 2.0)

    geometries = [mesh, transformed_frustums, raw_frustums, untransformed_frustums, center_links, world_frame]

    print("Open3D visualization legend:")
    print("  gray mesh   : transformed GeoSVR mesh (in mesh frame, as computed)")
    print("  green cams  : transformed MPSfM cameras (Polycam camera convention)")
    print("  red cams    : raw Polycam cameras AFTER applying mesh_info.alignmentTransform")
    print("  orange cams : raw Polycam cameras WITHOUT alignmentTransform (raw polycam world)")
    print("  yellow lines: matched camera-center offsets between green and red")

    if polycam_mesh_path is not None:
        polycam_mesh, loader_used = _load_polycam_mesh(polycam_mesh_path)
        if not polycam_mesh.has_vertex_normals():
            polycam_mesh.compute_vertex_normals()

        vertices = np.asarray(polycam_mesh.vertices)
        loaded_center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
        loaded_extent = vertices.max(axis=0) - vertices.min(axis=0)
        mesh_from_polycam_world, mesh_info_bbox_center = _load_mesh_info(mesh_info_path)
        print(f"Polycam mesh diagnostics (loader={loader_used}):")
        print(f"  loaded AABB center        : {loaded_center}")
        print(f"  loaded AABB extent        : {loaded_extent}")
        if mesh_info_bbox_center is not None:
            print(f"  mesh_info bboxCenter      : {mesh_info_bbox_center}")
            print(f"  center diff (loaded-info) : {loaded_center - mesh_info_bbox_center}")
        print(f"  mesh_from_polycam_world t : {mesh_from_polycam_world[:3, 3]}")

        polycam_mesh.paint_uniform_color([0.3, 0.55, 1.0])
        geometries.append(polycam_mesh)
        print("  blue mesh   : raw Polycam mesh (as loaded from --polycam-mesh-path)")

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Polycam alignment: mesh, transformed cameras, raw cameras",
    )


def align_mesh_and_cameras_to_polycam(
    *,
    reconstruction_dir,
    polycam_camera_dir,
    mesh_info_path,
    input_mesh_path,
    output_mesh_path,
    output_camera_path,
    output_alignment_path,
    correct_mesh_center=False,
):
    colmap_camera_to_world, colmap_centers = _read_colmap_camera_poses(reconstruction_dir)
    polycam_cameras = _files_by_stem(polycam_camera_dir, ".json")
    mesh_from_polycam_world, mesh_info_bbox_center = _load_mesh_info(mesh_info_path)

    frame_ids = sorted(set(colmap_centers) & set(polycam_cameras))
    if len(frame_ids) < 3:
        raise ValueError(
            "Need at least three matched cameras to align MPSfM to Polycam; "
            f"found {len(frame_ids)}."
        )

    source_poses = []
    target_poses = []
    polycam_camera_to_mesh = {}
    polycam_rotated_opencv_camera_to_mesh = {}
    for frame_id in frame_ids:
        polycam_pose = _polycam_camera_to_world(polycam_cameras[frame_id])
        # Step 1: convert the raw Polycam camera pose to the same rotated-image
        # OpenCV convention used by MPSfM. The input images were rotated 90deg
        # clockwise before reconstruction, so a plain Polycam->OpenCV basis
        # conversion is not sufficient.
        polycam_pose_rotated_opencv = polycam_pose.copy()
        polycam_pose_rotated_opencv[:3, :3] = (
            polycam_pose[:3, :3] @ POLYCAM_FROM_ROTATED_CW_OPENCV_CAMERA
        )

        # Steps 2-4 are baked into mesh_from_polycam_world:
        #   2. polycam_pose             (camera -> polycam world)
        #   3. alignmentTransform       (polycam world -> aligned mesh frame)
        #   4. POLYCAM_MESH_FROM_ALIGNED (+90 deg about X -> .glb mesh frame)
        polycam_pose_mesh = mesh_from_polycam_world @ polycam_pose
        polycam_pose_rotated_opencv_mesh = mesh_from_polycam_world @ polycam_pose_rotated_opencv
        polycam_camera_to_mesh[frame_id] = polycam_pose_mesh
        polycam_rotated_opencv_camera_to_mesh[frame_id] = polycam_pose_rotated_opencv_mesh

        source_poses.append(colmap_camera_to_world[frame_id])
        target_poses.append(polycam_pose_rotated_opencv_mesh)

    source_centers = np.stack([pose[:3, 3] for pose in source_poses])
    target_centers = np.stack([pose[:3, 3] for pose in target_poses])
    mesh_from_mpsfm, scale, rotation, translation = _similarity_from_pose_pairs(source_poses, target_poses)
    aligned_centers = _apply_transform(source_centers, mesh_from_mpsfm)
    errors = np.linalg.norm(aligned_centers - target_centers, axis=1)
    rotation_errors = []
    for source_pose, target_pose in zip(source_poses, target_poses):
        delta = target_pose[:3, :3].T @ rotation @ source_pose[:3, :3]
        cos_angle = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
        rotation_errors.append(float(np.rad2deg(np.arccos(cos_angle))))
    rotation_errors = np.array(rotation_errors, dtype=np.float64)

    # Mirror align_ego_camera.py:262 -- after estimating the rigid transform in
    # the scan2mesh (+90 deg about X) world, rotate it by -90 deg about X so the
    # final output lives in the alignmentTransform-aligned frame. Apply the same
    # rotation to the stored Polycam reference poses so every saved pose shares
    # one frame.
    aligned_from_mesh = POLYCAM_MESH_FROM_ALIGNED.T
    mesh_from_mpsfm = aligned_from_mesh @ mesh_from_mpsfm
    polycam_camera_to_mesh = {
        frame_id: aligned_from_mesh @ pose for frame_id, pose in polycam_camera_to_mesh.items()
    }
    polycam_rotated_opencv_camera_to_mesh = {
        frame_id: aligned_from_mesh @ pose
        for frame_id, pose in polycam_rotated_opencv_camera_to_mesh.items()
    }

    mesh_center_correction = np.zeros(3, dtype=np.float64)
    mesh_center_before_correction = _mesh_bbox_center(input_mesh_path, mesh_from_mpsfm)
    if correct_mesh_center and mesh_info_bbox_center is not None:
        mesh_center_correction = mesh_info_bbox_center - mesh_center_before_correction
        center_correction_transform = np.eye(4, dtype=np.float64)
        center_correction_transform[:3, 3] = mesh_center_correction
        mesh_from_mpsfm = center_correction_transform @ mesh_from_mpsfm

    _write_transformed_mesh(input_mesh_path, output_mesh_path, mesh_from_mpsfm)
    mesh_center_after_correction = _mesh_bbox_center(output_mesh_path)

    camera_records = {}
    linear = mesh_from_mpsfm[:3, :3]
    linear_rotation = linear / scale
    for frame_id in frame_ids:
        mpsfm_pose = colmap_camera_to_world[frame_id]
        target_pose_colmap_camera = np.eye(4, dtype=np.float64)
        target_pose_colmap_camera[:3, :3] = linear_rotation @ mpsfm_pose[:3, :3]
        target_pose_colmap_camera[:3, 3] = _apply_transform(mpsfm_pose[:3, 3][None], mesh_from_mpsfm)[0]

        target_pose_polycam_camera = target_pose_colmap_camera.copy()
        target_pose_polycam_camera[:3, :3] = (
            target_pose_colmap_camera[:3, :3] @ ROTATED_CW_OPENCV_FROM_POLYCAM_CAMERA
        )

        camera_records[frame_id] = {
            "image_name": f"{frame_id}",
            "mpsfm_camera_to_world_colmap": mpsfm_pose.tolist(),
            "mesh_camera_to_world_colmap_camera": target_pose_colmap_camera.tolist(),
            "mesh_camera_to_world_polycam_camera": target_pose_polycam_camera.tolist(),
            "reference_polycam_camera_to_world_mesh": polycam_camera_to_mesh[frame_id].tolist(),
            "reference_polycam_as_rotated_opencv_camera_to_world_mesh": polycam_rotated_opencv_camera_to_mesh[
                frame_id
            ].tolist(),
            "alignment_error": float(errors[frame_ids.index(frame_id)]),
            "rotation_alignment_error_degrees": float(rotation_errors[frame_ids.index(frame_id)]),
        }

    output_camera_path = Path(output_camera_path)
    output_camera_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_camera_path, "w") as f:
        json.dump(
            {
                "camera_convention": {
                    "polycam": "x-right, y-up, z-out-of-screen",
                    "mpsfm_colmap": "x-right, y-down, z-into-screen",
                    "polycam_to_mpsfm_camera_conversion": (
                        "raw Polycam camera pose is converted to the rotated-image OpenCV/COLMAP "
                        "camera axes via POLYCAM_FROM_ROTATED_CW_OPENCV_CAMERA before estimating "
                        "the relative world transform"
                    ),
                    "world_chain": (
                        "Targets used during similarity fit live in the scan2mesh world: "
                        "ROT90_X(+90deg) @ alignmentTransform @ polycam_pose @ "
                        "POLYCAM_FROM_ROTATED_CW_OPENCV_CAMERA. "
                        "After fitting, mesh_from_mpsfm is post-multiplied by ROT90_X(-90deg) "
                        "(mirroring align_ego_camera.py:262), so saved poses and mesh live in "
                        "the alignmentTransform-aligned frame."
                    ),
                },
                "cameras": camera_records,
            },
            f,
            indent=2,
        )
        f.write("\n")

    output_alignment_path = Path(output_alignment_path)
    output_alignment_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_alignment_path, "w") as f:
        json.dump(
            {
                "alignment_method": "pose_rotation_average_then_camera_center_scale_translation",
                "mesh_from_mpsfm": mesh_from_mpsfm.tolist(),
                "mesh_from_polycam_world": mesh_from_polycam_world.tolist(),
                "polycam_from_opencv_camera": POLYCAM_FROM_OPENCV_CAMERA.tolist(),
                "original_opencv_from_rotated_cw_opencv": ORIGINAL_OPENCV_FROM_ROTATED_CW_OPENCV.tolist(),
                "polycam_from_rotated_cw_opencv_camera": POLYCAM_FROM_ROTATED_CW_OPENCV_CAMERA.tolist(),
                "polycam_mesh_from_aligned": POLYCAM_MESH_FROM_ALIGNED.tolist(),
                "mesh_info_bbox_center": None if mesh_info_bbox_center is None else mesh_info_bbox_center.tolist(),
                "mesh_center_before_correction": mesh_center_before_correction.tolist(),
                "mesh_center_after_correction": mesh_center_after_correction.tolist(),
                "mesh_center_correction": mesh_center_correction.tolist(),
                "correct_mesh_center": correct_mesh_center,
                "scale": scale,
                "rotation": rotation.tolist(),
                "translation": translation.tolist(),
                "matched_frames": frame_ids,
                "num_matched_frames": len(frame_ids),
                "alignment_error_mean": float(errors.mean()),
                "alignment_error_median": float(np.median(errors)),
                "alignment_error_max": float(errors.max()),
                "rotation_alignment_error_degrees_mean": float(rotation_errors.mean()),
                "rotation_alignment_error_degrees_median": float(np.median(rotation_errors)),
                "rotation_alignment_error_degrees_max": float(rotation_errors.max()),
            },
            f,
            indent=2,
        )
        f.write("\n")

    return {
        "transformed_mesh": Path(output_mesh_path),
        "transformed_cameras": output_camera_path,
        "alignment": output_alignment_path,
        "alignment_error_mean": float(errors.mean()),
        "alignment_error_max": float(errors.max()),
    }


def main():
    parser = argparse.ArgumentParser(description="Align a GeoSVR/MPSfM mesh back to Polycam mesh coordinates.")
    parser.add_argument("--reconstruction-dir")
    parser.add_argument("--polycam-camera-dir")
    parser.add_argument("--mesh-info-path")
    parser.add_argument("--input-mesh-path")
    parser.add_argument("--output-mesh-path")
    parser.add_argument("--output-camera-path")
    parser.add_argument("--output-alignment-path")
    parser.add_argument(
        "--correct-mesh-center",
        action="store_true",
        help="Translate the final mesh bbox center to mesh_info.json bboxCenter after pose alignment.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open an Open3D visualization after alignment.",
    )
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only visualize existing --output-mesh-path and --output-camera-path outputs.",
    )
    parser.add_argument(
        "--polycam-mesh-path",
        help="Optional raw Polycam mesh (.glb/.ply/.obj) to overlay in the visualization.",
    )
    parser.add_argument(
        "--camera-scale",
        type=float,
        default=0.05,
        help="Camera frustum size for Open3D visualization.",
    )
    parser.add_argument(
        "--camera-stride",
        type=int,
        default=1,
        help="Draw every Nth camera in Open3D visualization.",
    )
    args = parser.parse_args()

    if args.camera_stride < 1:
        raise ValueError("--camera-stride must be >= 1.")

    if args.visualize_only:
        required = {
            "--output-mesh-path": args.output_mesh_path,
            "--output-camera-path": args.output_camera_path,
            "--polycam-camera-dir": args.polycam_camera_dir,
            "--mesh-info-path": args.mesh_info_path,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"Missing required visualization arguments: {', '.join(missing)}")
        visualize_polycam_alignment(
            mesh_path=args.output_mesh_path,
            transformed_camera_path=args.output_camera_path,
            polycam_camera_dir=args.polycam_camera_dir,
            mesh_info_path=args.mesh_info_path,
            polycam_mesh_path=args.polycam_mesh_path,
            camera_scale=args.camera_scale,
            camera_stride=args.camera_stride,
        )
        return

    required = {
        "--reconstruction-dir": args.reconstruction_dir,
        "--polycam-camera-dir": args.polycam_camera_dir,
        "--mesh-info-path": args.mesh_info_path,
        "--input-mesh-path": args.input_mesh_path,
        "--output-mesh-path": args.output_mesh_path,
        "--output-camera-path": args.output_camera_path,
        "--output-alignment-path": args.output_alignment_path,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing required alignment arguments: {', '.join(missing)}")

    result = align_mesh_and_cameras_to_polycam(
        reconstruction_dir=args.reconstruction_dir,
        polycam_camera_dir=args.polycam_camera_dir,
        mesh_info_path=args.mesh_info_path,
        input_mesh_path=args.input_mesh_path,
        output_mesh_path=args.output_mesh_path,
        output_camera_path=args.output_camera_path,
        output_alignment_path=args.output_alignment_path,
        correct_mesh_center=args.correct_mesh_center,
    )
    for key, value in result.items():
        print(f"{key}: {value}")

    if args.visualize:
        visualize_polycam_alignment(
            mesh_path=args.output_mesh_path,
            transformed_camera_path=args.output_camera_path,
            polycam_camera_dir=args.polycam_camera_dir,
            mesh_info_path=args.mesh_info_path,
            polycam_mesh_path=args.polycam_mesh_path,
            camera_scale=args.camera_scale,
            camera_stride=args.camera_stride,
        )


if __name__ == "__main__":
    main()
