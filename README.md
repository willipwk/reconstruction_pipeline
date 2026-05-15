# Polycam Photogrammetry Pipeline

This repository wraps a Polycam capture into a full reconstruction pipeline:

1. Rotate Polycam keyframes into the orientation expected by the downstream tools.
2. Run MPSfM to produce a COLMAP-compatible sparse reconstruction.
3. Run GeoSVR to train, render, and extract a TSDF mesh.
4. Simplify the mesh.
5. Transform the final mesh and camera poses back into the original Polycam mesh coordinate system.

The main entry point is:

```bash
python run_photogrammetry_pipeline.py <object_name>
```

`<object_name>` must correspond to a folder under `data/`, for example `data/microwave`.

Run the pipeline in an environment that has the required packages for all stages, including OpenCV, PyYAML, Open3D, pycolmap, fast_simplification, MPSfM, and GeoSVR. In the current setup this is typically the `reconstruction` conda environment.

## Quick Start

Put a Polycam export under:

```text
data/<object_name>/
  keyframes/
    images/ or corrected_images/
    cameras/ or corrected_cameras/
    depth/
  mesh_info.json
```

Then run:

```bash
python run_photogrammetry_pipeline.py microwave
```

By default this writes GeoSVR outputs to:

```text
GeoSVR/output/custom/<object_name>/
```

Useful variants:

```bash
# Stop after preprocessing, MPSfM, and GeoSVR data layout preparation.
python run_photogrammetry_pipeline.py microwave --skip-geosvr

# Re-run MPSfM even if sparse reconstruction files already exist.
python run_photogrammetry_pipeline.py microwave --force-sfm

# Re-run image/depth/camera rotation even if rotated outputs already exist.
python run_photogrammetry_pipeline.py microwave --force-preprocess

# Replace existing symlinks under mpsfm/data and GeoSVR/data/custom.
python run_photogrammetry_pipeline.py microwave --force-links

# Visualize the final Polycam alignment after mesh transformation.
python run_photogrammetry_pipeline.py microwave --visualize-polycam-alignment
```

## Main Pipeline

`run_photogrammetry_pipeline.py` coordinates the full workflow.

For an object named `microwave`, it assumes the raw input lives at:

```text
data/microwave/
```

The pipeline stages are:

1. `preprocess_polycam.py`
   Creates `data/microwave/keyframes_rot` by rotating raw images, depth maps, and camera intrinsics 90 degrees clockwise.

2. MPSfM
   Creates a symlink:

   ```text
   mpsfm/data/microwave -> data/microwave/keyframes_rot
   ```

   Then runs:

   ```bash
   cd mpsfm
   python reconstruct.py \
     --data_dir data/microwave \
     --images_dir data/microwave/<image_subdir> \
     --intrinsics_pth data/microwave/intrinsics.yaml \
     --conf sp-lg_m3dv2
   ```

   Sparse reconstruction is expected at:

   ```text
   data/microwave/keyframes_rot/sfm_outputs/rec/
   ```

3. GeoSVR layout
   Creates:

   ```text
   data/microwave/keyframes_rot/sparse/0 -> data/microwave/keyframes_rot/sfm_outputs/rec
   GeoSVR/data/custom/microwave -> data/microwave/keyframes_rot
   ```

4. GeoSVR training and mesh extraction
   Runs from `GeoSVR/`:

   ```bash
   python train.py --cfg_files <cfg> --source_path data/custom/microwave --model_path <output> --images <image_subdir>
   python render.py <output>
   PYTHONPATH=./ python mesh_extract/tsdf_mesh.py <output>
   ```

5. `mesh_simplification.py`
   Simplifies:

   ```text
   <output>/mesh/tsdf/tsdf_fusion_post.ply
   ```

   into:

   ```text
   <output>/mesh/tsdf/tsdf_fusion_post_simplified.ply
   ```

6. `polycam_alignment.py`
   Aligns the GeoSVR/MPSfM result back to the Polycam mesh coordinate system.

   Default aligned outputs:

   ```text
   <output>/mesh/tsdf/tsdf_fusion_post_simplified_polycam.ply
   <output>/mesh/tsdf/camera_pose_polycam.json
   <output>/mesh/tsdf/mpsfm_to_polycam_alignment.json
   ```

## Data Format

### Raw Polycam Input

The required raw folder layout is:

```text
data/<object_name>/
  keyframes/
    depth/
    images/
    cameras/
  mesh_info.json
```

If both of these folders exist, they are used instead of `images/` and `cameras/`:

```text
data/<object_name>/keyframes/corrected_images/
data/<object_name>/keyframes/corrected_cameras/
```

Image filenames and camera JSON filenames must share the same stem:

```text
keyframes/corrected_images/15467025843.jpg
keyframes/corrected_cameras/15467025843.json
```

Supported image extensions:

```text
.jpg, .jpeg, .png
```

Supported depth extensions:

```text
.png, .tif, .tiff, .exr
```

### Camera JSON

Each camera JSON must contain intrinsics:

```json
{
  "width": 1920,
  "height": 1440,
  "fx": 1234.0,
  "fy": 1234.0,
  "cx": 960.0,
  "cy": 720.0
}
```

For Polycam alignment, each camera JSON must also contain the camera-to-world transform fields:

```json
{
  "t_00": 1.0, "t_01": 0.0, "t_02": 0.0, "t_03": 0.0,
  "t_10": 0.0, "t_11": 1.0, "t_12": 0.0, "t_13": 0.0,
  "t_20": 0.0, "t_21": 0.0, "t_22": 1.0, "t_23": 0.0
}
```

Polycam camera convention is:

```text
x right, y up, z out of screen
```

MPSfM/COLMAP camera convention is OpenCV:

```text
x right, y down, z into screen
```

Because the pipeline rotates images 90 degrees clockwise before MPSfM, `polycam_alignment.py` aligns against the rotated-image OpenCV camera basis.

### mesh_info.json

`mesh_info.json` is required for final alignment. The pipeline uses:

```json
{
  "alignmentTransform": [16 numbers],
  "bboxCenter": [x, y, z]
}
```

`alignmentTransform` is interpreted as a Polycam/SceneKit-style column-major 4x4 transform. `bboxCenter` is optional unless `--correct-mesh-center` is used.

## Generated Data

After preprocessing:

```text
data/<object_name>/keyframes_rot/
  corrected_images/ or images/
  corrected_cameras/ or cameras/
  depth/
  intrinsics.yaml
  rotation_preprocess.yaml
```

`intrinsics.yaml` is generated for MPSfM. It stores each frame's `fx`, `fy`, `cx`, and `cy`, grouped by MPSfM camera index.

After MPSfM:

```text
data/<object_name>/keyframes_rot/sfm_outputs/rec/
  cameras.bin
  images.bin
  points3D.bin
```

Text COLMAP files are also accepted:

```text
cameras.txt
images.txt
points3D.txt
```

After GeoSVR:

```text
GeoSVR/output/custom/<object_name>/
  mesh/tsdf/tsdf_fusion_post.ply
  mesh/tsdf/tsdf_fusion_post_simplified.ply
  mesh/tsdf/tsdf_fusion_post_simplified_polycam.ply
  mesh/tsdf/camera_pose_polycam.json
  mesh/tsdf/mpsfm_to_polycam_alignment.json
```

## Idempotency and Reruns

The pipeline is designed to avoid repeating expensive work when outputs already exist.

`preprocess_polycam.py` skips rotation if all required rotated files already exist. If only some files are missing, it creates the missing files and keeps existing files.

MPSfM is skipped when a complete sparse reconstruction already exists under:

```text
data/<object_name>/keyframes_rot/sfm_outputs/rec/
```

Use these flags to force reruns:

```bash
--force-preprocess   # rotate raw data again
--force-sfm          # run MPSfM again
--force-links        # replace existing symlinks
```

Use these flags to skip stages manually:

```bash
--skip-preprocess
--skip-sfm
--skip-geosvr
--no-simplify-mesh
--no-transform-to-polycam
```

## Command-Line Options

Important options for `run_photogrammetry_pipeline.py`:

```text
--cfg-path                     GeoSVR config path. Relative paths are resolved from GeoSVR/.
--output-path                  GeoSVR output path. Defaults to GeoSVR/output/custom/<object>.
--mpsfm-conf                   MPSfM config name. Defaults to sp-lg_m3dv2.
--simplification-target-reduction
                               Fraction of mesh triangles to remove. Defaults to 0.9.
--correct-mesh-center          Translate final mesh bbox center to mesh_info.json bboxCenter.
--visualize-polycam-alignment  Open an Open3D alignment view.
--polycam-mesh-path            Optional raw Polycam mesh overlay for visualization.
--camera-scale                 Camera frustum size for visualization.
--camera-stride                Draw every Nth camera in visualization.
--geosvr-arg                   Extra token appended to GeoSVR train.py. Repeat for multiple tokens.
```

Example with a custom output path:

```bash
python run_photogrammetry_pipeline.py microwave \
  --output-path GeoSVR/output/custom/microwave_test
```

Example passing extra GeoSVR training arguments:

```bash
python run_photogrammetry_pipeline.py microwave \
  --geosvr-arg=--some_flag \
  --geosvr-arg some_value
```

## Alignment Visualization

To inspect alignment after a run:

```bash
python polycam_alignment.py \
  --visualize-only \
  --output-mesh-path GeoSVR/output/custom/microwave/mesh/tsdf/tsdf_fusion_post_simplified_polycam.ply \
  --output-camera-path GeoSVR/output/custom/microwave/mesh/tsdf/camera_pose_polycam.json \
  --polycam-camera-dir data/microwave/keyframes/corrected_cameras \
  --mesh-info-path data/microwave/mesh_info.json
```

Visualization colors:

```text
gray mesh     transformed GeoSVR mesh
green cameras transformed MPSfM cameras
red cameras   raw Polycam cameras after mesh_info transform
orange cameras raw Polycam cameras before mesh_info transform
yellow lines  matched camera-center offsets
blue mesh     optional raw Polycam mesh from --polycam-mesh-path
```

## Module Summary

`preprocess_polycam.py`

Rotates images, depth maps, and camera intrinsics 90 degrees clockwise. It prefers `corrected_images` and `corrected_cameras` when both are present. It writes `rotation_preprocess.yaml`.

`run_photogrammetry_pipeline.py`

Coordinates preprocessing, symlink setup, MPSfM reconstruction, GeoSVR training/rendering/mesh extraction, mesh simplification, and Polycam coordinate alignment.

`mesh_simplification.py`

Simplifies a triangle mesh using `fast_simplification` and Open3D.

`polycam_alignment.py`

Reads MPSfM/COLMAP camera poses, raw Polycam camera poses, and `mesh_info.json`. It estimates a similarity transform from MPSfM coordinates into the Polycam mesh coordinate system, writes an aligned mesh, writes transformed camera poses, and can visualize the result in Open3D.

`refine_depth.py`

Contains the lingbot-depth depth refinement workflow. It is separate from the main photogrammetry pipeline unless called independently.

## Practical Notes

Run commands from the repository root unless noted otherwise.

The pipeline creates symlinks into `mpsfm/data/` and `GeoSVR/data/custom/`. If a symlink exists but points somewhere else, use `--force-links`.

If `corrected_images` and `corrected_cameras` exist in the raw Polycam export, they must both exist. The pipeline only switches to corrected data when both folders are present.

The final aligned mesh is usually the mesh to use outside this pipeline:

```text
GeoSVR/output/custom/<object_name>/mesh/tsdf/tsdf_fusion_post_simplified_polycam.ply
```
