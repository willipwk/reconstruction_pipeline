# Polycam Photogrammetry Pipeline

This repository wraps a Polycam capture into a full reconstruction pipeline:

1. Rotate Polycam keyframes into the orientation expected by the downstream tools.
2. Run MPSfM or GlueMap to produce a COLMAP-compatible sparse reconstruction.
3. Run GeoSVR or PGSR to train, render, and extract a TSDF mesh.
4. Simplify the mesh.
5. Transform the final mesh and camera poses back into the original Polycam mesh coordinate system.


## Installation
We recommend using conda to manage the environment.
1. Create a conda environment
   ```bash
   git clone --recursive git@github.com:willipwk/reconstruction_pipeline.git
   cd reconstruction_pipeline
   conda create -n reconstruction python=3.11
   conda activate reconstruction
   ```
2. Install pytorch. The pipeline is tested on `torch==2.10.0+cu126`.
   ```bash
   pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126
   pip install hydra-core
   ```
3. Install mpsfm environment.
   1. Whether you have sudo or not, I recommend to install cpp libraries using conda. Then add paths to these libraries to `CMAKE_PREFIX`
      ```bash
      conda install -c conda-forge glog metis==5.1.0 suitesparse eigen==3.4.0 boost-cpp flann cgal gmp mpfr qt freeimage glew
      export CMAKE_PREFIX_PATH="$CONDA_PREFIX:$CMAKE_PREFIX_PATH"
      export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
      ```
   2. Now let's install pyceres. First you need to install the Ceres Solver following [the official instruction](http://ceres-solver.org/installation.html). I make some modifications to the installation scripts as I do not have sudo.
      ```bash
      cd third_party/mpsfm
      git clone https://github.com/cvg/pyceres.git
      cd pyceres
      git checkout v2.5

      wget http://ceres-solver.org/ceres-solver-2.2.0.tar.gz
      tar zxvf ceres-solver-2.2.0.tar.gz
      cd ceres-solver-2.2.0
      cmake -S . -B build -G Ninja \
         -DCMAKE_INSTALL_PREFIX=$HOME/local/ceres-solver \
         -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
         -DBLA_VENDOR=OpenBLAS \
         -DBLAS_LIBRARIES=$CONDA_PREFIX/lib/libopenblas.so \
         -DLAPACK_LIBRARIES=$CONDA_PREFIX/lib/libopenblas.so \
         -DSuiteSparse_DIR=$CONDA_PREFIX/lib/cmake/SuiteSparse \
         -DBUILD_TESTING=OFF \
         -DBUILD_EXAMPLES=OFF
      cmake --build build
      cmake --install build

      cd ..
      ```
   3. Then, let's install pyceres. Currently we use `pyceres==2.5`.
      ```bash
      export CMAKE_PREFIX_PATH=$HOME/local/ceres-solver:$CMAKE_PREFIX_PATH
      python -m pip install .
      cd ..
      ```
      If you see an error message like this: "Failed to find SuiteSparse - Did not find BLAS library", you can add paths manually in the `pyproject.toml` file like this BEFORE the `[project]` line:
      ```toml
      [tool.scikit-build.cmake]
      args=["-DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/blas/libblas.so", "-DLAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/lapack/liblapack.so"]
      ```
      Replace the paths to your BLAS and Lapack library files.
   4. After that, we need to build a customized colmap and pycolmap. You can follow [the official instructions](https://github.com/Zador-Pataki/colmap).
      ```bash
      git clone https://github.com/Zador-Pataki/colmap.git
      cd colmap

      mkdir build
      cd build
      cmake .. -GNinja \
         -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_INSTALL_PREFIX=$HOME/local/colmap \
         -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
         -DCMAKE_BUILD_RPATH="$CONDA_PREFIX/lib" \
         -DCMAKE_INSTALL_RPATH="$CONDA_PREFIX/lib" \
         -DGlog_DIR="$CONDA_PREFIX/lib/cmake/glog" \
         -DGflags_DIR="$CONDA_PREFIX/lib/cmake/gflags"
      ninja
      ninja install
      cd ..

      export CMAKE_PREFIX_PATH=$HOME/local/colmap:$CMAKE_PREFIX_PATH
      python -m pip install .
      cd ..
      ```
   5. Finally, let's install python packages for mpsfm
      ```bash
      pip install -r requirements.txt
      python -m pip install -e .
      cd ../..
      ```
4. Install GeoSVR environment.
    ```bash
    cd third_party/GeoSVR
    pip install yacs natsort imageio imageio-ffmpeg scikit-image plyfile shapely trimesh open3d gpytoolbox transformers==4.49.0 lpips pytorch-msssim fast_simplification hydra-core
    pip install --no-build-isolation git+https://github.com/rahul-goel/fused-ssim.git@3006269823fc28110ba44686a172cbd59ec01bc3
    pip install --no-build-isolation ./cuda
    cd ../..
    ```
5. Optional: install PGSR environment.
    ```bash
    cd third_party/pgsr
    pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
    pip install --no-build-isolation submodules/diff-plane-rasterization
    pip install --no-build-isolation submodules/simple-knn
    cd ../..
    ```
    PGSR in this repository expects per-image monocular depth maps. The pipeline
    generates those in-process with the Hugging Face Transformers Depth Anything V2
    model, so no extra repository clone or local checkpoint path is needed.
6. Optional: install GlueMap environment.
    ```bash
    cd third_party/gluemap
    CMAKE_PREFIX_PATH=$CONDA_PREFIX pip install -e .
    cd ../..
    ```
    The default GlueMap config expects checkpoints under `third_party/gluemap/checkpoints/`.
    See `third_party/gluemap/INSTALL.md` for the exact download commands.
Rightnow you should be able to run the code.
   

## Quick Start

First you need to use Polycam app to scan the object in lidar/space mode. Then, export the raw data from the app.

Put the raw data under `data/<object_name>`. THe file structure should like this

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
python run_photogrammetry_pipeline.py object=microwave
```

By default this writes GeoSVR outputs to:

```text
third_party/GeoSVR/output/custom/<object_name>/
```

To use PGSR instead of GeoSVR:

```bash
python run_photogrammetry_pipeline.py object=microwave renderer=pgsr
```

PGSR reuses the MPSfM sparse reconstruction directly, prepares data under
`third_party/pgsr/data/custom/<object_name>/`, and writes outputs to
`third_party/pgsr/output/custom/<object_name>/` by default.

To use GlueMap instead of MPSfM for sparse reconstruction:

```bash
python run_photogrammetry_pipeline.py object=microwave sparse_reconstructor=gluemap
```

GlueMap writes sparse outputs under:

```text
data/<object_name>/keyframes_rot/gluemap_outputs/
```

If `third_party/GeoSVR/output/custom/<object_name>/mesh/tsdf/tsdf_fusion_post.ply` already exists, the pipeline skips GeoSVR training/rendering/TSDF extraction and continues with mesh simplification and Polycam alignment.

Similarly, sparse reconstruction is skipped when the selected backend already has a valid COLMAP reconstruction. MPSfM checks `data/<object_name>/keyframes_rot/sfm_outputs/`; GlueMap checks `data/<object_name>/keyframes_rot/gluemap_outputs/`.

Useful variants:

```bash
# Stop after preprocessing, MPSfM, and renderer data layout preparation.
python run_photogrammetry_pipeline.py object=microwave skip_reconstruction=true

# Re-run MPSfM even if sparse reconstruction files already exist.
python run_photogrammetry_pipeline.py object=microwave force_sfm=true

# Use GlueMap sparse reconstruction with PGSR dense reconstruction.
python run_photogrammetry_pipeline.py object=microwave sparse_reconstructor=gluemap renderer=pgsr

# Keep the MPSfM monocular-prior cache for debugging.
python run_photogrammetry_pipeline.py object=microwave cleanup_mpsfm_cache=false

# Re-run image/depth/camera rotation even if rotated outputs already exist.
python run_photogrammetry_pipeline.py object=microwave force_preprocess=true

# Replace existing symlinks under third_party/mpsfm/data and renderer data folders.
python run_photogrammetry_pipeline.py object=microwave force_links=true

# Visualize the final Polycam alignment after mesh transformation.
python run_photogrammetry_pipeline.py object=microwave visualize_alignment=true
```

## Configuration

Pipeline defaults live in:

```text
configs/pipeline.yaml
configs/batch.yaml
```

Use Hydra overrides on the command line for one-off changes, or edit/copy the YAML files for repeated runs.

Sparse reconstruction knobs:

```text
sparse_reconstructor     mpsfm or gluemap. Defaults to mpsfm.
mpsfm_conf               MPSfM config name. Defaults to sp-lg_mogev2.
mpsfm_command            Optional Python command list for MPSfM. Defaults to ["python"].
gluemap_config           GlueMap config path. Relative paths resolve from third_party/gluemap/.
gluemap_command          Optional command list. Defaults to gluemap-demo when found, otherwise ["python", "demo.py"].
gluemap_intrinsics_mode  GlueMap intrinsics mode: SHARED, PER_FOLDER, or PER_CAMERA.
gluemap_extra_args       Extra tokens appended to gluemap-demo.
```

MPSfM and GlueMap require incompatible `pycolmap` builds. Keep them in
separate environments and select the backend command through Hydra:

```bash
python run_photogrammetry_pipeline.py \
  object=microwave \
  sparse_reconstructor=mpsfm \
  mpsfm_command='["conda","run","-n","mpsfm","python"]'

python run_photogrammetry_pipeline.py \
  object=microwave \
  sparse_reconstructor=gluemap \
  gluemap_command='["conda","run","-n","gluemap","gluemap-demo"]'
```

The launcher automatically inserts `--no-capture-output` for `conda run`
commands so GlueMap logs stream live. The first MapAnything run can still
spend a long time downloading/loading `facebook/map-anything` and building
CUDA/Torch caches before reconstruction starts.

Using absolute interpreters also works:

```bash
mpsfm_command='["/path/to/mpsfm/env/bin/python"]'
gluemap_command='["/path/to/gluemap/env/bin/gluemap-demo"]'
```

Example with a custom output path:

```bash
python run_photogrammetry_pipeline.py \
  object=microwave \
  output_path=third_party/GeoSVR/output/custom/microwave_test
```

Example passing extra GeoSVR training arguments:

```bash
python run_photogrammetry_pipeline.py \
  object=microwave \
  geosvr_extra_args='["--some_flag","some_value"]'
```

Example passing PGSR options:

```bash
python run_photogrammetry_pipeline.py \
  object=microwave \
  renderer=pgsr \
  pgsr_max_depth=2.0 \
  pgsr_voxel_size=0.005 \
  pgsr_extra_args='["--iterations","20000","--max_abs_split_points","0","--opacity_cull_threshold","0.01"]'
```

## Batch Reconstruction

To reconstruct every object folder under `data/`, run:

```bash
python run_all_photogrammetry_pipelines.py gpus='"0,1,2,3"'
```

The launcher uses `ThreadPoolExecutor` and assigns one GPU to one object at a time by setting `CUDA_VISIBLE_DEVICES` for each subprocess. When `GPUtil` is available, it checks GPU load and memory before starting a job on a reserved GPU. Inside a Slurm allocation this GPUtil waiting is disabled by default because Slurm already controls the GPU allocation and GPUtil can report global load in a misleading way. It writes per-object logs under `logs/photogrammetry/`.

Useful variants:

```bash
# Preview the schedule without running anything.
python run_all_photogrammetry_pipelines.py gpus='"0,1"' dry_run=true

# Process a subset of objects.
python run_all_photogrammetry_pipelines.py gpus='"0,1"' objects='["microwave","lamp"]'

# Forward Hydra overrides to run_photogrammetry_pipeline.py.
python run_all_photogrammetry_pipelines.py \
  gpus='"0,1"' \
  pipeline_overrides='["skip_sfm=true","renderer=pgsr"]'

# Wait for external GPU usage to drop before launching each job.
python run_all_photogrammetry_pipelines.py gpus='"0,1"' max_gpu_load=0.5 max_gpu_memory=0.5

# Disable GPUtil checks and only use exclusive scheduler slots.
python run_all_photogrammetry_pipelines.py gpus='"0,1"' no_gputil=true

# Force GPUtil load/memory checks even inside Slurm.
python run_all_photogrammetry_pipelines.py gpus='"0,1"' force_gputil_check=true
```

## Module Summary

`preprocess_polycam.py`

Rotates images, depth maps, and camera intrinsics 90 degrees clockwise. It prefers `corrected_images` and `corrected_cameras` when both are present. It writes `rotation_preprocess.yaml`.

`run_photogrammetry_pipeline.py`

Coordinates preprocessing, symlink setup, MPSfM/GlueMap sparse reconstruction, GeoSVR/PGSR training/rendering/mesh extraction, mesh simplification, and Polycam coordinate alignment.

`mesh_simplification.py`

Simplifies a triangle mesh using `fast_simplification` and Open3D.

`polycam_alignment.py`

Reads COLMAP camera poses from the selected sparse reconstructor, raw Polycam camera poses, and `mesh_info.json`. It estimates a similarity transform from sparse reconstruction coordinates into the Polycam mesh coordinate system, writes an aligned mesh, writes transformed camera poses, and can visualize the result in Open3D.

## Practical Notes

Run commands from the repository root unless noted otherwise.

The pipeline creates symlinks into `third_party/mpsfm/data/` when using MPSfM and into the selected renderer data folder. For GeoSVR this is `third_party/GeoSVR/data/custom/`; for PGSR this is `third_party/pgsr/data/custom/`. If a symlink exists but points somewhere else, use `force_links=true`.

If `corrected_images` and `corrected_cameras` exist in the raw Polycam export, they must both exist. The pipeline only switches to corrected data when both folders are present.

The final aligned mesh is usually the mesh to use outside this pipeline:

```text
third_party/GeoSVR/output/custom/<object_name>/mesh/tsdf/tsdf_fusion_post_simplified_polycam.ply
```

## Acknowledgments
This project is built on [mpsfm](https://github.com/cvg/mpsfm), [GeoSVR](https://github.com/Fictionarry/GeoSVR), [PGSR](https://github.com/zju3dv/PGSR), and [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2). We thank the authors for open sourcing their projects.
