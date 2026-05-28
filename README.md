# Polycam Photogrammetry Pipeline

This repository wraps a Polycam capture into a full reconstruction pipeline:

1. Rotate Polycam keyframes into the orientation expected by the downstream tools.
2. Run MPSfM to produce a COLMAP-compatible sparse reconstruction.
3. Run GeoSVR to train, render, and extract a TSDF mesh.
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
   ```
3. Install mpsfm environment.
   1. Whether you have sudo or not, I recommend to install cpp libraries using conda. Then add paths to these libraries to `CMAKE_PREFIX`
      ```bash
      conda install -c conda-forge glog metis suitesparse eigen==3.4.0 boost-cpp flann cgal gmp mpfr qt freeimage glew
      export CMAKE_PREFIX_PATH="$CONDA_PREFIX:$CMAKE_PREFIX_PATH"
      export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
      ```
   2. Now let's install pyceres. First you need to install the Ceres Solver following [the official instruction](http://ceres-solver.org/installation.html). I make some modifications to the installation scripts as I do not have sudo.
      ```bash
      cd mpsfm
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

      export CMAKE_PREFIX_PATH=$HOME/local/colmap:$CMAKE_PREFIX_PATH
      python -m pip install .
      cd ..
      ```
   5. Finally, let's install python packages for mpsfm
      ```bash
      pip install -r requirements.txt
      python -m pip install -e .
      ```
4. Install GeoSVR environment.
    ```bash
    cd GeoSVR
    pip install yacs natsort imageio imageio-ffmpeg scikit-image plyfile shapely trimesh open3d gpytoolbox transformers==4.49.0 lpips pytorch-msssim
    pip install --no-build-isolation git+https://github.com/rahul-goel/fused-ssim.git@3006269823fc28110ba44686a172cbd59ec01bc3
    pip install --no-build-isolation ./cuda
    cd ..
    ```
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
python run_photogrammetry_pipeline.py microwave
```

By default this writes GeoSVR outputs to:

```text
GeoSVR/output/custom/<object_name>/
```

If `GeoSVR/output/custom/<object_name>/mesh/tsdf/tsdf_fusion_post.ply` already exists, the pipeline skips GeoSVR training/rendering/TSDF extraction and continues with mesh simplification and Polycam alignment.

Similarly, MPSfM is skipped when `data/<object_name>/keyframes_rot/sfm_outputs/` already contains a valid COLMAP reconstruction, either directly in `sfm_outputs/` or in a child directory such as `sfm_outputs/rec/`.

Useful variants:

```bash
# Stop after preprocessing, MPSfM, and GeoSVR data layout preparation.
python run_photogrammetry_pipeline.py microwave --skip-geosvr

# Re-run MPSfM even if sparse reconstruction files already exist.
python run_photogrammetry_pipeline.py microwave --force-sfm

# Keep the MPSfM monocular-prior cache for debugging.
python run_photogrammetry_pipeline.py microwave --keep-mpsfm-cache

# Re-run image/depth/camera rotation even if rotated outputs already exist.
python run_photogrammetry_pipeline.py microwave --force-preprocess

# Replace existing symlinks under mpsfm/data and GeoSVR/data/custom.
python run_photogrammetry_pipeline.py microwave --force-links

# Visualize the final Polycam alignment after mesh transformation.
python run_photogrammetry_pipeline.py microwave --visualize-polycam-alignment
```

## Command-Line Options

Important options for `run_photogrammetry_pipeline.py`:

```text
--cfg-path                     GeoSVR config path. Relative paths are resolved from GeoSVR/.
--output-path                  GeoSVR output path. Defaults to GeoSVR/output/custom/<object>.
--mpsfm-conf                   MPSfM config name. Defaults to sp-lg_mogev2.
--keep-mpsfm-cache             Keep data/<object>/keyframes_rot/cache_dir after MPSfM finishes.
--simplification-target-reduction
                               Fraction of mesh triangles to remove. Defaults to 0.5.
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

## Batch Reconstruction

To reconstruct every object folder under `data/`, run:

```bash
python run_all_photogrammetry_pipelines.py --gpus 0,1,2,3
```

The launcher uses `ThreadPoolExecutor` and assigns one GPU to one object at a time by setting `CUDA_VISIBLE_DEVICES` for each subprocess. When `GPUtil` is available, it checks GPU load and memory before starting a job on a reserved GPU. Inside a Slurm allocation this GPUtil waiting is disabled by default because Slurm already controls the GPU allocation and GPUtil can report global load in a misleading way. It writes per-object logs under `logs/photogrammetry/`.

Useful variants:

```bash
# Preview the schedule without running anything.
python run_all_photogrammetry_pipelines.py --gpus 0,1 --dry-run

# Process a subset of objects.
python run_all_photogrammetry_pipelines.py --gpus 0,1 --objects microwave lamp

# Forward arguments to run_photogrammetry_pipeline.py after '--'.
python run_all_photogrammetry_pipelines.py --gpus 0,1 -- --skip-sfm

# Wait for external GPU usage to drop before launching each job.
python run_all_photogrammetry_pipelines.py --gpus 0,1 --max-gpu-load 0.5 --max-gpu-memory 0.5

# Disable GPUtil checks and only use exclusive scheduler slots.
python run_all_photogrammetry_pipelines.py --gpus 0,1 --no-gputil

# Force GPUtil load/memory checks even inside Slurm.
python run_all_photogrammetry_pipelines.py --gpus 0,1 --force-gputil-check
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

## Practical Notes

Run commands from the repository root unless noted otherwise.

The pipeline creates symlinks into `mpsfm/data/` and `GeoSVR/data/custom/`. If a symlink exists but points somewhere else, use `--force-links`.

If `corrected_images` and `corrected_cameras` exist in the raw Polycam export, they must both exist. The pipeline only switches to corrected data when both folders are present.

The final aligned mesh is usually the mesh to use outside this pipeline:

```text
GeoSVR/output/custom/<object_name>/mesh/tsdf/tsdf_fusion_post_simplified_polycam.ply
```

## Acknowledgments
This project is built on [mpsfm](https://github.com/cvg/mpsfm) and [GeoSVR](https://github.com/Fictionarry/GeoSVR). We thank the authors of these two paper for open sourcing their amazing project. 
