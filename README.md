# BEHAV3D offline reconstruction

Reconstruct calibrated BEHAV3D scans without ROS, a robot connection, or camera hardware. **Edit the settings at the top of `reconstruct.py`, then run that script.** It processes one scan or discovers all scans beneath a parent folder.

The default workflow uses calibrated depth/RGB undistortion, color-to-depth alignment, TSDF fusion, measured mesh colors, and removal of small disconnected mesh fragments. It preserves the tested J1 processing settings: 1.953125 mm voxels, extraction weight threshold 3, and a 100-triangle minimum component size. It does not smooth or decimate the exported mesh.

## Install

Use Python 3.11 and a virtual environment:

```bash
git clone https://github.com/josephnamar/behav3d-offline-reconstruction.git
cd behav3d-offline-reconstruction
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, use `py -3.11 -m venv .venv` and `.venv\Scripts\Activate.ps1` in PowerShell. On a minimal Ubuntu installation, Open3D may also need `sudo apt-get install libgl1 libgomp1`. CPU execution is the default; macOS does not require CUDA. Installation needs internet access; processing uses local files only.

## Input layout

Keep the same structure as the captured sessions:

```text
data/
  SESSION_ID/
    config/
      depth_intrinsics.yaml
      color_intrinsics.yaml
      extrinsics.yaml
      ir_intrinsics.yaml          # May be present; not required here.
    scan_yaml_targets/            # Any scan-folder name is accepted.
      manifest.yaml
      depth_raw/
        depth_i0.png
        ...
      color_raw/
        color_i0.png
        ...
      ir_raw/                     # May be present; not used here.
```

`input_path` may point to `data`, a session, a scan folder, or its manifest. Multiple scan folders within one session are supported. Existing `reconstruct` directories are ignored and aligned images are regenerated for each new run. The absolute `session.path` recorded in an old manifest is ignored, so captures can be moved between computers.

Each manifest capture must include its relative `depth` and `color` paths and an explicit `T_base_tool0` pose, with `position: [x, y, z]` in metres and `orientation_xyzw: [x, y, z, w]`. Images are single-channel uint16 depth and three-channel uint8 color. Intrinsics use the existing OpenCV YAML keys `image_width`, `image_height`, `camera_matrix`, and `distortion_coefficients`. Extrinsics use `frames.T_tool0_ir` and `frames.T_tool0_color`, each with `xyz` in metres and `quat_xyzw`.

Every selected scan is validated before any reconstruction starts. Missing calibration, missing poses, unreadable images, wrong image dimensions, duplicate frame indices, and repeated image paths are rejected.

## Run

Set `input_path` and, if needed, `output_root` in the **SETTINGS block in `reconstruct.py`**. All processing controls are in that block; there is no separate configuration file to maintain.

```bash
python reconstruct.py
```

Optional path overrides and a validation-only command:

```bash
python reconstruct.py --input /path/to/Scans --output /path/to/results
python reconstruct.py --input /path/to/Scans --validate-only
```

With empty `session_names` and `scan_folders` lists, all discovered scans are processed sequentially. Populate those lists to select particular sessions or scan types. Set `undistort` to `False` for a separate comparison run. Set `open_viewer` to `True` to inspect each colored mesh interactively; otherwise no display is required.

The existing depth correction is **retained**: `depth_bias_mm = -5.1`. For the usual millimetre uint16 input this rounds to **−5 mm**, applied along each camera's depth axis before fusion, not the robot's fixed world Z axis. Set it to `0.0` to disable it. This repository does not establish a new physical calibration.

`weight_threshold = 3` controls TSDF mesh extraction. `color_min_observations = 3` independently controls how many usable RGB observations are required for a colored point. Unsupported mesh vertices retain neutral gray; they are not removed merely because their color is missing.

The statistical/radius outlier filter affects the point clouds. Mesh noise removal deletes entire disconnected components smaller than `mesh_min_component_triangles`; set this to `0` to retain every component. It can also remove legitimate isolated features smaller than the chosen threshold. Optional center, world and adaptive object crops are disabled by default. Table-plane fitting saves a plane estimate and leaves the table in the geometry.

## Results

Every invocation creates a new UTC timestamped directory under `output_root`. Outputs must be outside source sessions, and captured inputs are never modified. Paths below each run preserve the selected session/scan hierarchy.

```text
results/RUN_ID/
  settings.json                   # Effective parameters, depth bias and versions.
  summary.json                    # Completion/failure status for every scan.
  .../SESSION_ID/SCAN_FOLDER/
    tsdf_surface_mesh.ply         # Full triangle mesh with measured vertex RGB.
    tsdf_surface_mesh.stl         # Same geometry, no color; coordinates in metres.
    tsdf_surface_rgb_colored.ply  # Filtered RGB point cloud.
    tsdf_surface_confidence_colored.ply
    preview.png                  # Headless, shaded color preview of the full scene.
    table_plane.json             # Present when plane fitting is enabled.
    color_in_depth_rectified/    # Or color_in_depth when undistort=False.
    input_provenance.json        # Source-file SHA-256 hashes.
    result.json                  # Mesh counts, bounds, timing and status.
    run.log
```

PLY/STL coordinates remain in metres in the robot base frame defined by the saved poses. The bounding dimensions in `result.json` are millimetres and describe the **full-scene axis-aligned bounding box**, including the table if present; they are not automatically isolated object dimensions. Confidence colors represent observation counts, not a probability or sensor confidence measurement. Preview lighting changes displayed brightness only; PLY retains the measured RGB values.

The script returns exit code 1 if validation or a selected reconstruction fails. `continue_on_error` lets later scans finish after a reconstruction failure; consult `summary.json` and each `result.json` before using partial output. Ctrl+C can leave a run marked `running`, which is not a completed result. A subsequent invocation always creates a separate directory.

## Verification and source

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover calibrated ray mapping, raw depth edge preservation, alignment completeness, RGB transfer, component cleanup, input validation, both undistortion modes, depth-bias toggling, and actual small TSDF reconstructions. See `VALIDATION.md` for the saved-scan regression results.

The numerical reconstruction core was extracted from `behav3d_sense` in [SDU-CREATE-BEHAV3D/behav3d_ws](https://github.com/SDU-CREATE-BEHAV3D/behav3d_ws), research commit `40d99a046e584f4705a49c1044d4fc47bf7f079c`, matching the production reconstruction changes in J1 commit `9716c8da05c077c93eaf4b6810a51d25efaf9fd3`. This repository adds explicit offline settings, safe output routing, validation, batch orchestration, confidence export, and CPU previews. No ROS package imports, capture data, or robot-control code are included.
