#!/usr/bin/env python3
"""Edit the settings below, then run: python reconstruct.py"""
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent

# ===================== ALL USER SETTINGS =====================
SETTINGS = {
    # One session, one scan folder, or a parent containing many sessions.
    "input_path": REPO_DIR / "data",
    "output_root": REPO_DIR / "results",  # Fresh timestamped directory each run.
    "session_names": [],  # [] = all; e.g. ["261113_150006"].
    "scan_folders": [],  # [] = all; e.g. ["scan_yaml_targets"].
    "continue_on_error": True,  # Record failed scans and finish the others.
    "validate_only": False,  # Check every image, calibration and pose; no fusion.

    # Calibrated image correction. False gives a comparison without undistortion.
    "undistort": True,
    "depth_scale": 1000.0,  # Raw depth units per metre (uint16 millimetres).
    "depth_bias_mm": -5.1,  # Existing J1 correction; set 0 to disable.
    # For uint16 mm input, -5.1 rounds to -5 mm, along each camera's depth axis.
    "depth_max_m": 0.9,
    "alignment_depth_max_m": 1.0,

    # TSDF geometry. These are the tested settings, including threshold 3.
    "voxel_size_m": 1 / 512,  # 1.953125 mm.
    "trunc_voxel_multiplier": 8.0,
    "weight_threshold": 3.0,  # Mesh extraction weight; separate from RGB count.
    "block_resolution": 12,
    "block_count": 10000,
    "device": "CPU:0",  # CUDA:0 requires a CUDA-enabled Open3D installation.
    "cpu_threads": 4,
    "random_seed": 0,

    # Measured RGB on the mesh and point cloud. Unsupported mesh vertices stay gray.
    "color_min_observations": 3,
    "color_depth_tolerance_m": 0.02,
    "color_sample_step": 1,
    "color_require_valid_mask": True,
    "confidence_clip_percentile": 100,

    # Remove disconnected mesh islands; no smoothing or decimation is applied.
    "mesh_min_component_triangles": 100,  # 0 retains all disconnected components.
    # The following outlier filter applies to POINT CLOUDS, not mesh vertices.
    "point_outlier_method": "statistical",  # "statistical", "radius", or "none".
    "point_neighbors": 35,
    "point_std_ratio": 1.5,
    "point_radius_min_neighbors": 12,
    "point_radius_m": 0.02,

    # Optional image/world crops; disabled for the validated full-frame workflow.
    "center_crop": False,
    "center_crop_width": 270,
    "center_crop_height": 290,
    "center_crop_depth": True,  # Only takes effect if center_crop is True.
    "world_crop": False,
    "world_crop_min_m": [-0.25, -1.1, -1.0],
    "world_crop_max_m": [0.3, -0.65, 0.5],
    "auto_object_crop": False,
    "object_min_height_m": 0.010,
    "object_cluster_eps_m": 0.020,
    "object_cluster_min_points": 30,
    "object_neighbor_max_gap_m": 0.100,
    "object_xy_margin_m": 0.020,
    "object_top_margin_m": 0.010,
    "object_table_below_margin_m": 0.010,

    # Table-plane fitting saves metadata; it does not remove the table by default.
    "fit_table_plane": True,
    "plane_downsample_m": 0.005,
    "plane_distance_threshold_m": 0.004,
    "plane_ransac_points": 3,
    "plane_iterations": 2000,
    "plane_up_axis": [0.0, 0.0, 1.0],
    "plane_max_tilt_deg": 25.0,

    # Saved preview works without a display; viewer opens only when requested.
    "save_preview": True,
    "preview_width": 960,
    "preview_height": 720,
    "preview_fov_deg": 55.0,
    "open_viewer": False,
}
# ============================================================


def main(argv=None):
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Override input_path above")
    parser.add_argument("--output", type=Path, help="Override output_root above")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs without reconstruction")
    args = parser.parse_args(argv)
    settings = dict(SETTINGS)
    if args.input is not None:
        settings["input_path"] = args.input
    if args.output is not None:
        settings["output_root"] = args.output
    if args.validate_only:
        settings["validate_only"] = True
    # Set before importing NumPy/Open3D. Existing shell limits take precedence.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, str(settings["cpu_threads"]))
    try:
        from offline_reconstruction.pipeline import execute
        return execute(settings)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"Reconstruction error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
