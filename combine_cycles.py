#!/usr/bin/env python3
"""Create both a complete mesh overlay and a fused reconstruction of all cycles."""
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent

# Edit this block, then run: python combine_cycles.py
SETTINGS = {
    "input_run": REPO_DIR / "results" / "YOUR_RUN_ID",  # Folder with summary.json.
    "output_root": REPO_DIR / "results" / "combined",
    "make_stacked": True,  # Preserve every cycle's vertices, faces, colors and pose.
    "make_fused": True,  # Integrate the original depth captures into one TSDF.
    "include_stacked_normals": False,  # Optional shading vectors; viewers can recompute them.
    "archive_stacked": False,  # Stream the full stacked PLY directly into a lossless ZIP.
    "cycle_names": [],  # [] = every cycle; e.g. ["cycle_0001", "cycle_0002"].
    # If the raw session moved, set its new path here (same folder level as the
    # original reconstruction's input_path). None uses the saved source paths.
    "source_root_override": None,
    # Fusion inherits ALL settings from the input run's settings.json.
    # Optional overrides: {"weight_threshold": 3.0, "voxel_size_m": 1 / 512}.
    # Undistortion, depth_scale and alignment_depth_max_m must match the saved
    # alignment. Rerun reconstruct.py to change those three settings.
    "fused_overrides": {},
    "cpu_threads": 4,
    # Preview sampling only: these never simplify the complete stacked mesh.
    "preview_voxel_size_m": 0.004,
    "preview_width": 1200,
    "preview_height": 900,
    "preview_fov_deg": 55.0,
}


def main(argv=None):
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Completed reconstruction run")
    parser.add_argument("--output", type=Path, help="Output root for new combined models")
    parser.add_argument("--source-root", type=Path, help="New location of the raw input folder")
    parser.add_argument("--mode", choices=("both", "stacked", "fused"), help="Override which outputs to produce")
    parser.add_argument("--archive-stacked", action="store_true", help="Write stacked/combined_mesh.zip without an unpacked temporary model")
    args = parser.parse_args(argv)
    settings = dict(SETTINGS)
    for argument, key in ((args.input, "input_run"), (args.output, "output_root"),
                          (args.source_root, "source_root_override")):
        if argument is not None:
            settings[key] = argument
    if args.mode:
        settings["make_stacked"] = args.mode in ("both", "stacked")
        settings["make_fused"] = args.mode in ("both", "fused")
    if args.archive_stacked:
        settings["archive_stacked"] = True
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, str(settings["cpu_threads"]))
    try:
        from offline_reconstruction.combine import execute
        return execute(settings)
    except (ValueError, OSError, ImportError) as exc:
        print(f"Combine error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
