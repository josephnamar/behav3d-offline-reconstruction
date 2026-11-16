"""Input discovery, strict validation and sequential offline reconstruction."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

import cv2
import numpy as np
import open3d as o3d
import yaml

from . import color_to_depth, fusion
from .utils.intrinsics import load_intrinsics
from .utils.rectification import rectify_images


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(paths, relative_to):
    return {str(path.relative_to(relative_to)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))}


def discover_scans(input_path, session_names=(), scan_folders=(), output_root=None):
    root = Path(input_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input does not exist: {root}. Set input_path in reconstruct.py or use --input.")
    if root.is_file():
        if root.name != "manifest.yaml":
            raise ValueError("Input file must be a manifest.yaml; otherwise select its folder.")
        manifests = [root]
    elif (root / "manifest.yaml").is_file():
        manifests = [root / "manifest.yaml"]
    else:
        manifests = []
        excluded = {".git", ".venv", "__pycache__", "reconstruct"}
        output = Path(output_root).expanduser().resolve() if output_root else None
        for directory, children, files in os.walk(root):
            directory = Path(directory)
            children[:] = sorted(name for name in children if name not in excluded
                                 and (directory / name).resolve() != output)
            if "manifest.yaml" in files:
                manifests.append(directory / "manifest.yaml")
    selected = sorted({p.parent for p in manifests
                       if (not session_names or p.parent.parent.name in session_names)
                       and (not scan_folders or p.parent.name in scan_folders)})
    if not selected:
        raise ValueError(f"No scans matched under {root}. Expected SESSION/SCAN/manifest.yaml.")
    return selected


def _finite_vector(value, length, label):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain {length} finite numbers") from exc
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain {length} finite numbers")
    return array


def validate_settings(s):
    positive = ("depth_scale", "depth_max_m", "alignment_depth_max_m", "voxel_size_m",
                "trunc_voxel_multiplier", "weight_threshold", "color_depth_tolerance_m",
                "point_std_ratio", "point_radius_m", "plane_distance_threshold_m",
                "object_cluster_eps_m")
    for key in positive:
        if not np.isfinite(s[key]) or s[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    integer_positive = ("block_count", "block_resolution", "cpu_threads", "color_min_observations",
                        "color_sample_step", "point_neighbors", "point_radius_min_neighbors",
                        "center_crop_width", "center_crop_height", "object_cluster_min_points",
                        "plane_ransac_points", "plane_iterations", "preview_width", "preview_height")
    for key in integer_positive:
        if isinstance(s[key], bool) or not isinstance(s[key], int) or s[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("mesh_min_component_triangles", "random_seed"):
        if isinstance(s[key], bool) or not isinstance(s[key], int) or s[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("plane_downsample_m", "object_min_height_m", "object_neighbor_max_gap_m",
                "object_xy_margin_m", "object_top_margin_m", "object_table_below_margin_m"):
        if not np.isfinite(s[key]) or s[key] < 0:
            raise ValueError(f"{key} must be nonnegative and finite")
    if not np.isfinite(s["depth_bias_mm"]):
        raise ValueError("depth_bias_mm must be finite")
    if s["point_outlier_method"] not in ("none", "statistical", "radius"):
        raise ValueError("point_outlier_method must be none, statistical or radius")
    if not 0 < s["confidence_clip_percentile"] <= 100:
        raise ValueError("confidence_clip_percentile must be in (0, 100]")
    if not 0 < s["preview_fov_deg"] < 150:
        raise ValueError("preview_fov_deg must be between 0 and 150")
    if not 0 <= s["plane_max_tilt_deg"] <= 90:
        raise ValueError("plane_max_tilt_deg must be in [0, 90]")
    if s["plane_ransac_points"] < 3:
        raise ValueError("plane_ransac_points must be at least 3")
    low = _finite_vector(s["world_crop_min_m"], 3, "world_crop_min_m")
    high = _finite_vector(s["world_crop_max_m"], 3, "world_crop_max_m")
    if np.any(high <= low):
        raise ValueError("world_crop_max_m must exceed world_crop_min_m on every axis")
    if np.linalg.norm(_finite_vector(s["plane_up_axis"], 3, "plane_up_axis")) == 0:
        raise ValueError("plane_up_axis cannot be zero")
    if s["auto_object_crop"] and not s["fit_table_plane"]:
        raise ValueError("auto_object_crop requires fit_table_plane")
    if s["device"] not in ("CPU:0", "CUDA:0"):
        raise ValueError("device must be CPU:0 or CUDA:0")
    if s["device"] == "CUDA:0" and not o3d.core.cuda.is_available():
        raise ValueError("CUDA requested but unavailable in this Open3D installation; choose CPU:0")
    for key in ("session_names", "scan_folders"):
        if not isinstance(s[key], (list, tuple)) or any(not isinstance(x, str) for x in s[key]):
            raise ValueError(f"{key} must be a list of folder names")


def validate_scan(scan):
    """Check every raw frame and explicit pose; never substitute an identity pose."""
    session = scan.parent
    files = [scan / "manifest.yaml"]
    calibration = {}
    for kind in ("depth", "color"):
        path = session / "config" / f"{kind}_intrinsics.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Missing calibration: {path}")
        width, height, matrix, distortion = load_intrinsics(str(path))
        rectify_images([], matrix, distortion, (width, height))
        calibration[kind] = (height, width)
        files.append(path)
    extrinsics_path = session / "config/extrinsics.yaml"
    extrinsics = yaml.safe_load(extrinsics_path.read_text())
    frames = extrinsics.get("frames", {}) if isinstance(extrinsics, dict) else {}
    for key in ("T_tool0_ir", "T_tool0_color"):
        frame = frames.get(key, {})
        _finite_vector(frame.get("xyz"), 3, key + ".xyz")
        if np.linalg.norm(_finite_vector(frame.get("quat_xyzw"), 4, key + ".quat_xyzw")) == 0:
            raise ValueError(f"Zero quaternion in {extrinsics_path}: {key}")
    files.append(extrinsics_path)
    manifest = yaml.safe_load((scan / "manifest.yaml").read_text())
    captures = manifest.get("captures") if isinstance(manifest, dict) else None
    if not isinstance(captures, list) or not captures:
        raise ValueError(f"No captures in {scan / 'manifest.yaml'}")
    frame_ids, paths_by_kind, pose_frames = set(), {"depth": set(), "color": set()}, set()
    for i, capture in enumerate(captures):
        label = f"{scan.name} capture {i}"
        if not isinstance(capture, dict):
            raise ValueError(f"{label}: capture must be a mapping")
        if "index" in capture:
            index = capture["index"]
            if not isinstance(index, int) or index in frame_ids:
                raise ValueError(f"{label}: duplicate/invalid capture index {index}")
            frame_ids.add(index)
        pose = capture.get("T_base_tool0", {})
        _finite_vector(pose.get("position"), 3, label + " position")
        if np.linalg.norm(_finite_vector(pose.get("orientation_xyzw"), 4, label + " quaternion")) == 0:
            raise ValueError(f"{label}: zero quaternion")
        pose_frames.add((pose.get("parent", "base"), pose.get("child", "tool0")))
        for kind in ("depth", "color"):
            relative = capture.get(kind)
            if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError(f"{label}: {kind} must be a relative image path within the scan")
            image_path = scan / relative
            if image_path in paths_by_kind[kind]:
                raise ValueError(f"{label}: duplicate {kind} image {relative}")
            paths_by_kind[kind].add(image_path)
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Unreadable {kind} image: {image_path}")
            if image.shape[:2] != calibration[kind]:
                raise ValueError(f"Image dimensions differ from {kind} calibration: {image_path}")
            if kind == "depth" and (image.dtype != np.uint16 or image.ndim != 2):
                raise ValueError(f"Depth must be single-channel uint16: {image_path}")
            if kind == "color" and (image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3):
                raise ValueError(f"Color must be three-channel uint8: {image_path}")
            files.append(image_path)
    if len(pose_frames) != 1:
        raise ValueError(f"Inconsistent robot pose frames in {scan}")
    return dict(frames=len(captures), coordinate_frame=next(iter(pose_frames))[0],
                files=files, hashes=fingerprint(files, session))


def configure_fusion(s):
    """Explicitly reset module settings for every sequential scan."""
    mapping = {
        "DEPTH_BIAS_MM": "depth_bias_mm", "C2D_REQUIRE_VALID_MASK_FOR_OBS": "color_require_valid_mask",
        "OUTLIER_METHOD": "point_outlier_method", "OUTLIER_NB_NEIGHBORS": "point_neighbors",
        "OUTLIER_STD_RATIO": "point_std_ratio", "OUTLIER_NB_POINTS": "point_radius_min_neighbors",
        "OUTLIER_RADIUS": "point_radius_m", "CONF_CLIP_PERCENTILE": "confidence_clip_percentile",
        "TABLE_PLANE_DOWNSAMPLE": "plane_downsample_m", "TABLE_PLANE_RANSAC_THRESH": "plane_distance_threshold_m",
        "TABLE_PLANE_RANSAC_N": "plane_ransac_points", "TABLE_PLANE_RANSAC_ITERS": "plane_iterations",
        "TABLE_PLANE_UP_AXIS": "plane_up_axis", "TABLE_PLANE_MAX_TILT_DEG": "plane_max_tilt_deg",
    }
    for attribute, key in mapping.items():
        setattr(fusion, attribute, s[key])
    fusion.C2D_ERODE_ENABLE = False
    fusion.MESH_REMOVE_SMALL_COMPONENTS = True
    fusion.TABLE_PLANE_MODE = "fit" if s["fit_table_plane"] else "off"
    fusion.TABLE_PLANE_VIS_ENABLE = False
    fusion.TABLE_SLICE_ENABLE = False


def reconstruct_scan(scan, output, s):
    configure_fusion(s)
    o3d.utility.random.seed(s["random_seed"])
    color_to_depth.run(str(scan.parent), scan.name, output_dir=output,
                       rectify_images=s["undistort"], depth_scale=s["depth_scale"],
                       depth_max=s["alignment_depth_max_m"])
    fusion.run(
        str(scan.parent), scan.name, visualize=False, output_dir=output,
        device=s["device"], rectify_images=s["undistort"],
        voxel_size=s["voxel_size_m"], block_count=s["block_count"], block_resolution=s["block_resolution"],
        depth_max=s["depth_max_m"], depth_scale=s["depth_scale"],
        trunc_voxel_multiplier=s["trunc_voxel_multiplier"], weight_threshold=s["weight_threshold"],
        color_sample_step=s["color_sample_step"], color_z_band_m=s["color_depth_tolerance_m"],
        color_min_obs=s["color_min_observations"], mesh_min_component_triangles=s["mesh_min_component_triangles"],
        center_crop_enable=s["center_crop"], center_crop_width=s["center_crop_width"],
        center_crop_height=s["center_crop_height"], center_crop_apply_to_depth=s["center_crop_depth"],
        aabb_crop_enable=s["world_crop"], aabb_crop_min=s["world_crop_min_m"], aabb_crop_max=s["world_crop_max_m"],
        auto_object_crop_enable=s["auto_object_crop"], auto_object_min_height_m=s["object_min_height_m"],
        auto_object_cluster_eps_m=s["object_cluster_eps_m"], auto_object_cluster_min_points=s["object_cluster_min_points"],
        auto_object_neighbor_max_gap_m=s["object_neighbor_max_gap_m"], auto_object_xy_margin_m=s["object_xy_margin_m"],
        auto_object_top_margin_m=s["object_top_margin_m"], auto_object_table_below_margin_m=s["object_table_below_margin_m"],
    )
    mesh = o3d.io.read_triangle_mesh(str(output / "tsdf_surface_mesh.ply"))
    pcd = o3d.io.read_point_cloud(str(output / "tsdf_surface_rgb_colored.ply"))
    if not mesh.has_vertex_colors() or not mesh.has_triangles():
        raise ValueError("Exported mesh is empty or missing RGB")
    if s["save_preview"]:
        from .preview import save_preview
        save_preview(mesh, output / "preview.png", s["preview_width"], s["preview_height"], s["preview_fov_deg"])
    bounds = mesh.get_axis_aligned_bounding_box()
    metrics = dict(vertices=len(mesh.vertices), triangles=len(mesh.triangles), colored_points=len(pcd.points),
                   mesh_bounds_min_m=bounds.min_bound.tolist(), mesh_bounds_max_m=bounds.max_bound.tolist(),
                   full_scene_aabb_dimensions_mm=(bounds.get_extent() * 1000).tolist())
    if s["open_viewer"]:
        o3d.visualization.draw_geometries([mesh, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)],
                                        window_name=f"{scan.parent.name}/{scan.name}")
    return metrics


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def execute(settings):
    validate_settings(settings)
    output_root = Path(settings["output_root"]).expanduser().resolve()
    scans = discover_scans(settings["input_path"], settings["session_names"], settings["scan_folders"], output_root)
    for scan in scans:
        if output_root == scan.parent or scan.parent in output_root.parents:
            raise ValueError(f"output_root must be outside the source session: {scan.parent}")
    print(f"Found {len(scans)} scan(s). Validating all raw frames, poses and calibration.")
    validated, failures = {}, []
    for scan in scans:
        try:
            validated[scan] = validate_scan(scan)
            print(f"Validated {scan.parent.name}/{scan.name}: {validated[scan]['frames']} captures")
        except Exception as exc:
            failures.append(f"{scan}: {exc}")
    if failures:
        raise ValueError("Input validation failed before reconstruction:\n" + "\n".join(failures))
    if settings["validate_only"]:
        print("Validation passed. No output files written.")
        return 0
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "open3d", "PyYAML")}
    versions.update(opencv=cv2.__version__, python=sys.version)
    write_json(run_dir / "settings.json", dict(settings=settings, versions=versions,
               effective_depth_bias_mm=round(settings["depth_bias_mm"] / 1000 * settings["depth_scale"]) / settings["depth_scale"] * 1000,
               thread_environment={key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}))
    common = Path(os.path.commonpath([str(scan.parent) for scan in scans]))
    summary = dict(status="running", scans=[], settings="settings.json")
    write_json(run_dir / "summary.json", summary)
    for scan in scans:
        relative = Path(common.name) / scan.relative_to(common)
        output = run_dir / relative
        output.mkdir(parents=True, exist_ok=False)
        info = validated[scan]
        write_json(output / "input_provenance.json", dict(source=str(scan), sha256=info["hashes"]))
        start = time.monotonic()
        row = dict(source=str(scan), output=str(relative), frames=info["frames"], coordinate_frame=info["coordinate_frame"], status="running")
        write_json(output / "result.json", row)
        with (output / "run.log").open("w") as logfile, redirect_stdout(Tee(sys.stdout, logfile)), redirect_stderr(Tee(sys.stderr, logfile)):
            print(f"\nReconstructing {scan}\nOutput: {output}")
            try:
                row.update(reconstruct_scan(scan, output, settings))
                after = fingerprint(info["files"], scan.parent)
                if after != info["hashes"]:
                    raise ValueError("Source files changed during reconstruction; result cannot be trusted.")
                row.update(status="complete", source_files_unchanged=True)
            except Exception as exc:
                row.update(status="failed", error=str(exc))
                traceback.print_exc()
            finally:
                row["elapsed_seconds"] = round(time.monotonic() - start, 3)
                write_json(output / "result.json", row)
                summary["scans"].append(row)
                write_json(run_dir / "summary.json", summary)
        if row["status"] == "failed" and not settings["continue_on_error"]:
            break
    failed = sum(row["status"] != "complete" for row in summary["scans"])
    summary.update(status="failed" if failed else "complete", selected_scans=len(scans),
                   completed_scans=len(summary["scans"]) - failed, failed_scans=failed,
                   unprocessed_scans=len(scans) - len(summary["scans"]))
    write_json(run_dir / "summary.json", summary)
    print(f"\nFinished: {summary['completed_scans']}/{len(scans)} succeeded. Results: {run_dir}")
    return 1 if failed else 0
