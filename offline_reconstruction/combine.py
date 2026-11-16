"""Stream complete cycle meshes and raw captures into two distinct 3D outcomes."""
from contextlib import ExitStack, contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import time
import traceback
import uuid
import zipfile

import cv2
import numpy as np
import open3d as o3d

from . import fusion
from .pipeline import (Tee, configure_fusion, fingerprint, session_for_scan,
                       validate_scan, validate_settings, write_json)
from .utils.session import Session


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_path(value):
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts or ":" in path.parts[0]:
        raise ValueError(f"Expected a relative result path: {value}")
    return Path(*path.parts)


def read_run(path, cycles=()):
    root = Path(path).expanduser().resolve()
    if root.name == "summary.json":
        root = root.parent
    summary = json.loads((root / "summary.json").read_text())
    if summary.get("status") != "complete":
        raise ValueError("Select a completed reconstruction run with summary.json; partial runs are not combined.")
    rows = summary.get("scans", [])
    if cycles:
        rows = [row for row in rows if any(name in relative_path(row["output"]).parts for name in cycles)]
        found = {name for name in cycles if any(name in relative_path(row["output"]).parts for row in rows)}
        if found != set(cycles):
            raise ValueError(f"Unknown cycle selection: {sorted(set(cycles) - found)}")
    if not rows or any(row.get("status") != "complete" for row in rows):
        raise ValueError("Every selected cycle must have a completed reconstruction.")
    frames = {row.get("coordinate_frame") for row in rows}
    if len(frames) != 1 or None in frames or "" in frames:
        raise ValueError("Cycles must explicitly share one coordinate frame; no automatic registration is applied.")
    folders = [root / relative_path(row["output"]) for row in rows]
    if len(set(folders)) != len(folders):
        raise ValueError("The run lists a cycle more than once.")
    for folder, row in zip(folders, rows):
        result = json.loads((folder / "result.json").read_text())
        for key in ("status", "vertices", "triangles", "coordinate_frame", "frames", "source"):
            if result.get(key) != row.get(key):
                raise ValueError(f"Cycle metadata differs from the run summary: {folder} ({key})")
    return root, rows, next(iter(frames))


@contextmanager
def artifact(root, row, name, scratch):
    folder = root / relative_path(row["output"])
    path = folder / relative_path(name)
    if path.is_file():
        yield path
        return
    archive = folder / relative_path(row.get("artifact_archive", "artifacts.zip"))
    with tempfile.TemporaryDirectory(dir=scratch) as temporary, zipfile.ZipFile(archive) as packed:
        target = Path(temporary) / Path(name).name
        with packed.open(name) as source, target.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        yield target


@contextmanager
def alignment(root, row, settings, scratch):
    name = "color_in_depth_rectified" if settings["undistort"] else "color_in_depth"
    folder = root / relative_path(row["output"])
    if (folder / name).is_dir():
        yield folder / name
        return
    archive = folder / relative_path(row.get("artifact_archive", "artifacts.zip"))
    with tempfile.TemporaryDirectory(dir=scratch) as temporary, zipfile.ZipFile(archive) as packed:
        target = Path(temporary) / name
        target.mkdir()
        for member in packed.namelist():
            if member.startswith(name + "/") and not member.endswith("/"):
                relative = relative_path(member)
                if len(relative.parts) != 2:
                    raise ValueError(f"Unexpected aligned-image archive path: {member}")
                with packed.open(member) as source, (target / relative.name).open("wb") as destination:
                    shutil.copyfileobj(source, destination)
        yield target


def mesh_preview_cloud(mesh, voxel):
    cloud = o3d.geometry.PointCloud()
    cloud.points = mesh.vertices
    cloud.colors = mesh.vertex_colors
    if mesh.has_vertex_normals():
        cloud.normals = mesh.vertex_normals
    return cloud.voxel_down_sample(voxel)


def save_cloud_preview(cloud, output, settings, bounds):
    """Identical CPU projection for both outcomes; only the preview is sampled."""
    width, height, fov = settings["preview_width"], settings["preview_height"], settings["preview_fov_deg"]
    low, high = np.asarray(bounds[0]), np.asarray(bounds[1])
    center = (low + high) / 2
    radius = max(np.linalg.norm(high - low) / 2, .001)
    direction = np.array([1., -1., .85]); direction /= np.linalg.norm(direction)
    half_fov = np.arctan(np.tan(np.radians(fov / 2)) * min(1, height / width))
    eye = center + direction * radius / np.sin(half_fov) * 1.12
    forward = -direction
    right = np.cross(forward, [0., 0., 1.]); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    xyz = np.asarray(cloud.points) - eye
    depth = xyz @ forward
    focal = width / (2 * np.tan(np.radians(fov / 2)))
    u = np.rint(width / 2 + focal * (xyz @ right) / depth).astype(int)
    v = np.rint(height / 2 - focal * (xyz @ up) / depth).astype(int)
    valid = (depth > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    indices = np.flatnonzero(valid)
    indices = indices[np.argsort(depth[indices], kind="stable")]
    _, first = np.unique(v[indices] * width + u[indices], return_index=True)
    indices = indices[first]
    rgb = np.asarray(cloud.colors)[indices].copy()
    if cloud.has_normals():
        normals = np.asarray(cloud.normals)[indices]
        lengths = np.linalg.norm(normals, axis=1)
        shade = .55 + .45 * np.abs((normals @ direction) / np.maximum(lengths, 1e-12))
        rgb *= shade[:, None]
    image = np.full((height, width, 3), 245, np.uint8)
    image[v[indices], u[indices]] = np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(output), image[:, :, ::-1]):
        raise IOError(f"Could not save preview: {output}")


def stack_meshes(root, rows, output, settings, scratch, bounds):
    n_vertices = sum(int(row["vertices"]) for row in rows)
    n_faces = sum(int(row["triangles"]) for row in rows)
    if n_vertices >= 2**31 or min(n_vertices, n_faces) <= 0:
        raise ValueError("Combined mesh exceeds the supported int32 face-index range or is empty.")
    coordinate_names = ("x", "y", "z", "nx", "ny", "nz") if settings["include_stacked_normals"] else ("x", "y", "z")
    vertex_type = np.dtype([(name, "<f8") for name in coordinate_names]
                           + [(name, "u1") for name in ("red", "green", "blue")])
    face_type = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
    header = ("ply\nformat binary_little_endian 1.0\ncomment Complete cycle mesh overlay, coordinates in metres\n"
              f"element vertex {n_vertices}\n"
              + "".join(f"property double {name}\n" for name in coordinate_names)
              + "".join(f"property uchar {name}\n" for name in ("red", "green", "blue"))
              + f"element face {n_faces}\nproperty list uchar int vertex_indices\nend_header\n").encode("ascii")
    face_start = len(header) + n_vertices * vertex_type.itemsize
    total_size = face_start + n_faces * face_type.itemsize
    archived = settings["archive_stacked"]
    if not archived and shutil.disk_usage(output).free < total_size + 300 * 1024**2:
        raise OSError(f"Stacked mesh requires {total_size / 1024**3:.2f} GiB plus 300 MiB working space.")
    destination = output / ("combined_mesh.zip" if archived else "combined_mesh.ply")
    temporary = output / (".combined_mesh.tmp.zip" if archived else ".combined_mesh.tmp.ply")
    ply_hash = hashlib.sha256()
    records = []
    preview = o3d.geometry.PointCloud()
    vertex_offset = face_offset = 0
    with ExitStack() as contexts:
        if archived:
            packed = contexts.enter_context(zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6))
            stream = contexts.enter_context(packed.open("combined_mesh.ply", "w", force_zip64=True))
        else:
            stream = contexts.enter_context(temporary.open("w+b"))
        stream.write(header)
        ply_hash.update(header)
        if not archived:
            stream.truncate(total_size)
        for number, row in enumerate(rows, 1):
            print(f"Stacking cycle {number}/{len(rows)}: {row['output']}", flush=True)
            with artifact(root, row, "tsdf_surface_mesh.ply", scratch) as source:
                source_hash = digest(source)
                mesh = o3d.io.read_triangle_mesh(str(source))
            xyz, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
            if len(xyz) != row["vertices"] or len(faces) != row["triangles"]:
                raise ValueError(f"Mesh counts differ from saved metadata: {row['output']}")
            if not mesh.has_vertex_colors() or not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(mesh.vertex_colors)):
                raise ValueError(f"Mesh has invalid vertices or missing colors: {row['output']}")
            if faces.min() < 0 or faces.max() >= len(xyz):
                raise ValueError(f"Invalid triangle indices: {row['output']}")
            vertices = np.zeros(len(xyz), dtype=vertex_type)
            for i, name in enumerate(("x", "y", "z")):
                vertices[name] = xyz[:, i]
            if settings["include_stacked_normals"] and mesh.has_vertex_normals():
                for i, name in enumerate(("nx", "ny", "nz")):
                    vertices[name] = np.asarray(mesh.vertex_normals)[:, i]
            rgb = np.clip(np.rint(np.asarray(mesh.vertex_colors) * 255), 0, 255).astype(np.uint8)
            for i, name in enumerate(("red", "green", "blue")):
                vertices[name] = rgb[:, i]
            triangles = np.empty(len(faces), dtype=face_type)
            triangles["count"] = 3
            triangles["indices"] = faces.astype(np.int64) + vertex_offset
            if archived:
                payload = vertices.tobytes()
                stream.write(payload)
                ply_hash.update(payload)
            else:
                stream.seek(len(header) + vertex_offset * vertex_type.itemsize)
                vertices.tofile(stream)
                stream.seek(face_start + face_offset * face_type.itemsize)
                triangles.tofile(stream)
            records.append(dict(cycle=row["output"], mesh_sha256=source_hash,
                                first_vertex=vertex_offset, vertices=len(xyz), first_triangle=face_offset, triangles=len(faces)))
            vertex_offset += len(xyz)
            face_offset += len(faces)
            preview += mesh_preview_cloud(mesh, settings["preview_voxel_size_m"])
            preview = preview.voxel_down_sample(settings["preview_voxel_size_m"])
        if archived:
            # PLY stores all vertices before its faces. A second streamed pass
            # avoids a full-size unpacked temporary file or in-memory mesh.
            for number, (row, record) in enumerate(zip(rows, records), 1):
                print(f"Archiving triangles {number}/{len(rows)}: {row['output']}", flush=True)
                with artifact(root, row, "tsdf_surface_mesh.ply", scratch) as source:
                    if digest(source) != record["mesh_sha256"]:
                        raise ValueError(f"Cycle mesh changed while combining: {row['output']}")
                    mesh = o3d.io.read_triangle_mesh(str(source))
                faces = np.asarray(mesh.triangles)
                triangles = np.empty(len(faces), dtype=face_type)
                triangles["count"] = 3
                triangles["indices"] = faces.astype(np.int64) + record["first_vertex"]
                payload = triangles.tobytes()
                stream.write(payload)
                ply_hash.update(payload)
    if archived:
        with zipfile.ZipFile(temporary) as packed:
            if packed.getinfo("combined_mesh.ply").file_size != total_size or packed.testzip() is not None:
                raise IOError("Incomplete stacked mesh archive")
    elif temporary.stat().st_size != total_size:
        raise IOError("Incomplete stacked PLY write")
    temporary.replace(destination)
    if not o3d.io.write_point_cloud(str(output / "preview_points.ply"), preview):
        raise IOError("Could not save stacked preview point cloud")
    save_cloud_preview(preview, output / "preview.png", settings, bounds)
    return dict(status="complete", vertices=n_vertices, triangles=n_faces, file=destination.name,
                bytes=destination.stat().st_size, uncompressed_bytes=total_size,
                sha256=digest(destination), ply_sha256=ply_hash.hexdigest() if archived else digest(destination), cycles=records,
                vertex_normals_included=settings["include_stacked_normals"],
                geometry="All input triangles and coordinates retained; overlapping and intermediate surfaces remain.")


def source_scan(row, saved_settings, override):
    if override is None:
        return Path(row["source"]).expanduser().resolve()
    original_root = PurePosixPath(str(saved_settings["input_path"]).replace("\\", "/"))
    if original_root.name == "manifest.yaml":
        original_root = original_root.parent
    relative = PurePosixPath(str(row["source"]).replace("\\", "/")).relative_to(original_root)
    return Path(override).expanduser().resolve().joinpath(*relative.parts)


def integration_for(scan, settings, grid=None):
    session = session_for_scan(scan)
    return fusion.TSDF_Integration(
        Session(str(session), str(scan.relative_to(session))), device=settings["device"],
        voxel_size=settings["voxel_size_m"], block_count=settings["block_count"],
        block_resolution=settings["block_resolution"], depth_max=settings["depth_max_m"],
        depth_scale=settings["depth_scale"], depth_bias_m=settings["depth_bias_mm"] / 1000,
        rectify_images=settings["undistort"], trunc_voxel_multiplier=settings["trunc_voxel_multiplier"],
        weight_threshold=settings["weight_threshold"], voxel_grid=grid,
    )


def cap_history_weight(grid, cap):
    """Discount accumulated evidence without changing the current TSDF values.

    The next cycle can then move an old surface more readily. This is not a
    sliding window: surfaces outside later views are not erased.
    """
    if cap is not None:
        # Flatten the shared view: Open3D 0.19's clip kernel cannot handle the
        # five-dimensional block attribute plus its scalar operands.
        grid.attribute("weight").reshape((-1,)).clip_(0.0, float(cap))


def fuse_captures(root, rows, output, options, scratch, bounds):
    saved = json.loads((root / "settings.json").read_text())["settings"]
    unknown = set(options["fused_overrides"]) - set(saved)
    if unknown:
        raise ValueError(f"Unknown fused override settings: {sorted(unknown)}")
    settings = dict(saved, **options["fused_overrides"])
    for key in ("undistort", "depth_scale", "alignment_depth_max_m"):
        if settings[key] != saved[key]:
            raise ValueError(f"{key} must match the saved alignment; rerun reconstruct.py to change it.")
    validate_settings(settings)
    history_cap = options.get("fused_history_weight_cap")
    if history_cap is not None and history_cap <= settings["weight_threshold"]:
        raise ValueError("fused_history_weight_cap must exceed weight_threshold so surfaces remain extractable")
    configure_fusion(settings)
    write_json(output / "fusion_settings.json", dict(settings, fused_history_weight_cap=history_cap))
    scans, evidence = [], []
    for number, row in enumerate(rows, 1):
        scan = source_scan(row, saved, options["source_root_override"])
        print(f"Validating fusion input {number}/{len(rows)}: {scan}", flush=True)
        info = validate_scan(scan)
        expected = json.loads((root / relative_path(row["output"]) / "input_provenance.json").read_text())["sha256"]
        if info["hashes"] != expected or info["frames"] != row["frames"] or info["coordinate_frame"] != row["coordinate_frame"]:
            raise ValueError(f"Raw input differs from the reconstructed cycle: {scan}")
        scans.append(scan)
        evidence.append(info)
    grid = None
    for number, (row, scan) in enumerate(zip(rows, scans), 1):
        print(f"Fusing cycle {number}/{len(rows)}: {row['output']}", flush=True)
        with alignment(root, row, settings, scratch) as aligned:
            fusion.C2D_DIR = aligned
            integrator = integration_for(scan, settings, grid)
            integrator.integrate_depths()
            grid = integrator.vbg
            cap_history_weight(grid, history_cap)
    mesh, surface = integrator.extract_tsdf_surface_mesh_and_pcd()
    if not mesh.has_triangles():
        raise ValueError("Joint TSDF extraction produced no surface.")
    color_sum = np.zeros((len(surface.points), 3), dtype=np.float64)
    observations = np.zeros(len(surface.points), dtype=np.int64)
    for number, (row, scan) in enumerate(zip(rows, scans), 1):
        print(f"Coloring fused model from cycle {number}/{len(rows)}", flush=True)
        with alignment(root, row, settings, scratch) as aligned:
            fusion.C2D_DIR = aligned
            integrator = integration_for(scan, settings, grid)
            rgb, counts = integrator.colorize_tsdf_surface_points(
                surface, sample_step=settings["color_sample_step"],
                z_band_m=settings["color_depth_tolerance_m"], return_statistics=True)
            color_sum += rgb
            observations += counts
    keep = observations >= settings["color_min_observations"]
    colors = np.full_like(color_sum, .8)
    colors[keep] = color_sum[keep] / observations[keep, None]
    surface.colors = o3d.utility.Vector3dVector(colors)
    colored = surface.select_by_index(np.flatnonzero(keep).tolist())
    fusion.output_folder = output
    fusion.TABLE_PLANE_FILE = output / "table_plane.json"
    o3d.utility.random.seed(settings["random_seed"])
    fusion.finish_reconstruction(integrator, mesh, surface, colored, observations[keep].astype(float),
                                 mesh_min_component_triangles=settings["mesh_min_component_triangles"])
    exported = o3d.io.read_triangle_mesh(str(output / "tsdf_surface_mesh.ply"))
    preview = mesh_preview_cloud(exported, options["preview_voxel_size_m"])
    if not o3d.io.write_point_cloud(str(output / "preview_points.ply"), preview):
        raise IOError("Could not save fused preview points")
    save_cloud_preview(preview, output / "preview.png", options, bounds)
    for scan, info in zip(scans, evidence):
        if fingerprint(info["files"], session_for_scan(scan)) != info["hashes"]:
            raise ValueError(f"Fusion source changed during processing: {scan}")
    return dict(status="complete", vertices=len(exported.vertices), triangles=len(exported.triangles),
                file="tsdf_surface_mesh.ply", sha256=digest(output / "tsdf_surface_mesh.ply"),
                cycles=len(rows), frames=sum(row["frames"] for row in rows), source_files_unchanged=True,
                history_weight_cap=history_cap, cycle_order=[row["output"] for row in rows],
                geometry=("Joint TSDF with bounded history weight; later cycles have more influence."
                          if history_cap is not None else
                          "Joint TSDF of all original depth captures; changing surfaces may be blended or suppressed."))


def execute(options):
    if not options["make_stacked"] and not options["make_fused"]:
        raise ValueError("Enable at least one output mode.")
    if not np.isfinite(options["preview_voxel_size_m"]) or options["preview_voxel_size_m"] <= 0:
        raise ValueError("preview_voxel_size_m must be positive and finite")
    for key in ("preview_width", "preview_height", "cpu_threads"):
        if isinstance(options[key], bool) or not isinstance(options[key], int) or options[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 < options["preview_fov_deg"] < 150:
        raise ValueError("preview_fov_deg must be between 0 and 150")
    if not isinstance(options["cycle_names"], (list, tuple)) or any(not isinstance(x, str) for x in options["cycle_names"]):
        raise ValueError("cycle_names must be a list of names")
    if not isinstance(options["fused_overrides"], dict):
        raise ValueError("fused_overrides must be a dictionary")
    history_cap = options.get("fused_history_weight_cap")
    if history_cap is not None and (isinstance(history_cap, bool)
                                   or not isinstance(history_cap, (int, float))
                                   or not np.isfinite(history_cap) or history_cap <= 0):
        raise ValueError("fused_history_weight_cap must be None or a positive finite number")
    root, rows, frame = read_run(options["input_run"], options["cycle_names"])
    low = np.min([row["mesh_bounds_min_m"] for row in rows], axis=0)
    high = np.max([row["mesh_bounds_max_m"] for row in rows], axis=0)
    bounds = (low, high)
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output = Path(options["output_root"]).expanduser().resolve() / run_name
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "combine_settings.json", options)
    report = dict(status="running", input_run=str(root), coordinate_frame=frame, units="metres",
                  selected_cycles=len(rows), selected_frames=sum(row["frames"] for row in rows),
                  mesh_bounds_min_m=low.tolist(), mesh_bounds_max_m=high.tolist(), outputs={})
    write_json(output / "combined_report.json", report)
    with tempfile.TemporaryDirectory(dir=output, prefix=".working_") as scratch, \
            (output / "run.log").open("w") as log, \
            redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        for mode, enabled, function in (("stacked", options["make_stacked"], stack_meshes),
                                        ("fused", options["make_fused"], fuse_captures)):
            if not enabled:
                continue
            folder = output / mode
            folder.mkdir()
            start = time.monotonic()
            try:
                result = function(root, rows, folder, options, scratch, bounds)
            except Exception as exc:
                traceback.print_exc()
                result = dict(status="failed", error=str(exc))
            result["elapsed_seconds"] = round(time.monotonic() - start, 3)
            report["outputs"][mode] = result
            write_json(output / "combined_report.json", report)
    failed = any(row["status"] != "complete" for row in report["outputs"].values())
    report["status"] = "failed" if failed else "complete"
    write_json(output / "combined_report.json", report)
    print(f"Combined outputs: {output}\nStatus: {report['status']}", flush=True)
    return 1 if failed else 0
