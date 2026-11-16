"""Exercise actual files, failure reporting and a small real TSDF reconstruction."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
import open3d as o3d
import pytest
import yaml

from reconstruct import SETTINGS
from offline_reconstruction.pipeline import discover_scans, execute, fingerprint, session_for_scan, validate_scan, validate_settings


@pytest.fixture
def scan(tmp_path):
    session = tmp_path / "data/session1"
    config = session / "config"
    config.mkdir(parents=True)
    for kind in ("depth", "color"):
        fs = cv2.FileStorage(str(config / f"{kind}_intrinsics.yaml"), cv2.FILE_STORAGE_WRITE)
        fs.write("image_width", 32)
        fs.write("image_height", 24)
        fs.write("camera_matrix", np.array([[50., 0, 15.5], [0, 50., 11.5], [0, 0, 1.]]))
        fs.write("distortion_coefficients", np.array([.1, -.02, 0, 0, 0]))
        fs.release()
    (config / "extrinsics.yaml").write_text(yaml.safe_dump({"frames": {
        key: {"xyz": [0, 0, 0], "quat_xyzw": [0, 0, 0, 1]} for key in ("T_tool0_ir", "T_tool0_color")}}))
    scan = session / "scan_yaml_targets"
    (scan / "depth_raw").mkdir(parents=True)
    (scan / "color_raw").mkdir()
    captures = []
    for i in range(5):
        depth, color = f"depth_raw/{i}.png", f"color_raw/{i}.png"
        cv2.imwrite(str(scan / depth), np.full((24, 32), 500, np.uint16))
        cv2.imwrite(str(scan / color), np.full((24, 32, 3), [25, 90, 180], np.uint8))
        captures.append(dict(index=i, depth=depth, color=color,
                             T_base_tool0=dict(position=[0, 0, 0], orientation_xyzw=[0, 0, 0, 1], parent="base", child="tool0")))
    (scan / "manifest.yaml").write_text(yaml.safe_dump(dict(captures=captures)))
    return scan


def settings_for(scan, tmp_path):
    settings = deepcopy(SETTINGS)
    settings.update(input_path=scan, output_root=tmp_path / "results", fit_table_plane=False,
                    voxel_size_m=.005, block_count=100, mesh_min_component_triangles=0,
                    preview_width=160, preview_height=120)
    return settings


def test_discovery_handles_parent_session_scan_and_manifest_without_stale_outputs(scan):
    for root in (scan.parent.parent, scan.parent, scan, scan / "manifest.yaml"):
        assert discover_scans(root) == [scan]
    (scan / "reconstruct/stale").mkdir(parents=True)
    (scan / "reconstruct/stale/manifest.yaml").write_text("captures: []")
    assert discover_scans(scan.parent) == [scan]
    with pytest.raises(ValueError, match="No scans matched"):
        discover_scans(scan.parent, session_names=["absent"])


@pytest.mark.parametrize("mutation,match", [
    (lambda c: c[0]["T_base_tool0"].pop("position"), "position"),
    (lambda c: c[0]["T_base_tool0"].update(orientation_xyzw=[0, 0, 0, 0]), "zero quaternion"),
    (lambda c: c[1].update(depth=c[0]["depth"]), "duplicate depth"),
    (lambda c: c[1].update(index=c[0]["index"]), "duplicate/invalid capture index"),
    (lambda c: c[0].update(depth="../outside.png"), "relative image path"),
])
def test_incomplete_or_ambiguous_capture_data_is_rejected(scan, mutation, match):
    manifest = yaml.safe_load((scan / "manifest.yaml").read_text())
    mutation(manifest["captures"])
    (scan / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    with pytest.raises(ValueError, match=match):
        validate_scan(scan)


def test_wrong_depth_type_fails_validation(scan):
    cv2.imwrite(str(scan / "depth_raw/0.png"), np.ones((24, 32), np.uint8))
    with pytest.raises(ValueError, match="uint16"):
        validate_scan(scan)


def test_validate_only_writes_nothing_and_rejects_output_in_original_session(scan, tmp_path):
    settings = settings_for(scan, tmp_path)
    settings["validate_only"] = True
    assert execute(settings) == 0
    assert not settings["output_root"].exists()
    settings["output_root"] = scan / "new_results"
    with pytest.raises(ValueError, match="outside the source session"):
        execute(settings)


@pytest.mark.parametrize("key,value", [("voxel_size_m", 0), ("weight_threshold", float("nan")),
    ("mesh_min_component_triangles", -1), ("color_min_observations", 0), ("color_sample_step", 1.5),
    ("point_outlier_method", "typo"), ("plane_up_axis", [0, 0, 0]), ("session_names", "session1")])
def test_invalid_processing_settings_fail_before_output(key, value):
    settings = deepcopy(SETTINGS)
    settings[key] = value
    with pytest.raises(ValueError):
        validate_settings(settings)


@pytest.mark.parametrize("undistort,bias,expected_z", [(True, -5.1, .495), (False, 0.0, .500)])
def test_real_tsdf_writes_colored_mesh_confidence_and_preview_without_changing_inputs(scan, tmp_path, undistort, bias, expected_z):
    info = validate_scan(scan)
    settings = settings_for(scan, tmp_path)
    settings.update(undistort=undistort, depth_bias_mm=bias)
    assert execute(settings) == 0
    result_path, = settings["output_root"].rglob("result.json")
    result = json.loads(result_path.read_text())
    assert result["status"] == "complete" and result["source_files_unchanged"]
    output = result_path.parent
    mesh = o3d.io.read_triangle_mesh(str(output / "tsdf_surface_mesh.ply"))
    assert len(mesh.triangles) > 0 and mesh.has_vertex_colors()
    assert np.median(np.asarray(mesh.vertices)[:, 2]) == pytest.approx(expected_z, abs=1e-5)
    assert (output / "tsdf_surface_mesh.stl").stat().st_size > 0
    assert (output / "tsdf_surface_confidence_colored.ply").stat().st_size > 0
    assert cv2.imread(str(output / "preview.png")).shape == (120, 160, 3)
    assert fingerprint(info["files"], scan.parent) == info["hashes"]
    assert not (scan / "reconstruct").exists()


def test_weight_threshold_is_honored_and_failure_is_not_reported_as_complete(scan, tmp_path):
    settings = settings_for(scan, tmp_path)
    settings["weight_threshold"] = 100
    assert execute(settings) == 1
    summary_path, = settings["output_root"].glob("*/summary.json")
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "failed" and summary["failed_scans"] == 1
    assert "no mesh" in summary["scans"][0]["error"]


def test_nested_print_cycles_use_root_calibration_and_preserve_hierarchy(scan, tmp_path):
    session = scan.parent
    nested = session / "print_yaml_scan/cycle_0001/scan"
    nested.parent.mkdir(parents=True)
    scan.rename(nested)
    second = session / "print_yaml_scan/cycle_0002/scan"
    shutil.copytree(nested, second)
    assert session_for_scan(nested) == session
    assert discover_scans(session, session_names=[session.name]) == [nested, second]
    assert discover_scans(session, scan_folders=["print_yaml_scan/cycle_0001/scan"]) == [nested]
    settings = settings_for(nested, tmp_path)
    settings["input_path"] = session
    assert execute(settings) == 0
    run, = settings["output_root"].iterdir()
    for cycle in ("cycle_0001", "cycle_0002"):
        output = run / session.name / "print_yaml_scan" / cycle / "scan"
        assert json.loads((output / "result.json").read_text())["source_files_unchanged"]
        provenance = json.loads((output / "input_provenance.json").read_text())
        assert "config/depth_intrinsics.yaml" in provenance["sha256"]
        assert (output / "tsdf_surface_mesh.ply").exists()
    settings["output_root"] = session / "print_yaml_scan/new_outputs"
    with pytest.raises(ValueError, match="outside the source session"):
        execute(settings)


def test_archived_reconstruction_can_be_restored_and_keeps_preview_and_metadata(scan, tmp_path):
    settings = settings_for(scan, tmp_path)
    settings["archive_outputs"] = True
    assert execute(settings) == 0
    result_path, = settings["output_root"].rglob("result.json")
    result = json.loads(result_path.read_text())
    output = result_path.parent
    assert result["status"] == "complete" and result["artifact_archive"] == "artifacts.zip"
    assert (output / "preview.png").is_file() and (output / "input_provenance.json").is_file()
    assert not (output / "tsdf_surface_mesh.ply").exists()
    restored = tmp_path / "restored"
    with zipfile.ZipFile(output / result["artifact_archive"]) as packed:
        assert packed.testzip() is None
        assert "color_in_depth_rectified/alignment.json" in packed.namelist()
        assert "tsdf_surface_confidence_colored.ply" in packed.namelist()
        packed.extractall(restored)
    mesh = o3d.io.read_triangle_mesh(str(restored / "tsdf_surface_mesh.ply"))
    assert len(mesh.triangles) == result["triangles"] and mesh.has_vertex_colors()
    assert np.median(np.asarray(mesh.vertices)[:, 2]) == pytest.approx(.495, abs=1e-5)
