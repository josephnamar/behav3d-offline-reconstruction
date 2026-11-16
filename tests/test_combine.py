"""Combining must preserve poses/colors and fuse observations across cycles."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import open3d as o3d
import pytest
import yaml

from combine_cycles import SETTINGS as COMBINE_SETTINGS
from offline_reconstruction.combine import execute as combine, read_run
from offline_reconstruction.pipeline import execute as reconstruct
from test_pipeline import scan, settings_for


@pytest.fixture
def run(scan, tmp_path):
    """Two cycles see the same plane, each with two frames and a different color."""
    import cv2

    session = scan.parent
    first = session / "print_yaml_scan/cycle_0001/scan"
    first.parent.mkdir(parents=True)
    scan.rename(first)
    second = session / "print_yaml_scan/cycle_0002/scan"
    shutil.copytree(first, second)
    for folder, bgr in ((first, [0, 0, 255]), (second, [255, 0, 0])):
        data = yaml.safe_load((folder / "manifest.yaml").read_text())
        data["captures"] = data["captures"][:2]
        (folder / "manifest.yaml").write_text(yaml.safe_dump(data))
        for capture in data["captures"]:
            cv2.imwrite(str(folder / capture["color"]), np.full((24, 32, 3), bgr, np.uint8))
    settings = settings_for(first, tmp_path)
    settings.update(input_path=session, weight_threshold=1.0, color_min_observations=1, archive_outputs=True)
    assert reconstruct(settings) == 0
    path, = settings["output_root"].iterdir()
    return path


def options(run, tmp_path):
    result = deepcopy(COMBINE_SETTINGS)
    result.update(input_run=run, output_root=tmp_path / "combined", preview_width=160, preview_height=120)
    return result


def load_source(root, row, temporary):
    with zipfile.ZipFile(root / row["output"] / "artifacts.zip") as z:
        z.extract("tsdf_surface_mesh.ply", temporary)
    return o3d.io.read_triangle_mesh(str(temporary / "tsdf_surface_mesh.ply"))


def test_both_outputs_preserve_stack_and_joint_fusion_accumulates_cross_cycle_evidence(run, tmp_path):
    s = options(run, tmp_path)
    # Two frames per cycle cannot independently meet this threshold. Together
    # they must produce a surface, with both red and blue observations counted.
    s["fused_overrides"] = {"weight_threshold": 3.0, "color_min_observations": 3}
    assert combine(s) == 0
    report_path, = s["output_root"].glob("*/combined_report.json")
    report = json.loads(report_path.read_text())
    output = report_path.parent
    stacked = o3d.io.read_triangle_mesh(str(output / "stacked/combined_mesh.ply"))
    source_rows = json.loads((run / "summary.json").read_text())["scans"]
    offset = face_offset = 0
    for row in source_rows:
        source = load_source(run, row, tmp_path)
        n, f = len(source.vertices), len(source.triangles)
        np.testing.assert_array_equal(np.asarray(stacked.vertices)[offset:offset+n], source.vertices)
        np.testing.assert_array_equal(np.asarray(stacked.vertex_colors)[offset:offset+n], source.vertex_colors)
        np.testing.assert_array_equal(np.asarray(stacked.triangles)[face_offset:face_offset+f], np.asarray(source.triangles) + offset)
        offset += n; face_offset += f
    assert len(stacked.vertices) == offset and len(stacked.triangles) == face_offset
    fused = o3d.io.read_triangle_mesh(str(output / "fused/tsdf_surface_mesh.ply"))
    assert 0 < len(fused.triangles) < len(stacked.triangles)
    assert np.median(np.asarray(fused.vertices)[:, 2]) == pytest.approx(.495, abs=1e-5)
    colors = np.asarray(fused.vertex_colors)
    supported = colors[:, 1] < .1
    assert np.count_nonzero(supported) > len(colors) / 2
    # Undistortion interpolates with the black border at the image edge, so
    # brightness can drop there; the two equally observed channels stay equal.
    np.testing.assert_array_equal(colors[supported, 0], colors[supported, 2])
    np.testing.assert_array_equal(colors[supported, 1], 0)
    assert np.median(colors[supported, 0]) == pytest.approx(.5, abs=1/255)
    assert report["outputs"]["fused"]["frames"] == 4
    assert report["outputs"]["fused"]["source_files_unchanged"]
    assert (output / "stacked/preview.png").is_file() and (output / "fused/preview.png").is_file()


def test_mixed_coordinate_frames_are_rejected(run):
    summary = json.loads((run / "summary.json").read_text())
    summary["scans"][1]["coordinate_frame"] = "different_base"
    (run / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="coordinate frame"):
        read_run(run)


def test_unknown_cycle_selection_does_not_silently_drop_requests(run):
    with pytest.raises(ValueError, match="Unknown cycle"):
        read_run(run, ["cycle_0001", "absent"])


def test_stack_still_works_without_raw_inputs_and_failed_fusion_is_reported(run, tmp_path):
    source = Path(json.loads((run / "summary.json").read_text())["scans"][0]["source"]).parents[2]
    source.rename(source.with_name("moved_source"))
    s = options(run, tmp_path)
    assert combine(s) == 1
    report_path, = s["output_root"].glob("*/combined_report.json")
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["outputs"]["stacked"]["status"] == "complete"
    assert report["outputs"]["fused"]["status"] == "failed"
    s.update(make_stacked=False, source_root_override=source.with_name("moved_source"))
    assert combine(s) == 0


def test_fusion_rejects_changed_calibration_or_incompatible_alignment(run, tmp_path):
    s = options(run, tmp_path)
    s.update(make_stacked=False, fused_overrides={"undistort": False})
    assert combine(s) == 1
    report_path, = s["output_root"].glob("*/combined_report.json")
    assert "must match the saved alignment" in json.loads(report_path.read_text())["outputs"]["fused"]["error"]
    s["fused_overrides"] = {}
    source = Path(json.loads((run / "summary.json").read_text())["scans"][0]["source"]).parents[2]
    calibration = source / "config/extrinsics.yaml"
    calibration.write_text(calibration.read_text() + "\n# changed calibration file\n")
    assert combine(s) == 1
    reports = [json.loads(p.read_text()) for p in s["output_root"].glob("*/combined_report.json")]
    assert any("differs from the reconstructed cycle" in r["outputs"]["fused"]["error"] for r in reports)
