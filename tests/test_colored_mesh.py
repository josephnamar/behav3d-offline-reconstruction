"""Measured mesh colors preserve unsupported geometry."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import open3d as o3d
import pytest

from offline_reconstruction.fusion import TSDF_Integration


def test_surface_rgb_uses_visible_observations_without_removing_uncolored_vertices():
    integration = TSDF_Integration.__new__(TSDF_Integration)
    integration.K = np.eye(3)
    integration.depth_scale = 1000.
    integration.depth_max = 2.
    integration.images = [np.full((4, 4), 1000, np.uint16) for _ in range(3)]
    integration.T_base_ir = [np.eye(4) for _ in range(3)]
    color = np.zeros((4, 4, 3), np.uint8)
    color[0, 0], color[0, 1], color[1, 0] = [0, 0, 255], [0, 255, 0], [255, 0, 0]
    integration.color_in_depth = [color.copy() for _ in range(3)]
    # Last point is occluded by the measured depth and must remain neutral.
    xyz = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.], [1.5, 1.5, 1.5]])
    surface = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    filtered, observations = integration.colorize_tsdf_surface_points(surface)
    np.testing.assert_array_equal(surface.points, xyz)
    np.testing.assert_allclose(surface.colors, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [.8, .8, .8]])
    np.testing.assert_array_equal(filtered.points, xyz[:3])
    np.testing.assert_array_equal(observations, [3, 3, 3])
