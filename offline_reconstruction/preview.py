"""CPU raycast preview with measured vertex colors; no GUI/OpenGL required."""
import cv2
import numpy as np
import open3d as o3d


def save_preview(mesh, path, width, height, fov):
    bounds = mesh.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    radius = max(np.linalg.norm(bounds.get_extent()) / 2, 0.001)
    half_fov = np.arctan(np.tan(np.radians(fov / 2)) * min(1, height / width))
    direction = np.array([1.0, -1.0, 0.85])
    direction /= np.linalg.norm(direction)
    eye = center + direction * radius / np.sin(half_fov) * 1.12
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rays = scene.create_rays_pinhole(fov_deg=fov, center=center, eye=eye, up=[0, 0, 1],
                                    width_px=width, height_px=height)
    hits = scene.cast_rays(rays)
    hit = np.isfinite(hits["t_hit"].numpy())
    image = np.full((height, width, 3), 245, np.uint8)
    face = hits["primitive_ids"].numpy()[hit].astype(int)
    uv = hits["primitive_uvs"].numpy()[hit]
    weights = np.column_stack((1 - uv.sum(axis=1), uv))
    colors = np.asarray(mesh.vertex_colors)[np.asarray(mesh.triangles)[face]]
    rgb = (colors * weights[:, :, None]).sum(axis=1)
    normals = hits["primitive_normals"].numpy()[hit]
    shade = 0.55 + 0.45 * np.abs(normals @ direction)
    image[hit] = np.clip(rgb * shade[:, None] * 255, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), image[:, :, ::-1]):
        raise IOError(f"Could not save preview: {path}")
