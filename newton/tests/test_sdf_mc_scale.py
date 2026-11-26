# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep marching-cubes contact geometry independent of length units."""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.sdf_hydroelastic import mc_calc_face_texture, vec8f
from newton._src.geometry.sdf_mc import get_mc_tables
from newton._src.geometry.sdf_texture import TextureSDFData
from newton._src.geometry.sdf_utils import _generate_dense_mc_kernel
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _faces(
    edges: wp.array[wp.vec2ub],
    scales: wp.array[float],
    areas: wp.array[float],
    normals: wp.array[wp.vec3],
):
    i = wp.tid()
    field = vec8f(-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    # Consistent affine pressure fields: their difference is proportional to
    # field, and both are negative at the extracted face.
    own = (vec8f(-2.0) - field) * 0.5
    other = (vec8f(-2.0) + field) * 0.5
    data = TextureSDFData()
    data.voxel_size = wp.vec3(scales[i], scales[i], scales[i])
    data.sdf_box_lower = wp.vec3(0.0)
    _force_area, area, normal, _center, _depth, _separation, _vertices = mc_calc_face_texture(
        edges, 0, field, own, other, data, 0, 0, 0, 0.0, 1.0
    )
    areas[i] = area
    normals[i] = normal


@wp.kernel
def _corner_faces(
    edges: wp.array[wp.vec2ub],
    corner_values: wp.array[float],
    areas: wp.array[float],
    normals: wp.array[wp.vec3],
):
    i = wp.tid()
    field = vec8f(-corner_values[i], 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    own = (vec8f(-2.0) - field) * 0.5
    other = (vec8f(-2.0) + field) * 0.5
    data = TextureSDFData()
    data.voxel_size = wp.vec3(2.5e-4, 2.5e-4, 2.5e-4)
    data.sdf_box_lower = wp.vec3(200.0, 100.0, 40.0)
    # A voxel far from the grid origin, with the edge clamp disabled.
    _force_area, area, normal, _center, _depth, _separation, _vertices = mc_calc_face_texture(
        edges, 0, field, own, other, data, 100, 40, 10, 0.0, 1.0
    )
    areas[i] = area
    normals[i] = normal


def test_triangle_face_scale(test, device):
    """Regular faces retain area and orientation; coincident faces retain a finite fallback."""
    scales_np = np.array([1.0e-15, 1.0e-7, 1.0e-5, 0.01, 1.0, 1.0e7, 1.0e15], dtype=np.float32)
    expected = np.sqrt(3.0) * scales_np.astype(np.float64) ** 2 / 8.0
    with wp.ScopedDevice(device):
        scales = wp.array(scales_np, dtype=float)
        area = wp.empty(len(scales_np), dtype=float)
        normal = wp.empty(len(scales_np), dtype=wp.vec3)
        for degenerate in (False, True):
            pairs = [[0, 1], [0, 1], [0, 1]] if degenerate else [[0, 1], [0, 3], [0, 4]]
            edges = wp.array(pairs, dtype=wp.vec2ub)
            wp.launch(_faces, len(scales_np), inputs=[edges, scales], outputs=[area, normal])
            actual_area, actual_normal = area.numpy().copy(), normal.numpy().copy()
            test.assertTrue(np.isfinite(actual_area).all())
            test.assertTrue(np.isfinite(actual_normal).all())
            if degenerate:
                np.testing.assert_array_equal(actual_area, 0.0)
                np.testing.assert_array_equal(actual_normal, np.tile([0.0, 0.0, 1.0], (len(scales_np), 1)))
            else:
                np.testing.assert_allclose(actual_area, expected, rtol=5.0e-7, atol=0.0)
                np.testing.assert_allclose(actual_normal, np.full_like(actual_normal, 1.0 / np.sqrt(3.0)), rtol=5.0e-7)
            if device.is_cuda:
                with wp.ScopedCapture() as capture:
                    wp.launch(_faces, len(scales_np), inputs=[edges, scales], outputs=[area, normal])
                for _ in range(3):
                    wp.capture_launch(capture.graph)
                    np.testing.assert_array_equal(area.numpy(), actual_area)
                    np.testing.assert_array_equal(normal.numpy(), actual_normal)


def test_small_face_far_from_grid_origin(test, device):
    """Small faces keep their orientation and area in a voxel far from the grid origin."""
    values = np.array([1.0, 1.0e-2, 1.0e-4, 2.0e-6], dtype=np.float32)
    # The face cuts the three edges at corner 0 at t = v / (1 + v) voxels from it.
    cut = values.astype(np.float64) / (1.0 + values.astype(np.float64)) * 2.5e-4
    with wp.ScopedDevice(device):
        edges = wp.array([[0, 1], [0, 3], [0, 4]], dtype=wp.vec2ub)
        area = wp.empty(len(values), dtype=float)
        normal = wp.empty(len(values), dtype=wp.vec3)
        wp.launch(_corner_faces, len(values), inputs=[edges, wp.array(values, dtype=float)], outputs=[area, normal])
        np.testing.assert_allclose(normal.numpy(), np.full((len(values), 3), 1.0 / np.sqrt(3.0)), rtol=1.0e-5)
        np.testing.assert_allclose(area.numpy(), 0.5 * np.sqrt(3.0) * cut**2, rtol=1.0e-5)


def test_dense_mc_small_face_far_from_origin(test, device):
    """Dense regular-grid marching cubes keeps small-face normals far from the world origin."""
    sdf_np = np.ones((2, 2, 2), dtype=np.float32)
    sdf_np[0, 0, 0] = -1.0e-6  # Clamped to MC_EDGE_CLAMP_MIN = 0.02 (cut = 5e-6 m at voxel_size = 2.5e-4 m).
    expected_normal = np.full((1, 3), -1.0 / np.sqrt(3.0), dtype=np.float32)
    with wp.ScopedDevice(device):
        tables = get_mc_tables(device)
        tri_range_table, flat_edge_verts_table = tables[0], tables[4]
        corner_offsets_table = tables[3]
        sdf_arr = wp.array(sdf_np.ravel(), dtype=wp.float32)
        for origin in ((0.0, 0.0, 0.0), (100.0, 40.0, 10.0), (200.0, 200.0, 200.0)):
            fc = wp.zeros(1, dtype=int)
            verts = wp.zeros(3, dtype=wp.vec3)
            norms = wp.zeros(1, dtype=wp.vec3)
            wp.launch(
                _generate_dense_mc_kernel,
                dim=(1, 1, 1),
                inputs=[
                    sdf_arr,
                    2,
                    2,
                    wp.vec3(*origin),
                    wp.vec3(2.5e-4, 2.5e-4, 2.5e-4),
                    tri_range_table,
                    flat_edge_verts_table,
                    corner_offsets_table,
                    fc,
                    verts,
                    norms,
                ],
            )
            np.testing.assert_array_equal(fc.numpy(), [1])
            np.testing.assert_allclose(norms.numpy(), expected_normal, rtol=1.0e-5, atol=1.0e-5)


class TestSDFMCScale(unittest.TestCase):
    pass


for name, function in (
    ("test_triangle_face_scale", test_triangle_face_scale),
    ("test_small_face_far_from_grid_origin", test_small_face_far_from_grid_origin),
    ("test_dense_mc_small_face_far_from_origin", test_dense_mc_small_face_far_from_origin),
):
    add_function_test(TestSDFMCScale, name, function, devices=get_test_devices(), check_output=False)


if __name__ == "__main__":
    unittest.main()
