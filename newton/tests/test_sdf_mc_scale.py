# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep marching-cubes contact geometry independent of length units."""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.sdf_hydroelastic import mc_calc_face_texture, vec8f
from newton._src.geometry.sdf_mc import get_triangle_fraction
from newton._src.geometry.sdf_texture import TextureSDFData
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _fractions(depths: wp.array[wp.vec3], output: wp.array[float]):
    i = wp.tid()
    d = depths[i]
    inside = int(d[0] < 0.0) + int(d[1] < 0.0) + int(d[2] < 0.0)
    output[i] = get_triangle_fraction(d, inside)


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


def test_triangle_fraction_scale(test, device):
    """Uniform depth scaling preserves analytic clipped areas, including zero vertices."""
    cases = [
        ((-1.0, 2.0, 3.0), 1.0 / 12.0),
        ((1.0, -2.0, -3.0), 11.0 / 12.0),
        ((-3.0, 1.0, 1.0), 9.0 / 16.0),
        ((3.0, -1.0, -1.0), 7.0 / 16.0),
        ((0.0, -1.0, -2.0), 1.0),
        ((-1.0, 0.0, 0.0), 1.0),
        ((0.0, 0.0, 0.0), 0.0),
        ((1.0, 2.0, 3.0), 0.0),
        ((-1.0, -2.0, -3.0), 1.0),
    ]
    depths, expected = [], []
    for values, fraction in cases:
        for rotation in range(3):
            for scale in (1.0e-30, 1.0e-15, 1.0e-5, 1.0, 1.0e15, 1.0e30):
                depths.append(np.roll(values, rotation) * scale)
                expected.append(fraction)
    with wp.ScopedDevice(device):
        inputs = wp.array(np.asarray(depths, dtype=np.float32), dtype=wp.vec3)
        result = wp.empty(len(depths), dtype=float)
        wp.launch(_fractions, len(depths), inputs=[inputs], outputs=[result])
        actual = result.numpy().copy()
        test.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-7)
        if device.is_cuda:
            with wp.ScopedCapture() as capture:
                wp.launch(_fractions, len(depths), inputs=[inputs], outputs=[result])
            for _ in range(3):
                wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(result.numpy(), actual)


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


class TestSDFMCScale(unittest.TestCase):
    pass


for name, function in (
    ("test_triangle_fraction_scale", test_triangle_fraction_scale),
    ("test_triangle_face_scale", test_triangle_face_scale),
):
    add_function_test(TestSDFMCScale, name, function, devices=get_test_devices(), check_output=False)


if __name__ == "__main__":
    unittest.main()
