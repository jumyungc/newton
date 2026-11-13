# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check clipped hydroelastic faces against independent polygon integration."""

import itertools
import unittest

import numpy as np
import warp as wp

from newton._src.geometry.sdf_hydroelastic import mc_calc_face_texture, vec8f
from newton._src.geometry.sdf_texture import TextureSDFData
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _clipped_faces(
    edges: wp.array[wp.vec2ub],
    depths: wp.array[wp.vec3],
    scales: wp.array[float],
    areas: wp.array[float],
    centers: wp.array[wp.vec3],
    pressures: wp.array[float],
):
    i = wp.tid()
    iso = vec8f(-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    pair = vec8f()
    pair[1] = 2.0 * depths[i][0]
    pair[3] = 2.0 * depths[i][1]
    pair[4] = 2.0 * depths[i][2]
    depth_scale = wp.max(wp.abs(depths[i][0]), wp.max(wp.abs(depths[i][1]), wp.abs(depths[i][2])))
    # For kh=2 on each shape, p_a-p_b is proportional to iso. At the
    # interface p_a=p_b=-pair_separation, so the analytic force is known.
    own = (pair - iso * depth_scale) * 0.5
    other = (pair + iso * depth_scale) * 0.5
    data = TextureSDFData()
    data.voxel_size = wp.vec3(scales[i], scales[i], scales[i])
    data.sdf_box_lower = wp.vec3(0.0)
    force_area, _area, _normal, center, depth, _separation, _vertices = mc_calc_face_texture(
        edges, 0, iso, own, other, data, 0, 0, 0, 0.0, 1.0
    )
    areas[i] = force_area
    centers[i] = center
    pressures[i] = wp.max(-2.0 * depth, 0.0)


def _polygon_integrals(vertices, depths):
    """Clip explicit vertices, then integrate each retained triangle in float64."""
    polygon = []
    for i, (end, de) in enumerate(zip(vertices, depths, strict=True)):
        start, ds = vertices[i - 1], depths[i - 1]
        if (ds < 0.0) != (de < 0.0):
            polygon.append((start + ds / (ds - de) * (end - start), 0.0))
        if de < 0.0:
            polygon.append((end, -de))
    area, force = 0.0, 0.0
    first_moment = np.zeros(3)
    for i in range(1, len(polygon) - 1):
        triangle = [polygon[0], polygon[i], polygon[i + 1]]
        points = np.array([p[0] for p in triangle])
        pressure = np.array([p[1] for p in triangle])
        triangle_area = 0.5 * np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0]))
        area += triangle_area
        force += triangle_area * pressure.mean()
        first_moment += triangle_area * points.mean(axis=0)
    return area, force, first_moment / area


def test_clipped_face_centroid(test, device):
    """Partial faces use their clipped centroid, including a positive uncut mean depth."""
    rows = []
    for base in ((-3.0, 1.0, 1.0), (-1.0, 1.0, 1.0), (3.0, -1.0, -1.0), (-1.0, -1.0, -1.0)):
        for values in sorted(set(itertools.permutations(base))):
            for depth_scale in (1.0e-3, 1.0e-5):
                for scale in (0.001, 1.0):
                    rows.append((np.array(values) * depth_scale, scale))
    depths_np = np.array([d for d, _ in rows], dtype=np.float32)
    scales_np = np.array([s for _, s in rows], dtype=np.float32)
    expected = []
    for depths, scale in zip(depths_np, scales_np, strict=True):
        vertices = np.eye(3) * float(scale) * 0.5
        expected.append(_polygon_integrals(vertices, depths.astype(np.float64)))
    expected_area = np.array([r[0] for r in expected])
    expected_force = np.array([r[1] for r in expected])
    expected_centroid = np.array([r[2] for r in expected])
    with wp.ScopedDevice(device):
        edges = wp.array([[0, 1], [0, 3], [0, 4]], dtype=wp.vec2ub)
        depths = wp.array(depths_np, dtype=wp.vec3)
        scales = wp.array(scales_np, dtype=float)
        area = wp.empty(len(rows), dtype=float)
        center = wp.empty(len(rows), dtype=wp.vec3)
        pressure = wp.empty(len(rows), dtype=float)
        inputs = [edges, depths, scales]
        outputs = [area, center, pressure]
        wp.launch(_clipped_faces, len(rows), inputs=inputs, outputs=outputs)
        actual = [a.numpy().copy() for a in outputs]
        test.assertTrue(all(np.isfinite(a).all() for a in actual))
        np.testing.assert_allclose(actual[0], expected_area, rtol=7.0e-7, atol=0.0)
        np.testing.assert_allclose(actual[0].astype(float) * actual[2], expected_force, rtol=7.0e-7, atol=0.0)
        np.testing.assert_allclose(actual[1] / scales_np[:, None], expected_centroid / scales_np[:, None], atol=1.0e-7)
        if device.is_cuda:
            with wp.ScopedCapture() as capture:
                wp.launch(_clipped_faces, len(rows), inputs=inputs, outputs=outputs)
            for _ in range(3):
                wp.capture_launch(capture.graph)
                for array, reference in zip(outputs, actual, strict=True):
                    np.testing.assert_array_equal(array.numpy(), reference)


class TestHydroelasticClippedFaces(unittest.TestCase):
    pass


add_function_test(
    TestHydroelasticClippedFaces,
    "test_clipped_face_centroid",
    test_clipped_face_centroid,
    devices=get_test_devices(),
    check_output=False,
)


if __name__ == "__main__":
    unittest.main()
