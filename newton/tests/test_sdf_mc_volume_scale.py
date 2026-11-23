# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise generic NanoVDB marching-cubes faces, not only the texture path."""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.sdf_mc import mc_calc_face, vec8f
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _fill_inside(volume: wp.uint64):
    x, y, z = wp.tid()
    wp.volume_store_f(volume, x, y, z, -1.0)


@wp.kernel
def _volume_faces(
    edges: wp.array[wp.vec2ub],
    corners: wp.array[wp.vec3ub],
    volumes: wp.array[wp.uint64],
    areas: wp.array[float],
    normals: wp.array[wp.vec3],
):
    i = wp.tid()
    field = vec8f(-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    area, normal, _center, _depth, _vertices = mc_calc_face(edges, corners, 0, field, volumes[i], 0, 0, 0)
    areas[i] = area
    normals[i] = normal


def test_generic_volume_face_scale(test, device):
    """Constant inside data gives analytic triangle areas at each volume scale."""
    scales = (1.0e-7, 1.0e-3, 1.0, 1.0e3)
    with wp.ScopedDevice(device):
        tiles = wp.array([[0, 0, 0]], dtype=wp.vec3i)
        volumes = [
            wp.Volume.allocate_by_tiles(tiles, voxel_size=scale, bg_value=-1.0, device=device) for scale in scales
        ]
        for volume in volumes:
            wp.launch(_fill_inside, dim=(8, 8, 8), inputs=[volume.id])
        ids = wp.array([volume.id for volume in volumes], dtype=wp.uint64)
        edges = wp.array([[0, 1], [0, 3], [0, 4]], dtype=wp.vec2ub)
        corners = wp.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
            dtype=wp.vec3ub,
        )
        areas = wp.empty(len(scales), dtype=float)
        normals = wp.empty(len(scales), dtype=wp.vec3)
        wp.launch(_volume_faces, dim=len(scales), inputs=[edges, corners, ids], outputs=[areas, normals])
        actual_area, actual_normal = areas.numpy().copy(), normals.numpy().copy()
        test.assertTrue(np.isfinite(actual_area).all())
        test.assertTrue(np.isfinite(actual_normal).all())
        expected = np.sqrt(3.0) * np.square(np.asarray(scales, dtype=np.float64)) / 8.0
        np.testing.assert_allclose(actual_area, expected, rtol=5.0e-7, atol=0.0)
        np.testing.assert_allclose(actual_normal, np.full_like(actual_normal, 1.0 / np.sqrt(3.0)), rtol=5.0e-7)
        with wp.ScopedCapture() as capture:
            wp.launch(_volume_faces, dim=len(scales), inputs=[edges, corners, ids], outputs=[areas, normals])
        for _ in range(2):
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(areas.numpy(), actual_area)
            np.testing.assert_array_equal(normals.numpy(), actual_normal)


@wp.kernel
def _volume_small_face_at(
    edges: wp.array[wp.vec2ub],
    corners: wp.array[wp.vec3ub],
    volume: wp.uint64,
    voxel: wp.vec3i,
    areas: wp.array[float],
    normals: wp.array[wp.vec3],
):
    field = vec8f(-1.0e-6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    area, normal, _center, _depth, _vertices = mc_calc_face(
        edges, corners, 0, field, volume, voxel[0], voxel[1], voxel[2]
    )
    areas[0] = area
    normals[0] = normal


def test_generic_volume_small_face_far_from_origin(test, device):
    """Generic NanoVDB mc_calc_face keeps clamped small-face area and normal far from the origin."""
    voxel_size = 2.5e-4
    # Clamped at MC_EDGE_CLAMP_MIN = 0.02 -> cut = 0.02 * 2.5e-4 = 5.0e-6 m.
    cut = 0.02 * voxel_size
    expected_area = 0.5 * np.sqrt(3.0) * cut * cut
    expected_normal = np.full((1, 3), 1.0 / np.sqrt(3.0), dtype=np.float32)
    with wp.ScopedDevice(device):
        edges = wp.array([[0, 1], [0, 3], [0, 4]], dtype=wp.vec2ub)
        corners = wp.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
            dtype=wp.vec3ub,
        )
        areas = wp.empty(1, dtype=float)
        normals = wp.empty(1, dtype=wp.vec3)

        for translation, tile in (
            ((0.0, 0.0, 0.0), (0, 0, 0)),
            ((200.0, 200.0, 200.0), (0, 0, 0)),
            ((0.0, 0.0, 0.0), (400000, 200000, 100000)),
        ):
            tiles = wp.array([list(tile)], dtype=wp.vec3i)
            volume = wp.Volume.allocate_by_tiles(
                tiles, voxel_size=voxel_size, translation=translation, bg_value=-1.0, device=device
            )
            wp.launch(_fill_inside, dim=(8, 8, 8), inputs=[volume.id])
            wp.launch(
                _volume_small_face_at,
                dim=1,
                inputs=[edges, corners, volume.id, wp.vec3i(*tile)],
                outputs=[areas, normals],
            )
            test.assertAlmostEqual(float(areas.numpy()[0]), expected_area, delta=expected_area * 1.0e-5)
            np.testing.assert_allclose(normals.numpy(), expected_normal, rtol=1.0e-5, atol=1.0e-5)


class TestSDFMCVolumeScale(unittest.TestCase):
    pass


# NanoVDB volume allocation requires CUDA; CPU texture-path coverage is separate.
for _name, _fn in (
    ("test_generic_volume_face_scale", test_generic_volume_face_scale),
    ("test_generic_volume_small_face_far_from_origin", test_generic_volume_small_face_far_from_origin),
):
    add_function_test(
        TestSDFMCVolumeScale,
        _name,
        _fn,
        devices=[device for device in get_test_devices() if device.is_cuda],
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
