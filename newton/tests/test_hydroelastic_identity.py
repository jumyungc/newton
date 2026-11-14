# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise hydroelastic identity capacity without constructing texture SDFs."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_data import make_contact_sort_key
from newton._src.geometry.sdf_hydroelastic import (
    _MAX_DETERMINISTIC_ISO_VOXELS_PER_PAIR,
    _hydroelastic_voxel_face_base,
    scatter_iso_subblock,
)
from newton._src.sim.collide import _record_reduction_overflow
from newton.geometry import HydroelasticSDF
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _export_voxel_identity_keys(
    records: wp.array[wp.vec3ui],
    pair_starts: wp.array[int],
    deterministic: int,
    overflows: wp.array[int],
    keys: wp.array[wp.int64],
):
    """Use the generation identity guard and production key writer for one face per voxel."""
    tid = wp.tid()
    pair = wp.int32(records[tid][2])
    base = _hydroelastic_voxel_face_base(tid, pair_starts[pair], deterministic, overflows)
    if base < 0:
        keys[tid] = wp.int64(-1)
    else:
        # The final possible MC face, exported from the voxel reduction entry.
        keys[tid] = make_contact_sort_key(2 * pair, 2 * pair + 1, ((base + 4) << 1) | 1, 4)


def test_hydroelastic_pair_local_capacity(test, device):
    """Keep a real final-scatter batch beyond the old global identity boundary."""
    hydro = HydroelasticSDF(
        num_shape_pairs=3,
        total_num_tiles=1024,
        total_num_active_tiles=2048,
        shape_material_kh=wp.ones(6, dtype=float, device=device),
        n_shapes=6,
        config=HydroelasticSDF.Config(reduce_contacts=False),
        device=device,
        deterministic=True,
    )
    limit = _MAX_DETERMINISTIC_ISO_VOXELS_PER_PAIR
    test.assertGreater(hydro.max_num_iso_voxels, limit)
    test.assertEqual(hydro.max_num_face_contacts, (1 << 20) - 1)
    # Each pair has fewer than 419,430 voxels, but the batch has 419,440.
    parents_per_pair = limit // 16 + 1
    parent_count = 2 * parents_per_pair + 3
    parents = np.zeros((parent_count, 3), dtype=np.uint32)
    masks = np.full(parent_count, 255, dtype=np.uint8)
    parents[parents_per_pair + 1 :, 2] = 1
    parents[parents_per_pair + 2 :, 2] = 2
    # Empty first parent of pair 0, empty entire pair 1, empty first parent of pair 2.
    masks[[0, parents_per_pair + 1, parents_per_pair + 2]] = 0
    # Coordinates remain distinct within each pair and fit the packed 16-bit axes.
    for start, end in ((0, parents_per_pair + 1), (parents_per_pair + 2, parent_count)):
        local = np.arange(end - start, dtype=np.uint32)
        parents[start:end, 0] = (local % 128) * 2 | (((local // 128) % 128) * 2 << 16)
        parents[start:end, 1] = (local // (128 * 128)) * 2
    counts = (masks != 0).astype(np.int32) * 8
    prefix = np.zeros(parent_count, dtype=np.int32)
    prefix[1:] = np.cumsum(counts[:-1])
    voxel_count = int(counts.sum())
    test.assertGreater(voxel_count, limit)
    half = voxel_count // 2
    previous_keys = None
    for reverse_pairs in (False, True):
        if reverse_pairs:
            parents[:, 2] = 2 - parents[:, 2]
        wp.launch(
            scatter_iso_subblock,
            dim=256,
            inputs=[
                256,
                wp.array([parent_count], dtype=int, device=device),
                wp.array(prefix, dtype=int, device=device),
                wp.array(masks, dtype=wp.uint8, device=device),
                wp.array(parents, dtype=wp.vec3ui, device=device),
                1,
                parent_count,
                hydro.max_num_iso_voxels,
            ],
            outputs=[hydro.iso_voxel_records, hydro.iso_pair_start],
            device=device,
        )
        records = hydro.iso_voxel_records[:voxel_count]
        keys = wp.empty(voxel_count, dtype=wp.int64, device=device)
        wp.launch(
            _export_voxel_identity_keys,
            dim=voxel_count,
            inputs=[records, hydro.iso_pair_start, 1, hydro.iso_fingerprint_overflows],
            outputs=[keys],
            device=device,
        )
        test.assertEqual(int(hydro.iso_fingerprint_overflows.numpy()[0]), 0)
        key_np = keys.numpy()
        test.assertEqual(len(np.unique(key_np)), voxel_count)
        subkeys = key_np & ((1 << 23) - 1)
        test.assertFalse(np.any(subkeys & (1 << 22)))
        # Both geometric pair copies retain the same subkeys despite their batch offset.
        np.testing.assert_array_equal(subkeys[:half], subkeys[half:])
        np.testing.assert_array_equal(
            hydro.iso_pair_start.numpy(), [half, half, 0] if reverse_pairs else [0, half, half]
        )
        test.assertEqual(int(records.numpy()[-1, 2]), 0 if reverse_pairs else 2)
        sorted_keys = np.sort(key_np)
        if previous_keys is not None:
            np.testing.assert_array_equal(sorted_keys, previous_keys)
        previous_keys = sorted_keys


def test_hydroelastic_pair_identity_overflow(test, device):
    """Reject only a pair's unrepresentable tail and count every skipped voxel."""
    limit = _MAX_DETERMINISTIC_ISO_VOXELS_PER_PAIR
    records_np = np.zeros((limit + 3, 3), dtype=np.uint32)
    # Pair 0 is one voxel too large; pair 2 starts afterwards and must still work.
    records_np[limit + 1 :, 2] = 2
    records = wp.array(records_np, dtype=wp.vec3ui, device=device)
    starts = wp.array([0, 0, limit + 1], dtype=int, device=device)
    overflows = wp.zeros(1, dtype=int, device=device)
    zero = wp.zeros(1, dtype=int, device=device)
    contacts = newton.Contacts(1, 0, device=device)
    keys = wp.empty(len(records_np), dtype=wp.int64, device=device)
    for _ in range(2):
        overflows.zero_()
        wp.launch(
            _export_voxel_identity_keys,
            dim=len(records_np),
            inputs=[records, starts, 1, overflows],
            outputs=[keys],
            device=device,
        )
        actual = keys.numpy()
        test.assertEqual(int(overflows.numpy()[0]), 1)
        np.testing.assert_array_equal(np.flatnonzero(actual == -1), [limit])
        test.assertEqual(len(np.unique(actual[actual >= 0])), limit + 2)
        test.assertLess(int(actual[limit - 1]) & ((1 << 23) - 1), 1 << 22)
        test.assertEqual(int(actual[limit + 1]) & ((1 << 23) - 1), 9)
        wp.launch(
            _record_reduction_overflow,
            dim=1,
            inputs=[zero, zero, overflows, contacts._reduction_overflow],
            device=device,
        )
        test.assertEqual(int(contacts._reduction_overflow.numpy()[0]), 1)
        # A non-hydro reducer must neither dereference the absent counter nor
        # erase an earlier loss on the same stream.
        wp.launch(
            _record_reduction_overflow,
            dim=1,
            inputs=[zero, zero, None, contacts._reduction_overflow],
            device=device,
        )
        test.assertEqual(int(contacts._reduction_overflow.numpy()[0]), 1)
        contacts.clear()
        overflows.zero_()
        wp.launch(
            _record_reduction_overflow,
            dim=1,
            inputs=[zero, zero, overflows, contacts._reduction_overflow],
            device=device,
        )
        test.assertEqual(int(contacts._reduction_overflow.numpy()[0]), 0)


class TestHydroelasticIdentity(unittest.TestCase):
    pass


for name, function in (
    ("test_hydroelastic_pair_local_capacity", test_hydroelastic_pair_local_capacity),
    ("test_hydroelastic_pair_identity_overflow", test_hydroelastic_pair_identity_overflow),
):
    add_function_test(TestHydroelasticIdentity, name, function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
