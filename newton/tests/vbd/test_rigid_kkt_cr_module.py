# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Lifetime and module-option contracts for the long-path scalar CR route."""

import unittest

import warp as wp

import newton
from newton._src.solvers.vbd import rigid_vbd_kkt
from newton.tests.test_rigid_vbd_kkt import _build_chain
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _cr_float_alias_lifetime(test, device):
    """Replanning keeps aliases attached to owned storage only on used routes."""
    for count in (1, 128, 129, 161, 257):
        path = rigid_vbd_kkt._Path(
            joints=list(range(count)),
            bodies=list(range(count + 1)),
            body_rows=[[] for _ in range(count + 1)],
        )
        bucket = rigid_vbd_kkt._PathBucket([path], device)
        for fused in (True, False, True):
            bucket.plan_cr(fused=fused)
            long_route = bool(bucket.cr_levels) or bucket.tail_paths is not None
            expected = device.is_cuda and fused and long_route
            test.assertEqual(bucket.cr_float_buffers is not None, expected)
            if expected:
                owned = (bucket.lower[0], bucket.diagonal[0], bucket.upper[0], bucket.rhs[0])
                for alias, array in zip(bucket.cr_float_buffers, owned, strict=True):
                    test.assertEqual(alias.ptr, array.ptr)
                    test.assertEqual(alias.capacity, array.capacity)


def _cr_unique_module_options(test, device):
    """Alternating solvers restore their own options on both unique CR modules."""
    model, _, _ = _build_chain(device, segments=4)
    solvers = [
        newton.solvers.SolverVBD(
            model,
            iterations=1,
            rigid_joint_global_iterations=1,
            deterministic=mode,
        )
        for mode in (wp.DeterministicMode.RUN_TO_RUN, wp.DeterministicMode.NOT_GUARANTEED)
    ]
    for solver in (solvers[0], solvers[1], solvers[0]):
        solver._apply_module_options()
        for kernel in (
            rigid_vbd_kkt.reduce_cr_in_place_rows,
            rigid_vbd_kkt.solve_cr_tail_persistent_rows,
        ):
            test.assertIn(kernel.module, solver._module_options)
            for key, value in solver._rigid_module_options.items():
                test.assertEqual(wp.get_module_options(kernel.module)[key], value)


class TestVBDKKTCRModule(unittest.TestCase):
    pass


add_function_test(
    TestVBDKKTCRModule, "test_cr_float_alias_lifetime", _cr_float_alias_lifetime, devices=get_test_devices()
)
add_function_test(
    TestVBDKKTCRModule, "test_cr_unique_module_options", _cr_unique_module_options, devices=get_test_devices()
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
