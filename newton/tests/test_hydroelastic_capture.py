# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA capture must not depend on the hydroelastic host-warning counter."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.test_hydroelastic import _build_margin_gap_boxes
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices


def test_capture_at_host_warning_boundary(test, device):
    model, state, _ = _build_margin_gap_boxes(device)
    pipeline = newton.CollisionPipeline(model, broad_phase="explicit", rigid_contact_max=20000, deterministic=True)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    hydro = pipeline.hydroelastic_sdf
    test.assertIsNotNone(hydro)
    hydro._launch_counter = hydro._host_warning_poll_interval - 1
    with wp.ScopedCapture(device=device) as capture:
        pipeline.collide(state, contacts)
    for _ in range(3):
        wp.capture_launch(capture.graph)
        count = int(contacts.rigid_contact_count.numpy()[0])
        test.assertGreater(count, 0)
        test.assertTrue(np.isfinite(contacts.rigid_contact_stiffness.numpy()[:count]).all())
    # The same polling boundary must still be usable outside capture.
    hydro._launch_counter = hydro._host_warning_poll_interval - 1
    pipeline.collide(state, contacts)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


class TestHydroelasticCapture(unittest.TestCase):
    pass


add_function_test(
    TestHydroelasticCapture,
    "test_capture_at_host_warning_boundary",
    test_capture_at_host_warning_boundary,
    devices=get_selected_cuda_test_devices(),
)

if __name__ == "__main__":
    unittest.main()
