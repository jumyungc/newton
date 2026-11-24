# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Supported mimic property replacements invalidate the per-color schedule."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_mimic_reference_notify_recapture(test, device):
    """Match a new solver after a mimic is rerouted, disabled and restored through notification."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        joints = []
        for i in range(4):
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01), lock_inertia=True)
            joints.append(
                builder.add_joint_prismatic(
                    -1,
                    body,
                    axis=newton.Axis.X,
                    parent_xform=wp.transform((0.0, 0.2 * i, 0.0), wp.quat_identity()),
                )
            )
        builder.add_articulation(joints)
        builder.set_joint_mimic(joints[3], joints[0], coeffs=(0.0, 1.0))
        builder.color()
        model = builder.finalize()
        # Valid over-coloring established before solver construction. Rerouting the
        # reference changes which color must receive its coupled reaction.
        model.body_color_groups = [wp.array([i], dtype=int) for i in range(4)]
        model.body_colors = wp.array([0, 1, 2, 3], dtype=int)
        model.joint_q.assign([0.01, -0.01, 0.03, 0.0])
        newton.eval_fk(model, model.joint_q, model.joint_qd, model)

        def make_solver():
            return newton.solvers.SolverVBD(
                model,
                iterations=8,
                rigid_contact_activation_guard=True,
                deterministic=wp.DeterministicMode.RUN_TO_RUN,
            )

        def solve_and_replay(selected):
            state_0, state_1 = model.state(), model.state()

            def reset():
                wp.copy(state_0.body_q, model.body_q)
                state_0.body_qd.zero_()
                selected.reset(state_0)

            def step():
                selected.step(state_0, state_1, None, None, 1.0 / 600.0)

            reset()
            step()
            eager = state_1.body_q.numpy().copy()
            if device.is_cuda:
                reset()
                with wp.ScopedCapture() as capture:
                    step()
                for _ in range(2):
                    reset()
                    wp.capture_launch(capture.graph)
                    np.testing.assert_array_equal(state_1.body_q.numpy(), eager)
            return eager

        solver = make_solver()
        for label, reference, enabled in (
            ("initial", 0, True),
            ("rerouted", 2, True),
            ("disabled", 2, False),
            ("restored", 0, True),
        ):
            with test.subTest(stage=label):
                references = model.joint_mimic_joint.numpy().copy()
                references[3] = reference
                model.joint_mimic_joint = wp.array(references, dtype=int)
                flags = model.joint_enabled.numpy().copy()
                flags[3] = enabled
                model.joint_enabled = wp.array(flags, dtype=bool)
                solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
                actual = solve_and_replay(solver)
                expected = solve_and_replay(make_solver())
                np.testing.assert_array_equal(actual, expected)
                if label == "rerouted":
                    test.assertGreater(abs(float(expected[2, 0]) - 0.03), 1.0e-5)
                    test.assertLess(abs(float(expected[0, 0]) - 0.01), 1.0e-7)


def test_mimic_notify_update_flags_and_graph_capture(test, device):
    """Honor CONSTRAINT_PROPERTIES for mimic coefficients and mark JOINT_PROPERTIES non-graphable."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        joints = [
            builder.add_joint_prismatic(
                -1,
                builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01), lock_inertia=True),
                axis=newton.Axis.X,
                parent_xform=wp.transform((0.0, 0.2 * i, 0.0), wp.quat_identity()),
            )
            for i in range(2)
        ]
        builder.add_articulation(joints)
        builder.set_joint_mimic(joints[1], joints[0], coeffs=(0.0, 1.0))
        builder.color()
        model = builder.finalize()
        model.joint_q.assign([0.02, 0.0])
        newton.eval_fk(model, model.joint_q, model.joint_qd, model)

        test.assertEqual(
            model.get_attribute_update_flags("joint_mimic_coeffs"),
            newton.ModelFlags.CONSTRAINT_PROPERTIES,
        )

        def make_solver():
            return newton.solvers.SolverVBD(
                model,
                iterations=8,
                deterministic=wp.DeterministicMode.RUN_TO_RUN,
            )

        solver = make_solver()
        test.assertFalse(solver.is_notify_model_changed_graphable(newton.ModelFlags.JOINT_PROPERTIES))
        test.assertTrue(solver.is_notify_model_changed_graphable(newton.ModelFlags.CONSTRAINT_PROPERTIES))

        state_0, state_1 = model.state(), model.state()
        solver.step(state_0, state_1, None, None, 1.0 / 600.0)
        test.assertGreater(float(np.abs(solver._joint_mimics.lambda_.numpy()).max()), 0.0)

        new_coeffs = model.joint_mimic_coeffs.numpy().copy()
        new_coeffs[joints[1]] = (0.01, -2.0)
        model.joint_mimic_coeffs.assign(new_coeffs)

        if device.is_cuda and wp.is_mempool_enabled(device):
            with wp.ScopedCapture(device) as notify_capture:
                solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)
            wp.capture_launch(notify_capture.graph)
        else:
            solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)

        np.testing.assert_array_equal(solver._joint_mimics.lambda_.numpy(), 0.0)
        np.testing.assert_array_equal(solver._joint_mimics.support.numpy(), 0.0)

        wp.copy(state_0.body_q, model.body_q)
        state_0.body_qd.zero_()
        solver.reset(state_0)
        solver.step(state_0, state_1, None, None, 1.0 / 600.0)

        ref_solver = make_solver()
        ref_0, ref_1 = model.state(), model.state()
        ref_solver.step(ref_0, ref_1, None, None, 1.0 / 600.0)
        np.testing.assert_array_equal(state_1.body_q.numpy(), ref_1.body_q.numpy())


class TestJointMimicLifecycle(unittest.TestCase):
    pass


add_function_test(
    TestJointMimicLifecycle,
    "test_mimic_reference_notify_recapture",
    test_mimic_reference_notify_recapture,
    devices=get_test_devices(),
)
add_function_test(
    TestJointMimicLifecycle,
    "test_mimic_notify_update_flags_and_graph_capture",
    test_mimic_notify_update_flags_and_graph_capture,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
