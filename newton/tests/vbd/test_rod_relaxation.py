# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Standard-linear-solid relaxation of VBD rods."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers.experimental.coupled import SolverCoupled
from newton.tests._usd_deformable_test_utils import _add_cable_curve, _deformable_stage
from newton.tests.unittest_utils import USD_AVAILABLE, add_function_test, get_test_devices


def _axial_rod(device, *, stiffness, damping, mass, ratio, tau, gravity, bend_ratio=0.0, inertia=1.0e-3, frame=None):
    """One rod joint from the world to a link, loaded along the joint +Z axis by gravity."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    newton.solvers.SolverVBD.register_custom_attributes(builder)
    origin = wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity())
    if frame is not None:
        origin = frame * origin
    link = builder.add_link(
        xform=origin,
        mass=mass,
        inertia=wp.mat33(np.eye(3) * inertia),
    )
    joint = builder.add_joint_rod(
        -1,
        link,
        parent_xform=origin,
        child_xform=wp.transform_identity(),
        stretch_stiffness=stiffness,
        stretch_damping=damping,
        bend_stiffness=1.0,
    )
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device=device)
    model.vbd.visco_stretch_ratio.fill_(ratio)
    model.vbd.visco_stretch_tau.fill_(tau)
    model.vbd.visco_bend_ratio.fill_(bend_ratio)
    model.vbd.visco_bend_tau.fill_(tau)
    return model, link


def _creep(model, link, solver, steps, dt, u=0.0, v=0.0, maxwell=0.0, *, stiffness, damping, mass, ratio, tau, load):
    """Step the solver and the float64 backward-Euler SLS recurrence together; return the reference state."""
    state_0, state_1 = model.state(), model.state()
    decay = tau / (tau + dt)
    for _ in range(steps):
        solver.step(state_0, state_1, None, None, dt)
        state_0, state_1 = state_1, state_0
        lhs = mass / dt**2 + damping / dt + stiffness * (1.0 + decay * ratio)
        rhs = mass / dt**2 * u + mass / dt * v + damping / dt * u + load + decay * (ratio * stiffness * u - maxwell)
        u_next = rhs / lhs
        maxwell = decay * (maxwell + ratio * stiffness * (u_next - u))
        v, u = (u_next - u) / dt, u_next
        np.testing.assert_allclose(state_0.body_q.numpy()[link, 2] - 0.1, u, atol=1.0e-7)
    return u


def _relative_angle(q_a, q_b):
    """Return the sign-invariant rotation angle [rad] between two unit xyzw quaternions."""
    return 2.0 * np.arccos(np.clip(abs(np.dot(q_a, q_b)), 0.0, 1.0))


def test_relaxation_parameters_enable_and_validate(test, device):
    """Activate a branch only for positive (ratio, tau) pairs and reject invalid values."""
    for ratio, tau, active in ((0.0, 0.5, False), (2.0, 0.0, False), (2.0, 0.5, True)):
        model, _ = _axial_rod(device, stiffness=1.0e3, damping=0.0, mass=1.0, ratio=ratio, tau=tau, gravity=0.0)
        test.assertEqual(newton.solvers.SolverVBD(model).joint_visco.shape[0] > 0, active)
    model, _ = _axial_rod(device, stiffness=1.0e3, damping=0.0, mass=1.0, ratio=2.0, tau=0.5, gravity=0.0)
    for name in ("visco_bend_ratio", "visco_bend_tau", "visco_stretch_ratio", "visco_stretch_tau"):
        array = getattr(model.vbd, name)
        valid = array.numpy().copy()
        for invalid in (-1.0, float("nan"), float("inf")):
            array.fill_(invalid)
            with test.assertRaisesRegex(ValueError, name):
                newton.solvers.SolverVBD(model)
        array.assign(valid)
    with test.assertRaisesRegex(ValueError, "integrate the rigid bodies"):
        newton.solvers.SolverVBD(model, integrate_with_external_rigid_solver=True)


def test_rod_creep_matches_backward_euler(test, device):
    """Match axial creep under a step load to the float64 backward-Euler SLS recurrence."""
    params = {"stiffness": 1.0e3, "damping": 5.0, "mass": 1.0e-3, "ratio": 3.0, "tau": 0.4}
    model, link = _axial_rod(device, gravity=9810.0, **params)
    solver = newton.solvers.SolverVBD(model, iterations=20)
    u = _creep(model, link, solver, 200, 0.01, load=9.81, **params)
    # After 1.25 retardation times tau * (1 + ratio) the strain has crept well past the instantaneous response.
    test.assertGreater(u, 3.0 * 9.81 / (1.0e3 * (1.0 + 3.0)))


def test_relaxation_rejects_coupled_state(test, device):
    """Reject coupled rod relaxation while allowing an elastic VBD entry to step."""
    model, link = _axial_rod(device, stiffness=1.0e3, damping=0.0, mass=1.0, ratio=2.0, tau=0.5, gravity=-9.81)
    for in_place in (False, True):
        with test.subTest(in_place=in_place):
            entry = SolverCoupled.Entry(
                name="rod",
                solver=lambda view: newton.solvers.SolverVBD(view, iterations=10),
                bodies=range(model.body_count),
                joints=range(model.joint_count),
                in_place=in_place,
            )
            model.vbd.visco_stretch_ratio.fill_(2.0)
            with test.assertRaisesRegex(ValueError, "Rod relaxation.*SolverCoupled.*SolverVBD directly"):
                SolverCoupled(model, entries=[entry])

            model.vbd.visco_stretch_ratio.zero_()
            coupled = SolverCoupled(model, entries=[entry])
            state_0, state_1 = model.state(), model.state()
            coupled.step(state_0, state_1, None, None, 0.01)
            test.assertLess(state_1.body_q.numpy()[link, 2], 0.1)
            np.testing.assert_array_equal(state_1.vbd.visco_history.numpy(), 0.0)


def test_held_bend_and_stretch_relaxation(test, device):
    """Balance the actual rod forces with analytic relaxing loads, including a changing timestep.

    Creep alone does not cover bending forces or verify that input history stays frozen during
    a solve. A single link held by independently known axial force and bending torque covers both.
    """
    ratio, tau, stiffness, theta, extension = 3.0, 0.4, 1.0e3, 0.2, 0.02
    frame = wp.transform(wp.vec3(0.15, -0.2, 0.25), wp.quat_from_axis_angle(wp.normalize(wp.vec3(1, 2, 3)), 0.7))
    model, link = _axial_rod(
        device,
        stiffness=stiffness,
        damping=0.0,
        mass=1.0,
        ratio=ratio,
        tau=tau,
        gravity=0.0,
        bend_ratio=ratio,
        inertia=0.1,
        frame=frame,
    )
    solver = newton.solvers.SolverVBD(model, iterations=20)
    state_0, state_1 = model.state(), model.state()
    pose = state_0.body_q.numpy().copy()
    pose[link] = frame * wp.transform(
        wp.vec3(0.0, 0.0, 0.1 + extension), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), theta)
    )
    state_0.body_q.assign(pose)
    # Planar geometric bend coordinate and its angle derivative, independent of Warp helpers.
    bend = 2.0 * np.tan(theta / 2)
    bend_jacobian = 1.0 / np.cos(theta / 2) ** 2
    decay_product = 1.0
    for dt in [0.01] * 10 + [0.025] * 10:
        before = state_0.vbd.visco_history.numpy().copy()
        decay_product *= tau / (tau + dt)
        factor = 1.0 + ratio * decay_product
        force = np.zeros((model.body_count, 6), dtype=np.float32)
        force[link, :3] = wp.quat_rotate(
            wp.transform_get_rotation(frame), wp.vec3(0, 0, stiffness * extension * factor)
        )
        # Bend stiffness is 1 N*m. World torque balances a moment in the material frame.
        force[link, 3:] = wp.quat_rotate(wp.transform_get_rotation(frame), wp.vec3(bend * bend_jacobian * factor, 0, 0))
        state_0.body_f.assign(force)
        solver.step(state_0, state_1, None, None, dt)
        np.testing.assert_array_equal(state_0.vbd.visco_history.numpy(), before)
        np.testing.assert_allclose(state_1.body_q.numpy()[link], pose[link], atol=2.0e-6)
        expected = np.array([bend, 0.0, extension]) * (1.0 - decay_product)
        np.testing.assert_allclose(state_1.vbd.visco_history.numpy()[0], expected, atol=2.0e-6)
        state_0, state_1 = state_1, state_0


def test_relaxation_edits_apply_through_notify(test, device):
    """Apply edited parameters through notify_model_changed without rebuilding the solver."""
    params = {"stiffness": 1.0e3, "damping": 5.0, "mass": 1.0e-3, "tau": 0.4}
    model, link = _axial_rod(device, gravity=9810.0, ratio=3.0, **params)
    solver = newton.solvers.SolverVBD(model, iterations=20)
    model.vbd.visco_stretch_ratio.fill_(1.5)
    solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
    _creep(model, link, solver, 50, 0.01, load=9.81, ratio=1.5, **params)
    # Enabling relaxation on a solver built without it requires a new solver.
    model, _ = _axial_rod(device, gravity=0.0, ratio=0.0, **params)
    solver = newton.solvers.SolverVBD(model)
    model.vbd.visco_stretch_ratio.fill_(1.0)
    solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)  # Frame edits do not re-read the material.
    with test.assertRaisesRegex(ValueError, "new SolverVBD"):
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)

    # Disabling axial relaxation must not depend on an unrelated, unstrained bend branch.
    released = []
    for keep_bend in (False, True):
        model, link = _axial_rod(
            device, stiffness=1.0e3, damping=0.0, mass=1.0, ratio=3.0, tau=0.4, gravity=0.0, bend_ratio=2.0
        )
        solver = newton.solvers.SolverVBD(model, iterations=0)
        state_0, state_1 = model.state(), model.state()
        model.vbd.visco_stretch_ratio.zero_()
        if not keep_bend:
            model.vbd.visco_bend_ratio.zero_()
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
        pose = state_0.body_q.numpy()
        pose[link, 2] += 0.04
        state_0.body_q.assign(pose)
        for _ in range(10):
            solver.step(state_0, state_1, None, None, 0.01)
            state_0, state_1 = state_1, state_0
        np.testing.assert_allclose(state_0.vbd.visco_history.numpy()[0, 2], 0.04, atol=1.0e-7)
        model.vbd.visco_stretch_ratio.fill_(3.0)
        model.vbd.visco_bend_ratio.zero_()
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
        solver.iterations = 20
        solver.step(state_0, state_1, None, None, 0.01)
        released.append(state_1.body_qd.numpy().copy())
    test.assertGreater(abs(released[0][link, 2]), 0.01)
    np.testing.assert_array_equal(released[0], released[1])


def test_bend_relaxation_leaves_twist_unchanged(test, device):
    """Relax bending only and leave a pure twist motion unchanged."""
    twist = []
    for bend_ratio in (0.0, 5.0):
        model, link = _axial_rod(
            device, stiffness=1.0e3, damping=0.0, mass=1.0, ratio=0.0, tau=0.5, gravity=0.0, bend_ratio=bend_ratio
        )
        solver = newton.solvers.SolverVBD(model, iterations=10)
        state_0, state_1 = model.state(), model.state()
        qd = state_0.body_qd.numpy()
        qd[link, 5] = 1.0
        state_0.body_qd.assign(qd)
        for _ in range(20):
            solver.step(state_0, state_1, None, None, 0.01)
            state_0, state_1 = state_1, state_0
        twist.append(state_0.body_q.numpy()[link])
    np.testing.assert_allclose(twist[0], twist[1], atol=1.0e-6)


def test_dahl_and_relaxation_compose(test, device):
    """Keep Dahl and bend relaxation histories independent on one rod, with both acting in a release."""
    runs = {}
    for name, eps_max, ratio in (("dahl", 0.05, 0.0), ("sls", 0.0, 5.0), ("both", 0.05, 5.0)):
        model, link = _axial_rod(
            device, stiffness=1.0e4, damping=0.0, mass=1.0, ratio=0.0, tau=0.3, gravity=0.0, bend_ratio=ratio
        )
        model.vbd.dahl_eps_max.fill_(eps_max)
        model.vbd.dahl_tau.fill_(0.2 if eps_max > 0.0 else 0.0)
        solver = newton.solvers.SolverVBD(model, iterations=0)
        state_0, state_1 = model.state(), model.state()
        # Common prescribed bending: histories are acquired without solving.
        for theta in np.concatenate((np.linspace(0.0, 0.3, 15), np.linspace(0.3, 0.12, 10))):
            pose = state_0.body_q.numpy()
            pose[link, 3:] = (np.sin(0.5 * theta), 0.0, 0.0, np.cos(0.5 * theta))
            state_0.body_q.assign(pose)
            state_0.body_qd.zero_()
            solver.step(state_0, state_1, None, None, 0.01)
            state_0, state_1 = state_1, state_0
        histories = (state_0.vbd.visco_history.numpy().copy(), solver.joint_sigma_prev.numpy().copy())
        # Release from the held pose.
        solver.iterations = 20
        for _ in range(30):
            solver.step(state_0, state_1, None, None, 0.01)
            state_0, state_1 = state_1, state_0
        runs[name] = (histories, state_0.body_q.numpy()[link].copy())

    # Both mechanisms acquire nonzero history, and neither changes the other's.
    test.assertGreater(np.abs(runs["sls"][0][0][0, :2]).max(), 0.05)
    test.assertGreater(np.abs(runs["dahl"][0][1][0]).max(), 5.0e-3)
    np.testing.assert_array_equal(runs["both"][0][0], runs["sls"][0][0])
    np.testing.assert_array_equal(runs["both"][0][1], runs["dahl"][0][1])
    test.assertTrue(np.all(np.isfinite(runs["both"][1])))
    # The release rotates the link about its fixed anchor; each mechanism changes that rotation.
    test.assertGreater(_relative_angle(runs["both"][1][3:], runs["dahl"][1][3:]), 0.03)
    test.assertGreater(_relative_angle(runs["both"][1][3:], runs["sls"][1][3:]), 1.0e-3)


def test_relaxation_history_reset_and_capture(test, device):
    """Restore only selected worlds on masked reset and match eager steps bitwise under capture."""
    template = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    newton.solvers.SolverVBD.register_custom_attributes(template)
    rod = newton.Rod.create_straight(
        start=wp.vec3(0.0, 0.0, 1.0), direction=wp.vec3(1.0, 0.0, 0.0), length=0.5, segment_count=10, radius=0.004
    )
    bodies, _ = template.add_rod(rod=rod, stretch_stiffness=2.0e3, bend_stiffness=0.05)
    template.body_flags[bodies[0]] = int(newton.BodyFlags.KINEMATIC)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    newton.solvers.SolverVBD.register_custom_attributes(builder)
    builder.add_world(template)
    builder.add_world(template, xform=wp.transform(wp.vec3(0.0, 1.0, 0.0), wp.quat_identity()))
    builder.color()

    results = []
    for captured in (False, True) if wp.get_device(device).is_cuda else (False,):
        model = builder.finalize(device=device)
        for name, value in (("visco_bend_ratio", 2.0), ("visco_bend_tau", 0.2)):
            getattr(model.vbd, name).fill_(value)
        for name, value in (("visco_stretch_ratio", 3.0), ("visco_stretch_tau", 0.3)):
            getattr(model.vbd, name).fill_(value)
        solver = newton.solvers.SolverVBD(model, iterations=10)
        state_0, state_1 = model.state(), model.state()
        dt = 1.0 / 600.0
        if captured:
            with wp.ScopedCapture(device=device) as capture:
                solver.step(state_0, state_1, None, None, dt)
                solver.step(state_1, state_0, None, None, dt)
        for step in range(30):
            if step in (10, 20):
                model.vbd.visco_stretch_ratio.fill_(0.0 if step == 10 else 3.0)
                solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
            if captured:
                wp.capture_launch(capture.graph)
            else:
                solver.step(state_0, state_1, None, None, dt)
                solver.step(state_1, state_0, None, None, dt)
        results.append((state_0.body_q.numpy().copy(), state_0.vbd.visco_history.numpy().copy()))
        copied = model.state()
        copied.assign(state_0)
        np.testing.assert_array_equal(copied.vbd.visco_history.numpy(), state_0.vbd.visco_history.numpy())

        # Pose-based initialization uses the same world selection and can be captured independently.
        world = model.joint_world.numpy()
        solver.relax_rod_history(copied)
        relaxed = copied.vbd.visco_history.numpy().copy()
        previous = state_0.vbd.visco_history.numpy().copy()
        mask = wp.array([True, False, False], dtype=wp.bool, device=device)
        if captured:
            with wp.ScopedCapture(device=device) as initialize:
                solver.relax_rod_history(state_0, world_mask=mask)
            wp.capture_launch(initialize.graph)
        else:
            solver.relax_rod_history(state_0, world_mask=mask)
        initialized = state_0.vbd.visco_history.numpy()
        np.testing.assert_array_equal(initialized[world == 0], relaxed[world == 0])
        np.testing.assert_array_equal(initialized[world == 1], previous[world == 1])

        before = state_0.vbd.visco_history.numpy().copy()
        solver.reset(state_0, world_mask=mask, flags=0)
        after = state_0.vbd.visco_history.numpy()
        test.assertGreater(np.abs(before[world == 0]).max(), 0.0)
        np.testing.assert_array_equal(after[world == 0], 0.0)
        np.testing.assert_array_equal(after[world == 1], before[world == 1])
        solver.reset(state_0, flags=0)
        np.testing.assert_array_equal(state_0.vbd.visco_history.numpy(), 0.0)

    if len(results) == 2:
        np.testing.assert_array_equal(results[0][0], results[1][0])
        np.testing.assert_array_equal(results[0][1], results[1][1])


def test_relax_history_holds_current_pose(test, device):
    """Start relaxed in a deformed pose: the equilibrium load alone then holds that pose.

    A virgin (zero) history in the same pose loads the transient branch, so the same load moves it.
    """
    ratio, tau, stiffness, theta, extension = 3.0, 0.4, 1.0e3, 0.2, 0.02
    frame = wp.transform(wp.vec3(0.15, -0.2, 0.25), wp.quat_from_axis_angle(wp.normalize(wp.vec3(1, 2, 3)), 0.7))
    bend = 2.0 * np.tan(theta / 2)
    bend_jacobian = 1.0 / np.cos(theta / 2) ** 2
    drift = {}
    for relaxed in (True, False):
        model, link = _axial_rod(
            device,
            stiffness=stiffness,
            damping=0.0,
            mass=1.0,
            ratio=ratio,
            tau=tau,
            gravity=0.0,
            bend_ratio=ratio,
            inertia=0.1,
            frame=frame,
        )
        solver = newton.solvers.SolverVBD(model, iterations=20)
        state_0, state_1 = model.state(), model.state()
        if relaxed:
            with test.assertRaisesRegex(ValueError, "required"):
                solver.relax_rod_history(None)
            state_1.body_q = None
            with test.assertRaisesRegex(ValueError, "body_q"):
                solver.relax_rod_history(state_1)
            state_1 = model.state()
        pose = state_0.body_q.numpy().copy()
        pose[link] = frame * wp.transform(
            wp.vec3(0.0, 0.0, 0.1 + extension), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), theta)
        )
        state_0.body_q.assign(pose)
        if relaxed:
            solver.relax_rod_history(state_0, world_mask=wp.array([False, False], dtype=wp.bool, device=device))
            np.testing.assert_array_equal(state_0.vbd.visco_history.numpy(), 0.0)
            solver.relax_rod_history(state_0)
            np.testing.assert_allclose(state_0.vbd.visco_history.numpy()[0], [bend, 0.0, extension], atol=2.0e-6)
        force = np.zeros((model.body_count, 6), dtype=np.float32)
        force[link, :3] = wp.quat_rotate(wp.transform_get_rotation(frame), wp.vec3(0, 0, stiffness * extension))
        force[link, 3:] = wp.quat_rotate(wp.transform_get_rotation(frame), wp.vec3(bend * bend_jacobian, 0, 0))
        drift[relaxed] = 0.0
        for dt in [0.01] * 5 + [0.025] * 5:
            state_0.body_f.assign(force)
            solver.step(state_0, state_1, None, None, dt)
            q = state_1.body_q.numpy()[link]
            moved = max(np.abs(q[:3] - pose[link, :3]).max(), _relative_angle(q[3:], pose[link, 3:]))
            drift[relaxed] = max(drift[relaxed], float(moved))
            state_0, state_1 = state_1, state_0
        if relaxed:
            np.testing.assert_allclose(state_0.vbd.visco_history.numpy()[0], [bend, 0.0, extension], atol=2.0e-6)
    test.assertLess(drift[True], 2.0e-6)
    test.assertGreater(drift[False], 1.0e-3)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
def test_usd_cable_curve_authors_relaxation(test, device):
    """Apply relaxation attributes authored on a USD cable curve to every rod joint generated from it."""
    from pxr import Sdf, UsdGeom

    stage = _deformable_stage()
    # Newton currently supports SI stages. Ratios are dimensionless and tau is seconds.
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    points = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.25, 0.0, 1.0), (0.3, 0.0, 1.0)]
    prim = _add_cable_curve(stage, "/World/Cable", points).GetPrim()
    authored = {"visco_bend_ratio": 9.9, "visco_bend_tau": 3.5, "visco_stretch_ratio": 2.0, "visco_stretch_tau": 0.5}
    for name, value in authored.items():
        prim.CreateAttribute(f"newton:vbd:{name}", Sdf.ValueTypeNames.Float).Set(value)
    stage.DefinePrim("/World/Copy").GetReferences().AddInternalReference("/World/Cable")
    builder = newton.ModelBuilder()
    newton.solvers.SolverVBD.register_custom_attributes(builder)
    # A wildcard joint attribute derives its value from joint prims and must not apply to curves.
    builder.add_custom_attribute(
        newton.ModelBuilder.CustomAttribute(
            name="wildcard",
            frequency=newton.Model.AttributeFrequency.JOINT,
            dtype=wp.float32,
            default=0.0,
            namespace="test",
            usd_attribute_name="*",
            usd_value_transformer=lambda _value, _context: 7.0,
        )
    )
    cable_map = builder.add_usd(stage, return_deformable_results=True)["path_cable_map"]
    joints = cable_map["/World/Cable"][1]
    builder.color()
    model = builder.finalize(device=device)
    test.assertEqual(len(joints), 2)
    np.testing.assert_allclose(model.vbd.visco_bend_ratio.numpy()[joints], 9.9, rtol=1.0e-6)
    np.testing.assert_allclose(model.vbd.visco_bend_tau.numpy()[joints], 3.5, rtol=1.0e-6)
    np.testing.assert_array_equal(model.vbd.visco_stretch_ratio.numpy()[joints], 2.0)
    np.testing.assert_array_equal(model.vbd.visco_stretch_tau.numpy()[joints], 0.5)
    np.testing.assert_array_equal(model.vbd.visco_history.numpy(), 0.0)
    np.testing.assert_array_equal(model.test.wildcard.numpy()[joints], 0.0)
    copied_joints = cable_map["/World/Copy"][1]
    np.testing.assert_array_equal(
        model.vbd.visco_bend_ratio.numpy()[joints], model.vbd.visco_bend_ratio.numpy()[copied_joints]
    )
    # Keep geometry fixed and author all four material values directly through builder overrides.
    direct = newton.ModelBuilder()
    newton.solvers.SolverVBD.register_custom_attributes(direct)
    direct.add_builder(builder)
    rod_indices = np.flatnonzero(model.joint_type.numpy() == int(newton.JointType.ROD))
    for name, value in authored.items():
        direct.custom_attributes[f"vbd:{name}"].values = dict.fromkeys(rod_indices, value)
    direct.color()
    copied_model = direct.finalize(device=device)
    results = []
    for candidate in (model, copied_model):
        solver = newton.solvers.SolverVBD(candidate, iterations=2)
        state_0, state_1 = candidate.state(), candidate.state()
        # Load a rod so equivalence exercises the material, not just free fall.
        q = state_0.body_q.numpy()
        child = int(candidate.joint_child.numpy()[joints[0]])
        q[child, 0] += 0.01
        q[child, 3:] = np.asarray(wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.1) * wp.quat(*q[child, 3:]))
        state_0.body_q.assign(q)
        solver.step(state_0, state_1, None, None, 0.001)
        history = state_1.vbd.visco_history.numpy()
        test.assertGreater(np.max(np.abs(history[joints, :2])), 1.0e-6)
        test.assertGreater(np.max(np.abs(history[joints, 2])), 1.0e-7)
        results.append((state_1.body_q.numpy().copy(), state_1.vbd.visco_history.numpy().copy()))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])


class TestVBDRelaxation(unittest.TestCase):
    pass


devices = get_test_devices()
for _name, _func in list(globals().items()):
    if _name.startswith("test_") and callable(_func):
        add_function_test(TestVBDRelaxation, _name, _func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
