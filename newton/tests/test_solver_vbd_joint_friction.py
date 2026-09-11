# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.joint_mimic import eval_joint_mimic_coordinate
from newton._src.solvers.vbd.joint_mimic import JointMimicSolver, _JointData
from newton.tests.unittest_utils import add_function_test, get_test_devices

_DT = 1.0 / 240.0


@wp.kernel
def _sample_coordinates(
    data: _JointData,
    joint: int,
    poses: wp.array[wp.transform],
    q: wp.array[float],
    gradients: wp.array2d[wp.spatial_vector],
):
    component = wp.tid()
    coordinate, parent, child = eval_joint_mimic_coordinate(
        joint,
        component,
        poses,
        data.body_com,
        data.joint_type,
        data.parent,
        data.child,
        data.X_p,
        data.X_c,
        data.qd_start,
        data.dof_dim,
        data.axis,
    )
    q[component] = coordinate
    gradients[component, 0] = parent
    gradients[component, 1] = child


def test_vbd_friction_coordinate_gradients(test, device):
    """Check the coordinate covectors used for friction and mimic virtual work."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), com=wp.vec3(0.1, 0.2, 0.3))
    child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), com=wp.vec3(-0.2, 0.1, 0.0))
    free = builder.add_joint_free(child=parent)
    config = newton.ModelBuilder.JointDofConfig
    joint = builder.add_joint_d6(
        parent=parent,
        child=child,
        linear_axes=[config(axis=axis) for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)],
        angular_axes=[config(axis=axis) for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)],
    )
    builder.add_articulation([free, joint])
    builder.joint_q[-6:] = [0.6, 0.4, -0.2, 0.3, 0.5, -0.4]
    model = builder.finalize(device=device)
    data = JointMimicSolver(model).data
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    original = state.body_q.numpy().copy()
    q = wp.empty(6, dtype=float, device=device)
    gradients = wp.empty((6, 2), dtype=wp.spatial_vector, device=device)

    def sample(poses):
        state.body_q.assign(poses)
        wp.launch(_sample_coordinates, dim=6, inputs=[data, joint, state.body_q], outputs=[q, gradients], device=device)
        return q.numpy().copy(), gradients.numpy().copy()

    _, analytic = sample(original)
    epsilon = 1.0e-3
    com = model.body_com.numpy()
    for body in range(2):
        for axis in range(6):
            samples = []
            for sign in (-1.0, 1.0):
                poses = original.copy()
                if axis < 3:
                    poses[body, axis] += sign * epsilon
                else:
                    rotation = wp.quat(*original[body, 3:])
                    direction = wp.vec3()
                    direction[axis - 3] = 1.0
                    perturbed = wp.quat_from_axis_angle(direction, sign * epsilon) * rotation
                    poses[body, :3] += np.asarray(wp.quat_rotate(rotation, wp.vec3(*com[body])))
                    poses[body, :3] -= np.asarray(wp.quat_rotate(perturbed, wp.vec3(*com[body])))
                    poses[body, 3:] = np.asarray(perturbed)
                samples.append(sample(poses)[0])
            numeric = (samples[1] - samples[0]) / (2.0 * epsilon)
            np.testing.assert_allclose(analytic[:, body, axis], numeric, atol=2.0e-4)


def _simulate(
    model,
    *,
    steps=120,
    dt=_DT,
    joint_force=None,
    iterations=6,
    capture=False,
    return_friction=False,
    return_position=False,
    solver_kwargs=None,
):
    """Simulate a VBD model and return its reconstructed joint velocities."""
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    if joint_force is not None:
        control.joint_f.assign(np.asarray(joint_force, dtype=np.float32))

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    solver = newton.solvers.SolverVBD(
        model, iterations=iterations, rigid_compliant_alm=True, **({} if solver_kwargs is None else solver_kwargs)
    )
    if capture and model.device.is_cuda:
        solver.step(state_in, state_out, control, None, dt)
        solver.reset(state_in)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        with wp.ScopedCapture(device=model.device) as graph:
            for _ in range(steps):
                solver.step(state_in, state_out, control, None, dt)
                state_in, state_out = state_out, state_in
        wp.capture_launch(graph.graph)
    else:
        for _ in range(steps):
            solver.step(state_in, state_out, control, None, dt)
            state_in, state_out = state_out, state_in

    joint_q = wp.empty_like(model.joint_q)
    joint_qd = wp.empty_like(model.joint_qd)
    newton.eval_ik(model, state_in, joint_q, joint_qd)
    position = joint_q.numpy()
    velocity = joint_qd.numpy()
    if return_position and return_friction:
        return position, velocity, solver.joint_friction_lambda.numpy()
    if return_position:
        return position, velocity
    if return_friction:
        return velocity, solver.joint_friction_lambda.numpy()
    return velocity


def _build_single_dof_model(device, joint_type, friction):
    """Build one free scalar joint with unit mass and inertia."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    kwargs = {
        "parent": -1,
        "child": body,
        "axis": newton.Axis.Z,
        "target_ke": 0.0,
        "target_kd": 0.0,
        "limit_ke": 0.0,
        "limit_kd": 0.0,
        "friction": friction,
    }
    if joint_type == newton.JointType.REVOLUTE:
        joint = builder.add_joint_revolute(**kwargs)
    else:
        joint = builder.add_joint_prismatic(**kwargs)
    builder.joint_qd[0] = 2.0
    builder.add_articulation([joint])
    builder.color()
    return builder.finalize(device=device)


def _build_actuated_mimic_model(device, follower_friction):
    """Build an actuated leader and equally geared follower with dry friction."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    leader_body = builder.add_link(
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    follower_body = builder.add_link(
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    common = {
        "parent": -1,
        "axis": newton.Axis.Z,
        "target_ke": 0.0,
        "target_kd": 0.0,
        "limit_ke": 0.0,
        "limit_kd": 0.0,
    }
    leader = builder.add_joint_revolute(child=leader_body, friction=0.4, **common)
    follower = builder.add_joint_revolute(child=follower_body, friction=follower_friction, **common)
    builder.joint_qd[:] = [0.5, 0.5]
    builder.add_articulation([leader, follower])
    builder.set_joint_mimic(follower, leader)
    builder.color()
    return builder.finalize(device=device)


def _build_d6_model(device):
    """Build a two-axis translational D6 joint with distinct friction values."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    axes = [
        newton.ModelBuilder.JointDofConfig(
            axis=newton.Axis.X,
            limit_ke=0.0,
            limit_kd=0.0,
            friction=0.5,
        ),
        newton.ModelBuilder.JointDofConfig(
            axis=newton.Axis.Y,
            limit_ke=0.0,
            limit_kd=0.0,
            friction=1.0,
        ),
    ]
    joint = builder.add_joint_d6(parent=-1, child=body, linear_axes=axes)
    builder.joint_qd[:] = [2.0, 2.0]
    builder.add_articulation([joint])
    builder.color()
    return builder.finalize(device=device)


def _build_angular_d6_model(device):
    """Build a three-axis rotational D6 joint with per-axis dry friction."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
    axes = [
        newton.ModelBuilder.JointDofConfig(
            axis=axis,
            target_ke=0.0,
            target_kd=0.0,
            limit_ke=0.0,
            limit_kd=0.0,
            friction=4.0,
        )
        for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
    ]
    joint = builder.add_joint_d6(parent=-1, child=body, angular_axes=axes)
    builder.add_articulation([joint])
    builder.color()
    return builder.finalize(device=device)


def _build_pendulum_model(device, *, gravity, friction):
    """Build a horizontal unit rod hinged about its end."""
    length = 1.0
    mass = 1.0
    inertia_com = mass * length * length / 12.0
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    body = builder.add_link(
        mass=mass,
        com=wp.vec3(length / 2.0, 0.0, 0.0),
        inertia=wp.mat33(inertia_com, 0.0, 0.0, 0.0, inertia_com, 0.0, 0.0, 0.0, inertia_com),
        lock_inertia=True,
    )
    joint = builder.add_joint_revolute(
        -1,
        body,
        axis=newton.Axis.Y,
        target_ke=0.0,
        target_kd=0.0,
        limit_ke=0.0,
        limit_kd=0.0,
        friction=friction,
    )
    builder.add_articulation([joint])
    builder.color()
    return builder.finalize(device=device)


def test_vbd_joint_friction_coast_down(test, device):
    """Dissipate motion with revolute and prismatic Coulomb friction."""
    for joint_type in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC):
        with test.subTest(joint_type=joint_type):
            model = _build_single_dof_model(device, joint_type, friction=1.0)
            joint_qd = _simulate(model)
            test.assertAlmostEqual(float(joint_qd[0]), 1.5, delta=0.1)


def test_vbd_d6_joint_friction(test, device):
    """Apply distinct Coulomb friction values to each free D6 axis."""
    model = _build_d6_model(device)
    joint_qd = _simulate(model)
    np.testing.assert_allclose(joint_qd, [1.75, 1.5], atol=0.1)


def test_vbd_d6_angular_joint_friction(test, device):
    """Exercise projected friction on the D6 Euler-coordinate covectors."""
    model = _build_angular_d6_model(device)
    velocity = _simulate(model, steps=1, joint_force=[2.0, -8.0, 0.0], iterations=12, capture=True)
    np.testing.assert_allclose(velocity, [0.0, -4.0 * _DT, 0.0], atol=3.0e-5)


def test_vbd_joint_friction_ignores_rigid_corotation(test, device):
    """Keep the friction reaction zero when parent and child rotate together."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
    child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
    root = builder.add_joint_free(child=parent)
    axes = [
        newton.ModelBuilder.JointDofConfig(
            axis=axis,
            target_ke=0.0,
            target_kd=0.0,
            limit_ke=0.0,
            limit_kd=0.0,
            friction=4.0,
        )
        for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
    ]
    relative = builder.add_joint_d6(parent=parent, child=child, angular_axes=axes)
    builder.joint_qd[3:6] = [0.4, -0.7, 1.1]
    builder.add_articulation([root, relative])
    builder.color()
    model = builder.finalize(device=device)
    velocity, friction_lambda = _simulate(model, steps=120, iterations=8, return_friction=True)
    np.testing.assert_allclose(velocity[-3:], 0.0, atol=2.0e-5)
    np.testing.assert_allclose(friction_lambda[-3:], 0.0, atol=2.0e-5)


def test_vbd_mimic_follower_friction(test, device):
    """Transfer follower friction through an actuated mimic relationship."""
    leader_only_model = _build_actuated_mimic_model(device, follower_friction=0.0)
    both_joints_model = _build_actuated_mimic_model(device, follower_friction=0.8)

    leader_only_qd = _simulate(leader_only_model, joint_force=[2.0, 0.0])
    both_joints_qd = _simulate(both_joints_model, joint_force=[2.0, 0.0])

    np.testing.assert_allclose(both_joints_qd[1], both_joints_qd[0], atol=2.0e-3)
    test.assertLess(float(both_joints_qd[0]), float(leader_only_qd[0]) - 0.1)


def test_vbd_joint_friction_stop(test, device):
    """Friction must not reverse or amplify a nearly stopped joint's velocity."""
    for joint_type in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC):
        for iterations in (1, 6, 7, 20):
            with test.subTest(joint_type=joint_type, iterations=iterations):
                model = _build_single_dof_model(device, joint_type, friction=100.0)
                model.joint_qd.assign([0.05])
                velocity = float(_simulate(model, steps=1, iterations=iterations)[0])
                test.assertGreaterEqual(velocity, -1.0e-5)
                test.assertLessEqual(velocity, 0.05)


def test_vbd_joint_friction_static_threshold(test, device):
    """Stick below the Coulomb threshold and slide above it in either direction."""
    friction = 4.0
    for joint_type in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC):
        for force in (-2.0, 2.0, -8.0, 8.0):
            with test.subTest(joint_type=joint_type, force=force):
                model = _build_single_dof_model(device, joint_type, friction=friction)
                model.joint_qd.zero_()
                velocity = _simulate(model, steps=1, joint_force=[force], iterations=12, capture=True)
                expected_velocity = np.sign(force) * max(abs(force) - friction, 0.0) * _DT
                test.assertAlmostEqual(float(velocity[0]), expected_velocity, delta=2.0e-5)


def test_vbd_joint_friction_gravity_matches_actuator(test, device):
    """Treat gravity torque and an equivalent actuator torque consistently."""
    gravity_torque = 9.81 * 0.5
    solver_kwargs = {"rigid_joint_linear_ke": 1.0e8, "rigid_joint_angular_ke": 1.0e8}
    for friction in (3.0, 6.0):
        with test.subTest(friction=friction):
            gravity_velocity = _simulate(
                _build_pendulum_model(device, gravity=-9.81, friction=friction),
                steps=1,
                iterations=12,
                capture=True,
                solver_kwargs=solver_kwargs,
            )[0]
            actuator_velocity = _simulate(
                _build_pendulum_model(device, gravity=0.0, friction=friction),
                steps=1,
                joint_force=[gravity_torque],
                iterations=12,
                capture=True,
                solver_kwargs=solver_kwargs,
            )[0]
            test.assertAlmostEqual(float(gravity_velocity), float(actuator_velocity), delta=5.0e-5)
            if friction < gravity_torque:
                expected = (gravity_torque - friction) * _DT / (1.0 / 3.0)
                test.assertAlmostEqual(float(gravity_velocity), expected, delta=5.0e-5)
            else:
                test.assertAlmostEqual(float(gravity_velocity), 0.0, delta=2.0e-5)


def test_vbd_joint_friction_uses_constrained_inertia(test, device):
    """Scale projected friction with the inertia of the actual free joint coordinate."""
    dt = _DT

    # An anisotropic, off-center revolute body exercises both the kinetic-energy
    # projection a^T I a and the parallel-axis contribution at the locked anchor.
    axis = np.asarray([1.0, 2.0, -0.5], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    com = np.asarray([0.35, -0.2, 0.1], dtype=np.float64)
    anchor = np.asarray([-0.1, 0.25, 0.3], dtype=np.float64)
    inertia = np.asarray([[2.0, 0.3, -0.1], [0.3, 3.0, 0.2], [-0.1, 0.2, 4.0]], dtype=np.float64)
    mass = 2.5

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(
        mass=mass,
        com=wp.vec3(*com),
        inertia=wp.mat33(inertia),
        lock_inertia=True,
    )
    joint = builder.add_joint_revolute(
        -1,
        body,
        child_xform=wp.transform(wp.vec3(*anchor), wp.quat_identity()),
        axis=wp.vec3(*axis),
        target_ke=0.0,
        target_kd=0.0,
        limit_ke=0.0,
        limit_kd=0.0,
        friction=1.0,
    )
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(model, iterations=0, rigid_compliant_alm=True)
    solver.step(model.state(), model.state(), model.control(), None, dt)

    model_inertia = model.body_inertia.numpy()[body].astype(np.float64)
    model_mass = float(model.body_mass.numpy()[body])
    lever = anchor - com
    expected_revolute_rho = (
        axis @ model_inertia @ axis + model_mass * np.dot(np.cross(lever, axis), np.cross(lever, axis))
    ) / (dt * dt)
    test.assertAlmostEqual(float(solver.joint_friction_rho.numpy()[0]), expected_revolute_rho, delta=0.05)

    # A one-angular-axis D6 has the same constrained coordinate as REVOLUTE.
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(
        mass=mass,
        com=wp.vec3(*com),
        inertia=wp.mat33(inertia),
        lock_inertia=True,
    )
    angular_axis = newton.ModelBuilder.JointDofConfig(
        axis=wp.vec3(*axis),
        target_ke=0.0,
        target_kd=0.0,
        limit_ke=0.0,
        limit_kd=0.0,
        friction=1.0,
    )
    joint = builder.add_joint_d6(
        -1,
        body,
        child_xform=wp.transform(wp.vec3(*anchor), wp.quat_identity()),
        angular_axes=[angular_axis],
    )
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(model, iterations=0, rigid_compliant_alm=True)
    solver.step(model.state(), model.state(), model.control(), None, dt)
    test.assertAlmostEqual(float(solver.joint_friction_rho.numpy()[0]), expected_revolute_rho, delta=0.05)

    # A prismatic joint locks rotation, so its free-coordinate inertia is mass,
    # independent of the center-of-mass offset.
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(
        mass=mass,
        com=wp.vec3(*com),
        inertia=wp.mat33(inertia),
        lock_inertia=True,
    )
    joint = builder.add_joint_prismatic(
        -1,
        body,
        axis=wp.vec3(*axis),
        target_ke=0.0,
        target_kd=0.0,
        limit_ke=0.0,
        limit_kd=0.0,
        friction=1.0,
    )
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(model, iterations=0, rigid_compliant_alm=True)
    solver.step(model.state(), model.state(), model.control(), None, dt)
    expected_prismatic_rho = float(model.body_mass.numpy()[body]) / (dt * dt)
    test.assertAlmostEqual(float(solver.joint_friction_rho.numpy()[0]), expected_prismatic_rho, delta=0.05)


def test_vbd_joint_friction_persists_and_rebalances(test, device):
    """Retain static reaction without creep, then remove and reverse it with the load."""
    for joint_type in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC):
        with test.subTest(joint_type=joint_type):
            model = _build_single_dof_model(device, joint_type, friction=4.0)
            model.joint_qd.zero_()
            state_in = model.state()
            state_out = model.state()
            control = model.control()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
            # A deliberately aggressive structural-history decay must not decay
            # the bounded dry-friction reaction.
            solver = newton.solvers.SolverVBD(
                model,
                iterations=4,
                rigid_compliant_alm=True,
                rigid_avbd_gamma=0.9,
            )
            joint_q = wp.empty_like(model.joint_q)
            joint_qd = wp.empty_like(model.joint_qd)

            def advance(
                force,
                steps,
                control=control,
                solver=solver,
                model=model,
                joint_q=joint_q,
                joint_qd=joint_qd,
            ):
                nonlocal state_in, state_out
                control.joint_f.assign([force])
                for _ in range(steps):
                    solver.step(state_in, state_out, control, None, _DT)
                    state_in, state_out = state_out, state_in
                newton.eval_ik(model, state_in, joint_q, joint_qd)
                return (
                    float(joint_q.numpy()[0]),
                    float(joint_qd.numpy()[0]),
                    float(solver.joint_friction_lambda.numpy()[0]),
                )

            loaded_q, loaded_qd, loaded_lambda = advance(2.0, 1000)
            test.assertAlmostEqual(loaded_qd, 0.0, delta=1.0e-6)
            test.assertAlmostEqual(loaded_lambda, 2.0, delta=1.0e-5)
            test.assertLess(abs(loaded_q), 1.0e-5)

            unloaded_q, unloaded_qd, unloaded_lambda = advance(0.0, 120)
            test.assertAlmostEqual(unloaded_qd, 0.0, delta=1.0e-6)
            test.assertAlmostEqual(unloaded_lambda, 0.0, delta=1.0e-5)
            test.assertLess(abs(unloaded_q - loaded_q), 1.5e-5)

            reversed_q, reversed_qd, reversed_lambda = advance(-2.0, 120)
            test.assertAlmostEqual(reversed_qd, 0.0, delta=1.0e-6)
            test.assertAlmostEqual(reversed_lambda, -2.0, delta=1.0e-5)
            test.assertLess(abs(reversed_q - unloaded_q), 1.0e-5)


def test_vbd_joint_friction_timestep_scaling(test, device):
    """Preserve static thresholds and stopping distance across timestep sizes."""
    stopping_positions = []
    for dt in (1.0 / 60.0, 1.0 / 120.0, 1.0 / 240.0, 1.0 / 480.0):
        with test.subTest(dt=dt, regime="stick"):
            model = _build_single_dof_model(device, newton.JointType.REVOLUTE, friction=4.0)
            model.joint_qd.zero_()
            position, velocity = _simulate(
                model,
                steps=round(0.5 / dt),
                dt=dt,
                joint_force=[2.0],
                iterations=4,
                return_position=True,
            )
            test.assertAlmostEqual(float(velocity[0]), 0.0, delta=2.0e-5)
            test.assertLess(abs(float(position[0])), 5.0e-5)

        with test.subTest(dt=dt, regime="slide-to-stop"):
            model = _build_single_dof_model(device, newton.JointType.REVOLUTE, friction=1.0)
            position, velocity = _simulate(
                model,
                steps=round(2.5 / dt),
                dt=dt,
                iterations=8,
                return_position=True,
            )
            test.assertAlmostEqual(float(velocity[0]), 0.0, delta=2.0e-5)
            stopping_positions.append(float(position[0]))

    # Unit inertia, initial speed 2 rad/s, and unit friction stop after 2 s
    # and travel 2 rad in continuous time. Semi-implicit time discretization
    # leaves the expected first-order position error while preserving the law.
    test.assertLess(max(stopping_positions) - min(stopping_positions), 0.03)
    np.testing.assert_allclose(stopping_positions, 2.0, atol=0.04)


def test_vbd_joint_friction_live_update_and_reset(test, device):
    """Read friction bounds dynamically under capture and clear their history on reset."""
    model = _build_single_dof_model(device, newton.JointType.REVOLUTE, friction=0.0)
    model.joint_qd.zero_()
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    control.joint_f.assign([2.0])
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    solver = newton.solvers.SolverVBD(model, iterations=12, rigid_compliant_alm=True)

    # Compile/lazily initialize before capture, then restore a clean fixed input.
    solver.step(state_in, state_out, control, None, _DT)
    solver.reset(state_in)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    if model.device.is_cuda:
        with wp.ScopedCapture(device=model.device) as capture:
            solver.step(state_in, state_out, control, None, _DT)

        def launch():
            wp.capture_launch(capture.graph)

    else:

        def launch():
            solver.step(state_in, state_out, control, None, _DT)

    joint_q = wp.empty_like(model.joint_q)
    joint_qd = wp.empty_like(model.joint_qd)

    def velocity_for(friction):
        model.joint_friction.assign([friction])
        launch()
        newton.eval_ik(model, state_out, joint_q, joint_qd)
        return float(joint_qd.numpy()[0])

    test.assertAlmostEqual(velocity_for(0.0), 2.0 * _DT, delta=2.0e-5)
    test.assertAlmostEqual(velocity_for(4.0), 0.0, delta=2.0e-5)
    friction_lambda = float(solver.joint_friction_lambda.numpy()[0])
    test.assertGreater(friction_lambda, 0.0)
    test.assertLessEqual(friction_lambda, 4.0 + 1.0e-6)
    test.assertAlmostEqual(velocity_for(1.0), 1.0 * _DT, delta=2.0e-5)
    test.assertLessEqual(abs(float(solver.joint_friction_lambda.numpy()[0])), 1.0 + 1.0e-6)

    model.joint_friction.assign([4.0])
    launch()
    test.assertGreater(abs(float(solver.joint_friction_lambda.numpy()[0])), 0.0)
    solver.reset(state_in, flags=0)
    test.assertAlmostEqual(float(solver.joint_friction_lambda.numpy()[0]), 0.0, delta=1.0e-7)


def test_vbd_mimic_friction_force_balance(test, device):
    """Both friction forces must enter the constrained pair's momentum balance."""
    model = _build_actuated_mimic_model(device, follower_friction=16.0)
    model.joint_friction.assign([4.0, 16.0])
    model.joint_qd.zero_()
    for ratio in (1.0, -1.0, -0.5, 2.0):
        for force in (-10.0, 10.0, 50.0):
            with test.subTest(ratio=ratio, force=force):
                model.joint_mimic_coeffs.assign([[0.0, 1.0], [0.0, ratio]])
                velocity, friction_lambda = _simulate(
                    model,
                    steps=1,
                    joint_force=[force, 0.0],
                    iterations=24,
                    capture=True,
                    return_friction=True,
                )
                # Reflect follower inertia and friction into the leader coordinate.
                momentum_change = (1.0 + ratio * ratio) * float(velocity[0]) / _DT
                net_force = force - float(friction_lambda[0]) - ratio * float(friction_lambda[1])
                np.testing.assert_allclose(ratio * velocity[0], velocity[1], atol=1.0e-5)
                np.testing.assert_array_less(np.abs(friction_lambda), np.array([4.0, 16.0]) + 1.0e-6)
                test.assertAlmostEqual(momentum_change, net_force, delta=0.05)


def test_vbd_multiple_mimic_followers_friction(test, device):
    """Shared leaders need sequential constraint updates and retained reactions."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    joints = []
    frictions = (1.0, 2.0, 3.0)
    for friction in frictions:
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
        joints.append(builder.add_joint_prismatic(-1, body, axis=newton.Axis.X, friction=friction))
    builder.add_articulation(joints)
    builder.set_joint_mimic(joints[1], joints[0], (0.0, -1.0))
    builder.set_joint_mimic(joints[2], joints[0], (0.0, 2.0))
    builder.color()
    model = builder.finalize(device=device)
    velocity, friction_lambda = _simulate(
        model, steps=1, iterations=24, joint_force=[4.0, 0.0, 0.0], capture=True, return_friction=True
    )
    ratios = np.array([1.0, -1.0, 2.0])
    np.testing.assert_allclose(velocity, ratios * velocity[0], atol=1.0e-5)
    net_force = 4.0 - np.dot(ratios, friction_lambda)
    np.testing.assert_array_less(np.abs(friction_lambda), np.asarray(frictions) + 1.0e-6)
    test.assertAlmostEqual(6.0 * float(velocity[0]) / _DT, net_force, delta=0.02)


def test_vbd_serial_mimic_and_passive_friction(test, device):
    """Balance friction on a serial mimic pair and a downstream passive hinge."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    joints = []
    parent = -1
    for friction in (1.0, 2.0, 3.0):
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
        joints.append(builder.add_joint_revolute(parent, body, axis=newton.Axis.Z, friction=friction))
        parent = body
    builder.add_articulation(joints)
    builder.set_joint_mimic(joints[1], joints[0])
    builder.color()
    model = builder.finalize(device=device)
    velocity, friction_lambda = _simulate(
        model, steps=1, iterations=64, joint_force=[4.0, 0.0, 0.0], capture=True, return_friction=True
    )
    np.testing.assert_allclose(velocity[0], velocity[1], atol=1.0e-5)
    # Body speeds are (v, 2v, 2v + w), so the reduced inertia is [[9,2],[2,1]].
    momentum = np.array([[9.0, 2.0], [2.0, 1.0]]) @ velocity[[0, 2]] / _DT
    net_force = [4.0 - friction_lambda[0] - friction_lambda[1], -friction_lambda[2]]
    np.testing.assert_array_less(np.abs(friction_lambda), np.array([1.0, 2.0, 3.0]) + 1.0e-6)
    np.testing.assert_allclose(momentum, net_force, atol=0.02)


class TestSolverVBDJointFriction(unittest.TestCase):
    pass


devices = get_test_devices(mode="basic")
add_function_test(
    TestSolverVBDJointFriction,
    "test_vbd_serial_mimic_and_passive_friction",
    test_vbd_serial_mimic_and_passive_friction,
    devices=devices,
)
add_function_test(
    TestSolverVBDJointFriction,
    "test_vbd_multiple_mimic_followers_friction",
    test_vbd_multiple_mimic_followers_friction,
    devices=devices,
)
add_function_test(
    TestSolverVBDJointFriction,
    "test_vbd_friction_coordinate_gradients",
    test_vbd_friction_coordinate_gradients,
    devices=devices,
)
add_function_test(
    TestSolverVBDJointFriction, "test_vbd_joint_friction_stop", test_vbd_joint_friction_stop, devices=devices
)
add_function_test(
    TestSolverVBDJointFriction,
    "test_vbd_joint_friction_static_threshold",
    test_vbd_joint_friction_static_threshold,
    devices=devices,
)
add_function_test(
    TestSolverVBDJointFriction,
    "test_vbd_joint_friction_gravity_matches_actuator",
    test_vbd_joint_friction_gravity_matches_actuator,
    devices=devices,
)
add_function_test(
    TestSolverVBDJointFriction,
    "test_vbd_joint_friction_uses_constrained_inertia",
    test_vbd_joint_friction_uses_constrained_inertia,
    devices=devices,
)
add_function_test(
    TestSolverVBDJointFriction,
    "test_vbd_joint_friction_persists_and_rebalances",
    test_vbd_joint_friction_persists_and_rebalances,
    devices=devices,
)
add_function_test(
    TestSolverVBDJointFriction,
    "test_vbd_joint_friction_timestep_scaling",
    test_vbd_joint_friction_timestep_scaling,
    devices=devices,
)
add_function_test(
    TestSolverVBDJointFriction,
    "test_vbd_joint_friction_live_update_and_reset",
    test_vbd_joint_friction_live_update_and_reset,
    devices=devices,
)
add_function_test(
    TestSolverVBDJointFriction,
    "test_vbd_mimic_friction_force_balance",
    test_vbd_mimic_friction_force_balance,
    devices=devices,
)
add_function_test(
    TestSolverVBDJointFriction,
    "test_vbd_joint_friction_coast_down",
    test_vbd_joint_friction_coast_down,
    devices=devices,
)
add_function_test(
    TestSolverVBDJointFriction,
    "test_vbd_mimic_follower_friction",
    test_vbd_mimic_follower_friction,
    devices=devices,
)
add_function_test(
    TestSolverVBDJointFriction,
    "test_vbd_d6_joint_friction",
    test_vbd_d6_joint_friction,
    devices=devices,
)
add_function_test(
    TestSolverVBDJointFriction,
    "test_vbd_d6_angular_joint_friction",
    test_vbd_d6_angular_joint_friction,
    devices=devices,
)
add_function_test(
    TestSolverVBDJointFriction,
    "test_vbd_joint_friction_ignores_rigid_corotation",
    test_vbd_joint_friction_ignores_rigid_corotation,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
