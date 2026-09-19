# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for XPBD's experimental global rigid-joint correction."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.xpbd.kernels import apply_body_deltas, solve_body_joints
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _build_fixed_chain(device, *, link_count=16, grounded=False, world_attached=True):
    builder = newton.ModelBuilder(
        gravity=(0.0, -9.81, 0.0) if grounded else (0.0, 0.0, 0.0),
        up_axis=newton.Axis.Y,
    )
    shape_cfg = builder.default_shape_cfg.copy()
    shape_cfg.ke = 1.0e5
    shape_cfg.kd = 1.0e3
    shape_cfg.mu = 0.6
    spacing = 0.15
    height = 0.08 if grounded else 0.0
    bodies = []
    joints = []
    for link in range(link_count):
        body = builder.add_link(
            xform=wp.transform(wp.vec3(spacing * link, height, 0.0), wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_box(body, hx=0.06, hy=0.08 if grounded else 0.02, hz=0.02, cfg=shape_cfg)
        bodies.append(body)
        if link == 0:
            if world_attached:
                joints.append(
                    builder.add_joint_fixed(
                        -1,
                        body,
                        parent_xform=wp.transform(wp.vec3(0.0, height, 0.0), wp.quat_identity()),
                    )
                )
        else:
            joints.append(
                builder.add_joint_fixed(
                    bodies[-2],
                    body,
                    parent_xform=wp.transform(wp.vec3(spacing, 0.0, 0.0), wp.quat_identity()),
                )
            )
    builder.add_articulation(joints)
    if grounded:
        builder.add_ground_plane(cfg=shape_cfg)
    return builder.finalize(device=device), np.asarray(bodies, dtype=np.int32)


def _build_branched_fixed_tree(device):
    builder = newton.ModelBuilder(gravity=(0.0, -9.81, 0.0), up_axis=newton.Axis.Y)
    positions = np.asarray(
        [
            (0.0, 2.0, 0.0),
            (-0.4, 1.6, 0.0),
            (0.4, 1.6, 0.0),
            (-0.8, 1.2, 0.0),
            (-0.4, 1.2, 0.0),
            (0.4, 1.2, 0.0),
            (0.8, 1.2, 0.0),
        ],
        dtype=np.float32,
    )
    bodies = []
    for position in positions:
        body = builder.add_link(xform=wp.transform(wp.vec3(*position), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.12, hy=0.12, hz=0.12)
        bodies.append(body)
    joints = [
        builder.add_joint_fixed(
            -1,
            bodies[0],
            parent_xform=wp.transform(wp.vec3(*positions[0]), wp.quat_identity()),
        )
    ]
    for parent, child in ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)):
        offset = positions[child] - positions[parent]
        joints.append(
            builder.add_joint_fixed(
                bodies[parent],
                bodies[child],
                parent_xform=wp.transform(wp.vec3(*offset), wp.quat_identity()),
            )
        )
    builder.add_articulation(joints)
    return builder.finalize(device=device)


def _build_ball_loop(device):
    builder = newton.ModelBuilder(gravity=(0.0, -9.81, 0.0), up_axis=newton.Axis.Y)
    positions = np.asarray(
        [(-0.5, 1.5, 0.0), (0.5, 1.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0)],
        dtype=np.float32,
    )
    bodies = []
    for position in positions:
        body = builder.add_link(xform=wp.transform(wp.vec3(*position), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.12)
        bodies.append(body)
    root = builder.add_joint_ball(
        -1,
        bodies[0],
        parent_xform=wp.transform(wp.vec3(*positions[0]), wp.quat_identity()),
    )

    tree_joints = []
    for parent, child in ((0, 1), (1, 2), (2, 3)):
        midpoint = 0.5 * (positions[parent] + positions[child])
        tree_joints.append(
            builder.add_joint_ball(
                bodies[parent],
                bodies[child],
                parent_xform=wp.transform(wp.vec3(*(midpoint - positions[parent])), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(*(midpoint - positions[child])), wp.quat_identity()),
            )
        )
    builder.add_articulation([root, *tree_joints])

    parent = 3
    child = 0
    midpoint = 0.5 * (positions[parent] + positions[child])
    builder.add_joint_ball(
        bodies[parent],
        bodies[child],
        parent_xform=wp.transform(wp.vec3(*(midpoint - positions[parent])), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(*(midpoint - positions[child])), wp.quat_identity()),
    )
    return builder.finalize(device=device)


def _build_mixed_unsupported_island(device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent = builder.add_link(xform=wp.transform_identity(), mass=1.0)
    child = builder.add_link(xform=wp.transform(wp.vec3(0.3, 0.0, 0.0), wp.quat_identity()), mass=1.0)
    builder.add_shape_sphere(parent, radius=0.05)
    builder.add_shape_sphere(child, radius=0.05)
    root = builder.add_joint_fixed(-1, parent)
    rod = builder.add_joint_rod(parent, child)
    builder.add_articulation([root, rod])
    return builder.finalize(device=device)


def _build_free_root_fixed_chain(device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    bodies = []
    for link in range(4):
        body = builder.add_link(
            xform=wp.transform(wp.vec3(0.2 * link, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_box(body, hx=0.08, hy=0.03, hz=0.03)
        bodies.append(body)
    free = builder.add_joint_free(child=bodies[0])
    joints = [free]
    for link in range(1, len(bodies)):
        joints.append(
            builder.add_joint_fixed(
                bodies[link - 1],
                bodies[link],
                parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
            )
        )
    builder.add_articulation(joints)
    return builder.finalize(device=device)


def _build_rod_only_island(device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent = builder.add_link(mass=1.0)
    child = builder.add_link(
        xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
    )
    builder.add_shape_box(parent, hx=0.08, hy=0.03, hz=0.03)
    builder.add_shape_box(child, hx=0.08, hy=0.03, hz=0.03)
    joint = builder.add_joint_rod(
        parent,
        child,
        parent_xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.1, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _build_supported_joint_gallery(device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    dof = newton.ModelBuilder.JointDofConfig
    joints = []
    for index, kind in enumerate(("revolute", "prismatic", "d6")):
        body = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.5 * index, 0.0), wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_box(body, hx=0.08, hy=0.06, hz=0.04)
        if kind == "revolute":
            joint = builder.add_joint_revolute(
                -1,
                body,
                parent_xform=wp.transform(wp.vec3(0.0, 0.5 * index, 0.0), wp.quat_identity()),
                axis=newton.Axis.Z,
                target_pos=0.1,
                target_ke=1.0e4,
                target_kd=20.0,
            )
        elif kind == "prismatic":
            joint = builder.add_joint_prismatic(
                -1,
                body,
                parent_xform=wp.transform(wp.vec3(0.0, 0.5 * index, 0.0), wp.quat_identity()),
                axis=newton.Axis.X,
                target_pos=0.1,
                target_ke=1.0e4,
                target_kd=20.0,
            )
        else:
            joint = builder.add_joint_d6(
                -1,
                body,
                parent_xform=wp.transform(wp.vec3(0.0, 0.5 * index, 0.0), wp.quat_identity()),
                linear_axes=[dof(axis=newton.Axis.X, limit_lower=-0.2, limit_upper=0.2)],
                angular_axes=[dof(axis=newton.Axis.Z, limit_lower=-0.4, limit_upper=0.4)],
            )
        joints.append(joint)
        builder.add_articulation([joint])
    return builder.finalize(device=device)


def _perturbed_chain_result(device, *, global_iterations):
    model, bodies = _build_fixed_chain(device, link_count=32)
    state_in = model.state()
    state_out = model.state()
    pose = state_in.body_q.numpy()
    pose[bodies, 1] = 0.05 * np.sin(0.4 * np.arange(len(bodies)))
    state_in.body_q.assign(pose)
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=2,
        rigid_joint_global_iterations=global_iterations,
    )
    solver.step(state_in, state_out, None, None, 1.0 / 120.0)
    wp.synchronize_device(device)
    target = model.body_q.numpy()[bodies, :3]
    actual = state_out.body_q.numpy()[bodies, :3]
    return float(np.max(np.linalg.norm(actual - target, axis=1))), actual, state_out.body_qd.numpy()


def test_global_joint_iteration_validation(test, device):
    """Preserve G=0 behavior and validate only the experimental opt-in."""
    model, _ = _build_fixed_chain(device, link_count=4)
    newton.solvers.SolverXPBD(model, iterations=0)

    with test.assertRaisesRegex(TypeError, "must be an integer"):
        newton.solvers.SolverXPBD(model, rigid_joint_global_iterations=True)
    with test.assertRaisesRegex(ValueError, "non-negative"):
        newton.solvers.SolverXPBD(model, rigid_joint_global_iterations=-1)
    with test.assertRaisesRegex(ValueError, "less than iterations"):
        newton.solvers.SolverXPBD(model, iterations=1, rigid_joint_global_iterations=1)

    solver = newton.solvers.SolverXPBD(model, iterations=5, rigid_joint_global_iterations=2)
    test.assertEqual(solver._rigid_joint_global_iteration_indices, {0, 2})

    differentiable_state = model.state(requires_grad=True)
    with test.assertRaisesRegex(NotImplementedError, "differentiable"):
        solver.step(differentiable_state, model.state(requires_grad=True), None, None, 1.0 / 120.0)


def test_global_fixed_chain_improves_convergence(test, device):
    """Reduce a system-wide fixed-chain defect with one coupled correction."""
    local_error, _, _ = _perturbed_chain_result(device, global_iterations=0)
    global_error, global_pose, global_velocity = _perturbed_chain_result(device, global_iterations=1)

    test.assertTrue(np.isfinite(global_pose).all())
    test.assertTrue(np.isfinite(global_velocity).all())
    test.assertLess(global_error, 0.35 * local_error)
    test.assertGreater(local_error, 0.04)


def _max_joint_anchor_gap(model, state):
    poses = state.body_q.numpy()
    parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
    parent_frame, child_frame = model.joint_X_p.numpy(), model.joint_X_c.numpy()
    gaps = []
    for joint in range(model.joint_count):
        xp = wp.vec3(*parent_frame[joint, :3])
        if parent[joint] >= 0:
            pose = poses[parent[joint]]
            xp = wp.transform_point(wp.transform(pose[:3], pose[3:]), xp)
        pose = poses[child[joint]]
        xc = wp.transform_point(wp.transform(pose[:3], pose[3:]), wp.vec3(*child_frame[joint, :3]))
        gaps.append(float(wp.length(xc - xp)))
    return max(gaps, default=0.0)


def test_global_joint_topology_routes(test, device):
    """Route rigid paths, branches, and loops through their coupled backends."""
    path_model, _ = _build_fixed_chain(device, link_count=8)
    path_solver = newton.solvers.SolverXPBD(path_model, iterations=2, rigid_joint_global_iterations=1)
    path_backend = path_solver._rigid_joint_global_solver.backend
    test.assertEqual((len(path_backend.path_buckets), len(path_backend.tree_buckets)), (1, 0))

    tree_model = _build_branched_fixed_tree(device)
    tree_solver = newton.solvers.SolverXPBD(tree_model, iterations=2, rigid_joint_global_iterations=1)
    tree_backend = tree_solver._rigid_joint_global_solver.backend
    test.assertEqual((len(tree_backend.path_buckets), len(tree_backend.tree_buckets)), (0, 1))

    loop_model = _build_ball_loop(device)
    loop_solver = newton.solvers.SolverXPBD(loop_model, iterations=2, rigid_joint_global_iterations=1)
    loop_backend = loop_solver._rigid_joint_global_solver.backend
    test.assertEqual(len(loop_backend.closed_tree_buckets), 1)

    for model, solver in ((tree_model, tree_solver), (loop_model, loop_solver)):
        state_in = model.state()
        state_out = model.state()
        pose = state_in.body_q.numpy()
        pose[-1, 0] += 0.02
        state_in.body_q.assign(pose)
        local_in, local_out = model.state(), model.state()
        local_in.body_q.assign(pose)
        newton.solvers.SolverXPBD(model, iterations=2).step(local_in, local_out, None, None, 1.0 / 120.0)
        solver.step(state_in, state_out, None, None, 1.0 / 120.0)
        test.assertLess(_max_joint_anchor_gap(model, state_out), 0.25 * _max_joint_anchor_gap(model, local_out))
        state_in, state_out = state_out, state_in
        for _ in range(19):
            solver.step(state_in, state_out, None, None, 1.0 / 120.0)
            state_in, state_out = state_out, state_in
        test.assertTrue(np.isfinite(state_in.body_q.numpy()).all())
        test.assertTrue(np.isfinite(state_in.body_qd.numpy()).all())


def test_unsupported_mixed_island_remains_local(test, device):
    """Leave an entire island local when one of its joints is unsupported."""
    model = _build_mixed_unsupported_island(device)
    local_state_in = model.state()
    local_state_out = model.state()
    global_state_in = model.state()
    global_state_out = model.state()
    local_solver = newton.solvers.SolverXPBD(model, iterations=2)
    global_solver = newton.solvers.SolverXPBD(model, iterations=2, rigid_joint_global_iterations=1)
    test.assertFalse(global_solver._rigid_joint_global_solver.active)

    for _ in range(3):
        local_solver.step(local_state_in, local_state_out, None, None, 1.0 / 120.0)
        global_solver.step(global_state_in, global_state_out, None, None, 1.0 / 120.0)
        local_state_in, local_state_out = local_state_out, local_state_in
        global_state_in, global_state_out = global_state_out, global_state_in

    np.testing.assert_array_equal(global_state_in.body_q.numpy(), local_state_in.body_q.numpy())
    np.testing.assert_array_equal(global_state_in.body_qd.numpy(), local_state_in.body_qd.numpy())

    rod_model = _build_rod_only_island(device)
    rod_solver = newton.solvers.SolverXPBD(rod_model, iterations=2, rigid_joint_global_iterations=1)
    test.assertFalse(rod_solver._rigid_joint_global_solver.active)


def test_free_root_does_not_hide_supported_constraints(test, device):
    """Ignore a constraint-free root while coupling its supported descendants."""
    model = _build_free_root_fixed_chain(device)
    solver = newton.solvers.SolverXPBD(model, iterations=2, rigid_joint_global_iterations=1)
    backend = solver._rigid_joint_global_solver.backend
    test.assertTrue(backend.active)
    test.assertEqual(backend.joint_count, 3)
    test.assertEqual(len(backend.path_buckets), 1)


def test_supported_joint_controls_and_compliance(test, device):
    """Solve supported limits and drives with finite XPBD compliance."""
    model = _build_supported_joint_gallery(device)
    state_in = model.state()
    state_out = model.state()
    pose = state_in.body_q.numpy()
    pose[:, 0] += 0.15
    pose[:, 2] += 0.05
    state_in.body_q.assign(pose)
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=3,
        joint_linear_compliance=1.0e-6,
        joint_angular_compliance=1.0e-6,
        rigid_joint_global_iterations=1,
    )

    for _ in range(8):
        solver.step(state_in, state_out, None, None, 1.0 / 120.0)
        state_in, state_out = state_out, state_in

    test.assertEqual(solver._rigid_joint_global_solver.backend.joint_count, 3)
    test.assertTrue(np.isfinite(state_in.body_q.numpy()).all())
    test.assertTrue(np.isfinite(state_in.body_qd.numpy()).all())


def test_global_joint_cuda_graph_capture(test, device):
    """Run the global rigid-joint correction inside a CUDA graph."""
    model, _ = _build_fixed_chain(device, link_count=16)
    solver = newton.solvers.SolverXPBD(model, iterations=2, rigid_joint_global_iterations=1)
    state_0 = model.state()
    state_1 = model.state()
    solver.step(state_0, state_1, None, None, 1.0 / 120.0)
    with wp.ScopedCapture(device) as capture:
        solver.step(state_1, state_0, None, None, 1.0 / 120.0)
        solver.step(state_0, state_1, None, None, 1.0 / 120.0)
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    test.assertTrue(np.isfinite(state_1.body_q.numpy()).all())


def test_global_joint_inertial_refresh_rebuilds_dynamic_partition(test, device):
    """Rebuild topology only when an inertial edit changes dynamic participation."""
    model, _ = _build_fixed_chain(device, link_count=4)
    solver = newton.solvers.SolverXPBD(model, iterations=2, rigid_joint_global_iterations=1)
    original_backend = solver._rigid_joint_global_solver.backend
    test.assertFalse(solver.coupling_supports_inertial_property_refresh())

    inverse_mass = model.body_inv_mass.numpy()
    inverse_mass[0] *= 0.5
    model.body_inv_mass.assign(inverse_mass)
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    test.assertIs(solver._rigid_joint_global_solver.backend, original_backend)

    inverse_mass[0] = 0.0
    model.body_inv_mass.assign(inverse_mass)
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    rebuilt_backend = solver._rigid_joint_global_solver.backend
    test.assertIsNot(rebuilt_backend, original_backend)
    test.assertFalse(bool(rebuilt_backend.dynamic_body_mask_host[0]))


def test_global_joint_contact_reconciliation(test, device):
    """Resolve contacts locally after the coupled rigid-joint correction."""
    model, _ = _build_fixed_chain(device, link_count=12, grounded=True, world_attached=False)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in = model.state()
    state_out = model.state()
    solver = newton.solvers.SolverXPBD(model, iterations=4, rigid_joint_global_iterations=1)

    def pair():
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, 1.0 / 240.0)
        pipeline.collide(state_out, contacts)
        solver.step(state_out, state_in, None, contacts, 1.0 / 240.0)

    pair()
    with wp.ScopedCapture(device) as capture:
        pair()
    for _ in range(29):
        wp.capture_launch(capture.graph)

    pose = state_in.body_q.numpy()
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
    test.assertTrue(np.isfinite(pose).all())
    test.assertGreater(float(np.min(pose[:, 1])), 0.07)


def test_global_joint_parent_force(test, device):
    """Include the coupled correction in the reported inbound joint wrench."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)
    builder.request_state_attributes("body_parent_f")
    body = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, -1.0), wp.quat_identity()),
        mass=2.0,
    )
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_fixed(
        -1,
        body,
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
    )
    builder.add_articulation([joint])
    model = builder.finalize(device=device)
    state_in = model.state()
    state_out = model.state()
    solver = newton.solvers.SolverXPBD(model, iterations=2, rigid_joint_global_iterations=1)
    solver.step(state_in, state_out, None, None, 1.0 / 240.0)
    wp.synchronize_device(device)

    parent_force = state_out.body_parent_f.numpy()[body]
    expected_weight = float(model.body_mass.numpy()[body]) * 9.81
    np.testing.assert_allclose(parent_force[:2], 0.0, atol=1.0e-5)
    test.assertAlmostEqual(float(parent_force[2]), expected_weight, delta=0.001 * expected_weight)
    np.testing.assert_allclose(parent_force[3:], 0.0, atol=1.0e-5)


def _single_joint_local_result(model, state, dt):
    """Evaluate an unrelaxed local correction without advancing time."""
    control = model.control()
    delta = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=model.device)
    result = model.state()
    wp.launch(
        solve_body_joints,
        model.joint_count,
        inputs=[
            state.body_q,
            state.body_qd,
            model.body_com,
            model.body_inv_mass,
            model.body_inv_inertia,
            model.joint_type,
            model.joint_enabled,
            model.joint_parent,
            model.joint_child,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_limit_lower,
            model.joint_limit_upper,
            model.joint_qd_start,
            model.joint_target_q_start,
            model.joint_dof_dim,
            model.joint_axis,
            control.joint_target_q,
            control.joint_target_qd,
            model.joint_target_ke,
            model.joint_target_kd,
            0.0,
            0.0,
            1.0,
            1.0,
            dt,
        ],
        outputs=[delta, None],
        device=model.device,
    )
    wp.launch(
        apply_body_deltas,
        model.body_count,
        inputs=[
            state.body_q,
            state.body_qd,
            model.body_com,
            model.body_inertia,
            model.body_inv_mass,
            model.body_inv_inertia,
            delta,
            None,
            dt,
        ],
        outputs=[result.body_q, result.body_qd],
        device=model.device,
    )
    return result


def test_global_joint_preserves_nonzero_limits(test, device):
    """Match local projection for joint intervals that exclude zero."""
    for kind in ("prismatic", "revolute"):
        for sign in (-1.0, 1.0):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            body = builder.add_link(mass=1.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
            lower, upper = sorted((sign * 0.2, sign * 0.4))
            joint = getattr(builder, f"add_joint_{kind}")(
                -1,
                body,
                axis=newton.Axis.X,
                limit_lower=lower,
                limit_upper=upper,
            )
            builder.add_articulation([joint])
            model = builder.finalize(device=device)
            state = model.state()
            expected = _single_joint_local_result(model, state, 1.0 / 120.0)
            solver = newton.solvers.SolverXPBD(model, rigid_joint_global_iterations=1)
            solver._solve_rigid_joint_global(state.body_q, state.body_qd, model.control(), None, 1.0 / 120.0)
            np.testing.assert_allclose(state.body_q.numpy(), expected.body_q.numpy(), atol=2.0e-6)
            np.testing.assert_allclose(state.body_qd.numpy(), expected.body_qd.numpy(), atol=2.0e-4)


def test_global_joint_preserves_gyroscopic_update(test, device):
    """Match local impulse application for a spinning asymmetric body."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(
        mass=1.0,
        inertia=wp.mat33(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0),
        xform=wp.transform(wp.vec3(0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.1)),
    )
    joint = builder.add_joint_fixed(-1, body)
    builder.add_articulation([joint])
    model = builder.finalize(device=device)
    state = model.state()
    state.body_qd.assign(np.asarray([[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]], dtype=np.float32))
    expected = _single_joint_local_result(model, state, 1.0 / 120.0)
    solver = newton.solvers.SolverXPBD(model, rigid_joint_global_iterations=1)
    solver._solve_rigid_joint_global(state.body_q, state.body_qd, model.control(), None, 1.0 / 120.0)
    np.testing.assert_allclose(state.body_q.numpy(), expected.body_q.numpy(), atol=2.0e-6)
    np.testing.assert_allclose(state.body_qd.numpy(), expected.body_qd.numpy(), atol=2.0e-4)


def test_global_static_contact_matches_dense_system(test, device):
    """Condensed static normals match an independent joint/contact KKT solve."""
    model, _ = _build_fixed_chain(device, link_count=2, grounded=True, world_attached=False)
    state = model.state()
    poses = state.body_q.numpy()
    poses[:, 1] -= 0.01
    poses[1, 0] += 0.003
    state.body_q.assign(poses)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    dt = 0.01
    matrix = np.zeros((12, 12))
    for body in range(2):
        matrix[6 * body : 6 * body + 3, 6 * body : 6 * body + 3] = np.eye(3) * model.body_mass.numpy()[body] / dt
        matrix[6 * body + 3 : 6 * body + 6, 6 * body + 3 : 6 * body + 6] = (
            model.body_inertia.numpy()[body].astype(float) / dt
        )
    rhs = np.zeros(12)
    normal_rows, gaps, shape0_rows = [], [], []
    normals = contacts.rigid_contact_normal.numpy()[:count].astype(float)
    margins = contacts.rigid_contact_margin0.numpy()[:count] + contacts.rigid_contact_margin1.numpy()[:count]
    bodies, points = [], []
    for endpoint in (0, 1):
        shapes = getattr(contacts, f"rigid_contact_shape{endpoint}").numpy()[:count]
        body = model.shape_body.numpy()[shapes]
        point = getattr(contacts, f"rigid_contact_point{endpoint}").numpy()[:count].astype(float)
        moving = body >= 0
        point[moving] += poses[body[moving], :3]
        bodies.append(body)
        points.append(point)
    for c in range(count):
        gap = float(normals[c] @ (points[1][c] - points[0][c]) - margins[c])
        test.assertLess(gap, 0.0)
        row = np.zeros(12)
        jac_a = np.r_[-normals[c], -np.cross(points[0][c], normals[c])]
        for endpoint, sign in ((0, -1), (1, 1)):
            body = int(bodies[endpoint][c])
            if body >= 0:
                jac = np.r_[sign * normals[c], sign * np.cross(points[endpoint][c] - poses[body, :3], normals[c])]
                row[6 * body : 6 * body + 6] = jac
                if endpoint == 0:
                    jac_a = jac
        matrix += np.outer(row, row) / 1.0e-6
        rhs -= row * gap / 1.0e-6
        normal_rows.append(row)
        shape0_rows.append(jac_a)
        gaps.append(gap)

    # Fixed-joint rows at identity: the local swing coordinates are half angles.
    joint = np.zeros((6, 12))
    joint[:3, :3], joint[:3, 6:9] = -np.eye(3), np.eye(3)
    joint[1, 5], joint[2, 4] = -0.15, 0.15
    joint[3:, 3:6] = -np.diag([1.0, 0.5, 0.5])
    joint[3:, 9:12] = np.diag([1.0, 0.5, 0.5])
    error = np.zeros(6)
    error[:3] = poses[1, :3].astype(float) - poses[0, :3] - np.array([0.15, 0.0, 0.0])
    system = np.block([[matrix, joint.T], [joint, np.zeros((6, 6))]])
    exact = np.linalg.solve(system, np.r_[rhs, -error])[:12]
    impulse = wp.zeros(contacts.rigid_contact_max, dtype=wp.spatial_vector, device=device)
    solver = newton.solvers.SolverXPBD(model, rigid_joint_global_iterations=1)
    solver._solve_rigid_joint_global(state.body_q, state.body_qd, model.control(), None, dt, contacts, impulse)
    correction = solver._rigid_joint_global_solver.backend.body_correction.numpy().ravel()
    np.testing.assert_allclose(correction, exact, atol=3.0e-6, rtol=3.0e-3)
    multipliers = -(np.asarray(gaps) + np.asarray(normal_rows) @ exact) / 1.0e-6
    np.testing.assert_allclose(
        impulse.numpy()[:count], multipliers[:, None] * np.asarray(shape0_rows), atol=0.03, rtol=0.03
    )
    # A reused contact allocation with count zero must not retain its old metric.
    contacts.rigid_contact_count.zero_()
    state.body_q.assign(poses)
    state.body_qd.zero_()
    impulse.zero_()
    solver._solve_rigid_joint_global(state.body_q, state.body_qd, model.control(), None, dt, contacts, impulse)
    correction = solver._rigid_joint_global_solver.backend.body_correction.numpy()
    np.testing.assert_allclose(correction[:, 1], 0.0, atol=1.0e-6)
    np.testing.assert_array_equal(impulse.numpy(), 0.0)


def test_global_distance_matches_scalar_projection(test, device):
    """Preserve radial limits, inactive rows, damping and coincident anchors."""
    for position in (0.0, 0.1, 0.3, 0.6):
        for stiffness, damping in ((0.0, 0.0), (400.0, 7.0)):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            body = builder.add_link(
                mass=2.0,
                inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                xform=wp.transform(wp.vec3(position, 0.0, 0.0), wp.quat_identity()),
            )
            joint = builder.add_joint_distance(
                -1,
                body,
                min_distance=0.2,
                max_distance=0.4,
            )
            builder.joint_target_ke[builder.joint_qd_start[joint]] = stiffness
            builder.joint_target_kd[builder.joint_qd_start[joint]] = damping
            builder.add_articulation([joint])
            model = builder.finalize(device=device)
            state = model.state()
            state.body_qd.assign(np.asarray([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
            expected = _single_joint_local_result(model, state, 1.0 / 120.0)
            solver = newton.solvers.SolverXPBD(model, rigid_joint_global_iterations=1)
            test.assertTrue(solver._rigid_joint_global_solver.active)
            solver._solve_rigid_joint_global(state.body_q, state.body_qd, model.control(), None, 1.0 / 120.0)
            np.testing.assert_allclose(state.body_q.numpy(), expected.body_q.numpy(), atol=2.0e-6)
            np.testing.assert_allclose(state.body_qd.numpy(), expected.body_qd.numpy(), atol=2.0e-4)


def test_global_distance_matches_dense_system(test, device):
    """Check coupled damping/mass scaling against an independent FP64 solve."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    masses = np.asarray([1.0, 3.0, 0.5])
    positions = np.asarray([0.25, 0.55, 0.8])
    velocities = np.asarray([0.1, -0.2, 0.3])
    stiffness = np.asarray([200.0, 800.0, 400.0])
    damping = np.asarray([3.0, 10.0, 2.0])
    joints = []
    parent = -1
    for index in range(3):
        body = builder.add_link(
            mass=masses[index],
            xform=wp.transform(wp.vec3(positions[index], 0.0, 0.0), wp.quat_identity()),
            inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
        joint = builder.add_joint_distance(parent, body, min_distance=0.2, max_distance=0.2)
        joints.append(joint)
        builder.joint_target_ke[builder.joint_qd_start[joint]] = stiffness[index]
        builder.joint_target_kd[builder.joint_qd_start[joint]] = damping[index]
        parent = body
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    state = model.state()
    qd = np.zeros((3, 6), dtype=np.float32)
    qd[:, 0] = velocities
    state.body_qd.assign(qd)
    dt = 1.0 / 120.0
    jacobian = np.eye(3) - np.eye(3, k=-1)
    gamma = damping / stiffness
    scale = np.sqrt(1.0 + gamma / dt)
    scaled_jacobian = scale[:, None] * jacobian
    residual = (jacobian @ positions - 0.2 + gamma * (jacobian @ velocities)) / scale
    inverse_metric = np.diag(dt / masses)
    system = np.diag(1.0 / (stiffness * dt)) + scaled_jacobian @ inverse_metric @ scaled_jacobian.T
    delta = -inverse_metric @ scaled_jacobian.T @ np.linalg.solve(system, residual)
    solver = newton.solvers.SolverXPBD(model, rigid_joint_global_iterations=1)
    test.assertTrue(solver._rigid_joint_global_solver.active)
    solver._solve_rigid_joint_global(state.body_q, state.body_qd, model.control(), None, dt)
    np.testing.assert_allclose(state.body_q.numpy()[:, 0], positions + delta, atol=2.0e-6)
    np.testing.assert_allclose(state.body_qd.numpy()[:, 0], velocities + delta / dt, atol=2.0e-4)


def test_global_distance_capture_updates_active_set(test, device):
    """Reuse one graph as a radial limit activates, releases and reverses."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=1.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    builder.add_articulation([builder.add_joint_distance(-1, body, min_distance=0.2, max_distance=0.4)])
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverXPBD(model, rigid_joint_global_iterations=1)
    state = model.state()
    control = model.control()
    dt = 1.0 / 120.0
    solver._solve_rigid_joint_global(state.body_q, state.body_qd, control, None, dt)
    with wp.ScopedCapture(device) as capture:
        solver._solve_rigid_joint_global(state.body_q, state.body_qd, control, None, dt)
    for position in (0.0, 0.3, 0.6, 0.3, -0.1):
        state.body_q.assign(np.asarray([[position, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
        state.body_qd.zero_()
        expected = _single_joint_local_result(model, state, dt)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(state.body_q.numpy(), expected.body_q.numpy(), atol=2.0e-6)
        np.testing.assert_allclose(state.body_qd.numpy(), expected.body_qd.numpy(), atol=2.0e-4)


class TestRigidXPBDKKT(unittest.TestCase):
    pass


for _test in (
    test_global_joint_preserves_nonzero_limits,
    test_global_joint_preserves_gyroscopic_update,
    test_global_static_contact_matches_dense_system,
    test_global_distance_matches_scalar_projection,
    test_global_distance_matches_dense_system,
):
    add_function_test(TestRigidXPBDKKT, _test.__name__, _test, devices=get_test_devices())


add_function_test(
    TestRigidXPBDKKT,
    "test_global_distance_capture_updates_active_set",
    test_global_distance_capture_updates_active_set,
    devices=get_cuda_test_devices(),
)


add_function_test(
    TestRigidXPBDKKT,
    "test_global_joint_iteration_validation",
    test_global_joint_iteration_validation,
    devices=get_test_devices(),
)
add_function_test(
    TestRigidXPBDKKT,
    "test_global_fixed_chain_improves_convergence",
    test_global_fixed_chain_improves_convergence,
    devices=get_test_devices(),
)
add_function_test(
    TestRigidXPBDKKT,
    "test_global_joint_topology_routes",
    test_global_joint_topology_routes,
    devices=get_test_devices(),
)
add_function_test(
    TestRigidXPBDKKT,
    "test_unsupported_mixed_island_remains_local",
    test_unsupported_mixed_island_remains_local,
    devices=get_test_devices(),
)
add_function_test(
    TestRigidXPBDKKT,
    "test_supported_joint_controls_and_compliance",
    test_supported_joint_controls_and_compliance,
    devices=get_test_devices(),
)
add_function_test(
    TestRigidXPBDKKT,
    "test_free_root_does_not_hide_supported_constraints",
    test_free_root_does_not_hide_supported_constraints,
    devices=get_test_devices(),
)
add_function_test(
    TestRigidXPBDKKT,
    "test_global_joint_cuda_graph_capture",
    test_global_joint_cuda_graph_capture,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestRigidXPBDKKT,
    "test_global_joint_inertial_refresh_rebuilds_dynamic_partition",
    test_global_joint_inertial_refresh_rebuilds_dynamic_partition,
    devices=get_test_devices(),
)
add_function_test(
    TestRigidXPBDKKT,
    "test_global_joint_contact_reconciliation",
    test_global_joint_contact_reconciliation,
    devices=get_cuda_test_devices(),
    check_output=False,
)
add_function_test(
    TestRigidXPBDKKT,
    "test_global_joint_parent_force",
    test_global_joint_parent_force,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
