# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for VBD's topology-partitioned pair-contact blocks."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd import rigid_pair_contact, rigid_vbd_kernels
from newton._src.solvers.vbd.rigid_pair_contact import RigidPairContact
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    accumulate_body_body_contacts_per_body,
    color_rigid_bodies_from_contacts,
)
from newton._src.solvers.xpbd import kernels as xpbd_kernels
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices, get_test_devices

newton.use_coord_layout_targets = True


def _make_pipeline(model, capacity):
    return newton.CollisionPipeline(
        model,
        broad_phase="sap",
        rigid_contact_max=capacity,
        contact_matching="sticky",
        deterministic=True,
    )


def test_pair_preparation_refreshes_material_and_previous_pose(test, device):
    """Refresh material, anchors and prescribed surface motion on captured sweeps."""
    model = SimpleNamespace(
        body_count=1,
        device=device,
        shape_body=wp.array([-1, 0], dtype=int, device=device),
        body_com=wp.zeros(1, dtype=wp.vec3, device=device),
    )
    inverse_mass = wp.ones(1, dtype=float, device=device)
    pair = rigid_pair_contact.RigidPairContact(
        model,
        inverse_mass,
        wp.zeros(1, dtype=wp.mat33, device=device),
        wp.ones(1, dtype=int, device=device),
        1,
    )
    contacts = SimpleNamespace(
        rigid_contact_max=1,
        rigid_contact_count=wp.ones(1, dtype=int, device=device),
        rigid_contact_shape0=wp.zeros(1, dtype=int, device=device),
        rigid_contact_shape1=wp.ones(1, dtype=int, device=device),
        rigid_contact_point0=wp.zeros(1, dtype=wp.vec3, device=device),
        rigid_contact_point1=wp.zeros(1, dtype=wp.vec3, device=device),
        rigid_contact_surface_velocity=wp.zeros(1, dtype=wp.vec3, device=device),
        rigid_contact_offset0=wp.zeros(1, dtype=wp.vec3, device=device),
        rigid_contact_offset1=wp.zeros(1, dtype=wp.vec3, device=device),
        rigid_contact_normal=wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3, device=device),
        rigid_contact_margin0=wp.zeros(1, dtype=float, device=device),
        rigid_contact_margin1=wp.full(1, 0.1, dtype=float, device=device),
    )
    pair.build(contacts)
    seed = wp.array([[0.02, 0.0, 0.09, 0.0, 0.0, 0.0, 1.0]], dtype=wp.transform, device=device)
    pose = wp.clone(seed)
    previous_pose = wp.clone(seed)
    stiffness = wp.ones(1, dtype=float, device=device)
    damping = wp.zeros(1, dtype=float, device=device)
    friction = wp.full(1, 1.0e6, dtype=float, device=device)
    multiplier = wp.zeros(1, dtype=wp.vec3, device=device)
    dt = 0.01

    def sweep():
        wp.copy(pose, seed)
        multiplier.zero_()
        pair.solve(pose, previous_pose, contacts, dt, stiffness, damping, friction, multiplier, 0, True)

    sweep()
    graph = None
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            sweep()
        graph = capture.graph
    for ke, kd, previous_x, gap, surface_velocity in [
        (1000.0, 0.0, 0.0, 0.01, (0.0, 0.0, 0.0)),
        (2000.0, 10.0, 0.01, 0.004, (0.5, 0.0, 1.0)),
        (0.0, 0.0, -0.01, 0.02, (-0.5, 0.0, 0.0)),
        (500.0, 0.0, -0.02, 0.01, (-0.5, 0.0, -1.0)),
    ]:
        contacts.rigid_contact_surface_velocity.assign(np.asarray([surface_velocity], dtype=np.float32))
        stiffness.fill_(ke)
        damping.fill_(kd)
        previous_pose.assign(np.asarray([[previous_x, 0.0, 0.1 - gap, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
        if graph is None:
            sweep()
        else:
            wp.capture_launch(graph)
        # Implicit Kelvin--Voigt normal response with unit inverse mass.
        effective_stiffness = ke + kd / dt
        normal_force = max(0.0, (ke * 0.01 + kd / dt * (0.01 - gap)) / (1.0 + effective_stiffness * dt * dt))
        # Large friction is intentionally non-sliding: remove relative translation.
        expected_x = previous_x - dt * surface_velocity[0] if normal_force > 0.0 else 0.02
        actual = pose.numpy()[0]
        np.testing.assert_allclose(actual[:3], [expected_x, 0.0, 0.09 + dt * dt * normal_force], atol=3.0e-7, rtol=0.0)
        np.testing.assert_allclose(multiplier.numpy()[0, 2], normal_force, atol=3.0e-5, rtol=1.0e-6)
        np.testing.assert_array_equal(actual[3:], [0.0, 0.0, 0.0, 1.0])


def test_pair_contacts_are_limited_to_free_body_systems(test, device):
    """Keep pair contacts exclusive to joint-free collision worlds."""

    def make_solver(jointed, compliant=False, history=False, hard=True):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        bodies = []
        for x in (0.0, 2.0):
            body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()))
            builder.add_shape_sphere(body, radius=0.5)
            bodies.append(body)
        if jointed:
            builder.add_joint_fixed(parent=bodies[0], child=bodies[1])
        builder.color()
        model = builder.finalize(device=device)
        return newton.solvers.SolverVBD(
            model,
            iterations=1,
            rigid_compliant_alm=compliant,
            rigid_contact_history=history,
            rigid_contact_hard=hard,
        )

    test.assertIsNotNone(make_solver(jointed=False)._rigid_pair_contacts)
    test.assertIsNone(make_solver(jointed=True)._rigid_pair_contacts)
    test.assertFalse(make_solver(jointed=False, compliant=True)._pair_contacts_enabled_this_step)
    test.assertTrue(make_solver(jointed=False, compliant=True, history=True)._pair_contacts_enabled_this_step)
    legacy_soft = make_solver(jointed=False, compliant=False, hard=False)
    test.assertIsNone(legacy_soft._rigid_pair_contacts)
    np.testing.assert_array_equal(legacy_soft.body_uses_pair_contacts.numpy(), np.zeros(2, dtype=np.int32))


def test_mixed_joint_and_free_bodies_share_per_body_world(test, device):
    """Keep both endpoints of a mixed contact in the per-body solve."""

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, ke=1.0e6, kd=1.0e3, mu=0.0)
    bodies = []
    for x in (-2.0, 0.0, 0.9):
        body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.5, cfg=cfg)
        bodies.append(body)
    builder.add_joint_fixed(
        parent=bodies[0],
        child=bodies[1],
        parent_xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)
    pipeline = _make_pipeline(model, 32)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=4,
        rigid_compliant_alm=True,
        rigid_contact_history=False,
    )
    state_in = model.state()
    state_out = model.state()

    np.testing.assert_array_equal(solver.body_uses_pair_contacts.numpy(), np.zeros(3, dtype=np.int32))
    structural_groups = np.concatenate([group.numpy() for group in solver._structural_body_color_groups])
    np.testing.assert_array_equal(np.sort(structural_groups), np.arange(3))
    test.assertIsNone(solver._rigid_pair_contacts)

    pipeline.collide(state_in, contacts)
    before = state_in.body_q.numpy().copy()
    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 120.0)
    after = state_out.body_q.numpy().copy()
    test.assertGreater(after[2, 0], before[2, 0] + 1.0e-5)
    test.assertTrue(np.isfinite(after).all())


def test_world_partition_keeps_pair_contacts_in_joint_free_worlds(test, device):
    """Use pair contacts in a collision-isolated world without structural joints."""

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    bodies = []
    builder.begin_world(label="jointed")
    for x in (0.0, 1.9):
        body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=1.0)
        bodies.append(body)
    builder.add_joint_fixed(parent=bodies[0], child=bodies[1])
    builder.end_world()
    builder.begin_world(label="free")
    for x in (0.0, 1.9):
        body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=1.0)
        bodies.append(body)
    builder.end_world()
    builder.color()
    model = builder.finalize(device=device)
    pipeline = _make_pipeline(model, 32)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(model, iterations=1, rigid_compliant_alm=False)
    state_in = model.state()
    state_out = model.state()

    np.testing.assert_array_equal(solver.body_uses_pair_contacts.numpy(), np.asarray([0, 0, 1, 1]))
    structural_groups = np.concatenate([group.numpy() for group in solver._structural_body_color_groups])
    np.testing.assert_array_equal(np.sort(structural_groups), np.asarray([0, 1]))
    test.assertIsNotNone(solver._rigid_pair_contacts)
    test.assertEqual(solver._rigid_pair_contacts.pair_body_count, 2)

    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 120.0)
    pair = solver._rigid_pair_contacts
    test.assertEqual(int(pair.group_count.numpy()[0]), 1)
    groups = np.flatnonzero(pair.group_color.numpy() >= 0)
    test.assertEqual(groups.size, 1)
    group = groups[0]
    test.assertEqual(
        {int(pair.group_body0.numpy()[group]), int(pair.group_body1.numpy()[group])},
        {2, 3},
    )
    test.assertTrue(np.isfinite(state_out.body_q.numpy()).all())


def test_legacy_iteration_collision_falls_back_to_complete_body_schedule(test, device):
    """Do not omit pair-partitioned bodies when legacy iteration detection disables the pair lane."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, ke=1.0e6, kd=1.0e3, mu=0.0)

    builder.begin_world(label="jointed")
    jointed = []
    for x in (-2.0, 0.0):
        body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.5, cfg=cfg)
        jointed.append(body)
    builder.add_joint_fixed(
        parent=jointed[0],
        child=jointed[1],
        parent_xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.end_world()

    builder.begin_world(label="free")
    free = []
    for x in (0.0, 0.9):
        body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.5, cfg=cfg)
        free.append(body)
    builder.end_world()
    builder.color()
    model = builder.finalize(device=device)
    pipeline = _make_pipeline(model, 32)
    slot = newton.solvers.SolverBase.CollisionSlot.RIGID
    frequency = newton.solvers.SolverBase.CollisionFrequencyType.ITERATIONS
    solver = newton.solvers.SolverVBD(
        model,
        iterations=2,
        rigid_compliant_alm=False,
        rigid_contact_hard=True,
        rigid_contact_coloring="refined",
        collision_pipeline=pipeline,
        collision_frequency={slot: 1},
        collision_frequency_type={slot: frequency},
    )
    state_in, state_out = model.state(), model.state()
    before = state_in.body_q.numpy().copy()
    solver.step(state_in, state_out, None, None, 1.0 / 120.0)
    after = state_out.body_q.numpy()

    test.assertFalse(solver._pair_contacts_enabled_this_step)
    before_separation = before[free[1], 0] - before[free[0], 0]
    after_separation = after[free[1], 0] - after[free[0], 0]
    test.assertGreater(after_separation, before_separation + 1.0e-5)
    test.assertTrue(np.isfinite(after).all())


def test_per_body_contact_lists_are_sorted_without_pair_manager(test, device):
    """Sort jointed-body contact lists even when no pair manager exists."""

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, ke=1.0e6, kd=1.0e3, mu=0.0)
    bodies = []
    for index in range(10):
        angle = 2.0 * math.pi * index / 10
        position = wp.vec3(1.5 * math.cos(angle), 1.5 * math.sin(angle), 0.0)
        body = builder.add_body(xform=wp.transform(position, wp.quat_identity()))
        builder.add_shape_sphere(body, radius=1.0, cfg=cfg)
        bodies.append(body)
        if index:
            builder.add_joint_fixed(parent=bodies[index - 1], child=body)
    builder.color()
    model = builder.finalize(device=device)
    pipeline = _make_pipeline(model, 256)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=False,
        rigid_body_contact_buffer_size=64,
    )
    test.assertIsNone(solver._rigid_pair_contacts)
    state_in = model.state()
    state_out = model.state()

    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 120.0)
    counts = solver.body_body_contact_counts.numpy()
    indices = solver.body_body_contact_indices.numpy().reshape(model.body_count, -1)
    test.assertGreater(int(np.max(counts)), 1)
    for body, count in enumerate(counts):
        active = indices[body, : min(int(count), indices.shape[1])]
        np.testing.assert_array_equal(active, np.sort(active))


def test_pair_warmstart_respects_normal_and_tangent_moment_arms(test, device):
    """Keep sphere normal warm starts torque-free while preserving friction torque."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.99), wp.quat_identity()))
    builder.add_shape_sphere(body, radius=1.0)
    builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device=device)
    pipeline = _make_pipeline(model, 16)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(model, iterations=1, rigid_compliant_alm=True, rigid_contact_history=True)
    state_in, state_out = model.state(), model.state()
    control = model.control()
    dt = 1.0 / 120.0
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, control, contacts, dt)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)

    shape0 = int(contacts.rigid_contact_shape0.numpy()[0])
    body_is_first = int(model.shape_body.numpy()[shape0]) == body
    sign = -1.0 if body_is_first else 1.0
    normal = contacts.rigid_contact_normal.numpy()[0]
    offsets = contacts.rigid_contact_offset0 if body_is_first else contacts.rigid_contact_offset1
    rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 4.0)
    arm = np.asarray(wp.quat_rotate(rotation, wp.vec3(*offsets.numpy()[0])))
    poses = state_in.body_q.numpy().copy()
    poses[body, 3:] = np.asarray(rotation)
    q_seed = wp.array(poses, dtype=wp.transform, device=device)

    for tangent in (np.zeros(3), np.array([3.0, 0.0, 0.0])):
        force = sign * normal * 10.0 + tangent
        multipliers = np.zeros((contacts.rigid_contact_max, 3), dtype=np.float32)
        multipliers[0] = sign * force
        lambda_seed = wp.array(multipliers, dtype=wp.vec3, device=device)

        def primal(seed=lambda_seed):
            wp.copy(state_in.body_q, q_seed)
            wp.copy(solver.body_body_contact_lambda, seed)
            solver._solve_rigid_body_iteration(state_in, state_out, control, contacts, dt)

        primal()
        np.testing.assert_allclose(solver.body_forces.numpy()[body], force, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(solver.body_torques.numpy()[body], np.cross(arm, tangent), atol=1.0e-6)
        if device.is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                primal()
            for _ in range(3):
                wp.capture_launch(capture.graph)
                np.testing.assert_allclose(solver.body_torques.numpy()[body], np.cross(arm, tangent), atol=1.0e-6)


def test_pair_primal_is_seeded_only_for_retained_multipliers(test, device):
    """Seed retained multipliers once and rebuild coloring only on refresh."""

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for x in (0.0, 1.9):
        body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=1.0)
    builder.color()
    model = builder.finalize(device=device)
    pipeline = _make_pipeline(model, 16)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=3,
        rigid_compliant_alm=False,
        rigid_contact_history=False,
    )
    state_in = model.state()
    state_out = model.state()

    include_pair_primal = []
    pair_build_count = 0
    scratch_clear_count = 0
    launch_kernel = wp.launch

    def record_launch(*args, **kwargs):
        nonlocal scratch_clear_count
        if kwargs.get("kernel") is rigid_vbd_kernels.clear_rigid_body_contact_scratch:
            scratch_clear_count += 1
        return launch_kernel(*args, **kwargs)

    solve_iteration = solver._solve_rigid_body_iteration
    pair_build = solver._rigid_pair_contacts.build

    def record_iteration(*args, **kwargs):
        include_pair_primal.append(kwargs["include_pair_primal"])
        return solve_iteration(*args, **kwargs)

    def record_pair_build(*args, **kwargs):
        nonlocal pair_build_count
        pair_build_count += 1
        return pair_build(*args, **kwargs)

    solver._solve_rigid_body_iteration = record_iteration
    solver._rigid_pair_contacts.build = record_pair_build
    pipeline.collide(state_in, contacts)
    with patch.object(wp, "launch", side_effect=record_launch) as launch:
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 120.0)
    test.assertFalse(
        any(call.kwargs.get("kernel") is rigid_vbd_kernels.update_duals_joint for call in launch.call_args_list)
    )
    if device.is_cuda:
        test.assertFalse(
            any(
                call.kwargs.get("kernel") is rigid_pair_contact.solve_constraint_contacts_serial
                for call in launch.call_args_list
            )
        )
    test.assertEqual(include_pair_primal, [False, False, False])
    test.assertEqual(pair_build_count, 1)
    test.assertEqual(scratch_clear_count, 0)

    include_pair_primal.clear()
    solver.set_rigid_history_update(False)
    with patch.object(wp, "launch", side_effect=record_launch):
        solver.step(state_out, state_in, model.control(), contacts, 1.0 / 120.0)
    test.assertEqual(include_pair_primal, [True, False, False])
    test.assertEqual(pair_build_count, 1)
    test.assertEqual(scratch_clear_count, 1)

    include_pair_primal.clear()
    pipeline.collide(state_in, contacts)
    with patch.object(wp, "launch", side_effect=record_launch):
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 120.0)
    test.assertEqual(include_pair_primal, [False, False, False])
    test.assertEqual(pair_build_count, 2)
    test.assertEqual(scratch_clear_count, 1)


def _parallel_chain_snapshot(device):
    body_count = max(300, 2 * device.sm_count + 2)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = newton.ModelBuilder.ShapeConfig(density=1.0, ke=1.0e6, kd=1.0e3, mu=0.0)
    for body_index in range(body_count):
        body = builder.add_body(xform=wp.transform(wp.vec3(1.9 * body_index, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=1.0, cfg=cfg)
    builder.color()
    model = builder.finalize(device=device)
    capacity = body_count * 4
    pipeline = _make_pipeline(model, capacity)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=False,
        rigid_contact_history=False,
    )
    state_in = model.state()
    state_out = model.state()

    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 120.0)
    pair = solver._rigid_pair_contacts
    return {
        "body_count": body_count,
        "contact_count": int(contacts.rigid_contact_count.numpy()[0]),
        "group_count": int(pair.group_count.numpy()[0]),
        "overflow_count": int(pair.overflow_count.numpy()[0]),
        "group_color": pair.group_color.numpy().copy(),
        "group_body0": pair.group_body0.numpy().copy(),
        "group_body1": pair.group_body1.numpy().copy(),
    }


def test_parallel_pair_coloring_is_conflict_free_and_repeatable(test, device):
    """Produce a repeatable, conflict-free parallel pair coloring."""

    first = _parallel_chain_snapshot(device)
    second = _parallel_chain_snapshot(device)

    test.assertEqual(first["contact_count"], first["body_count"] - 1)
    test.assertEqual(first["group_count"], first["contact_count"])
    test.assertEqual(first["overflow_count"], 0)
    for name in ("group_color", "group_body0", "group_body1"):
        np.testing.assert_array_equal(first[name], second[name])

    colors = first["group_color"]
    for color in np.unique(colors[colors >= 0]):
        groups = np.flatnonzero(colors == color)
        endpoints = np.concatenate((first["group_body0"][groups], first["group_body1"][groups]))
        test.assertEqual(np.unique(endpoints).size, endpoints.size)


def _overflow_snapshot(device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, ke=1.0e6, kd=1.0e3, mu=0.0)
    center = builder.add_body(xform=wp.transform_identity())
    builder.add_shape_sphere(center, radius=1.0, cfg=cfg)
    satellite_count = 40
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(satellite_count):
        z = 1.0 - 2.0 * (index + 0.5) / satellite_count
        radius_xy = math.sqrt(max(0.0, 1.0 - z * z))
        theta = index * golden_angle
        direction = wp.vec3(radius_xy * math.cos(theta), radius_xy * math.sin(theta), z)
        body = builder.add_body(xform=wp.transform(1.1 * direction, wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.12, cfg=cfg)
    builder.color()
    model = builder.finalize(device=device)
    pipeline = _make_pipeline(model, 1024)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=2,
        rigid_compliant_alm=False,
        rigid_contact_history=True,
    )
    state_in = model.state()
    state_out = model.state()

    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 120.0)
    pair = solver._rigid_pair_contacts
    return (
        int(contacts.rigid_contact_count.numpy()[0]),
        int(pair.group_count.numpy()[0]),
        int(pair.overflow_count.numpy()[0]),
        pair.color_counts.numpy().copy(),
        state_out.body_q.numpy().copy(),
        state_out.body_qd.numpy().copy(),
    )


def test_pair_color_overflow_is_complete_and_repeatable(test, device):
    """Solve color overflow completely and repeatably."""

    first = _overflow_snapshot(device)
    second = _overflow_snapshot(device)
    test.assertEqual(first[0], 40)
    test.assertEqual(first[1], 40)
    test.assertEqual(first[2], 10)
    test.assertEqual(int(np.sum(first[3])), 30)
    test.assertTrue(np.isfinite(first[4]).all())
    test.assertTrue(np.isfinite(first[5]).all())
    for first_value, second_value in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_value, second_value)


def _make_jointed_contact_lattice(device, ny, nz):
    """Build joint chains with four-point face manifolds between chains."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.rigid_gap = 0.0
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, ke=1.0e5, kd=100.0, mu=0.5)
    nx = 4
    for iz in range(nz):
        for iy in range(ny):
            parent = -1
            for ix in range(nx):
                body = builder.add_body(
                    xform=wp.transform(wp.vec3(0.96 * ix, 0.96 * iy, 0.96 * iz), wp.quat_identity())
                )
                builder.add_shape_box(body, hx=0.45, hy=0.5, hz=0.5, cfg=cfg)
                if parent >= 0:
                    builder.add_joint_fixed(
                        parent=parent,
                        child=body,
                        parent_xform=wp.transform(wp.vec3(0.96, 0.0, 0.0), wp.quat_identity()),
                    )
                parent = body
    builder.color()
    model = builder.finalize(device=device)
    pipeline = _make_pipeline(model, model.body_count * 64)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_contact_history=False,
        rigid_body_contact_buffer_size=128,
        rigid_contact_coloring="auto",
    )
    return model, pipeline, contacts, solver


def _dynamic_contact_pairs(model, contacts):
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape_body = model.shape_body.numpy()
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    body0 = np.where(shape0 >= 0, shape_body[np.maximum(shape0, 0)], -1)
    body1 = np.where(shape1 >= 0, shape_body[np.maximum(shape1, 0)], -1)
    keep = (body0 >= 0) & (body1 >= 0) & (body0 != body1)
    pairs = np.stack((np.minimum(body0[keep], body1[keep]), np.maximum(body0[keep], body1[keep])), axis=1)
    return count, np.unique(pairs, axis=0)


def test_jointed_contact_refinement_counts_manifolds_not_points(test, device):
    """Four points on one body pair remain one non-branching graph edge."""
    model, pipeline, contacts, solver = _make_jointed_contact_lattice(device, ny=2, nz=1)
    state_in = model.state()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, model.state(), model.control(), contacts, 1.0 / 600.0)
    count, pairs = _dynamic_contact_pairs(model, contacts)

    test.assertGreater(count, pairs.shape[0])
    test.assertEqual(int(solver._rigid_body_max_same_color_degree.numpy()[0]), 1)
    test.assertEqual(int(solver._rigid_body_refinement_active.numpy()[0]), 0)


def test_jointed_contact_refinement_preserves_joint_colors(test, device):
    """Branching contacts are exactly colored without weakening joint independence."""
    model, pipeline, contacts, solver = _make_jointed_contact_lattice(device, ny=4, nz=2)
    state_in = model.state()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, model.state(), model.control(), contacts, 1.0 / 600.0)

    test.assertGreater(int(solver._rigid_body_max_same_color_degree.numpy()[0]), 2)
    test.assertEqual(int(solver._rigid_body_refinement_active.numpy()[0]), 1)
    test.assertEqual(int(solver._rigid_body_refinement_overflow.numpy()[0]), 0)

    colors = solver._rigid_body_colors.numpy()
    active = solver._rigid_body_schedule_active.numpy().astype(bool)
    test.assertTrue(np.all(colors[active] >= 0))
    for parent, child in zip(model.joint_parent.numpy(), model.joint_child.numpy(), strict=True):
        if parent >= 0 and child >= 0 and active[parent] and active[child]:
            test.assertNotEqual(colors[parent], colors[child])

    _, pairs = _dynamic_contact_pairs(model, contacts)
    test.assertTrue(np.all(colors[pairs[:, 0]] != colors[pairs[:, 1]]))


def test_jointed_contact_refinement_is_cuda_graph_repeatable(test, device):
    """Dynamic analysis, coloring, and nested solve selection replay entirely on-device."""
    model, pipeline, contacts, solver = _make_jointed_contact_lattice(device, ny=4, nz=2)
    state_in = model.state()
    state_out = model.state()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 600.0)

    state_in = model.state()
    state_out = model.state()
    pipeline.collide(state_in, contacts)
    with wp.ScopedCapture(device=device) as capture:
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 600.0)

    wp.capture_launch(capture.graph)
    first = solver._rigid_body_colors.numpy().copy()
    wp.capture_launch(capture.graph)
    second = solver._rigid_body_colors.numpy().copy()
    np.testing.assert_array_equal(first, second)
    test.assertEqual(int(solver._rigid_body_refinement_active.numpy()[0]), 1)


def test_jointed_contact_auto_without_conditional_graphs(test, device):
    """Retain contact refinement when CUDA conditional nodes are unavailable."""

    def run(policy, conditional):
        model, pipeline, contacts, solver = _make_jointed_contact_lattice(device, ny=4, nz=2)
        solver.rigid_contact_coloring = policy
        solver.iterations = 4
        state_in, state_out = model.state(), model.state()
        pipeline.collide(state_in, contacts)
        # Warm the kernels without advancing the measured input state.
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 600.0)
        state_in, state_out = model.state(), model.state()
        with patch.object(wp, "is_conditional_graph_supported", return_value=conditional):
            with wp.ScopedCapture(device=device) as capture:
                solver.step(state_in, state_out, model.control(), contacts, 1.0 / 600.0)
        wp.capture_launch(capture.graph)
        return state_out.body_q.numpy(), state_out.body_qd.numpy()

    expected = run("refined", False)
    actual = run("auto", False)
    for left, right in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(left, right)


def test_jointed_contact_graph_changes_during_capture(test, device):
    """Switch dense, empty, and dense contacts within one captured schedule."""
    model, pipeline, contacts, solver = _make_jointed_contact_lattice(device, ny=4, nz=2)
    solver.iterations = 4
    state_in, state_out = model.state(), model.state()
    pipeline.collide(state_in, contacts)
    contact_count = int(contacts.rigid_contact_count.numpy()[0])
    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 600.0)
    state_in, state_out = model.state(), model.state()
    # Restore the entire direct state, including colors, between comparisons.
    arrays = []
    seen = set()
    for obj in (solver, state_in, state_out):
        for value in vars(obj).values():
            if isinstance(value, wp.array) and value.ptr and value.ptr not in seen:
                seen.add(value.ptr)
                arrays.append((value, wp.clone(value)))

    def restore():
        for target, seed in arrays:
            wp.copy(target, seed)

    with wp.ScopedCapture(device=device) as capture:
        restore()
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 600.0)
    for count in (contact_count, 0, contact_count):
        contacts.rigid_contact_count.assign(np.array([count], dtype=np.int32))
        restore()
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 600.0)
        expected_q = state_out.body_q.numpy()
        expected_qd = state_out.body_qd.numpy()
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(state_out.body_q.numpy(), expected_q)
        np.testing.assert_array_equal(state_out.body_qd.numpy(), expected_qd)
        test.assertEqual(int(solver._rigid_body_refinement_active.numpy()[0]), int(count > 0))


def test_jointed_contact_refinement_overflow_is_complete(test, device):
    """Keep every jointed body and deterministic output when contact colors run out."""

    def run():
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, ke=1.0e5, mu=0.0)
        for i in range(40):
            position = wp.vec3(0.1 * math.cos(i), 0.1 * math.sin(i), 0.01 * i)
            body = builder.add_link(xform=wp.transform(position, wp.quat_identity()))
            builder.add_shape_sphere(body, radius=0.5, cfg=cfg)
            builder.add_joint_fixed(parent=-1, child=body, parent_xform=wp.transform(position, wp.quat_identity()))
        builder.color()
        model = builder.finalize(device=device)
        pipeline = _make_pipeline(model, 2048)
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverVBD(
            model,
            iterations=2,
            rigid_contact_history=False,
            rigid_compliant_alm=True,
            rigid_contact_coloring="refined",
            rigid_body_contact_buffer_size=64,
        )
        state_in, state_out = model.state(), model.state()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 600.0)
        test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 780)
        test.assertGreater(int(solver._rigid_body_refinement_overflow.numpy()[0]), 0)
        ids = solver._rigid_body_ids_by_color.numpy()
        ends = solver._rigid_body_color_starts.numpy()
        active_count = int(ends[solver._rigid_body_color_budget])
        np.testing.assert_array_equal(np.sort(ids[:active_count]), np.arange(40))
        colors = solver._rigid_body_colors.numpy()
        test.assertTrue(np.all((colors >= 0) & (colors < solver._rigid_body_color_budget)))
        test.assertTrue(np.isfinite(state_out.body_q.numpy()).all())
        return state_out.body_q.numpy(), state_out.body_qd.numpy(), colors

    for first, second in zip(run(), run(), strict=True):
        np.testing.assert_array_equal(first, second)


def test_body_contact_tile_matches_serial_accumulation(test, device):
    """Compare contact tiles with a serial sum and repeat captured results exactly."""
    model, pipeline, contacts, solver = _make_jointed_contact_lattice(device, ny=4, nz=2)
    inverse_mass = solver.body_inv_mass_effective.numpy()
    inverse_mass[::7] = 0.0
    solver.body_inv_mass_effective.assign(inverse_mass)
    solver.iterations = 2
    state_in, state_out = model.state(), model.state()
    pipeline.collide(state_in, contacts)
    original_launch = wp.launch_tiled
    checked = False

    def checked_launch(*args, **kwargs):
        nonlocal checked
        if checked or kwargs.get("kernel") is not accumulate_body_body_contacts_per_body:
            return original_launch(*args, **kwargs)

        # Isolate these five contact outputs from any preceding contributions.
        outputs = kwargs["outputs"]

        def evaluate(width):
            scratch = [wp.zeros_like(value) for value in outputs]
            call = dict(kwargs, outputs=scratch, block_dim=width)
            original_launch(*args, **call)
            return scratch, call, [value.numpy() for value in scratch]

        _, _, reference = evaluate(1)
        if not any(np.any(value) for value in reference):
            return original_launch(*args, **kwargs)
        checked = True
        for width in (4, 8, 16, 32, 64, 128, 256):
            scratch, call, expected = evaluate(width)
            for actual, serial in zip(expected, reference, strict=True):
                # Different reduction trees can round differently. Bound that
                # difference at float32 scale; do not require equality across widths.
                scale = max(1.0, float(np.max(np.abs(serial))))
                np.testing.assert_allclose(actual, serial, rtol=2.0e-6, atol=8.0 * np.finfo(np.float32).eps * scale)
                np.testing.assert_array_equal(actual[::7], np.zeros_like(actual[::7]))
            with wp.ScopedCapture(device=device) as capture:
                for value in scratch:
                    value.zero_()
                original_launch(*args, **call)
            for _ in range(3):
                wp.capture_launch(capture.graph)
                for value, eager in zip(scratch, expected, strict=True):
                    np.testing.assert_array_equal(value.numpy().view(np.uint8), eager.view(np.uint8))
        return original_launch(*args, **kwargs)

    with patch.object(wp, "launch_tiled", side_effect=checked_launch):
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 600.0)
    test.assertTrue(checked, "the test must exercise nonzero rigid contact contributions")


def test_jointed_color_mask_matches_greedy_reference(test, device):
    """Match CPU/CUDA greedy repair, including exhausted mandatory-joint colors."""
    budget = 32

    def reference(seed, active, authored, contacts, joints):
        colors = np.full(len(seed), -1, dtype=np.int32)
        for body, color in enumerate(seed):
            if active[body] and 0 <= color < budget:
                if not any(
                    other < body and active[other] and seed[other] == color for other in contacts[body] | joints[body]
                ):
                    colors[body] = color
        overflow, joint_overflow = 0, 0
        for body in range(len(seed)):
            if not active[body] or colors[body] >= 0:
                continue
            mandatory = {int(colors[other]) for other in joints[body] if active[other]}
            used = mandatory | {int(colors[other]) for other in contacts[body] if active[other]}
            choices = [color for color in range(budget) if color not in used]
            if not choices:
                overflow += 1
                choices = [color for color in range(budget) if color not in mandatory]
                if not choices:
                    joint_overflow = 1
                    choices = [0]
            colors[body] = choices[0]
        if joint_overflow:
            colors = np.where(active, authored, -1).astype(np.int32)
        return colors, overflow, joint_overflow

    for body_count, kind in (
        (40, "clique"),
        (64, "grid"),
        (513, "grid"),
        (769, "grid"),
        (33, "joint-star"),
        (0, "empty"),
        (1, "empty"),
        (17, "inactive"),
    ):
        if kind == "clique":
            edges = [(a, b) for a in range(body_count) for b in range(a + 1, body_count)]
            joint_edges = []
        elif kind == "grid":
            edges = [
                (a, b)
                for a in range(body_count)
                for b in range(a + 1, body_count)
                if abs(a // 8 - b // 8) <= 1 and abs(a % 8 - b % 8) <= 1
            ]
            joint_edges = [(b, b + 1) for b in range(body_count - 1)]
        elif kind == "joint-star":
            edges = []
            joint_edges = [(b, body_count - 1) for b in range(body_count - 1)]
        else:
            edges, joint_edges = [], []
        rows = np.repeat(np.asarray(edges, dtype=np.int32).reshape(-1, 2), 3, axis=0)
        contacts = [set() for _ in range(body_count)]
        joints = [set() for _ in range(body_count)]
        contact_rows = [[] for _ in range(body_count)]
        joint_rows = [[] for _ in range(body_count)]
        for index, (a, b) in enumerate(rows):
            contacts[a].add(int(b))
            contacts[b].add(int(a))
            contact_rows[a].append(index)
            contact_rows[b].append(index)
        for index, (a, b) in enumerate(joint_edges):
            joints[a].add(b)
            joints[b].add(a)
            joint_rows[a].append(index)
            joint_rows[b].append(index)
        capacity = max([1, *map(len, contact_rows)])
        indices = np.zeros((body_count, capacity), dtype=np.int32)
        for body, row in enumerate(contact_rows):
            indices[body, : len(row)] = row
        inverse_mass = np.ones(body_count, dtype=np.float32)
        if kind == "grid":
            inverse_mass[::7] = 0.0
        elif kind == "inactive":
            inverse_mass.fill(0.0)
        active = (inverse_mass > 0.0).astype(np.int32)
        authored = np.arange(body_count, dtype=np.int32) % 2 if kind == "grid" else np.zeros(body_count, dtype=np.int32)
        if kind == "joint-star":
            authored[-1] = 1

        def array(value):
            return wp.array(np.asarray(value, dtype=np.int32), device=device)

        endpoints = np.asarray(joint_edges, dtype=np.int32).reshape(-1, 2)
        inputs = [
            body_count,
            budget,
            512 if device.is_cuda else 1,
            wp.array(inverse_mass, device=device),
            wp.zeros(body_count, dtype=wp.int32, device=device),
            array(active),
            capacity,
            array(list(map(len, contact_rows))),
            array(indices.ravel()),
            array(rows[:, 0]),
            array(rows[:, 1]),
            array(np.arange(body_count)),
            array(authored),
            array(np.cumsum([0, *map(len, joint_rows)])),
            array([joint for row in joint_rows for joint in row]),
            array(endpoints[:, 0]),
            array(endpoints[:, 1]),
            array([int(newton.JointType.FIXED)] * len(joint_edges)),
            False,  # These cold probes replace colors independently of the packed schedule.
        ]
        colors = wp.full(body_count, -1, dtype=wp.int32, device=device)
        ids = wp.full(body_count, -1, dtype=wp.int32, device=device)
        starts = wp.zeros(max(body_count, budget) + 1, dtype=wp.int32, device=device)
        counters = [wp.zeros(1, dtype=wp.int32, device=device) for _ in range(6)]
        outputs = [colors, ids, starts, *counters]
        seeds = [np.full(body_count, -1, dtype=np.int32), authored.copy()]
        if kind == "joint-star":
            seeds.append(np.append(np.arange(32, dtype=np.int32), -1))
        for seed in seeds:
            expected, expected_overflow, expected_joint_overflow = reference(seed, active, authored, contacts, joints)
            seed_array = array(seed)

            def run(colors, seed_array, inputs, outputs):
                wp.copy(colors, seed_array)
                wp.launch(
                    color_rigid_bodies_from_contacts,
                    dim=512 if device.is_cuda else 1,
                    block_dim=512 if device.is_cuda else 1,
                    max_blocks=1,
                    inputs=inputs,
                    outputs=outputs,
                    device=device,
                )

            run(colors, seed_array, inputs, outputs)
            graph = None
            if device.is_cuda:
                with wp.ScopedCapture(device=device) as capture:
                    run(colors, seed_array, inputs, outputs)
                graph = capture.graph
            for _ in range(3):
                if graph is None:
                    run(colors, seed_array, inputs, outputs)
                else:
                    wp.capture_launch(graph)
                np.testing.assert_array_equal(colors.numpy(), expected)
                expected_count = int(expected.max()) + 1 if expected.size else 0
                test.assertEqual(int(counters[0].numpy()[0]), expected_count)
                for counter, threshold in zip(counters[3:], (4, 8, 16), strict=True):
                    test.assertEqual(int(counter.numpy()[0]), int(expected_count > threshold))
                test.assertEqual(int(counters[1].numpy()[0]), expected_overflow)
                test.assertEqual(int(counters[2].numpy()[0]), expected_joint_overflow)
                expected_ids = np.asarray(
                    [body for color in range(budget) for body in range(body_count) if expected[body] == color],
                    dtype=np.int32,
                )
                np.testing.assert_array_equal(ids.numpy()[: len(expected_ids)], expected_ids)
                expected_starts = np.cumsum([0] + [np.count_nonzero(expected == color) for color in range(budget)])
                np.testing.assert_array_equal(starts.numpy()[: budget + 1], expected_starts)
                for a, b in joint_edges:
                    if active[a] and active[b]:
                        test.assertNotEqual(expected[a], expected[b])
                if kind == "joint-star" and np.array_equal(seed[:-1], np.arange(32)):
                    test.assertEqual(expected_joint_overflow, 1)

        # Reuse the packed schedule while replaying changed graph/active inputs.
        inputs[-1] = True

        def refresh(inputs, outputs):
            wp.launch(
                color_rigid_bodies_from_contacts,
                dim=512 if device.is_cuda else 1,
                block_dim=512 if device.is_cuda else 1,
                max_blocks=1,
                inputs=inputs,
                outputs=outputs,
                device=device,
            )

        reuse_graph = None
        if device.is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                refresh(inputs, outputs)
            reuse_graph = capture.graph
        for mutation in ("same", "same", "no-contacts", "restore", "inactive", "restore", "mass", "restore"):
            current_mass = inverse_mass.copy()
            current_active = active.copy()
            if mutation == "inactive":
                current_active[::3] = 0
            if mutation == "mass":
                current_mass[::5] = 0.0
            current_contacts = [set() for _ in range(body_count)] if mutation == "no-contacts" else contacts
            inputs[3].assign(current_mass)
            inputs[5].assign(current_active)
            inputs[7].assign(
                np.zeros(body_count, dtype=np.int32)
                if mutation == "no-contacts"
                else np.asarray(list(map(len, contact_rows)), dtype=np.int32)
            )
            expected, expected_overflow, expected_joint_overflow = reference(
                colors.numpy(), current_active & (current_mass > 0.0), authored, current_contacts, joints
            )
            if reuse_graph is None:
                refresh(inputs, outputs)
            else:
                wp.capture_launch(reuse_graph)
            np.testing.assert_array_equal(colors.numpy(), expected)
            expected_ids = np.full(body_count, -1, dtype=np.int32)
            packed = [body for color in range(budget) for body in range(body_count) if expected[body] == color]
            expected_ids[: len(packed)] = packed
            np.testing.assert_array_equal(ids.numpy(), expected_ids)
            expected_starts = np.cumsum([0] + [np.count_nonzero(expected == color) for color in range(budget)])
            np.testing.assert_array_equal(starts.numpy()[: budget + 1], expected_starts)
            expected_count = int(expected.max()) + 1 if expected.size else 0
            expected_counters = (
                expected_count,
                expected_overflow,
                expected_joint_overflow,
                int(expected_count > 4),
                int(expected_count > 8),
                int(expected_count > 16),
            )
            for counter, value in zip(counters, expected_counters, strict=True):
                test.assertEqual(int(counter.numpy()[0]), value)


def test_pair_coloring_clamps_overreported_contact_count(test, device):
    """Never scan beyond contact capacity when narrow phase reports overflow."""
    capacity = 40
    body_count = max(capacity + 1, device.sm_count + 2 if device.is_cuda else 0)
    model = SimpleNamespace(
        body_count=body_count,
        device=device,
        shape_body=wp.array(np.arange(body_count, dtype=np.int32), device=device),
    )
    pair = RigidPairContact(
        model,
        wp.ones(body_count, dtype=float, device=device),
        wp.zeros(body_count, dtype=wp.mat33, device=device),
        wp.ones(body_count, dtype=wp.int32, device=device),
        body_count,
    )
    contacts = SimpleNamespace(
        rigid_contact_max=capacity,
        rigid_contact_count=wp.array([capacity + 5], dtype=wp.int32, device=device),
        rigid_contact_shape0=wp.zeros(capacity, dtype=wp.int32, device=device),
        rigid_contact_shape1=wp.array(np.arange(1, capacity + 1, dtype=np.int32), device=device),
    )

    pair.build(contacts)
    test.assertEqual(int(pair.group_count.numpy()[0]), capacity)
    test.assertEqual(int(pair.overflow_count.numpy()[0]), capacity - rigid_pair_contact._COLOR_COUNT)
    test.assertLessEqual(int(np.max(pair.group_end.numpy())), capacity)

    if device.is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            pair.build(contacts)
        for _ in range(3):
            wp.capture_launch(capture.graph)
            test.assertEqual(int(pair.group_count.numpy()[0]), capacity)
            test.assertEqual(int(pair.overflow_count.numpy()[0]), capacity - rigid_pair_contact._COLOR_COUNT)


def test_serial_pair_coloring_reuses_capacity_safely(test, device):
    """Clear old compact groups when contact sets shrink, empty, grow, or resize."""
    body_count = 8
    model = SimpleNamespace(
        body_count=body_count,
        device=device,
        shape_body=wp.array(np.arange(body_count, dtype=np.int32), device=device),
    )
    pair = RigidPairContact(
        model,
        wp.ones(body_count, dtype=float, device=device),
        wp.zeros(body_count, dtype=wp.mat33, device=device),
        wp.ones(body_count, dtype=wp.int32, device=device),
        body_count,
    )
    dense = [(0, body) for body in range(1, body_count)]

    for capacity in (64, 8192):
        contacts = SimpleNamespace(
            rigid_contact_max=capacity,
            rigid_contact_count=wp.zeros(1, dtype=wp.int32, device=device),
            rigid_contact_shape0=wp.full(capacity, -1, dtype=wp.int32, device=device),
            rigid_contact_shape1=wp.full(capacity, -1, dtype=wp.int32, device=device),
        )
        pair.build(contacts)
        graph = None
        if device.is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                pair.build(contacts)
            graph = capture.graph

        for edges in (dense, [(0, 1)], [], [(-1, 2), (2, 3)], dense):
            rows = np.repeat(np.asarray(edges, dtype=np.int32).reshape(-1, 2), 4, axis=0)
            shape0 = np.full(capacity, -1, dtype=np.int32)
            shape1 = np.full(capacity, -1, dtype=np.int32)
            shape0[: len(rows)], shape1[: len(rows)] = rows[:, 0], rows[:, 1]
            contacts.rigid_contact_shape0.assign(shape0)
            contacts.rigid_contact_shape1.assign(shape1)
            contacts.rigid_contact_count.assign(np.asarray([len(rows)], dtype=np.int32))
            for _ in range(2):
                if graph is None:
                    pair.build(contacts)
                else:
                    wp.capture_launch(graph)
                test.assertEqual(int(pair.group_count.numpy()[0]), len(edges))
                test.assertEqual(int(pair.overflow_count.numpy()[0]), 0)
                colors = pair.group_color.numpy()
                test.assertTrue(np.all(colors[: len(edges)] >= 0))
                test.assertTrue(np.all(colors[len(edges) :] == -2))
                np.testing.assert_array_equal(
                    np.stack((pair.group_body0.numpy()[: len(edges)], pair.group_body1.numpy()[: len(edges)]), axis=1),
                    np.asarray(edges, dtype=np.int32).reshape(-1, 2),
                )
                np.testing.assert_array_equal(pair.group_start.numpy()[: len(edges)], np.arange(len(edges)) * 4)
                np.testing.assert_array_equal(pair.group_end.numpy()[: len(edges)], np.arange(1, len(edges) + 1) * 4)


def test_parallel_pair_completion_replays_changed_graphs(test, device):
    """Keep pair coloring complete through dense, empty, and overflow graph replays."""
    body_count = max(256, ((device.sm_count + 15) // 16) * 16)
    inverse = np.ones(body_count, dtype=np.float32)
    inverse[17::17] = 0.0
    inverse_array = wp.array(inverse, device=device)
    model = SimpleNamespace(
        body_count=body_count,
        device=device,
        shape_body=wp.array(np.arange(-1, body_count, dtype=np.int32), device=device),
    )
    pair = RigidPairContact(
        model,
        inverse_array,
        wp.zeros(body_count, dtype=wp.mat33, device=device),
        wp.ones(body_count, dtype=wp.int32, device=device),
        body_count,
    )
    dense = sorted(
        {
            (body, body + offset)
            for body in range(body_count)
            for offset in (1, 15, 16, 17)
            if body + offset < body_count
        }
    )
    star = [(0, body) for body in range(1, body_count)]
    capacity = len(dense) * 4
    contacts = SimpleNamespace(
        rigid_contact_max=capacity,
        rigid_contact_count=wp.zeros(1, dtype=wp.int32, device=device),
        rigid_contact_shape0=wp.full(capacity, -1, dtype=wp.int32, device=device),
        rigid_contact_shape1=wp.full(capacity, -1, dtype=wp.int32, device=device),
    )
    pair.resize(capacity)

    def assign(edges):
        rows = np.repeat(np.asarray(edges, dtype=np.int32).reshape(-1, 2), 4, axis=0)
        shape0 = np.full(capacity, -1, dtype=np.int32)
        shape1 = np.full(capacity, -1, dtype=np.int32)
        shape0[: len(rows)], shape1[: len(rows)] = rows[:, 0] + 1, rows[:, 1] + 1
        contacts.rigid_contact_shape0.assign(shape0)
        contacts.rigid_contact_shape1.assign(shape1)
        contacts.rigid_contact_count.assign(np.asarray([len(rows)], dtype=np.int32))

    def snapshot(edges):
        colors = pair.group_color.numpy()
        leaders = np.flatnonzero(colors >= -1)
        endpoints = np.stack(
            (pair.group_body0.numpy()[leaders], pair.group_body1.numpy()[leaders]),
            axis=1,
        )
        expected = np.asarray(
            [edge for edge in edges if any(inverse[body] > 0.0 for body in edge)],
            dtype=np.int32,
        ).reshape(-1, 2)
        np.testing.assert_array_equal(endpoints, expected)
        test.assertEqual(int(pair.group_count.numpy()[0]), len(expected))
        test.assertEqual(int(pair.overflow_count.numpy()[0]), int(np.count_nonzero(colors[leaders] < 0)))
        overflow = np.flatnonzero(colors == -1)
        np.testing.assert_array_equal(pair.group_list.numpy()[pair.capacity - len(overflow) :], overflow)
        for color in np.unique(colors[colors >= 0]):
            bodies = endpoints[colors[leaders] == color].ravel()
            dynamic = bodies[inverse[bodies] > 0.0]
            test.assertEqual(len(np.unique(dynamic)), len(dynamic))
        return colors

    for conditional in (False, True):
        if conditional and not wp.is_conditional_graph_supported():
            continue
        assign(dense)
        pair.build(contacts)
        with patch.object(wp, "is_conditional_graph_supported", return_value=conditional):
            with wp.ScopedCapture(device=device) as capture:
                pair.build(contacts)
        for edges in (dense, [], star, dense):
            assign(edges)
            pair.build(contacts)
            expected = snapshot(edges)
            for _ in range(3):
                wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(snapshot(edges), expected)
            if edges is star:
                test.assertGreater(int(pair.overflow_count.numpy()[0]), 0)


def test_rigid_determinism_survives_particle_module_reuse(test, device):
    """Keep rigid contact output repeatable when another solver changes shared options."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e5, kd=100.0, mu=0.5)
    for i in range(4):
        body = builder.add_body(xform=wp.transform(wp.vec3(0.9 * i, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.5, cfg=cfg)
    builder.color()
    model = builder.finalize(device=device)
    pipeline = _make_pipeline(model, 32)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=2,
        rigid_compliant_alm=True,
        rigid_contact_history=True,
        rigid_body_contact_buffer_size=128,
        deterministic=wp.DeterministicMode.RUN_TO_RUN,
    )
    state_in, state_out = model.state(), model.state()
    pipeline.collide(state_in, contacts)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
    solver.step(state_in, state_out, None, contacts, 1.0 / 600.0)
    arrays = []
    seen = set()
    for obj in (solver, solver._rigid_pair_contacts, state_in, state_out):
        for value in vars(obj).values():
            if isinstance(value, wp.array) and value.ptr and value.ptr not in seen:
                seen.add(value.ptr)
                arrays.append((value, wp.clone(value)))

    def restore():
        for target, seed in arrays:
            wp.copy(target, seed)

    solver.step(state_in, state_out, None, contacts, 1.0 / 600.0)
    expected = (state_out.body_q.numpy(), state_out.body_qd.numpy(), solver.body_body_contact_lambda.numpy())

    builder = newton.ModelBuilder()
    body = builder.add_body()
    builder.add_shape_sphere(body, radius=0.5)
    builder.add_particle(pos=wp.vec3(0.0, 0.0, 2.0), vel=wp.vec3(0.0), mass=1.0)
    builder.color()
    coupled_model = builder.finalize(device=device)
    coupled = newton.solvers.SolverVBD(
        coupled_model, iterations=1, rigid_compliant_alm=True, deterministic=wp.DeterministicMode.RUN_TO_RUN
    )
    coupled.step(coupled_model.state(), coupled_model.state(), None, None, 1.0 / 600.0)
    test.assertEqual(wp.get_module_options(module=rigid_vbd_kernels)["deterministic"], wp.DeterministicMode.RUN_TO_RUN)
    restore()
    solver.step(state_in, state_out, None, contacts, 1.0 / 600.0)
    # Pair reservations only permute independent work; all numerical gathers
    # preserve the requested lowering when switching between shared modules.
    test.assertEqual(wp.get_module_options(module=rigid_pair_contact)["deterministic"], wp.DeterministicMode.RUN_TO_RUN)
    test.assertEqual(wp.get_module_options(module=rigid_vbd_kernels)["deterministic"], wp.DeterministicMode.RUN_TO_RUN)
    test.assertEqual(wp.get_module_options(module=xpbd_kernels)["deterministic"], wp.DeterministicMode.RUN_TO_RUN)
    for actual, reference in zip(
        (state_out.body_q.numpy(), state_out.body_qd.numpy(), solver.body_body_contact_lambda.numpy()),
        expected,
        strict=True,
    ):
        np.testing.assert_array_equal(actual.view(np.uint8), reference.view(np.uint8))
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            restore()
            solver.step(state_in, state_out, None, contacts, 1.0 / 600.0)
        for _ in range(3):
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(state_out.body_q.numpy().view(np.uint8), expected[0].view(np.uint8))
            np.testing.assert_array_equal(state_out.body_qd.numpy().view(np.uint8), expected[1].view(np.uint8))


def test_history_disabled_free_contacts_use_body_coloring(test, device):
    """Apply body-color policies to cold-start compliant contacts, including capture."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.rigid_gap = 0.0
    cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e5, kd=100.0, mu=0.5)
    for z in range(4):
        for y in range(4):
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.96 * y, 0.96 * z), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5, cfg=cfg)
    builder.color()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, contact_matching="disabled", deterministic=True, rigid_contact_max=1024)
    contacts = pipeline.contacts()
    control = model.control()
    for policy in ("model", "auto", "refined"):
        with test.subTest(policy=policy):
            solver = newton.solvers.SolverVBD(
                model,
                iterations=2,
                rigid_compliant_alm=True,
                rigid_contact_history=False,
                rigid_contact_coloring=policy,
                deterministic=wp.DeterministicMode.RUN_TO_RUN,
            )
            test.assertIsNone(solver._rigid_pair_contacts)
            test.assertTrue(solver._rigid_body_refinement_supported)
            np.testing.assert_array_equal(solver.body_uses_pair_contacts.numpy(), np.zeros(model.body_count))
            state_in, state_out = model.state(), model.state()
            pipeline.collide(state_in, contacts)
            count, pairs = _dynamic_contact_pairs(model, contacts)
            test.assertGreater(count, 0)
            test.assertGreater(len(pairs), 0)
            solver.step(state_in, state_out, control, contacts, 1.0 / 600.0)
            arrays = []
            seen = set()
            for obj in (solver, state_in, state_out, contacts):
                for value in vars(obj).values():
                    if isinstance(value, wp.array) and value.ptr and value.ptr not in seen:
                        seen.add(value.ptr)
                        arrays.append((value, wp.clone(value)))

            def run(solver, state_in, state_out, arrays):
                for target, seed in arrays:
                    wp.copy(target, seed)
                solver.step(state_in, state_out, control, contacts, 1.0 / 600.0)

            run(solver, state_in, state_out, arrays)
            expected_q, expected_qd = state_out.body_q.numpy(), state_out.body_qd.numpy()
            refined = policy != "model"
            test.assertEqual(int(solver._rigid_body_refinement_active.numpy()[0]), int(refined))
            if refined:
                colors = solver._rigid_body_colors.numpy()
                test.assertTrue(np.all(colors >= 0))
                test.assertTrue(np.all(colors[pairs[:, 0]] != colors[pairs[:, 1]]))
                test.assertEqual(int(solver._rigid_body_refinement_overflow.numpy()[0]), 0)
            graph = None
            if wp.get_device(device).is_cuda:
                with wp.ScopedCapture(device=device) as capture:
                    run(solver, state_in, state_out, arrays)
                graph = capture.graph
            for _ in range(3):
                if graph is not None:
                    wp.capture_launch(graph)
                else:
                    run(solver, state_in, state_out, arrays)
                q, qd = state_out.body_q.numpy(), state_out.body_qd.numpy()
                test.assertTrue(np.isfinite(q).all() and np.isfinite(qd).all())
                np.testing.assert_array_equal(q.view(np.uint8), expected_q.view(np.uint8))
                np.testing.assert_array_equal(qd.view(np.uint8), expected_qd.view(np.uint8))


def test_contact_color_growth_replays_every_body(test, device):
    """Solve newly required high colors on the first replay after graph growth."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e5, kd=100.0, mu=0.5)
    for i in range(6):
        position = wp.vec3(0.1 * math.cos(i), 0.1 * math.sin(i), 0.01 * i)
        body = builder.add_link(xform=wp.transform(position, wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.5, cfg=cfg)
        builder.add_joint_fixed(parent=-1, child=body, parent_xform=wp.transform(position, wp.quat_identity()))
    builder.color()
    model = builder.finalize(device=device)
    pipeline = _make_pipeline(model, 128)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=2,
        rigid_compliant_alm=True,
        rigid_contact_history=False,
        rigid_contact_coloring="refined",
        deterministic=wp.DeterministicMode.RUN_TO_RUN,
    )
    state_in, state_out = model.state(), model.state()
    control = model.control()
    pipeline.collide(state_in, contacts)
    dense_count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertEqual(dense_count, 15)
    # Warm allocation with an empty graph, whose valid schedule needs one color.
    contacts.rigid_contact_count.zero_()
    solver.step(state_in, state_out, control, contacts, 1.0 / 600.0)
    test.assertEqual(int(solver._rigid_body_refinement_gt4.numpy()[0]), 0)
    arrays = []
    seen = set()
    for obj in (solver, state_in, state_out):
        for value in vars(obj).values():
            if isinstance(value, wp.array) and value.ptr and value.ptr not in seen:
                seen.add(value.ptr)
                arrays.append((value, wp.clone(value)))

    def restore():
        for target, seed in arrays:
            wp.copy(target, seed)

    for policy in ("auto", "refined"):
        with test.subTest(policy=policy):
            solver.rigid_contact_coloring = policy
            # Compile the populated path, then capture from the previous empty graph.
            contacts.rigid_contact_count.assign(np.asarray([dense_count], dtype=np.int32))
            solver.step(state_in, state_out, control, contacts, 1.0 / 600.0)
            restore()
            with wp.ScopedCapture(device=device) as capture:
                restore()
                solver.step(state_in, state_out, control, contacts, 1.0 / 600.0)
            for count in (dense_count, 0, dense_count):
                contacts.rigid_contact_count.assign(np.asarray([count], dtype=np.int32))
                restore()
                # Eager forced refinement executes the complete fixed envelope.
                solver.rigid_contact_coloring = "refined"
                solver.step(state_in, state_out, control, contacts, 1.0 / 600.0)
                expected = (state_out.body_q.numpy(), state_out.body_qd.numpy())
                solver.rigid_contact_coloring = policy
                for _ in range(3):
                    wp.capture_launch(capture.graph)
                    test.assertEqual(int(solver._rigid_body_num_colors.numpy()[0]), 6 if count else 1)
                    for actual, reference in zip(
                        (state_out.body_q.numpy(), state_out.body_qd.numpy()), expected, strict=True
                    ):
                        np.testing.assert_array_equal(actual.view(np.uint8), reference.view(np.uint8))


def test_mixed_world_force_query_keeps_each_contact_law(test, device):
    """Report body-ALM forces unchanged when an isolated world uses pair contacts."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e5, kd=100.0, mu=0.5)
    for world in range(2):
        builder.begin_world()
        bodies = []
        for x in (-2.0, 0.0, 0.9) if world == 0 else (0.0, 0.9):
            body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()))
            builder.add_shape_sphere(body, radius=0.5, cfg=cfg)
            bodies.append(body)
        if world == 0:
            builder.add_joint_fixed(
                parent=bodies[0],
                child=bodies[1],
                parent_xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()),
            )
        builder.end_world()
    builder.color()
    model = builder.finalize(device=device)
    pipeline = _make_pipeline(model, 32)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_contact_history=True,
        deterministic=wp.DeterministicMode.RUN_TO_RUN,
    )
    state_in, state_out = model.state(), model.state()
    previous = wp.clone(state_in.body_q)
    pipeline.collide(state_in, contacts)
    observables = solver.observables({newton.solvers.SolverObservableFlags.CONTACT_F})
    solver.step(state_in, state_out, None, contacts, 1.0 / 600.0, observables=observables)
    test.assertTrue(solver._pair_contacts_enabled_this_step)
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape_body = model.shape_body.numpy()
    body0 = shape_body[contacts.rigid_contact_shape0.numpy()[:count]]
    mask = body0 < 3
    test.assertTrue(mask.any() and (~mask).any())
    solver._pair_contacts_enabled_this_step = False
    expected = (
        solver.collect_rigid_contact_forces(state_out.body_q, previous, contacts, 1.0 / 600.0)[4].numpy()[:count].copy()
    )
    solver._pair_contacts_enabled_this_step = True

    def query():
        return solver.collect_rigid_contact_forces(state_out.body_q, previous, contacts, 1.0 / 600.0)[4]

    actual = query().numpy()[:count]
    np.testing.assert_array_equal(actual[mask], expected[mask])
    np.testing.assert_array_equal(actual[~mask], solver.body_body_contact_lambda.numpy()[:count][~mask])
    # New upstream observables report the force on body 0, the opposite of the
    # collector's force on body 1, while sharing the same contact evaluator.
    np.testing.assert_array_equal(observables.contact_f.numpy()[:count, :3], -actual)
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            query()
        for _ in range(3):
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(contacts.rigid_contact_force.numpy()[:count], actual)


def test_zero_compliant_contact_material_is_not_hard(test, device):
    """Keep zero stiffness and damping force-free in every matching mode."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = newton.ModelBuilder.ShapeConfig(ke=0.0, kd=0.0, mu=0.5)
    for x in (0.0, 0.9):
        body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.5, cfg=cfg)
    builder.color()
    model = builder.finalize(device=device)
    for matching in ("disabled", "latest", "sticky"):
        with test.subTest(matching=matching):
            pipeline = newton.CollisionPipeline(
                model, contact_matching=matching, deterministic=True, rigid_contact_max=32
            )
            contacts = pipeline.contacts()
            solver = newton.solvers.SolverVBD(
                model,
                iterations=2,
                rigid_compliant_alm=True,
                rigid_contact_history=matching != "disabled",
                deterministic=wp.DeterministicMode.RUN_TO_RUN,
            )
            state_in, state_out = model.state(), model.state()
            initial = state_in.body_q.numpy()
            pipeline.collide(state_in, contacts)
            test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
            solver.step(state_in, state_out, None, contacts, 1.0 / 600.0)
            np.testing.assert_allclose(state_out.body_q.numpy(), initial, rtol=0.0, atol=1.0e-7)
            # Body ALM can retain an auxiliary multiplier whose material-weighted
            # force is zero. Pair multipliers are the applied force itself.
            if solver._pair_contacts_enabled_this_step:
                np.testing.assert_array_equal(solver.body_body_contact_lambda.numpy(), 0.0)
            np.testing.assert_array_equal(state_out.body_qd.numpy(), 0.0)


def test_body_contact_color_budget_is_bounded_and_joint_safe(test, device):
    """A requested cap bounds body work without overriding mandatory joint colors."""
    model, pipeline, contacts, _ = _make_jointed_contact_lattice(device, ny=4, nz=2)
    for invalid in (0, -1, 33, 4.5, True):
        with test.assertRaisesRegex(ValueError, "rigid_body_contact_color_budget"):
            newton.solvers.SolverVBD(model, rigid_compliant_alm=True, rigid_body_contact_color_budget=invalid)
    for budget in (1, 4, 8, 32):
        with test.subTest(budget=budget):
            solver = newton.solvers.SolverVBD(
                model,
                iterations=2,
                rigid_compliant_alm=True,
                rigid_contact_history=True,
                rigid_contact_coloring="refined",
                rigid_body_contact_buffer_size=64,
                rigid_body_contact_color_budget=budget,
                deterministic=wp.DeterministicMode.RUN_TO_RUN,
            )
            state_in, state_out = model.state(), model.state()
            control = model.control()
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, 1.0 / 600.0)
            effective = max(budget, len(model.body_color_groups))
            test.assertEqual(solver._rigid_body_color_budget, effective)
            colors = solver._rigid_body_colors.numpy()
            active = solver._rigid_body_schedule_active.numpy().astype(bool)
            test.assertTrue(np.all((colors[active] >= 0) & (colors[active] < effective)))
            for parent, child in zip(model.joint_parent.numpy(), model.joint_child.numpy(), strict=True):
                if parent >= 0 and child >= 0 and active[parent] and active[child]:
                    test.assertNotEqual(colors[parent], colors[child])
            np.testing.assert_array_equal(
                np.sort(solver._rigid_body_ids_by_color.numpy()[: np.count_nonzero(active)]),
                np.flatnonzero(active),
            )

            copies = [
                (value, wp.clone(value))
                for owner in (state_in, state_out, solver, contacts)
                for value in vars(owner).values()
                if isinstance(value, wp.array)
            ]

            def restore(copies=copies):
                for value, seed in copies:
                    wp.copy(value, seed)

            restore()
            solver.step(state_in, state_out, control, contacts, 1.0 / 600.0)
            expected = (state_out.body_q.numpy().tobytes(), state_out.body_qd.numpy().tobytes())
            restore()
            if device.is_cuda:
                with wp.ScopedCapture(device=device) as capture:
                    solver.step(state_in, state_out, control, contacts, 1.0 / 600.0)
            for _ in range(3):
                restore()
                if device.is_cuda:
                    wp.capture_launch(capture.graph)
                else:
                    solver.step(state_in, state_out, control, contacts, 1.0 / 600.0)
                test.assertEqual((state_out.body_q.numpy().tobytes(), state_out.body_qd.numpy().tobytes()), expected)


def test_disjoint_joint_force_controls_preserve_wrenches(test, device):
    """Preserve nonzero controls and mutable enables without cross-body reduction."""
    for shared in (False, True):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for x in (0.0, 3.0):
            body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()))
            builder.add_shape_sphere(body, radius=0.5)
        if shared:
            builder.add_joint_fixed(parent=0, child=1)
        builder.color()
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverVBD(
            model, iterations=1, rigid_compliant_alm=True, deterministic=wp.DeterministicMode.RUN_TO_RUN
        )
        selected = getattr(solver, "_joint_force_kernel", xpbd_kernels.apply_joint_forces)
        if device.is_cuda and not shared:
            test.assertIsNot(selected, xpbd_kernels.apply_joint_forces)
        else:
            test.assertIs(selected, xpbd_kernels.apply_joint_forces)
        if shared:
            continue
        control = model.control()
        state = model.state()
        body_seed = np.asarray([[3, -2, 7, 4, -5, 6], [-1, 2, -3, 4, -5, 6]], dtype=np.float32)
        expected = wp.array(body_seed, dtype=wp.spatial_vector, device=device)
        actual = wp.clone(expected)
        seed = wp.clone(expected)
        inputs = [
            state.body_q,
            model.body_com,
            model.joint_type,
            model.joint_enabled,
            model.joint_parent,
            model.joint_child,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_qd_start,
            model.joint_dof_dim,
            model.joint_axis,
            control.joint_f,
            1.0 / 120.0,
        ]

        def launch(actual=actual, seed=seed, selected=selected, inputs=inputs, count=model.joint_count):
            wp.copy(actual, seed)
            wp.launch(selected, dim=count, inputs=inputs, outputs=[actual, None], device=device)

        launch()
        capture = None
        if device.is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                launch()
        for enabled in (True, False, True):
            control.joint_f.assign(np.linspace(-11, 17, model.joint_dof_count, dtype=np.float32))
            model.joint_enabled.assign(np.full(model.joint_count, enabled, dtype=bool))
            expected.assign(body_seed)
            solver._apply_module_options()
            wp.launch(
                xpbd_kernels.apply_joint_forces,
                dim=model.joint_count,
                inputs=inputs,
                outputs=[expected, None],
                device=device,
            )
            for _ in range(3):
                if capture is not None:
                    wp.capture_launch(capture.graph)
                else:
                    launch()
                np.testing.assert_array_equal(actual.numpy(), expected.numpy())
            if not enabled:
                np.testing.assert_array_equal(actual.numpy(), body_seed)
        test.assertEqual(wp.get_module_options(xpbd_kernels)["deterministic"], wp.DeterministicMode.RUN_TO_RUN)
        strict = newton.solvers.SolverVBD(
            model, iterations=1, rigid_compliant_alm=True, deterministic=wp.DeterministicMode.GPU_TO_GPU
        )
        test.assertIs(strict._joint_force_kernel, xpbd_kernels.apply_joint_forces)


def test_pair_parallel_tail_replays_population_changes(test, device):
    """Refresh the execution flag when dense late colors appear and disappear."""
    body_count = 1536
    model = SimpleNamespace(
        body_count=body_count, device=device, shape_body=wp.array(np.arange(body_count, dtype=np.int32), device=device)
    )
    pair = RigidPairContact(
        model,
        wp.ones(body_count, dtype=float, device=device),
        wp.zeros(body_count, dtype=wp.mat33, device=device),
        wp.ones(body_count, dtype=wp.int32, device=device),
        body_count,
    )
    edges = np.asarray(
        [(start + i, start + j) for start in range(0, body_count, 12) for i in range(12) for j in range(i + 1, 12)],
        dtype=np.int32,
    )
    contacts = SimpleNamespace(
        rigid_contact_max=len(edges),
        rigid_contact_count=wp.array([len(edges)], dtype=wp.int32, device=device),
        rigid_contact_shape0=wp.array(edges[:, 0], device=device),
        rigid_contact_shape1=wp.array(edges[:, 1], device=device),
    )
    pair.build(contacts)
    with wp.ScopedCapture(device=device) as capture:
        pair.build(contacts)
    for count in (len(edges), 0, 1, len(edges)):
        contacts.rigid_contact_count.assign(np.asarray([count], dtype=np.int32))
        wp.capture_launch(capture.graph)
        counts = pair.color_counts.numpy()
        expected = int(np.any(counts[8:16] > rigid_pair_contact._FUSED_THREADS))
        test.assertEqual(int(pair.parallel_tail.numpy()[0]), expected)
        test.assertEqual(expected, int(count == len(edges)))


def test_pair_tail_dispatch_is_once_per_substep(test, device):
    """A fixed pair graph must not pay a conditional node for every sweep."""
    if not wp.is_conditional_graph_supported():
        test.skipTest("Conditional CUDA graphs unavailable")
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for index in range(device.sm_count + 1):
        body = builder.add_body(xform=wp.transform(wp.vec3(2.0 * index, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.5)
    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=3,
        rigid_compliant_alm=True,
        rigid_contact_history=True,
        deterministic=wp.DeterministicMode.RUN_TO_RUN,
    )
    pipeline = _make_pipeline(model, 8)
    contacts = pipeline.contacts()
    states = [model.state(), model.state()]
    control = model.control()
    pipeline.collide(states[0], contacts)
    solver.step(states[0], states[1], control, contacts, 1.0 / 120.0)
    expected = states[1].body_q.numpy().copy()
    original = wp.capture_if
    calls = []

    def capture_if(condition, *args, **kwargs):
        if condition is solver._rigid_pair_contacts.parallel_tail:
            calls.append(condition)
        return original(condition, *args, **kwargs)

    with patch.object(wp, "capture_if", capture_if):
        with wp.ScopedCapture(device=device) as capture:
            solver.step(states[0], states[1], control, contacts, 1.0 / 120.0)
    test.assertEqual(len(calls), 1)
    for _ in range(3):
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(states[1].body_q.numpy(), expected)


def test_particle_attachments_follow_refined_contact_schedule(test, device):
    """Keep attached worlds per-body and honor mutable packed color slices."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.begin_world()
    for i in range(3):
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        particle = builder.add_particle(pos=wp.vec3(i + 1.0, 0.0, 0.0), vel=wp.vec3(), mass=1.0)
        builder.add_attachment_particle_body(particle, body, stiffness=10.0, enabled=i != 2)
    builder.end_world()
    builder.begin_world()
    builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
    builder.end_world()
    builder.color()
    model = builder.finalize(device=device)
    pipeline = _make_pipeline(model, 4)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(model, iterations=1, rigid_contact_history=True, rigid_contact_coloring="refined")
    # Even a disabled attachment may be enabled during graph replay.
    np.testing.assert_array_equal(solver.body_uses_pair_contacts.numpy(), [0, 0, 0, 1])
    ids = wp.array([2, 0, 1], dtype=int, device=device)
    starts = wp.array([0, 1, 3], dtype=int, device=device)
    outputs = [
        wp.zeros(model.body_count, dtype=dtype, device=device)
        for dtype in (wp.vec3, wp.vec3, wp.mat33, wp.mat33, wp.mat33)
    ]

    def gather():
        for output in outputs:
            output.zero_()
        wp.launch(
            rigid_vbd_kernels.accumulate_body_particle_attachments_per_body,
            dim=3,
            inputs=[
                1.0,
                model.body_color_groups[0],
                ids,
                starts,
                0,
                model.particle_q,
                model.particle_q,
                model.body_q,
                model.body_q,
                model.body_com,
                model.body_inv_mass,
                model.attachment_particle_body_particle,
                model.attachment_particle_body_body_point,
                model.attachment_particle_body_stiffness,
                model.attachment_particle_body_damping,
                model.attachment_particle_body_enabled,
                solver.body_particle_attachment_offsets,
                solver.body_particle_attachment_indices,
            ],
            outputs=outputs,
            device=device,
        )

    gather()
    graph = None
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            gather()
        graph = capture.graph
    for order, end, enabled in [([2, 0, 1], 1, False), ([2, 0, 1], 2, True), ([1, 0, 2], 1, True)]:
        ids.assign(np.asarray(order, dtype=np.int32))
        starts.assign(np.asarray([0, end, 3], dtype=np.int32))
        model.attachment_particle_body_enabled.assign(np.asarray([True, True, enabled]))
        for _ in range(2):
            gather() if graph is None else wp.capture_launch(graph)
            expected = np.zeros((4, 3), dtype=np.float32)
            for body in order[:end]:
                if body != 2 or enabled:
                    expected[body, 0] = 10.0 * (body + 1)
            np.testing.assert_array_equal(outputs[0].numpy(), expected)
    # Exercise the complete host launch path as well as the adversarial packed slices.
    state_in, state_out = model.state(), model.state()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, None, contacts, 0.01)
    test.assertTrue(np.isfinite(state_out.body_q.numpy()).all())


class TestVBDRigidPairContact(unittest.TestCase):
    def test_rigid_block_dim_preserves_launch_policy(self):
        """Choose the largest supported block meeting the half-SM target."""
        widths = (4, 8, 16, 32, 64, 128, 256)
        for sm_count in (1, 2, 3, 16, 71, 142):
            device = SimpleNamespace(is_cuda=True, sm_count=sm_count)
            target = (sm_count + 1) // 2
            for launch_dim in (0, 1, 4, 17, 63, 128, 257, 4096, 1000000):
                with self.subTest(sm_count=sm_count, launch_dim=launch_dim):
                    valid = [width for width in widths if (launch_dim + width - 1) // width >= target]
                    self.assertEqual(
                        rigid_vbd_kernels.select_rigid_block_dim(launch_dim, device),
                        max(valid, default=4),
                    )
        self.assertEqual(rigid_vbd_kernels.select_rigid_block_dim(0, SimpleNamespace(is_cuda=False)), 256)


devices = get_test_devices()
cuda_devices = get_selected_cuda_test_devices()

add_function_test(
    TestVBDRigidPairContact,
    "test_particle_attachments_follow_refined_contact_schedule",
    test_particle_attachments_follow_refined_contact_schedule,
    devices=devices,
)

add_function_test(
    TestVBDRigidPairContact,
    "test_pair_tail_dispatch_is_once_per_substep",
    test_pair_tail_dispatch_is_once_per_substep,
    devices=cuda_devices,
)

add_function_test(
    TestVBDRigidPairContact,
    "test_rigid_determinism_survives_particle_module_reuse",
    test_rigid_determinism_survives_particle_module_reuse,
    devices=devices,
)
add_function_test(
    TestVBDRigidPairContact,
    "test_history_disabled_free_contacts_use_body_coloring",
    test_history_disabled_free_contacts_use_body_coloring,
    devices=devices,
)
add_function_test(
    TestVBDRigidPairContact,
    "test_serial_pair_coloring_reuses_capacity_safely",
    test_serial_pair_coloring_reuses_capacity_safely,
    devices=devices,
)
add_function_test(
    TestVBDRigidPairContact,
    "test_parallel_pair_completion_replays_changed_graphs",
    test_parallel_pair_completion_replays_changed_graphs,
    devices=cuda_devices,
)
add_function_test(
    TestVBDRigidPairContact,
    "test_jointed_color_mask_matches_greedy_reference",
    test_jointed_color_mask_matches_greedy_reference,
    devices=devices,
)
add_function_test(
    TestVBDRigidPairContact,
    "test_body_contact_tile_matches_serial_accumulation",
    test_body_contact_tile_matches_serial_accumulation,
    devices=cuda_devices,
)
add_function_test(
    TestVBDRigidPairContact,
    "test_jointed_contact_graph_changes_during_capture",
    test_jointed_contact_graph_changes_during_capture,
    devices=cuda_devices,
)
add_function_test(
    TestVBDRigidPairContact,
    "test_jointed_contact_refinement_overflow_is_complete",
    test_jointed_contact_refinement_overflow_is_complete,
    devices=devices,
)
add_function_test(
    TestVBDRigidPairContact,
    "test_jointed_contact_auto_without_conditional_graphs",
    test_jointed_contact_auto_without_conditional_graphs,
    devices=cuda_devices,
)
add_function_test(
    TestVBDRigidPairContact,
    "test_pair_contacts_are_limited_to_free_body_systems",
    test_pair_contacts_are_limited_to_free_body_systems,
    devices=devices,
)
add_function_test(
    TestVBDRigidPairContact,
    "test_mixed_joint_and_free_bodies_share_per_body_world",
    test_mixed_joint_and_free_bodies_share_per_body_world,
    devices=devices,
)
add_function_test(
    TestVBDRigidPairContact,
    "test_world_partition_keeps_pair_contacts_in_joint_free_worlds",
    test_world_partition_keeps_pair_contacts_in_joint_free_worlds,
    devices=devices,
)
add_function_test(
    TestVBDRigidPairContact,
    "test_legacy_iteration_collision_falls_back_to_complete_body_schedule",
    test_legacy_iteration_collision_falls_back_to_complete_body_schedule,
    devices=devices,
)
add_function_test(
    TestVBDRigidPairContact,
    "test_per_body_contact_lists_are_sorted_without_pair_manager",
    test_per_body_contact_lists_are_sorted_without_pair_manager,
    devices=devices,
)
add_function_test(
    TestVBDRigidPairContact,
    "test_pair_warmstart_respects_normal_and_tangent_moment_arms",
    test_pair_warmstart_respects_normal_and_tangent_moment_arms,
    devices=devices,
)
add_function_test(
    TestVBDRigidPairContact,
    "test_pair_primal_is_seeded_only_for_retained_multipliers",
    test_pair_primal_is_seeded_only_for_retained_multipliers,
    devices=devices,
)
add_function_test(
    TestVBDRigidPairContact,
    "test_parallel_pair_coloring_is_conflict_free_and_repeatable",
    test_parallel_pair_coloring_is_conflict_free_and_repeatable,
    devices=cuda_devices,
)
add_function_test(
    TestVBDRigidPairContact,
    "test_pair_coloring_clamps_overreported_contact_count",
    test_pair_coloring_clamps_overreported_contact_count,
    devices=devices,
)
add_function_test(
    TestVBDRigidPairContact,
    "test_jointed_contact_refinement_counts_manifolds_not_points",
    test_jointed_contact_refinement_counts_manifolds_not_points,
    devices=devices,
)
add_function_test(
    TestVBDRigidPairContact,
    "test_jointed_contact_refinement_preserves_joint_colors",
    test_jointed_contact_refinement_preserves_joint_colors,
    devices=devices,
)
add_function_test(
    TestVBDRigidPairContact,
    "test_jointed_contact_refinement_is_cuda_graph_repeatable",
    test_jointed_contact_refinement_is_cuda_graph_repeatable,
    devices=cuda_devices,
)
add_function_test(
    TestVBDRigidPairContact,
    "test_pair_color_overflow_is_complete_and_repeatable",
    test_pair_color_overflow_is_complete_and_repeatable,
    devices=cuda_devices,
)


add_function_test(
    TestVBDRigidPairContact,
    "test_contact_color_growth_replays_every_body",
    test_contact_color_growth_replays_every_body,
    devices=cuda_devices,
)


add_function_test(
    TestVBDRigidPairContact,
    "test_mixed_world_force_query_keeps_each_contact_law",
    test_mixed_world_force_query_keeps_each_contact_law,
    devices=devices,
)
add_function_test(
    TestVBDRigidPairContact,
    "test_zero_compliant_contact_material_is_not_hard",
    test_zero_compliant_contact_material_is_not_hard,
    devices=devices,
)


add_function_test(
    TestVBDRigidPairContact,
    "test_body_contact_color_budget_is_bounded_and_joint_safe",
    test_body_contact_color_budget_is_bounded_and_joint_safe,
    devices=devices,
)


add_function_test(
    TestVBDRigidPairContact,
    "test_disjoint_joint_force_controls_preserve_wrenches",
    test_disjoint_joint_force_controls_preserve_wrenches,
    devices=devices,
)
add_function_test(
    TestVBDRigidPairContact,
    "test_pair_parallel_tail_replays_population_changes",
    test_pair_parallel_tail_replays_population_changes,
    devices=cuda_devices,
)


add_function_test(
    TestVBDRigidPairContact,
    "test_pair_preparation_refreshes_material_and_previous_pose",
    test_pair_preparation_refreshes_material_and_previous_pose,
    devices=devices,
)


def test_pair_sliding_incline_preserves_momentum(test, device):
    """Keep Coulomb acceleration independent of the world-space pose origin."""
    theta = math.radians(35.0)
    normal = np.array([math.sin(theta), 0.0, math.cos(theta)])
    downhill = np.array([math.cos(theta), 0.0, -math.sin(theta)])
    rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), theta)
    dt = 1.0 / 600.0
    steps = 300
    expected_speed = 9.81 * (math.sin(theta) - 0.5 * math.cos(theta)) * (steps * dt)
    for history in ("latest", "sticky"):
        for offset in (0.0, -25.0):
            with test.subTest(history=history, offset=offset):
                builder = newton.ModelBuilder()
                builder.rigid_gap = 0.0
                cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e6, kd=5000.0, mu=0.5, density=1000.0)
                builder.add_shape_plane(xform=wp.transform(wp.vec3(0.0), rotation), width=0.0, length=0.0, cfg=cfg)
                center = normal * 0.1 + downhill * offset
                body = builder.add_body(xform=wp.transform(wp.vec3(*center), rotation))
                builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
                builder.color()
                model = builder.finalize(device=device)
                pipeline = newton.CollisionPipeline(
                    model, broad_phase="sap", contact_matching=history, deterministic=True, rigid_contact_max=32
                )
                contacts = pipeline.contacts()
                solver = newton.solvers.SolverVBD(
                    model,
                    iterations=8,
                    rigid_compliant_alm=True,
                    rigid_contact_history=True,
                    deterministic=wp.DeterministicMode.RUN_TO_RUN,
                )
                states = [model.state(), model.state()]
                control = model.control()

                def advance(pipeline=pipeline, states=states, contacts=contacts, solver=solver, control=control):
                    for _ in range(2):
                        pipeline.collide(states[0], contacts)
                        solver.step(states[0], states[1], control, contacts, dt)
                        states.reverse()

                advance()
                if device.is_cuda:
                    with wp.ScopedCapture(device=device) as capture:
                        advance()
                    for _ in range(steps // 2 - 1):
                        wp.capture_launch(capture.graph)
                else:
                    for _ in range(steps // 2 - 1):
                        advance()
                velocity = states[0].body_qd.numpy()[0, :3]
                np.testing.assert_allclose(np.dot(velocity, downhill), expected_speed, rtol=0.005, atol=0.001)
                test.assertLess(abs(np.dot(velocity, normal)), 0.003)


add_function_test(
    TestVBDRigidPairContact,
    "test_pair_sliding_incline_preserves_momentum",
    test_pair_sliding_incline_preserves_momentum,
    devices=devices,
)


def test_body_contact_sort_replays_changed_counts(test, device):
    """Preserve bounded rows and padding across empty, short and long captured lists."""
    rng = np.random.default_rng(20260927)
    body_count = 13
    for capacity in (1, 7, 32, 96, 1024):
        counts = wp.zeros(body_count, dtype=wp.int32, device=device)
        indices = wp.zeros(body_count * capacity, dtype=wp.int32, device=device)

        def sort_rows(capacity=capacity, counts=counts, indices=indices):
            launch = wp.launch_tiled if device.is_cuda else wp.launch
            launch(
                kernel=(
                    rigid_vbd_kernels.sort_body_body_contact_lists_tiled
                    if device.is_cuda
                    else rigid_vbd_kernels.sort_body_body_contact_lists
                ),
                dim=body_count,
                block_dim=32 if device.is_cuda else 256,
                inputs=[capacity, counts, indices],
                device=device,
            )

        sort_rows()
        graph = None
        if device.is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                sort_rows()
            graph = capture.graph
        for replay in range(6):
            host_counts = rng.integers(0, capacity + 4, body_count, dtype=np.int32)
            host_counts[:7] = [0, 1, 31, 32, 33, capacity, capacity + 3]
            if replay == 2:
                host_counts.fill(0)
            original = rng.integers(0, 4096, (body_count, capacity), dtype=np.int32)
            expected = original.copy()
            for body, count in enumerate(host_counts):
                expected[body, : min(int(count), capacity)].sort()
            counts.assign(host_counts)
            indices.assign(original.ravel())
            if graph is None:
                sort_rows()
            else:
                wp.capture_launch(graph)
            np.testing.assert_array_equal(indices.numpy(), expected.ravel())
            np.testing.assert_array_equal(counts.numpy(), host_counts)


add_function_test(
    TestVBDRigidPairContact,
    "test_body_contact_sort_replays_changed_counts",
    test_body_contact_sort_replays_changed_counts,
    devices=devices,
)


def test_body_contact_scratch_overwrite_and_accumulate(test, device):
    """Initialize empty body rows and preserve prior coupling contributions on replay."""
    model, pipeline, contacts, solver = _make_jointed_contact_lattice(device, ny=4, nz=2)
    inverse_mass = solver.body_inv_mass_effective.numpy()
    inverse_mass[::7] = 0.0
    solver.body_inv_mass_effective.assign(inverse_mass)
    state_in, state_out = model.state(), model.state()
    pipeline.collide(state_in, contacts)
    launch_tiled = wp.launch_tiled
    calls = []

    def record(*args, **kwargs):
        if kwargs.get("kernel") is accumulate_body_body_contacts_per_body and not calls:
            calls.append(kwargs.copy())
        return launch_tiled(*args, **kwargs)

    with patch.object(wp, "launch_tiled", side_effect=record):
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 600.0)
    test.assertTrue(calls)
    call = calls[0]
    inputs = list(call["inputs"])
    active_color = inputs[4]
    if active_color >= 0:
        starts = inputs[3].numpy()
        selected = inputs[2].numpy()[starts[active_color] : starts[active_color + 1]]
    else:
        selected = inputs[1].numpy()
    selected = selected[inverse_mass[selected] > 0.0]
    counts = solver.body_body_contact_counts
    original_counts = counts.numpy().copy()
    scratch = [wp.zeros_like(value) for value in call["outputs"]]
    reference = [wp.zeros_like(value) for value in scratch]
    reference_call = dict(call, inputs=[*inputs[:-1], True], outputs=reference)
    try:
        for accumulate in (False, True):
            candidate_call = dict(call, inputs=[*inputs[:-1], accumulate], outputs=scratch)
            launch_tiled(**candidate_call)
            graph = None
            if device.is_cuda:
                with wp.ScopedCapture(device=device) as capture:
                    launch_tiled(**candidate_call)
                graph = capture.graph
            for empty in (False, True, False):
                counts.assign(np.zeros_like(original_counts) if empty else original_counts)
                for value in reference:
                    value.zero_()
                launch_tiled(**reference_call)
                if not empty:
                    test.assertTrue(any(np.any(value.numpy()[selected]) for value in reference))
                for value in scratch:
                    value.fill_(7.0)
                if graph is None:
                    launch_tiled(**candidate_call)
                else:
                    wp.capture_launch(graph)
                for value, contribution in zip(scratch, reference, strict=True):
                    expected = np.full_like(value.numpy(), 7.0)
                    expected[selected] = contribution.numpy()[selected] + (7.0 if accumulate else 0.0)
                    np.testing.assert_array_equal(value.numpy(), expected)
    finally:
        counts.assign(original_counts)


add_function_test(
    TestVBDRigidPairContact,
    "test_body_contact_scratch_overwrite_and_accumulate",
    test_body_contact_scratch_overwrite_and_accumulate,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
