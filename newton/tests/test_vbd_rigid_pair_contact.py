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
    color_rigid_bodies_from_contacts_parallel,
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


def test_pair_contacts_are_limited_to_free_body_systems(test, device):
    """Keep pair contacts exclusive to joint-free collision worlds."""

    def make_solver(jointed, compliant=False, history=False):
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
        )

    test.assertIsNotNone(make_solver(jointed=False)._rigid_pair_contacts)
    test.assertIsNone(make_solver(jointed=True)._rigid_pair_contacts)
    test.assertFalse(make_solver(jointed=False, compliant=True)._pair_contacts_enabled_this_step)
    test.assertTrue(make_solver(jointed=False, compliant=True, history=True)._pair_contacts_enabled_this_step)


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
    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 120.0)
    test.assertEqual(include_pair_primal, [False, False, False])
    test.assertEqual(pair_build_count, 1)

    include_pair_primal.clear()
    solver.set_rigid_history_update(False)
    solver.step(state_out, state_in, model.control(), contacts, 1.0 / 120.0)
    test.assertEqual(include_pair_primal, [True, False, False])
    test.assertEqual(pair_build_count, 1)

    include_pair_primal.clear()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 120.0)
    test.assertEqual(include_pair_primal, [False, False, False])
    test.assertEqual(pair_build_count, 2)


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

    base_colors = model.body_colors.numpy()
    colors = solver._rigid_body_colors.numpy()
    subcolors = solver._rigid_body_contact_subcolor_count
    active = solver._rigid_body_schedule_active.numpy().astype(bool)
    np.testing.assert_array_equal(colors[active] // subcolors, base_colors[active])

    _, pairs = _dynamic_contact_pairs(model, contacts)
    same_base = base_colors[pairs[:, 0]] == base_colors[pairs[:, 1]]
    test.assertTrue(np.all(colors[pairs[same_base, 0]] != colors[pairs[same_base, 1]]))


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
    """Keep every jointed body and deterministic output when subcolors run out."""

    def run():
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, ke=1.0e5, mu=0.0)
        for i in range(10):
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
            rigid_contact_history=False,
            rigid_compliant_alm=True,
            rigid_contact_coloring="refined",
            rigid_body_contact_buffer_size=64,
        )
        state_in, state_out = model.state(), model.state()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 600.0)
        test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 45)
        test.assertGreater(int(solver._rigid_body_refinement_overflow.numpy()[0]), 0)
        ids = solver._rigid_body_ids_by_color.numpy()
        ends = solver._rigid_body_color_starts.numpy()
        active_count = int(ends[solver._rigid_body_color_budget])
        np.testing.assert_array_equal(np.sort(ids[:active_count]), np.arange(10))
        colors = solver._rigid_body_colors.numpy()
        np.testing.assert_array_equal(colors // solver._rigid_body_contact_subcolor_count, model.body_colors.numpy())
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

        # These five arrays are additive outputs only. Zero scratch buffers isolate
        # contact contributions from the joint forces already assembled by the solver.
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
    """Match serial greedy decisions for cold, warm, inactive, and overflow graphs."""

    def reference(seed, active, authored, neighbors):
        body_count = len(seed)
        budget, subcolors = 16, 8
        retained = np.full(body_count, -1, dtype=np.int32)
        for body in range(body_count):
            color = seed[body]
            if active[body] and 0 <= color < budget and color // subcolors == authored[body]:
                if not any(other < body and active[other] and seed[other] == color for other in neighbors[body]):
                    retained[body] = color
        overflow_count = 0
        for body in range(body_count):
            if not active[body] or retained[body] >= 0:
                continue
            first = int(authored[body]) * subcolors
            used = {int(retained[other]) for other in neighbors[body] if active[other]}
            available = [color for color in range(first, first + subcolors) if color not in used]
            retained[body] = available[0] if available else first
            overflow_count += not available
        return retained, overflow_count

    def run(colors, seed_array, inputs, outputs):
        wp.copy(colors, seed_array)
        wp.launch(
            color_rigid_bodies_from_contacts_parallel,
            dim=512,
            block_dim=512,
            inputs=inputs,
            outputs=outputs,
            device=device,
        )

    for body_count, clique in ((20, True), (64, False)):
        pairs = [
            (a, b)
            for a in range(body_count)
            for b in range(a + 1, body_count)
            if clique or (abs(a // 8 - b // 8) <= 1 and abs(a % 8 - b % 8) <= 1)
        ]
        pairs.extend((-1, body) for body in range(0, body_count, 5))
        contacts = np.repeat(np.asarray(sorted(pairs), dtype=np.int32), 3, axis=0)
        neighbors = [set() for _ in range(body_count)]
        rows = [[] for _ in range(body_count)]
        for contact, (a, b) in enumerate(contacts):
            for body, other in ((a, b), (b, a)):
                if body >= 0:
                    rows[body].append(contact)
                    if other >= 0:
                        neighbors[body].add(int(other))
        capacity = max(map(len, rows))
        indices = np.zeros((body_count, capacity), dtype=np.int32)
        for body, row in enumerate(rows):
            indices[body, : len(row)] = row
        inverse_mass = np.ones(body_count, dtype=np.float32)
        inverse_mass[::7] = 0.0
        active = (inverse_mass > 0.0).astype(np.int32)
        authored = np.arange(body_count, dtype=np.int32) % 2
        budget, subcolors = 16, 8
        colors = wp.full(body_count, -1, dtype=wp.int32, device=device)
        body_ids = wp.full(body_count, -1, dtype=wp.int32, device=device)
        starts = wp.zeros(body_count + 1, dtype=wp.int32, device=device)
        count = wp.zeros(1, dtype=wp.int32, device=device)
        overflow = wp.zeros(1, dtype=wp.int32, device=device)
        high = wp.zeros(1, dtype=wp.int32, device=device)
        inputs = [
            body_count,
            budget,
            2,
            wp.array(inverse_mass, device=device),
            wp.zeros(body_count, dtype=wp.int32, device=device),
            wp.array(active, device=device),
            capacity,
            wp.array(np.asarray(list(map(len, rows))), dtype=wp.int32, device=device),
            wp.array(indices.ravel(), device=device),
            wp.array(contacts[:, 0] + 1, device=device),
            wp.array(contacts[:, 1] + 1, device=device),
            wp.array(np.arange(-1, body_count, dtype=np.int32), device=device),
            wp.array(authored, device=device),
        ]

        for seed in (
            np.full(body_count, -1, dtype=np.int32),
            authored * subcolors,
            authored * subcolors + np.arange(body_count, dtype=np.int32) % subcolors,
        ):
            expected, expected_overflow = reference(seed, active, authored, neighbors)
            seed_array = wp.array(seed, device=device)

            run(colors, seed_array, inputs, [colors, body_ids, starts, count, overflow, high])
            with wp.ScopedCapture(device=device) as capture:
                run(colors, seed_array, inputs, [colors, body_ids, starts, count, overflow, high])
            for _ in range(3):
                wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(colors.numpy(), expected)
                test.assertEqual(int(overflow.numpy()[0]), expected_overflow)
                test.assertEqual(int(count.numpy()[0]), int(expected.max()) + 1)
                expected_ids = np.asarray(
                    [body for color in range(budget) for body in range(body_count) if expected[body] == color],
                    dtype=np.int32,
                )
                np.testing.assert_array_equal(body_ids.numpy()[: len(expected_ids)], expected_ids)
                expected_starts = np.cumsum([0] + [int(np.count_nonzero(expected == color)) for color in range(budget)])
                np.testing.assert_array_equal(starts.numpy()[: budget + 1], expected_starts)


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
    # Pair reservations only permute independent work; preserve the requested
    # lowering on the shared rigid and joint modules when switching solvers.
    test.assertEqual(
        wp.get_module_options(module=rigid_pair_contact)["deterministic"], wp.DeterministicMode.NOT_GUARANTEED
    )
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


class TestVBDRigidPairContact(unittest.TestCase):
    pass


devices = get_test_devices()
cuda_devices = get_selected_cuda_test_devices()

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
    "test_parallel_pair_completion_replays_changed_graphs",
    test_parallel_pair_completion_replays_changed_graphs,
    devices=cuda_devices,
)
add_function_test(
    TestVBDRigidPairContact,
    "test_jointed_color_mask_matches_greedy_reference",
    test_jointed_color_mask_matches_greedy_reference,
    devices=cuda_devices,
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
    "test_per_body_contact_lists_are_sorted_without_pair_manager",
    test_per_body_contact_lists_are_sorted_without_pair_manager,
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
