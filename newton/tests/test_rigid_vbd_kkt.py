# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the private VBD structural compliance-KKT backend."""

import unittest
import warnings
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton import NewtonWarning
from newton._src.geometry.inertia import _inverse_inertia, compute_inertia_capsule
from newton._src.solvers.vbd import rigid_vbd_kkt, rigid_vbd_kkt_chains
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    compute_kappa_and_jacobian,
    evaluate_angular_constraint_force_hessian,
    evaluate_rigid_contact_from_collision,
    ldlt6_solve,
    update_body_velocity,
)
from newton._src.solvers.vbd.rigid_vbd_kkt import (
    _CR_PERSISTENT_MAX_ROWS,
    _FUSED_TREE_MAX_LEVEL_WIDTH,
    _FUSED_TREE_MIN_LEVELS,
    _PAIRED_COARSE_MIN_CLOSURES,
    _corrected_pose,
    _fused_tree_levels_supported,
    _inverse_spatial_robust,
    add_joint_stress_majorizer,
    apply_global_correction,
    assemble_closure_schur,
    back_substitute_tree_backbone_cr_in_place,
    build_body_surrogate,
    classify_global_contact_islands,
    limit_dynamic_contact_jacobi_step,
    linearize_joint_path_rows,
    suppress_nonfinite_correction,
)
from newton.tests.unittest_utils import (
    add_function_test,
    get_cuda_test_devices,
    get_test_devices,
)


@wp.kernel
def _probe_local_factor_recovery(
    angular_hessian: wp.array[wp.mat33],
    default_step: wp.array[wp.spatial_vector],
    disabled_step: wp.array[wp.spatial_vector],
    enabled_step: wp.array[wp.spatial_vector],
):
    index = wp.tid()
    linear = wp.identity(3, float)
    cross = wp.mat33(0.0)
    force = wp.vec3(1.0, 0.0, 0.0)
    x, w = ldlt6_solve(linear, angular_hessian[index], cross, force, force)
    default_step[index] = wp.spatial_vector(x, w)
    x, w = ldlt6_solve(linear, angular_hessian[index], cross, force, force, False)
    disabled_step[index] = wp.spatial_vector(x, w)
    x, w = ldlt6_solve(linear, angular_hessian[index], cross, force, force, True)
    enabled_step[index] = wp.spatial_vector(x, w)


def _local_factor_recovery_is_opt_in(test, device):
    """Keep the main factorization unless global numerical recovery is enabled."""
    with wp.ScopedDevice(device):
        angular = wp.array(
            [np.eye(3), np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]])],
            dtype=wp.mat33,
        )
        default, disabled, enabled = (wp.empty(2, dtype=wp.spatial_vector) for _ in range(3))
        wp.launch(_probe_local_factor_recovery, 2, inputs=[angular], outputs=[default, disabled, enabled])
        actual_default, actual_disabled, actual_enabled = default.numpy(), disabled.numpy(), enabled.numpy()
    np.testing.assert_array_equal(actual_default, actual_disabled)
    np.testing.assert_array_equal(actual_default[0], actual_enabled[0])
    # A finite positive diagonal with an indefinite angular block models lost
    # weak curvature. Only the opt-in path substitutes the diagonal majorizer.
    np.testing.assert_allclose(actual_default[1, 3:], [-1.0 / 3.0, 2.0 / 3.0, 0.0], atol=1.0e-7)
    np.testing.assert_allclose(actual_enabled[1, 3:], [1.0 / 3.0, 0.0, 0.0], atol=1.0e-7)
    test.assertTrue(np.isfinite(actual_enabled).all())


@wp.kernel
def _invert_spatial_matrix(
    matrix: wp.array[wp.spatial_matrix],
    inverse: wp.array[wp.spatial_matrix],
):
    inverse[0] = _inverse_spatial_robust(matrix[0])


@wp.kernel
def _perturb_body_pose(
    body: int,
    delta: wp.spatial_vector,
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
):
    if wp.tid() == 0:
        body_q[body] = _corrected_pose(body_q[body], delta, body_com[body], 1.0)


@wp.kernel
def _apply_body_force(body: int, force: wp.vec3, body_f: wp.array[wp.spatial_vector]):
    body_f[body] = wp.spatial_vector(force[0], force[1], force[2], 0.0, 0.0, 0.0)


@wp.kernel
def _evaluate_local_angular_damping(
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    dt: float,
    result: wp.array[wp.vec3],
):
    kappa, jacobian = compute_kappa_and_jacobian(
        wp.transform_get_rotation(body_q[0]),
        wp.transform_get_rotation(body_q[1]),
        wp.quat_identity(),
        wp.quat_identity(),
    )
    torque, _hessian = evaluate_angular_constraint_force_hessian(
        wp.transform_get_rotation(body_q[0]),
        wp.transform_get_rotation(body_q[1]),
        kappa,
        jacobian,
        wp.transform_get_rotation(body_q_prev[0]),
        wp.transform_get_rotation(body_q_prev[1]),
        False,
        1.0,
        0.0,
        wp.identity(3, float),
        wp.vec3(0.0),
        wp.vec3(0.0),
        0.0,
        2.0,
        1,
        dt,
    )
    result[0] = torque


def _pin_body(builder: newton.ModelBuilder, body: int) -> None:
    builder.body_mass[body] = 0.0
    builder.body_inv_mass[body] = 0.0
    builder.body_inertia[body] = wp.mat33(0.0)
    builder.body_inv_inertia[body] = wp.mat33(0.0)


def _use_capsule_rod_mass(builder: newton.ModelBuilder, bodies, density: float) -> None:
    """Restore pre-#5328 capsule rod mass and inertia for legacy pinned fixtures."""
    for body in bodies:
        shape = builder.body_shapes[int(body)][0]
        radius, half_height = float(builder.shape_scale[shape][0]), float(builder.shape_scale[shape][1])
        mass, _, inertia = compute_inertia_capsule(density, radius, half_height)
        builder.body_mass[body] = mass
        builder.body_inv_mass[body] = 1.0 / mass
        builder.body_inertia[body] = inertia
        builder.body_inv_inertia[body] = _inverse_inertia(inertia)


def _build_chain(
    device,
    *,
    segments=16,
    stiffness=1.0e7,
    dahl=False,
    pinned=True,
    with_particle=False,
    capsule_mass=False,
):
    builder = newton.ModelBuilder()
    if dahl:
        newton.solvers.SolverVBD.register_custom_attributes(builder)
    rod = newton.Rod.create_straight(
        start=wp.vec3(-0.5 * segments * 0.03, 0.0, 1.0),
        direction=wp.vec3(1.0, 0.0, 0.0),
        length=segments * 0.03,
        segment_count=segments,
        radius=0.01,
    )
    bodies, joints = builder.add_rod(
        rod=rod,
        cfg=builder.default_shape_cfg.copy(),
        stretch_stiffness=stiffness,
        stretch_damping=0.0,
        bend_stiffness=1.0e4,
        bend_damping=1.0e3,
        wrap_in_articulation=True,
        body_frame_origin="com",
    )
    if capsule_mass:
        _use_capsule_rod_mass(builder, bodies, builder.default_shape_cfg.density)
    if pinned:
        _pin_body(builder, int(bodies[0]))
    if with_particle:
        builder.add_particle(pos=(0.0, 0.0, 2.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.01)
    builder.color(balance_colors=False)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))
    if dahl:
        model.vbd.dahl_eps_max.fill_(2.0)
        model.vbd.dahl_tau.fill_(0.1)
    return model, np.asarray(bodies, dtype=np.int32), np.asarray(joints, dtype=np.int32)


def _build_chains(device, segment_counts, *, stiffness=1.0e7):
    """Build pinned chains of different lengths in one model."""
    builder = newton.ModelBuilder()
    for index, segments in enumerate(segment_counts):
        rod = newton.Rod.create_straight(
            start=wp.vec3(0.0, 0.1 * index, 1.0),
            direction=wp.vec3(1.0, 0.0, 0.0),
            length=segments * 0.03,
            segment_count=segments,
            radius=0.01,
        )
        bodies, _ = builder.add_rod(
            rod=rod,
            cfg=builder.default_shape_cfg.copy(),
            stretch_stiffness=stiffness,
            stretch_damping=0.0,
            bend_stiffness=1.0e4,
            bend_damping=1.0e3,
            wrap_in_articulation=True,
            body_frame_origin="com",
        )
        _pin_body(builder, int(bodies[0]))
    builder.color(balance_colors=False)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))
    return model


def _build_y_tree(device, *, segments_per_branch=6, stiffness=1.0e9, interior_pins=()):
    builder = newton.ModelBuilder()
    positions = [wp.vec3(0.0, 0.0, 1.0)]
    edges = []
    directions = (
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(-0.5, 0.8660254, 0.0),
        wp.vec3(-0.5, -0.8660254, 0.0),
    )
    for direction in directions:
        previous = 0
        for segment in range(1, segments_per_branch + 1):
            positions.append(positions[0] + direction * (0.03 * segment))
            current = len(positions) - 1
            edges.append((previous, current))
            previous = current
    bodies, joints = builder.add_rod(
        rod=newton.Rod(positions, edges=edges, radius=0.01),
        cfg=builder.default_shape_cfg.copy(),
        stretch_stiffness=stiffness,
        bend_stiffness=1.0e4,
        bend_damping=1.0e3,
        wrap_in_articulation=True,
        body_frame_origin="com",
    )
    _pin_body(builder, int(bodies[segments_per_branch - 1]))
    for index in interior_pins:
        _pin_body(builder, int(bodies[index]))
    builder.color(balance_colors=False)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))
    return model, np.asarray(bodies, dtype=np.int32), np.asarray(joints, dtype=np.int32)


def _build_y_trees(device, branch_segments, *, stiffness=1.0e7):
    """Build pinned Y trees whose three branches have the given segment counts."""
    builder = newton.ModelBuilder()
    directions = (
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(-0.5, 0.8660254, 0.0),
        wp.vec3(-0.5, -0.8660254, 0.0),
    )
    for index, segments in enumerate(branch_segments):
        positions = [wp.vec3(0.0, 1.0 * index, 1.0)]
        edges = []
        for direction, count in zip(directions, segments, strict=True):
            previous = 0
            for segment in range(1, count + 1):
                positions.append(positions[0] + direction * (0.03 * segment))
                edges.append((previous, len(positions) - 1))
                previous = len(positions) - 1
        bodies, _ = builder.add_rod(
            rod=newton.Rod(positions, edges=edges, radius=0.01),
            cfg=builder.default_shape_cfg.copy(),
            stretch_stiffness=stiffness,
            bend_stiffness=1.0e4,
            bend_damping=1.0e3,
            wrap_in_articulation=True,
            body_frame_origin="com",
        )
        _pin_body(builder, int(bodies[segments[0] - 1]))
    builder.color(balance_colors=False)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))
    return model


def _build_spurred_tree(device, *, arm_segments=8, stiffness=1.0e9):
    """Build a long open tree that falsifies unsafe unpaired backbone pivots."""
    builder = newton.ModelBuilder()
    positions = [wp.vec3(0.0, 0.0, 1.0)]
    edges = []
    for direction, segment_count in (
        (wp.vec3(1.0, 0.0, 0.0), arm_segments),
        (wp.vec3(-1.0, 0.0, 0.0), arm_segments),
        (wp.vec3(0.0, 1.0, 0.0), 1),
    ):
        previous = 0
        for segment in range(1, segment_count + 1):
            positions.append(positions[0] + direction * (0.03 * segment))
            current = len(positions) - 1
            edges.append((previous, current))
            previous = current
    bodies, joints = builder.add_rod(
        rod=newton.Rod(positions, edges=edges, radius=0.01),
        cfg=builder.default_shape_cfg.copy(),
        stretch_stiffness=stiffness,
        bend_stiffness=1.0e4,
        bend_damping=1.0e3,
        wrap_in_articulation=True,
        body_frame_origin="com",
    )
    _pin_body(builder, int(bodies[arm_segments - 1]))
    builder.color(balance_colors=False)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))
    return model, np.asarray(bodies, dtype=np.int32), np.asarray(joints, dtype=np.int32)


def _build_loop_with_branch(
    device,
    *,
    ring_segments=8,
    stiffness=1.0e9,
    bend_stiffness=1.0e4,
    add_chord=False,
    grounded=False,
    capsule_mass=False,
):
    """Build a one-loop cable island with a branched, world-attached tail."""
    builder = newton.ModelBuilder()
    shape = builder.default_shape_cfg.copy()
    if grounded:
        shape.mu = 0.45
        shape.ke = 1.0e5
        shape.kd = 100.0
    radius = 0.12
    cable_radius = 0.01
    height = cable_radius + 0.001 if grounded else 1.0
    positions = [
        wp.vec3(
            radius * np.cos(2.0 * np.pi * index / ring_segments),
            radius * np.sin(2.0 * np.pi * index / ring_segments),
            height,
        )
        for index in range(ring_segments)
    ]
    positions.extend((wp.vec3(radius + 0.06, 0.0, height), wp.vec3(radius + 0.12, 0.0, height)))
    edges = [(index, (index + 1) % ring_segments) for index in range(ring_segments)]
    edges.extend(((0, ring_segments), (ring_segments, ring_segments + 1)))
    # This fixture deliberately restores the omitted closure below.
    with unittest.TestCase().assertWarnsRegex(NewtonWarning, "^Rod graph contains a cycle"):
        bodies, joints = builder.add_rod(
            rod=newton.Rod(positions, edges=edges, radius=cable_radius),
            cfg=shape,
            stretch_stiffness=stiffness,
            bend_stiffness=bend_stiffness,
            bend_damping=1.0e3,
            wrap_in_articulation=True,
            body_frame_origin="com",
        )
    if capsule_mass:
        _use_capsule_rod_mass(builder, bodies, shape.density)
    jointed_pairs = {frozenset((int(builder.joint_parent[joint]), int(builder.joint_child[joint]))) for joint in joints}
    for node in range(ring_segments):
        previous_edge = (node - 1) % ring_segments
        next_edge = node
        if frozenset((int(bodies[previous_edge]), int(bodies[next_edge]))) in jointed_pairs:
            continue
        previous_length = float(wp.length(positions[node] - positions[previous_edge]))
        next_length = float(wp.length(positions[(node + 1) % ring_segments] - positions[node]))
        closure = builder.add_joint_rod(
            parent=int(bodies[previous_edge]),
            child=int(bodies[next_edge]),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * previous_length), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.5 * next_length), wp.quat_identity()),
            stretch_stiffness=stiffness,
            bend_stiffness=bend_stiffness,
            bend_damping=1.0e3,
        )
        joints.append(closure)
        break
    else:
        raise RuntimeError("Expected one omitted ring closure in the spanning articulation")
    if add_chord:
        parent_body = int(bodies[0])
        child_body = int(bodies[ring_segments // 2])
        anchor = 0.5 * (
            wp.transform_get_translation(builder.body_q[parent_body])
            + wp.transform_get_translation(builder.body_q[child_body])
        )
        parent_anchor = wp.transform_point(wp.transform_inverse(builder.body_q[parent_body]), anchor)
        child_anchor = wp.transform_point(wp.transform_inverse(builder.body_q[child_body]), anchor)
        # Keep the two material frames aligned at rest. With the split
        # bend/twist cable model, leaving both local rotations at identity
        # would make this diametric chord nearly antiparallel and place the
        # DER curvature-binormal measure at its 180-degree singularity.
        parent_rotation = wp.transform_get_rotation(builder.body_q[parent_body])
        child_rotation = wp.transform_get_rotation(builder.body_q[child_body])
        child_anchor_rotation = wp.quat_inverse(child_rotation) * parent_rotation
        chord = builder.add_joint_rod(
            parent=parent_body,
            child=child_body,
            parent_xform=wp.transform(parent_anchor, wp.quat_identity()),
            child_xform=wp.transform(child_anchor, child_anchor_rotation),
            stretch_stiffness=stiffness,
            bend_stiffness=bend_stiffness,
            bend_damping=1.0e3,
        )
        joints.append(chord)
    branch_length = float(wp.length(positions[-1] - positions[-2]))
    attachment = builder.add_joint_rod(
        parent=-1,
        child=int(bodies[-1]),
        parent_xform=wp.transform(positions[-1], wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * branch_length), wp.quat_identity()),
        stretch_stiffness=stiffness,
        bend_stiffness=bend_stiffness,
        bend_damping=1.0e3,
    )
    joints.append(attachment)
    if grounded:
        builder.add_ground_plane(cfg=shape)
    builder.color(balance_colors=False)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))
    return model, np.asarray(bodies, dtype=np.int32), np.asarray(joints, dtype=np.int32)


def _build_fixed_chain(device, *, link_count=24):
    builder = newton.ModelBuilder()
    bodies = []
    joints = []
    spacing = 0.08
    for link in range(link_count):
        body = builder.add_link(
            xform=wp.transform(wp.vec3(spacing * link, 0.0, 1.0), wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_box(body, hx=0.04, hy=0.01, hz=0.01)
        bodies.append(body)
    for link, body in enumerate(bodies):
        if link == 0:
            joint = builder.add_joint_fixed(
                parent=-1,
                child=body,
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
            )
        else:
            joint = builder.add_joint_fixed(
                parent=bodies[link - 1],
                child=body,
                parent_xform=wp.transform(wp.vec3(spacing, 0.0, 0.0), wp.quat_identity()),
            )
        joints.append(joint)
    builder.add_articulation(joints)
    builder.color(balance_colors=False)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))
    return model, np.asarray(bodies, dtype=np.int32), np.asarray(joints, dtype=np.int32)


def _build_fixed_topology_template(*, link_count, closure_count):
    """Build one fixed-joint topology for replicated and mixed-island tests."""
    template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    bodies = []
    joints = []
    spacing = 0.08
    for link in range(link_count):
        position = wp.vec3(spacing * link, 0.0, 1.0)
        body = template.add_link(xform=wp.transform(position, wp.quat_identity()), mass=1.0)
        template.add_shape_box(body, hx=0.03, hy=0.01, hz=0.01)
        bodies.append(body)
        if link == 0:
            joints.append(
                template.add_joint_fixed(
                    parent=-1,
                    child=body,
                    parent_xform=wp.transform(position, wp.quat_identity()),
                )
            )
        else:
            joints.append(
                template.add_joint_fixed(
                    parent=bodies[-2],
                    child=body,
                    parent_xform=wp.transform((spacing, 0.0, 0.0), wp.quat_identity()),
                )
            )
    template.add_articulation(joints)
    if closure_count:
        with warnings.catch_warnings():
            if link_count == 2:
                # Two links closed into a cycle carry two joints between the same pair on purpose.
                warnings.filterwarnings("ignore", message="Adding a FIXED joint", category=NewtonWarning)
            template.add_joint_fixed(
                parent=bodies[-1],
                child=bodies[0],
                parent_xform=wp.transform((-spacing * (link_count - 1), 0.0, 0.0), wp.quat_identity()),
            )
        for closure in range(1, closure_count):
            child_link = 1 + (closure - 1) % (link_count - 2)
            template.add_joint_fixed(
                parent=bodies[-1],
                child=bodies[child_link],
                parent_xform=wp.transform(
                    (spacing * (child_link - (link_count - 1)), 0.0, 0.0),
                    wp.quat_identity(),
                ),
            )

    return template, np.asarray(bodies, dtype=np.int32)


def _build_replicated_fixed_topology(device, *, closed: bool, worlds=4, link_count=6, closure_count=None):
    """Build equal-shape rigid islands to exercise topology batching."""
    if closure_count is None:
        closure_count = int(closed)
    if closure_count < 0 or (not closed and closure_count):
        raise ValueError("Closure count must agree with closed topology")
    template, _ = _build_fixed_topology_template(link_count=link_count, closure_count=closure_count)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for world in range(worlds):
        builder.add_builder(template, xform=wp.transform((0.0, 0.2 * world, 0.0), wp.quat_identity()))
    builder.color(balance_colors=False)
    return builder.finalize(device=device)


def _build_mixed_fixed_topologies(device):
    """Build one healthy tree and one independent high-rank cyclic island."""
    healthy_template, healthy_template_bodies = _build_fixed_topology_template(link_count=6, closure_count=0)
    oversized_template, _ = _build_fixed_topology_template(link_count=20, closure_count=16)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    # Give the oversized cycle lower model IDs. The policy must reject it and
    # continue to the healthy tree rather than relying on a tree-first order.
    builder.add_builder(oversized_template, xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()))
    healthy_start = builder.body_count
    builder.add_builder(healthy_template, xform=wp.transform((0.0, 1.0, 0.0), wp.quat_identity()))
    healthy_bodies = healthy_start + healthy_template_bodies
    builder.color(balance_colors=False)
    return builder.finalize(device=device), healthy_bodies


def _build_structural_backend_with_intercepted_bytes(model, *, solver=None, **backend_options):
    """Count allocations through the backend or its solver-owned rebuild path."""
    allocations = []
    allocation_functions = (
        "array",
        "empty",
        "empty_like",
        "zeros",
        "zeros_like",
        "ones",
    )
    originals = {name: getattr(wp, name) for name in allocation_functions}
    with ExitStack() as patches:
        for name, original in originals.items():

            def record_allocation(*args, _original=original, **kwargs):
                result = _original(*args, **kwargs)
                allocations.append((str(result.device), int(result.ptr), int(result.capacity)))
                return result

            patches.enter_context(mock.patch.object(wp, name, record_allocation))
        if solver is None:
            backend = rigid_vbd_kkt.StructuralGraphKKT(
                model,
                model.body_inv_mass,
                ignore_free_completion_joints=True,
                **backend_options,
            )
        else:
            solver._rebuild_structural_graph_kkt()
            backend = solver._structural_graph_kkt
    target_device = str(model.device)
    unique_allocations = {
        (pointer, capacity) for device, pointer, capacity in allocations if device == target_device and capacity > 0
    }
    return backend, sum(capacity for _, capacity in unique_allocations)


def _closed_prefix_payload_budget(model, count):
    """Return the exact active payload for a prefix of one closed topology batch."""
    body_inv_mass_host = np.asarray(model.body_inv_mass.numpy(), dtype=float)
    paths = rigid_vbd_kkt._build_paths(model, body_inv_mass_host)
    trees = rigid_vbd_kkt._build_trees(
        model,
        {frozenset(path.joints) for path in paths},
        body_inv_mass_host,
    )
    handled = {frozenset(component.joints) for component in [*paths, *trees]}
    closed = rigid_vbd_kkt._build_closed_trees(model, handled, body_inv_mass_host)
    parent = np.asarray(model.joint_parent.numpy(), dtype=np.int32)
    child = np.asarray(model.joint_child.numpy(), dtype=np.int32)
    ordered = sorted(closed, key=rigid_vbd_kkt._component_order_key)
    prefix = ordered[:count]
    symbolics = rigid_vbd_kkt._tree_bucket_symbolics(prefix[0].tree, include_contraction=False)
    diagnostics = rigid_vbd_kkt._closed_tree_bucket_diagnostics(
        prefix,
        parent,
        child,
        model.device,
        tree_symbolics=symbolics,
    )
    body_ids = {body for component in prefix for body in component.bodies}
    joint_ids = {joint for component in prefix for joint in component.joints}
    return diagnostics.estimated_bytes + rigid_vbd_kkt._shared_structural_payload_bytes(
        model.body_count,
        model.joint_count,
        len(body_ids),
        count,
        len(joint_ids),
        is_cuda=model.device.is_cuda,
    )


def _build_prismatic_material_chain(device, *, kind, stiffness=1.0e5, link_count=12):
    """Build a serial finite drive or unilateral-limit equilibrium problem."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    bodies = []
    joints = []
    spacing = 0.08
    damping = 2.0 * np.sqrt(stiffness)
    for link in range(link_count):
        body = builder.add_link(
            xform=wp.transform(wp.vec3(spacing * link, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_box(body, hx=0.03, hy=0.015, hz=0.015)
        bodies.append(body)
        parent = -1 if link == 0 else bodies[link - 1]
        parent_anchor = wp.vec3(spacing * link, 0.0, 0.0) if parent < 0 else wp.vec3(spacing, 0.0, 0.0)
        options = {}
        if kind == "drive":
            options.update(target_ke=stiffness, target_kd=damping)
        elif kind == "limit":
            options.update(
                limit_lower=-0.015,
                limit_upper=0.015,
                limit_ke=stiffness,
                limit_kd=damping,
            )
        else:
            raise ValueError(kind)
        joints.append(
            builder.add_joint_prismatic(
                parent=parent,
                child=body,
                parent_xform=wp.transform(parent_anchor, wp.quat_identity()),
                child_xform=wp.transform_identity(),
                axis=newton.Axis.X,
                **options,
            )
        )
    builder.add_articulation(joints)
    builder.color(balance_colors=False)
    return builder.finalize(device=device), np.asarray(bodies, dtype=np.int32), spacing


def _simulate_prismatic_material_chain(device, *, kind, stiffness, global_iterations, iterations, steps):
    model, bodies, spacing = _build_prismatic_material_chain(device, kind=kind, stiffness=stiffness)
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    if kind == "drive":
        control.joint_target_q.fill_(0.02)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=iterations,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=global_iterations,
    )
    dt = 1.0 / 600.0
    for _ in range(steps):
        state_in.clear_forces()
        wp.launch(
            _apply_body_force,
            1,
            inputs=[int(bodies[-1]), wp.vec3(24.0, 0.0, 0.0)],
            outputs=[state_in.body_f],
            device=device,
        )
        solver.step(state_in, state_out, control, None, dt)
        state_in, state_out = state_out, state_in
    body_q = state_in.body_q.numpy()
    x = body_q[bodies, 0].astype(np.float64)
    coordinates = np.empty_like(x)
    coordinates[0] = x[0]
    coordinates[1:] = x[1:] - x[:-1] - spacing
    return body_q, coordinates


def _build_joint_pair(
    device,
    joint_type,
    *,
    finite_limit=False,
    kinematic_parent=False,
    static_child=False,
):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent = builder.add_link(xform=wp.transform_identity(), mass=1.0, is_kinematic=kinematic_parent)
    child = builder.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(parent, hx=0.04, hy=0.03, hz=0.02)
    builder.add_shape_box(child, hx=0.04, hy=0.03, hz=0.02)
    parent_xform = wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity())
    child_xform = wp.transform_identity()
    if joint_type == newton.JointType.ROD:
        joint = builder.add_joint_rod(
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            stretch_stiffness=1.0e6,
            shear_stiffness=2.0e5,
            bend_stiffness=4.0e4,
            twist_stiffness=8.0e3,
        )
    elif joint_type == newton.JointType.BALL:
        joint = builder.add_joint_ball(parent, child, parent_xform=parent_xform, child_xform=child_xform)
    elif joint_type == newton.JointType.FIXED:
        joint = builder.add_joint_fixed(parent, child, parent_xform=parent_xform, child_xform=child_xform)
    elif joint_type == newton.JointType.REVOLUTE:
        limit_options = {"limit_lower": -0.5, "limit_upper": 0.5} if finite_limit else {}
        joint = builder.add_joint_revolute(
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            axis=newton.Axis.Z,
            **limit_options,
        )
    elif joint_type == newton.JointType.PRISMATIC:
        joint = builder.add_joint_prismatic(
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            axis=newton.Axis.X,
        )
    elif joint_type == newton.JointType.D6:
        config = newton.ModelBuilder.JointDofConfig
        joint = builder.add_joint_d6(
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            linear_axes=[config.create_unlimited(newton.Axis.X)],
            angular_axes=[config.create_unlimited(newton.Axis.Z)],
        )
    else:
        raise ValueError(f"Unsupported test joint type {joint_type}")
    if static_child:
        _pin_body(builder, child)
    builder.add_articulation([joint])
    builder.color(balance_colors=False)
    return builder.finalize(device=device), parent, child, joint


def _build_grounded_chain(
    device,
    *,
    global_iterations,
    iterations=5,
    segments=32,
    contact_history=False,
    contact_kd=100.0,
    contact_buffer_size=128,
    deterministic=None,
    capsule_mass=False,
):
    builder = newton.ModelBuilder()
    shape = builder.default_shape_cfg.copy()
    shape.mu = 0.45
    shape.ke = 1.0e5
    shape.kd = contact_kd
    radius = 0.01
    rod = newton.Rod.create_straight(
        start=wp.vec3(-0.5 * segments * 0.03, 0.0, radius + 0.001),
        direction=wp.vec3(1.0, 0.0, 0.0),
        length=segments * 0.03,
        segment_count=segments,
        radius=radius,
    )
    bodies, joints = builder.add_rod(
        rod=rod,
        cfg=shape,
        stretch_stiffness=1.0e9,
        bend_stiffness=1.0e4,
        bend_damping=1.0e3,
        wrap_in_articulation=True,
        body_frame_origin="com",
    )
    if capsule_mass:
        _use_capsule_rod_mass(builder, bodies, shape.density)
    builder.add_ground_plane(cfg=shape)
    builder.color(balance_colors=False)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))
    pipeline = newton.CollisionPipeline(model, contact_matching="latest")
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=iterations,
        rigid_compliant_alm=True,
        rigid_contact_history=contact_history,
        rigid_joint_global_iterations=global_iterations,
        rigid_body_contact_buffer_size=contact_buffer_size,
        deterministic=deterministic,
    )
    return (
        model,
        pipeline,
        contacts,
        solver,
        np.asarray(bodies, dtype=np.int32),
        np.asarray(joints),
        radius,
    )


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quaternion[:3]
    intermediate = 2.0 * np.cross(xyz, vector)
    return vector + quaternion[3] * intermediate + np.cross(xyz, intermediate)


def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return transform[:3] + _quat_rotate(transform[3:7], point)


def _max_joint_gap(model, body_q: np.ndarray, joint_ids: np.ndarray) -> float:
    parents = model.joint_parent.numpy()
    children = model.joint_child.numpy()
    parent_frames = model.joint_X_p.numpy()
    child_frames = model.joint_X_c.numpy()
    result = 0.0
    for joint in joint_ids:
        parent = int(parents[joint])
        child = int(children[joint])
        parent_anchor = parent_frames[joint, :3]
        if parent >= 0:
            parent_anchor = _transform_point(body_q[parent], parent_anchor)
        child_anchor = _transform_point(body_q[child], child_frames[joint, :3])
        result = max(result, float(np.linalg.norm(child_anchor - parent_anchor)))
    return result


def _simulate(model, solver, steps, dt, contacts=None):
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    for _ in range(steps):
        state_in.clear_forces()
        if contacts is not None:
            model.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, dt)
        state_in, state_out = state_out, state_in
    return state_in


def _relative_norm(value, reference):
    """Return a relative Euclidean norm with a finite zero-reference fallback."""
    return float(np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1.0e-30))


def _null_space_error_metrics(operator, value, reference):
    """Decompose an error into the observable range and orthogonal null space."""
    error = value - reference
    _, singular_values, right_vectors_transpose = np.linalg.svd(operator, full_matrices=True)
    tolerance = max(operator.shape) * np.finfo(np.float64).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    null_basis = right_vectors_transpose[rank:].T
    null_error = null_basis @ (null_basis.T @ error)
    range_error = error - null_error
    error_norm = max(np.linalg.norm(error), 1.0e-30)
    reference_norm = max(np.linalg.norm(reference), 1.0e-30)
    return {
        "reaction_null_fraction": float(np.linalg.norm(null_error) / error_norm),
        "reaction_null_error": float(np.linalg.norm(null_error) / reference_norm),
        "reaction_range_error": float(np.linalg.norm(range_error) / reference_norm),
    }


def _robust_history_growth_factor(values):
    """Estimate end-to-end exponential growth from the median pairwise log slope."""
    log_values = np.log(np.maximum(np.asarray(values, dtype=np.float64), np.finfo(np.float64).tiny))
    slopes = [
        (log_values[end] - log_values[start]) / (end - start)
        for start in range(len(log_values))
        for end in range(start + 1, len(log_values))
    ]
    return float(np.exp(np.median(slopes) * (len(log_values) - 1)))


def _assemble_float64_structural_kkt(
    model,
    backend,
    body_matrix,
    body_rhs,
    joint_ids,
    jacobian_parent,
    jacobian_child,
    compliance,
    residual,
):
    """Assemble the physical frozen structural KKT system in float64."""
    body_count = backend.graph_body_count
    row_count = len(joint_ids)
    body_slot_by_id = backend.body_slot_by_id.numpy()
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()

    body_block = np.zeros((6 * body_count, 6 * body_count), dtype=np.float64)
    row_block = np.zeros((6 * row_count, 6 * row_count), dtype=np.float64)
    jacobian = np.zeros((6 * row_count, 6 * body_count), dtype=np.float64)
    for body in range(body_count):
        body_slice = slice(6 * body, 6 * body + 6)
        body_block[body_slice, body_slice] = body_matrix[body]
    for row, joint in enumerate(joint_ids):
        row_slice = slice(6 * row, 6 * row + 6)
        row_block[row_slice, row_slice] = compliance[row]
        for body, endpoint_jacobian in (
            (int(joint_parent[joint]), jacobian_parent[row]),
            (int(joint_child[joint]), jacobian_child[row]),
        ):
            body_slot = int(body_slot_by_id[body]) if body >= 0 else -1
            if body_slot >= 0:
                body_slice = slice(6 * body_slot, 6 * body_slot + 6)
                jacobian[row_slice, body_slice] += endpoint_jacobian

    matrix = np.block([[body_block, jacobian.T], [jacobian, -row_block]])
    rhs = np.concatenate((body_rhs.reshape(-1), -residual.reshape(-1)))
    return matrix, rhs


def _equilibrated_kkt_residual(matrix, rhs, physical_solution, scales):
    """Measure residual in the diagonal coordinates used by the tree kernels."""
    scaled_matrix = scales[:, None] * matrix * scales[None, :]
    scaled_rhs = scales * rhs
    scaled_solution = physical_solution / scales
    return float(
        np.linalg.norm(scaled_matrix @ scaled_solution - scaled_rhs) / max(np.linalg.norm(scaled_rhs), 1.0e-30)
    )


def _path_float64_oracle_metrics(device, stiffness, *, segments=16, model=None, capsule_mass=False):
    """Compare one production path Schur solve with independent float64 assembly."""
    if model is None:
        model, _, _ = _build_chain(device, segments=segments, stiffness=stiffness, capsule_mass=capsule_mass)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    # The solver plans routes and factor precisions at its first step outside capture and may then replace
    # the backend; plan for the step of this probe before taking references into it.
    solver._plan_structural_graph_kkt(1.0 / 600.0)
    backend = solver._structural_graph_kkt
    bucket = backend.path_buckets[0]

    # Production aliases compliance/diagonal and residual/rhs because their
    # lifetimes do not overlap. Give the test its own linearization storage so
    # the independent formula can inspect both sides of the in-place assembly.
    bucket.compliance = wp.zeros(bucket.size, dtype=wp.spatial_matrix, device=device)
    bucket.residual = wp.zeros(bucket.size, dtype=wp.spatial_vector, device=device)

    captured = {}
    solve_rows = bucket.solve_rows

    def capture_and_solve_rows():
        for name, array in (
            ("lower", bucket.lower[0]),
            ("diagonal", bucket.diagonal[0]),
            ("upper", bucket.upper[0]),
            ("rhs", bucket.rhs[0]),
            ("jacobian_parent", bucket.jacobian_parent),
            ("jacobian_child", bucket.jacobian_child),
            ("compliance", bucket.compliance),
            ("residual", bucket.residual),
            ("row_body", bucket.row_body),
        ):
            captured[name] = array.numpy().copy().astype(np.float64)
        captured["body_inverse"] = backend.body_inverse.numpy().copy().astype(np.float64)
        captured["body_free"] = backend.body_free.numpy().copy().astype(np.float64)
        solve_rows()
        captured["multiplier"] = bucket.solution[0].numpy().copy().astype(np.float64)

    bucket.solve_rows = capture_and_solve_rows
    solve = backend.solve

    def capture_solve(**kwargs):
        solve(**kwargs)
        if not kwargs.get("translation_only", False):
            # The later translation reconciliation reuses this scratch even
            # when every currently enabled boundary row blocks translation.
            captured["body_correction"] = backend.body_correction.numpy().astype(np.float64)

    backend.solve = capture_solve
    _simulate(model, solver, 1, 1.0 / 600.0)

    row_count = bucket.size
    matrix = np.zeros((6 * row_count, 6 * row_count), dtype=np.float64)
    for row in range(row_count):
        row_slice = slice(6 * row, 6 * row + 6)
        matrix[row_slice, row_slice] = captured["diagonal"][row]
        if row > 0:
            matrix[row_slice, slice(6 * (row - 1), 6 * row)] = captured["lower"][row]
        if row + 1 < row_count:
            matrix[row_slice, slice(6 * (row + 1), 6 * (row + 2))] = captured["upper"][row]
    rhs = captured["rhs"].reshape(-1)

    # Independently recover C + J B^-1 J^T and c + J B^-1 f from the actual
    # linearized row arrays. This tests the Galerkin/Delassus identity as well
    # as the CR factorization that consumes the assembled blocks.
    formula_matrix = np.zeros_like(matrix)
    formula_rhs = captured["residual"].copy()
    endpoints = []
    for row in range(row_count):
        formula_matrix[6 * row : 6 * row + 6, 6 * row : 6 * row + 6] += captured["compliance"][row]
        row_endpoints = []
        for endpoint_jacobian, body_slot in (
            (captured["jacobian_parent"][row], int(captured["row_body"][row, 0])),
            (captured["jacobian_child"][row], int(captured["row_body"][row, 1])),
        ):
            if body_slot >= 0:
                row_endpoints.append((endpoint_jacobian, body_slot))
                formula_rhs[row] += endpoint_jacobian @ captured["body_free"][body_slot]
        endpoints.append(row_endpoints)
    for row, row_endpoints in enumerate(endpoints):
        for column, column_endpoints in enumerate(endpoints):
            block = np.zeros((6, 6), dtype=np.float64)
            for row_jacobian, row_body in row_endpoints:
                for column_jacobian, column_body in column_endpoints:
                    if row_body == column_body:
                        block += row_jacobian @ captured["body_inverse"][row_body] @ column_jacobian.T
            formula_matrix[6 * row : 6 * row + 6, 6 * column : 6 * column + 6] += block

    reference_multiplier = np.linalg.solve(matrix, rhs).reshape(row_count, 6)
    production_multiplier = captured["multiplier"]
    reference_correction = captured["body_free"].copy()
    for row, row_endpoints in enumerate(endpoints):
        for jacobian, body_slot in row_endpoints:
            reference_correction[body_slot] -= (
                captured["body_inverse"][body_slot] @ jacobian.T @ reference_multiplier[row]
            )
    production_correction = captured["body_correction"]

    return {
        "condition": float(np.linalg.cond(matrix)),
        "formula_matrix_error": _relative_norm(formula_matrix, matrix),
        "formula_rhs_error": _relative_norm(formula_rhs.reshape(-1), rhs),
        "residual": float(
            np.linalg.norm(matrix @ production_multiplier.reshape(-1) - rhs) / max(np.linalg.norm(rhs), 1.0e-30)
        ),
        "multiplier_error": _relative_norm(production_multiplier, reference_multiplier),
        "correction_error": _relative_norm(production_correction, reference_correction),
    }


def _tree_float64_oracle_metrics(
    device,
    stiffness,
    *,
    closed,
    tree_segments_per_branch=4,
    ring_segments=8,
    add_chord=True,
    bend_stiffness=1.0e4,
    open_shape="y",
    capsule_mass=False,
):
    """Compare a production open/cyclic tree solve with a float64 physical KKT."""
    if closed:
        model, _, _ = _build_loop_with_branch(
            device,
            ring_segments=ring_segments,
            stiffness=stiffness,
            bend_stiffness=bend_stiffness,
            add_chord=add_chord,
            capsule_mass=capsule_mass,
        )
    elif open_shape == "y":
        model, _, _ = _build_y_tree(device, segments_per_branch=tree_segments_per_branch, stiffness=stiffness)
    elif open_shape == "spurred":
        model, _, _ = _build_spurred_tree(device, arm_segments=tree_segments_per_branch, stiffness=stiffness)
    else:
        raise ValueError(f"Unknown open tree shape {open_shape}")
    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    backend = solver._structural_graph_kkt
    bucket = backend.closed_tree_buckets[0] if closed else backend.tree_buckets[0]
    tree = bucket.tree if closed else bucket
    captured = {}
    solve_tree = bucket.solve_tree
    if closed:
        solve_closure_schur = bucket.solve_closure_schur

        def capture_and_solve_closure_schur():
            captured["closure_schur"] = bucket.closure_schur.numpy().copy().astype(np.float64)
            captured["closure_rhs_assembled"] = bucket.closure_rhs.numpy().copy().astype(np.float64)
            solve_closure_schur()

        bucket.solve_closure_schur = capture_and_solve_closure_schur

    def capture_and_solve_tree(body_matrix, body_rhs, body_scale, body_correction, **kwargs):
        arrays = [
            ("body_matrix", body_matrix),
            ("body_rhs", body_rhs),
            ("tree_jacobian_parent", tree.jacobian_parent),
            ("tree_jacobian_child", tree.jacobian_child),
            ("tree_compliance", tree.compliance),
            ("tree_residual", tree.residual),
        ]
        if closed:
            arrays.extend(
                (
                    ("closure_jacobian_parent", bucket.closure_jacobian_parent),
                    ("closure_jacobian_child", bucket.closure_jacobian_child),
                    ("closure_compliance", bucket.closure_compliance),
                    ("closure_residual", bucket.closure_residual),
                )
            )
        for name, array in arrays:
            captured[name] = array.numpy().copy().astype(np.float64)

        solve_tree(body_matrix, body_rhs, body_scale, body_correction, **kwargs)
        for name, array in (
            ("body_correction", body_correction),
            ("tree_solution", tree.solution),
            ("tree_row_scale", tree.row_scale),
            ("tree_node_row", tree.node_row),
        ):
            captured[name] = array.numpy().copy().astype(np.float64)
        if closed:
            for name, array in (
                ("closure_response", bucket.response_solution),
                ("closure_multiplier", bucket.closure_multiplier),
                ("closure_row_scale", bucket.closure_row_scale),
            ):
                captured[name] = array.numpy().copy().astype(np.float64)

    bucket.solve_tree = capture_and_solve_tree
    _simulate(model, solver, 1, 1.0 / 600.0)

    joint_ids = tree.joint_ids_host
    jacobian_parent = captured["tree_jacobian_parent"]
    jacobian_child = captured["tree_jacobian_child"]
    compliance = captured["tree_compliance"]
    residual = captured["tree_residual"]
    if closed:
        joint_ids = np.concatenate((joint_ids, bucket.closure_joint_ids.numpy().astype(np.int64)))
        jacobian_parent = np.concatenate((jacobian_parent, captured["closure_jacobian_parent"]))
        jacobian_child = np.concatenate((jacobian_child, captured["closure_jacobian_child"]))
        compliance = np.concatenate((compliance, captured["closure_compliance"]))
        residual = np.concatenate((residual, captured["closure_residual"]))

    matrix, rhs = _assemble_float64_structural_kkt(
        model,
        backend,
        captured["body_matrix"],
        captured["body_rhs"],
        joint_ids,
        jacobian_parent,
        jacobian_child,
        compliance,
        residual,
    )
    reference = np.linalg.solve(matrix, rhs)

    tree_multiplier = np.zeros((tree.row_count, 6), dtype=np.float64)
    closure_count = bucket.closure_count if closed else 0
    for node, row in enumerate(captured["tree_node_row"].astype(np.int64)):
        if row < 0:
            continue
        value = captured["tree_solution"][node].copy()
        if closed:
            for closure in range(closure_count):
                value -= (
                    captured["closure_response"][node * closure_count + closure]
                    @ captured["closure_multiplier"][closure]
                )
        tree_multiplier[row] = captured["tree_row_scale"][row] * value
    multiplier = tree_multiplier
    row_scales = captured["tree_row_scale"]
    if closed:
        closure_multiplier = captured["closure_row_scale"] * captured["closure_multiplier"]
        multiplier = np.concatenate((multiplier, closure_multiplier))
        row_scales = np.concatenate((row_scales, captured["closure_row_scale"]))

    body_count = backend.graph_body_count
    body_correction = captured["body_correction"][:body_count]
    production = np.concatenate((body_correction.reshape(-1), multiplier.reshape(-1)))
    # body_scale aliases body_correction in production after factorization, so
    # reconstruct the exact equilibration rule from the saved physical blocks.
    body_scales = 1.0 / np.sqrt(
        np.maximum(
            np.abs(np.diagonal(captured["body_matrix"][:body_count], axis1=1, axis2=2)),
            1.0e-30,
        )
    )
    scales = np.concatenate((body_scales.reshape(-1), row_scales.reshape(-1)))
    reference_correction = reference[: 6 * body_count].reshape(body_count, 6)
    reference_multiplier = reference[6 * body_count :].reshape(len(joint_ids), 6)
    jacobian = matrix[6 * body_count :, : 6 * body_count]
    scaled_matrix = scales[:, None] * matrix * scales[None, :]
    scaled_rhs = scales * rhs
    scaled_solution = production / scales
    scaled_residual = scaled_matrix @ scaled_solution - scaled_rhs
    scaled_rhs_norm = max(np.linalg.norm(scaled_rhs), 1.0e-30)
    body_dofs = 6 * body_count
    anchored = np.minimum(model.joint_parent.numpy()[joint_ids], model.joint_child.numpy()[joint_ids]) < 0
    internal_multiplier = np.where(anchored[:, None], reference_multiplier, multiplier)
    metrics = {
        "condition": float(np.linalg.cond(matrix)),
        "physical_residual": float(np.linalg.norm(matrix @ production - rhs) / max(np.linalg.norm(rhs), 1.0e-30)),
        "equilibrated_residual": _equilibrated_kkt_residual(matrix, rhs, production, scales),
        "equilibrated_body_residual": float(np.linalg.norm(scaled_residual[:body_dofs]) / scaled_rhs_norm),
        "equilibrated_row_residual": float(np.linalg.norm(scaled_residual[body_dofs:]) / scaled_rhs_norm),
        "correction_error": _relative_norm(body_correction, reference_correction),
        "multiplier_error": _relative_norm(multiplier, reference_multiplier),
        # Raw reaction coordinates include redundant cycle-space modes.  This
        # separate metric measures the body-affecting reaction J^T mu, which is
        # the quantity that changes the primal correction.
        "body_reaction_error": _relative_norm(
            jacobian.T @ multiplier.reshape(-1),
            jacobian.T @ reference_multiplier.reshape(-1),
        ),
        "internal_reaction_error": _relative_norm(
            jacobian.T @ internal_multiplier.reshape(-1),
            jacobian.T @ reference_multiplier.reshape(-1),
        ),
        "uses_backbone_cr": bool(not closed and bucket.use_backbone_cr),
        "uses_paired_backbone": bool(closed and bucket.use_paired_backbone),
    }
    metrics.update(
        _null_space_error_metrics(
            jacobian.T,
            multiplier.reshape(-1),
            reference_multiplier.reshape(-1),
        )
    )
    if closed:
        # Independently eliminate the body/tree block. This distinguishes an
        # inaccurate closure solve from an inaccurate tree response used to
        # assemble that small Schur system.
        tree_dofs = 6 * tree.row_count
        base_dofs = body_dofs + tree_dofs
        base_matrix = matrix[:base_dofs, :base_dofs]
        closure_coupling = matrix[base_dofs:, :base_dofs]
        exact_closure_schur = -matrix[base_dofs:, base_dofs:] + closure_coupling @ np.linalg.solve(
            base_matrix, closure_coupling.T
        )
        exact_closure_rhs = -rhs[base_dofs:] + closure_coupling @ np.linalg.solve(base_matrix, rhs[:base_dofs])
        closure_scales = captured["closure_row_scale"].reshape(-1)
        exact_closure_schur = closure_scales[:, None] * exact_closure_schur * closure_scales[None, :]
        exact_closure_rhs = closure_scales * exact_closure_rhs
        production_closure_schur = np.zeros_like(exact_closure_schur)
        for row in range(closure_count):
            for column in range(closure_count):
                production_closure_schur[
                    6 * row : 6 * row + 6,
                    6 * column : 6 * column + 6,
                ] = captured["closure_schur"][row * closure_count + column]
        exact_closure_multiplier = np.linalg.solve(exact_closure_schur, exact_closure_rhs)
        production_closure_multiplier = captured["closure_multiplier"].reshape(-1)
        closure_rhs_norm = max(np.linalg.norm(exact_closure_rhs), 1.0e-30)
        metrics.update(
            {
                "closure_schur_error": _relative_norm(production_closure_schur, exact_closure_schur),
                "closure_schur_action_error": float(
                    np.linalg.norm((production_closure_schur - exact_closure_schur) @ exact_closure_multiplier)
                    / closure_rhs_norm
                ),
                "closure_rhs_error": _relative_norm(captured["closure_rhs_assembled"].reshape(-1), exact_closure_rhs),
                "closure_multiplier_error": _relative_norm(
                    production_closure_multiplier,
                    exact_closure_multiplier,
                ),
                "closure_exact_system_residual": float(
                    np.linalg.norm(exact_closure_schur @ production_closure_multiplier - exact_closure_rhs)
                    / closure_rhs_norm
                ),
            }
        )
    return metrics


def _linearize_bucket(model, solver, bucket, body_q, dt, *, track_limits=False):
    control = model.control()
    backend = solver._structural_graph_kkt
    wp.launch(
        linearize_joint_path_rows,
        bucket.size,
        inputs=[
            bucket.joint_ids,
            wp.array(np.arange(bucket.size, dtype=np.int32), dtype=wp.int32, device=model.device),
            model.joint_type,
            model.joint_enabled,
            model.joint_parent,
            model.joint_child,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_axis,
            solver.joint_rod_rest_kb_local,
            solver.joint_rod_rest_twist,
            model.joint_qd_start,
            model.joint_target_q_start,
            model.joint_dof_dim,
            solver.joint_constraint_start,
            solver.joint_material_k,
            solver.joint_rho,
            solver.joint_penalty_kd,
            solver.joint_lambda_lin,
            solver.joint_lambda_ang,
            solver.joint_C0_lin,
            solver.joint_C0_ang,
            solver.joint_sigma_start,
            solver.joint_C_fric,
            model.joint_target_ke,
            model.joint_target_kd,
            control.joint_target_q,
            control.joint_target_qd,
            model.joint_limit_lower,
            model.joint_limit_upper,
            model.joint_limit_ke,
            model.joint_limit_kd,
            solver.joint_rest_angle,
            solver.joint_angle_prev,
            solver.joint_drive_limit_support,
            solver.joint_drive_lambda,
            solver.joint_limit_lambda,
            solver.rigid_joint_alpha,
            body_q,
            solver.body_q_prev,
            model.body_q,
            model.body_com,
            dt,
            False,
            backend.body_slot_by_id,
            backend.graph_body_island,
            backend.body_correction,
            backend.line_search_slope,
        ],
        outputs=[
            bucket.jacobian_parent,
            bucket.jacobian_child,
            bucket.compliance,
            bucket.residual,
            bucket.row_active,
            False,
            backend.joint_stress_slot,
            model.body_inv_mass,
            backend.body_stress,
            False,
            0,
            backend.slope_jacobian_parent,
            backend.slope_jacobian_child,
            backend.slope_compliance,
            backend.slope_residual,
            backend.slope_objective,
            backend.slope_mode,
            backend.limit_search_needed if track_limits else None,
            backend.limit_search_islands if track_limits else None,
        ],
        device=model.device,
    )


def _perturbed_linearization(model, solver, bucket, base_body_q, body, direction, scale, dt):
    body_q = wp.array(base_body_q, dtype=wp.transform, device=model.device)
    delta = np.zeros(6, dtype=np.float32)
    delta[direction] = scale
    wp.launch(
        _perturb_body_pose,
        1,
        inputs=[body, wp.spatial_vector(*delta), model.body_com],
        outputs=[body_q],
        device=model.device,
    )
    _linearize_bucket(model, solver, bucket, body_q, dt)
    return bucket.residual.numpy().astype(np.float64)


def _simulate_ground_drag(
    device,
    global_iterations,
    *,
    substeps=1,
    iterations=5,
    contact_history=False,
    force=45.0,
    reverse_at=None,
    weight_fraction=None,
    frames=240,
    capsule_mass=False,
):
    model, pipeline, contacts, solver, bodies, joints, radius = _build_grounded_chain(
        device,
        global_iterations=global_iterations,
        iterations=iterations,
        contact_history=contact_history,
        deterministic=wp.DeterministicMode.RUN_TO_RUN,
        capsule_mass=capsule_mass,
    )
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    dt = 1.0 / (600.0 * substeps)
    if weight_fraction is not None:
        force = weight_fraction * float(model.body_mass.numpy()[bodies].sum()) * 9.81
    initial_center = 0.0
    minimum_z = np.inf
    maximum_center_z = -np.inf
    for step in range(frames * substeps):
        state_in.clear_forces()
        if step >= 40 * substeps:
            wp.launch(
                _apply_body_force,
                1,
                inputs=[
                    int(bodies[-1]),
                    wp.vec3(
                        force if reverse_at is None or step < reverse_at * substeps else -force,
                        0.0,
                        0.0,
                    ),
                ],
                outputs=[state_in.body_f],
                device=device,
            )
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, dt)
        state_in, state_out = state_out, state_in
        if step == 40 * substeps - 1:
            initial_center = float(state_in.body_q.numpy()[bodies, 0].mean())
        elif step >= 40 * substeps:
            positions = state_in.body_q.numpy()[bodies, 2]
            minimum_z = min(minimum_z, float(positions.min()))
            maximum_center_z = max(maximum_center_z, float(positions.mean()))
    body_q = state_in.body_q.numpy()
    return {
        "motion": float(body_q[bodies, 0].mean()) - initial_center,
        "penetration": max(0.0, radius - minimum_z),
        "rise": max(0.0, maximum_center_z - radius),
        "gap": _max_joint_gap(model, body_q, joints),
        "mass": float(model.body_mass.numpy()[bodies].sum()),
    }


def _simulate_grounded_loop_load(device, global_iterations):
    model, bodies, joints = _build_loop_with_branch(device, ring_segments=16, grounded=True)
    pipeline = newton.CollisionPipeline(model, contact_matching="latest")
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=2,
        rigid_compliant_alm=True,
        rigid_contact_history=False,
        rigid_joint_global_iterations=global_iterations,
        rigid_body_contact_buffer_size=256,
        deterministic=wp.DeterministicMode.RUN_TO_RUN,
    )
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    dt = 1.0 / 600.0
    minimum_z = np.inf
    for step in range(120):
        state_in.clear_forces()
        if step >= 20:
            wp.launch(
                _apply_body_force,
                1,
                inputs=[int(bodies[8]), wp.vec3(-35.0, 0.0, 0.0)],
                outputs=[state_in.body_f],
                device=device,
            )
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, dt)
        state_in, state_out = state_out, state_in
        minimum_z = min(minimum_z, float(state_in.body_q.numpy()[bodies, 2].min()))
    body_q = state_in.body_q.numpy()
    return {
        "gap": _max_joint_gap(model, body_q, joints),
        "penetration": max(0.0, 0.01 - minimum_z),
    }


def _structural_kkt_selects_supported_complete_graphs(test, device):
    """Select complete supported joint graphs and their limit metadata."""
    elastic, _, _ = _build_chain(device)
    elastic_solver = newton.solvers.SolverVBD(
        elastic,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    test.assertIsNotNone(elastic_solver._structural_graph_kkt)
    test.assertEqual(elastic_solver._structural_graph_kkt.island_count, 1)

    dahl, _, _ = _build_chain(device, dahl=True)
    dahl_solver = newton.solvers.SolverVBD(
        dahl,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    test.assertIsNotNone(dahl_solver._structural_graph_kkt)

    limited, _, _, _ = _build_joint_pair(device, newton.JointType.REVOLUTE, finite_limit=True)
    limited_solver = newton.solvers.SolverVBD(
        limited,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    test.assertIsNotNone(limited_solver._structural_graph_kkt)
    limited_state = _simulate(limited, limited_solver, 2, 1.0 / 600.0)
    test.assertTrue(np.isfinite(limited_state.body_q.numpy()).all())


def _structural_kkt_classifies_dynamic_contact_topology(test, device):
    """Only active dynamic pairs mark structural islands for relaxation."""
    contact_count = wp.array([1], dtype=wp.int32, device=device)
    shape0 = wp.array([0], dtype=wp.int32, device=device)
    shape1 = wp.array([1], dtype=wp.int32, device=device)
    point = wp.zeros(1, dtype=wp.vec3, device=device)
    normal = wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
    margin = wp.full(1, value=0.1, dtype=float, device=device)
    shape_body = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    body_inv_mass = wp.array([1.0, 1.0, 0.0], dtype=float, device=device)
    contact_lambda = wp.zeros(1, dtype=wp.vec3, device=device)
    body_slot = wp.array([0, 1, -1], dtype=wp.int32, device=device)
    body_island = wp.array([0, 1], dtype=wp.int32, device=device)
    body_contact_counts = wp.array([1, 1, 0], dtype=wp.int32, device=device)
    contact_state = wp.ones(2, dtype=wp.int32, device=device)

    def classify(body_q):
        wp.launch(
            classify_global_contact_islands,
            1,
            inputs=[
                contact_count,
                shape0,
                shape1,
                point,
                point,
                normal,
                margin,
                margin,
                shape_body,
                body_q,
                body_inv_mass,
                contact_lambda,
                body_slot,
                body_island,
                1,
                body_contact_counts,
            ],
            outputs=[contact_state],
            device=device,
        )

    separated_q = wp.array(
        [
            wp.transform(wp.vec3(0.0), wp.quat_identity()),
            wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
            wp.transform(wp.vec3(0.0), wp.quat_identity()),
        ],
        dtype=wp.transform,
        device=device,
    )
    classify(separated_q)
    np.testing.assert_array_equal(contact_state.numpy(), [1, 1])

    active_q = wp.array(
        [
            wp.transform(wp.vec3(0.0), wp.quat_identity()),
            wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()),
            wp.transform(wp.vec3(0.0), wp.quat_identity()),
        ],
        dtype=wp.transform,
        device=device,
    )
    classify(active_q)
    np.testing.assert_array_equal(contact_state.numpy(), [-1, -1])

    # A dynamic endpoint outside the structural backend is fixed during this
    # global pass, so its contact is exact one-body support rather than a
    # concurrently solved pair edge.
    body_slot.assign([0, -1, -1])
    contact_state.fill_(1)
    classify(active_q)
    np.testing.assert_array_equal(contact_state.numpy(), [1, 1])

    # Truncation makes the frozen objective incomplete even when this sampled
    # row is otherwise classifiable, so only the affected island is suppressed.
    body_slot.assign([0, 1, -1])
    body_contact_counts.assign([2, 1, 0])
    contact_state.fill_(1)
    classify(active_q)
    np.testing.assert_array_equal(contact_state.numpy(), [-2, -1])

    # The overflow sentinel must suppress only its island at application time;
    # a healthy dynamic-contact island still receives the unmodified correction.
    body_ids = wp.array([0, 1], dtype=wp.int32, device=device)
    correction = wp.array(
        [
            wp.spatial_vector(0.25, 0.0, 0.0, 0.0, 0.0, 0.0),
            wp.spatial_vector(0.0, 0.25, 0.0, 0.0, 0.0, 0.0),
        ],
        dtype=wp.spatial_vector,
        device=device,
    )
    island_scale = wp.ones(2, dtype=float, device=device)
    body_com = wp.zeros(2, dtype=wp.vec3, device=device)
    pose_before = active_q.numpy().copy()
    wp.launch(
        apply_global_correction,
        2,
        inputs=[
            body_ids,
            body_island,
            contact_state,
            correction,
            island_scale,
            body_com,
            0,
            wp.zeros_like(active_q),
        ],
        outputs=[
            active_q,
            wp.zeros(active_q.shape[0], dtype=wp.vec3, device=device),
            wp.zeros(active_q.shape[0], dtype=wp.vec3, device=device),
        ],
        device=device,
    )
    pose_after = active_q.numpy()
    np.testing.assert_array_equal(pose_after[0], pose_before[0])
    np.testing.assert_allclose(pose_after[1, :3], pose_before[1, :3] + [0.0, 0.25, 0.0], rtol=0.0, atol=0.0)


def _structural_kkt_relaxes_dynamic_contact_by_curvature(test, device):
    """Relax global corrections only on islands with active dynamic contact."""
    body_ids = wp.array([0], dtype=wp.int32, device=device)
    body_island = wp.array([0], dtype=wp.int32, device=device)
    contact_state = wp.array([-1], dtype=wp.int32, device=device)
    correction = wp.array(
        [wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
        dtype=wp.spatial_vector,
        device=device,
    )
    dynamic_contact_hessian_host = np.zeros((1, 6, 6), dtype=np.float32)
    dynamic_contact_hessian_host[0, 0, 0] = 1.0
    dynamic_contact_hessian = wp.array(
        dynamic_contact_hessian_host,
        dtype=wp.spatial_matrix,
        device=device,
    )
    body_q = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
    body_mass = wp.array([2.0], dtype=float, device=device)
    body_inertia = wp.array(
        [wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)],
        dtype=wp.mat33,
        device=device,
    )
    step_scale = wp.ones(1, dtype=float, device=device)

    wp.launch(
        limit_dynamic_contact_jacobi_step,
        1,
        inputs=[
            body_ids,
            body_island,
            contact_state,
            correction,
            dynamic_contact_hessian,
            1.0,
            body_q,
            body_mass,
            body_inertia,
        ],
        outputs=[step_scale],
        device=device,
    )
    # Dynamic and inertial directional curvatures are both two, hence
    # overlap=1/2 and omega=1/(1+1/2)=2/3.
    np.testing.assert_allclose(step_scale.numpy(), [2.0 / 3.0], rtol=1.0e-6)

    contact_state.fill_(1)
    step_scale.fill_(1.0)
    wp.launch(
        limit_dynamic_contact_jacobi_step,
        1,
        inputs=[
            body_ids,
            body_island,
            contact_state,
            correction,
            dynamic_contact_hessian,
            1.0,
            body_q,
            body_mass,
            body_inertia,
        ],
        outputs=[step_scale],
        device=device,
    )
    np.testing.assert_array_equal(step_scale.numpy(), [1.0])


def _structural_kkt_majorizes_only_represented_dynamic_contact(test, device):
    """Use H_static+2H_dynamic without doubling unrelated static support."""
    body_ids = wp.array([0], dtype=wp.int32, device=device)
    pose = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
    body_mass = wp.array([1.0], dtype=float, device=device)
    body_inv_mass = wp.array([1.0], dtype=float, device=device)
    body_inertia = wp.array([wp.mat33(1.0)], dtype=wp.mat33, device=device)
    body_com = wp.zeros(1, dtype=wp.vec3, device=device)
    contact_hessian_ll = wp.array(
        [wp.mat33(3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
        dtype=wp.mat33,
        device=device,
    )
    contact_hessian_al = wp.zeros(1, dtype=wp.mat33, device=device)
    contact_hessian_aa = wp.zeros(1, dtype=wp.mat33, device=device)
    contact_force = wp.zeros(1, dtype=wp.vec3, device=device)
    dynamic_host = np.zeros((1, 6, 6), dtype=np.float32)
    dynamic_host[0, 0, 0] = 1.0
    dynamic_hessian = wp.array(dynamic_host, dtype=wp.spatial_matrix, device=device)
    matrix = wp.empty(1, dtype=wp.spatial_matrix, device=device)
    rhs = wp.empty(1, dtype=wp.spatial_vector, device=device)

    wp.launch(
        build_body_surrogate,
        1,
        inputs=[
            body_ids,
            1.0,
            pose,
            pose,
            wp.zeros(pose.shape[0], dtype=wp.vec3, device=device),
            wp.zeros(pose.shape[0], dtype=wp.vec3, device=device),
            0,
            body_mass,
            body_inv_mass,
            body_inertia,
            body_com,
            contact_hessian_ll,
            contact_hessian_al,
            contact_hessian_aa,
            contact_force,
            contact_force,
            dynamic_hessian,
            1,
            0,
            wp.zeros(2, dtype=wp.int32, device=device),
            wp.zeros(0, dtype=wp.mat33, device=device),
            wp.zeros(1, dtype=wp.int32, device=device),
        ],
        outputs=[matrix, rhs],
        device=device,
    )
    value = matrix.numpy()[0]
    # Unit inertia + H_total(3) + one extra represented-pair H(1).
    test.assertAlmostEqual(float(value[0, 0]), 5.0, places=5)
    test.assertAlmostEqual(float(value[1, 1]), 1.0, places=5)


def _structural_kkt_bounds_material_frame_rod_stress(test, device):
    """Bound rod stress without resisting common translation or changing forces."""
    rng = np.random.default_rng(9183)
    poses_host = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.001, 0.002, 0.0005, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    com_host = np.array([[0.0002, -0.0001, 0.0], [0.0, 0.0001, -0.0002]], dtype=np.float32)
    anchor_host = np.array([[0.0002, 0.0, -0.0005, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    rp = poses_host[1, :3].astype(np.float64) + anchor_host[0, :3] - com_host[0]
    rc = anchor_host[0, :3].astype(np.float64) - com_host[1]
    length = np.linalg.norm(rp) + np.linalg.norm(rc)
    ids = wp.array([0], dtype=int, device=device)
    kind = wp.array([int(newton.JointType.ROD)], dtype=int, device=device)
    child = wp.array([1], dtype=int, device=device)
    anchors = wp.array(anchor_host, dtype=wp.transform, device=device)
    poses = wp.array(poses_host, dtype=wp.transform, device=device)
    com = wp.array(com_host, dtype=wp.vec3, device=device)
    inverse_mass = wp.ones(2, dtype=float, device=device)
    slots = wp.array([0, 1], dtype=int, device=device)
    for mask in (np.ones(3), np.array([0.0, 0.0, 1.0])):
        with test.subTest(mask=mask):
            reaction = np.array([2.0, -3.0, 5.0]) * mask
            parent_jacobian = np.zeros((6, 6), dtype=np.float32)
            child_jacobian = np.zeros((6, 6), dtype=np.float32)
            parent_jacobian[:3, :3] = -np.diag(mask)
            parent_jacobian[:3, 3:] = np.diag(mask) @ np.asarray(wp.skew(wp.vec3(rp))).reshape(3, 3)
            child_jacobian[:3, :3] = np.diag(mask)
            child_jacobian[:3, 3:] = -np.diag(mask) @ np.asarray(wp.skew(wp.vec3(rc))).reshape(3, 3)
            before_jacobian = np.hstack([parent_jacobian, child_jacobian]).astype(np.float64)
            before_residual = np.r_[reaction, np.zeros(3)]
            jp = wp.array([parent_jacobian], dtype=wp.spatial_matrix, device=device)
            jc = wp.array([child_jacobian], dtype=wp.spatial_matrix, device=device)
            C = wp.array([np.eye(6)], dtype=wp.spatial_matrix, device=device)
            c = wp.array([before_residual], dtype=wp.spatial_vector, device=device)
            row_stress = wp.zeros(2, dtype=wp.mat33, device=device)
            inputs = [
                wp.array([wp.vec2i(0, 1)], dtype=wp.vec2i, device=device),
                ids,
                wp.array([0], dtype=wp.int32, device=device),
                kind,
                ids,
                child,
                anchors,
                anchors,
                poses,
                com,
                inverse_mass,
                slots,
                jp,
                jc,
                C,
                c,
            ]
            if device.is_cuda:
                with wp.ScopedCapture(device=device) as capture:
                    wp.launch(
                        add_joint_stress_majorizer,
                        1,
                        inputs=inputs,
                        outputs=[row_stress],
                        device=device,
                    )
                wp.capture_launch(capture.graph)
            else:
                wp.launch(
                    add_joint_stress_majorizer,
                    1,
                    inputs=inputs,
                    outputs=[row_stress],
                    device=device,
                )
            value = np.zeros((2, 6, 6))
            value[:, 3:, 3:] = row_stress.numpy()
            jacobian = np.hstack([jp.numpy()[0], jc.numpy()[0]]).astype(np.float64)
            stiffness = np.linalg.inv(C.numpy()[0].astype(np.float64))
            np.testing.assert_allclose(
                jacobian.T @ stiffness @ c.numpy()[0],
                before_jacobian.T @ before_residual,
                rtol=2e-6,
                atol=1e-7,
            )
            bound_matrix = jacobian.T @ stiffness @ jacobian - before_jacobian.T @ before_jacobian
            bound_matrix[:6, :6] += value[0]
            bound_matrix[6:, 6:] += value[1]
            # A common translation changes no relative strain, even when only
            # stretch is enabled. The majorizer must not introduce a world pin.
            for axis in range(3):
                translation = np.zeros(12)
                translation[axis] = translation[axis + 6] = 1.0
                np.testing.assert_allclose(bound_matrix @ translation, 0.0, atol=1e-7)
            # Material-frame stress couples rotation only to motion transverse
            # to the force. Do not damp axial convergence with a fictitious spring.
            axial = np.zeros(12)
            axial[6:9] = reaction / np.linalg.norm(reaction)
            test.assertLess(abs(axial @ bound_matrix @ axial), 1.0e-3)
            for _ in range(128):
                direction = rng.normal(size=(2, 6))
                direction[:, :3] *= length
                vp, vc = direction[:, :3]
                omega_p, omega_c = direction[:, 3:]
                stress = reaction @ (
                    np.cross(omega_p, np.cross(omega_p, rp))
                    - 2 * np.cross(omega_p, vc - vp)
                    - 2 * np.cross(omega_p, np.cross(omega_c, rc))
                    + np.cross(omega_c, np.cross(omega_c, rc))
                )
                bound = direction.ravel() @ bound_matrix @ direction.ravel()
                test.assertLessEqual(abs(stress), bound + 1e-7)
            c.zero_()
            row_stress.zero_()
            before = (jp.numpy(), jc.numpy(), C.numpy())
            wp.launch(
                add_joint_stress_majorizer,
                1,
                inputs=inputs,
                outputs=[row_stress],
                device=device,
            )
            np.testing.assert_array_equal(row_stress.numpy(), 0.0)
            for expected, actual in zip(before, (jp.numpy(), jc.numpy(), C.numpy()), strict=True):
                np.testing.assert_array_equal(actual, expected)

            # Rod rows are diagonal before whitening, including anisotropic materials.
            # Check force preservation against a float64 diagonal solve, not the old inverse.
            for diagonal in ([1.0e-6, 0.03, 1.0e3, 1, 1, 1], [0.2, 7, 1.0e-4, 1, 1, 1]):
                original_compliance = np.diag(diagonal).astype(np.float32)
                original_residual = (np.asarray(diagonal) * before_residual).astype(np.float32)
                jp.assign([parent_jacobian])
                jc.assign([child_jacobian])
                C.assign([original_compliance])
                c.assign([original_residual])
                wp.launch(
                    add_joint_stress_majorizer,
                    1,
                    inputs=inputs,
                    outputs=[row_stress],
                    device=device,
                )
                after_jacobian = np.hstack([jp.numpy()[0], jc.numpy()[0]]).astype(np.float64)
                expected_force = before_jacobian.T @ (
                    original_residual.astype(np.float64) / np.diag(original_compliance).astype(np.float64)
                )
                actual_force = after_jacobian.T @ np.linalg.solve(C.numpy()[0].astype(np.float64), c.numpy()[0])
                np.testing.assert_allclose(actual_force, expected_force, rtol=2.0e-6, atol=1.0e-7)


def _structural_kkt_limit_requests_follow_dynamic_endpoint(test, device):
    """Route limited prescribed-child rows to their represented parent's island."""
    for joint_type in (newton.JointType.PRISMATIC, newton.JointType.REVOLUTE):
        for kinematic in (False, True):
            with test.subTest(joint_type=joint_type, kinematic=kinematic):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                bodies = []
                for position in ((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 1.0, 0.0), (0.2, 1.0, 0.0)):
                    body = builder.add_link(xform=wp.transform(wp.vec3(*position), wp.quat_identity()), mass=1.0)
                    builder.add_shape_box(body, hx=0.04, hy=0.03, hz=0.02)
                    bodies.append(body)
                parent, child, other_parent, other_child = bodies
                if not kinematic:
                    _pin_body(builder, child)
                add_joint = (
                    builder.add_joint_prismatic
                    if joint_type == newton.JointType.PRISMATIC
                    else builder.add_joint_revolute
                )
                parent_xform = wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity())
                limited_joint = add_joint(
                    parent,
                    child,
                    parent_xform=parent_xform,
                    child_xform=wp.transform_identity(),
                    axis=newton.Axis.X if joint_type == newton.JointType.PRISMATIC else newton.Axis.Z,
                    limit_lower=-0.01,
                    limit_upper=0.01,
                    limit_ke=1.0e5,
                    limit_kd=0.0,
                )
                other_joint = builder.add_joint_fixed(
                    other_parent, other_child, parent_xform=parent_xform, child_xform=wp.transform_identity()
                )
                builder.add_articulation([limited_joint])
                builder.add_articulation([other_joint])
                builder.color(balance_colors=False)
                model = builder.finalize(device=device)
                if kinematic:
                    flags = model.body_flags.numpy().copy()
                    flags[child] |= int(newton.BodyFlags.KINEMATIC)
                    model.body_flags.assign(flags)
                solver = newton.solvers.SolverVBD(
                    model, iterations=1, rigid_compliant_alm=True, rigid_joint_global_iterations=1
                )
                backend = solver._structural_graph_kkt
                test.assertIsNotNone(backend)
                test.assertEqual(backend.island_count, 2)
                slots = backend.body_slot_by_id.numpy()
                islands = backend.graph_body_island.numpy()
                test.assertEqual(slots[child], -1)
                requested_island = islands[slots[parent]]
                test.assertNotEqual(requested_island, islands[-1])
                expected = np.zeros(backend.island_count, dtype=np.int32)
                expected[requested_island] = 1
                solver.joint_rho.fill_(1.0e6)
                solver.joint_drive_limit_support.fill_(1.0e6)
                wp.copy(solver.body_q_prev, model.body_q)
                for loaded in (True, False):
                    solver.joint_limit_lambda.fill_(1.0 if loaded else 0.0)
                    backend.limit_search_needed.zero_()
                    backend.limit_search_islands.zero_()
                    for bucket in backend.buckets:
                        _linearize_bucket(model, solver, bucket, model.body_q, 1.0e-3, track_limits=True)
                    np.testing.assert_array_equal(
                        backend.limit_search_islands.numpy(), expected if loaded else np.zeros_like(expected)
                    )
                    test.assertEqual(int(backend.limit_search_needed.numpy()[0]), int(loaded))


def _structural_kkt_rebuild_accounts_for_global_storage(test, device):
    """The solver must not allocate extra global payload outside backend admission."""
    chain, _, _ = _build_prismatic_material_chain(device, kind="limit", link_count=4)
    mixed, _ = _build_mixed_fixed_topologies(device)
    for model in (chain, mixed):
        with test.subTest(body_count=model.body_count):
            solver = newton.solvers.SolverVBD(
                model, iterations=1, rigid_compliant_alm=True, rigid_joint_global_iterations=1
            )
            backend, allocated = _build_structural_backend_with_intercepted_bytes(model, solver=solver)
            test.assertIsNotNone(backend)
            test.assertEqual(allocated, backend._estimated_bytes)
            test.assertEqual(allocated, solver._structural_graph_kkt_estimated_bytes)
            parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
            inverse_mass = solver.body_inv_mass_effective.numpy()
            dynamic_side = np.zeros(model.joint_count, dtype=bool)
            for endpoint in (parent, child):
                valid = endpoint >= 0
                dynamic_side[valid] |= inverse_mass[endpoint[valid]] > 0.0
            expected = np.isin(np.arange(model.joint_count), backend.graph_joint_ids.numpy()) & dynamic_side
            np.testing.assert_array_equal(np.any(backend.joint_stress_slot.numpy() >= 0, axis=1), expected)
            with (
                mock.patch.object(rigid_vbd_kkt, "_structural_payload_budget", return_value=0),
                test.assertWarnsRegex(RuntimeWarning, "payload budget"),
            ):
                rejected, allocated = _build_structural_backend_with_intercepted_bytes(model, solver=solver)
            test.assertIsNone(rejected)
            test.assertEqual(allocated, 0)
            test.assertEqual(solver._structural_graph_kkt_estimated_bytes, 0)
            solver.rigid_joint_global_iterations = 0
            inactive, allocated = _build_structural_backend_with_intercepted_bytes(model, solver=solver)
            test.assertIsNone(inactive)
            test.assertEqual(allocated, 0)


def _structural_kkt_limit_capture_fallback_preserves_state(test, device):
    """Fixed CUDA capture needs no readback and preserves unrequested islands."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for island in range(2):
        origin = wp.vec3(0.0, float(island), 0.0)
        body = builder.add_link(xform=wp.transform(origin, wp.quat_identity()), mass=1.0)
        builder.add_shape_box(body, hx=0.03, hy=0.015, hz=0.015)
        width = 0.01 if island == 0 else 1000.0
        joint = builder.add_joint_prismatic(
            parent=-1,
            child=body,
            parent_xform=wp.transform(origin, wp.quat_identity()),
            child_xform=wp.transform_identity(),
            axis=newton.Axis.X,
            limit_lower=-width,
            limit_upper=width,
            limit_ke=1.0e4,
            limit_kd=100.0,
        )
        builder.add_articulation([joint])
    builder.color(balance_colors=False)
    model = builder.finalize(device=device)
    initial = model.body_q.numpy().copy()
    dt = 1.0 / 60.0

    def simulate(loaded, mode):
        solver = newton.solvers.SolverVBD(
            model, iterations=2, rigid_compliant_alm=True, rigid_joint_global_iterations=1
        )
        backend = solver._structural_graph_kkt
        test.assertEqual(backend.island_count, 2)
        test.assertTrue(backend.has_limit_rows)
        a, b = model.state(), model.state()
        pose = initial.copy()
        pose[0, 0] = 0.02 if loaded else 0.0
        a.body_q.assign(pose)
        control = model.control()
        force = np.zeros((model.body_count, 6), dtype=np.float32)
        force[0, 0] = 30.0 if loaded else 0.0
        force[1, 0] = 7.0
        a.body_f.assign(force)
        solver.step(a, b, control, None, dt)
        a, b = b, a
        records, graphs = [], {}
        for step in range(12):
            a.clear_forces()
            a.body_f.assign(force)
            if mode == "eager":
                solver.step(a, b, control, None, dt)
            else:
                parity = step % 2
                if parity not in graphs:
                    with ExitStack() as patches:
                        if mode == "fixed":
                            patches.enter_context(
                                mock.patch.object(wp, "is_conditional_graph_supported", return_value=False)
                            )
                        patches.enter_context(
                            mock.patch.object(
                                wp.array, "numpy", side_effect=AssertionError("host readback during capture")
                            )
                        )
                        with wp.ScopedCapture(device) as capture:
                            solver.step(a, b, control, None, dt)
                    graphs[parity] = capture.graph
                wp.capture_launch(graphs[parity])
            a, b = b, a
            records.append(
                tuple(
                    array.numpy().copy()
                    for array in (
                        a.body_q,
                        a.body_qd,
                        solver.body_q_prev,
                        solver.joint_lambda_lin,
                        solver.joint_lambda_ang,
                        solver.joint_drive_lambda,
                        solver.joint_limit_lambda,
                        solver.joint_rho,
                        solver.joint_drive_limit_support,
                    )
                )
            )
        test.assertGreater(abs(records[-1][0][1, 0] - records[0][0][1, 0]), 0.0)
        test.assertEqual(bool(np.any(records[0][6] != 0.0)), loaded)
        return records

    for loaded in (False, True):
        reference = simulate(loaded, "eager")
        for mode in ("captured", "fixed"):
            with test.subTest(loaded=loaded, mode=mode):
                for actual, expected in zip(simulate(loaded, mode), reference, strict=True):
                    for value, expected_val in zip(actual, expected, strict=True):
                        test.assertTrue(np.isfinite(value).all())
                        np.testing.assert_array_equal(value, expected_val)


def _structural_kkt_inactive_limit_adds_curvature_without_force(test, device):
    """An open limit adds no curvature when unloaded, and retains curvature without ghost force when dual-loaded."""
    model, _, _, _ = _build_joint_pair(device, newton.JointType.REVOLUTE, finite_limit=True)
    solver = newton.solvers.SolverVBD(model, iterations=1, rigid_compliant_alm=True, rigid_joint_global_iterations=1)
    _simulate(model, solver, 1, 1.0 / 600.0)
    solver.joint_drive_limit_support.fill_(100.0)
    solver.joint_drive_lambda.zero_()
    solver.joint_limit_lambda.zero_()
    bucket = solver._structural_graph_kkt.buckets[0]
    _linearize_bucket(model, solver, bucket, model.body_q, 1.0 / 600.0)
    np.testing.assert_array_equal(bucket.jacobian_child.numpy()[0, 5, 5], 0.0)
    np.testing.assert_array_equal(bucket.residual.numpy()[0, 5], 0.0)
    solver.joint_limit_lambda.fill_(1.0)
    _linearize_bucket(model, solver, bucket, model.body_q, 1.0 / 600.0)
    jacobian = bucket.jacobian_child.numpy()[0]
    test.assertGreater(abs(float(jacobian[5, 5])), 0.99)
    np.testing.assert_array_equal(bucket.residual.numpy()[0, 5], 0.0)
    test.assertGreater(float(bucket.compliance.numpy()[0, 5, 5]), 0.0)


def _structural_kkt_open_contact_activates_on_penetration(test, device):
    """An open candidate is inactive; refreshing a penetrating trial activates it."""
    model, pipeline, contacts, solver, bodies, _, _ = _build_grounded_chain(
        device, global_iterations=1, iterations=1, segments=4, contact_kd=0.0
    )
    model.set_gravity((0.0, 0.0, 0.0))
    state = model.state()
    pipeline.collide(state, contacts)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
    solver.step(state, model.state(), model.control(), contacts, 1.0 / 600.0)
    solver.body_body_contact_lambda.zero_()
    solver.body_body_contact_C0.zero_()
    solver._reset_structural_solve_state()
    solver._refresh_structural_contact_objective(state, contacts, 1.0 / 600.0)
    np.testing.assert_array_equal(solver.body_hessian_ll.numpy()[bodies], 0.0)
    np.testing.assert_array_equal(solver.body_hessian_al.numpy()[bodies], 0.0)
    np.testing.assert_array_equal(solver.body_hessian_aa.numpy()[bodies], 0.0)
    np.testing.assert_array_equal(solver.body_forces.numpy()[bodies], 0.0)
    np.testing.assert_array_equal(solver.body_torques.numpy()[bodies], 0.0)
    np.testing.assert_array_equal(solver._structural_graph_kkt.island_contact_state.numpy(), 1)

    # Reuse the same collision candidates, as the global line search does.
    # Move through the initial 1 mm clearance without changing contact history.
    pose = state.body_q.numpy()
    pose[bodies, 2] -= 0.002
    state.body_q.assign(pose)
    solver._reset_structural_solve_state()
    solver._refresh_structural_contact_objective(state, contacts, 1.0 / 600.0)
    test.assertGreater(float(np.max(solver.body_hessian_ll.numpy()[bodies, 2, 2])), 0.0)
    test.assertGreater(float(np.sum(solver.body_forces.numpy()[bodies, 2])), 0.0)
    np.testing.assert_array_equal(solver.body_body_contact_lambda.numpy(), 0.0)


def _structural_kkt_inactive_trees_do_not_apply_free_body_corrections(test, device):
    """Suppress tree/closure scatter when no incident structural row is active."""
    for closed in (False, True):
        model = _build_replicated_fixed_topology(device, closed=closed, worlds=1)
        solver = newton.solvers.SolverVBD(
            model,
            iterations=2,
            rigid_compliant_alm=True,
            rigid_joint_global_iterations=1,
            rigid_joint_linear_ke=0.0,
            rigid_joint_angular_ke=0.0,
            rigid_joint_linear_kd=0.0,
            rigid_joint_angular_kd=0.0,
        )
        solver.joint_material_k.zero_()
        solver.joint_penalty_kd.zero_()
        captured = {}
        global_solve = solver._solve_structural_graph_kkt

        def capture_global(*args, _solve=global_solve, _captured=captured, _solver=solver, **kwargs):
            _solve(*args, **kwargs)
            _captured["correction"] = _solver._structural_graph_kkt.body_correction.numpy().copy()

        solver._solve_structural_graph_kkt = capture_global
        state_in = model.state()
        state_out = model.state()
        state_in.clear_forces()
        wp.launch(
            _apply_body_force,
            1,
            inputs=[model.body_count - 1, wp.vec3(3.0, 0.0, 0.0)],
            outputs=[state_in.body_f],
            device=device,
        )
        solver.step(state_in, state_out, model.control(), None, 1.0 / 600.0)
        with test.subTest(closed=closed):
            test.assertIn("correction", captured)
            np.testing.assert_array_equal(captured["correction"], 0.0)


def _structural_kkt_reuses_compact_scratch(test, device):
    """Reuse compact graph-local workspaces across non-overlapping lifetimes."""
    builder = newton.ModelBuilder()
    for index in range(8):
        builder.add_body(
            xform=wp.transform(wp.vec3(float(index), 0.0, 2.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
    rod = newton.Rod.create_straight(
        start=wp.vec3(0.0, 0.0, 1.0),
        direction=wp.vec3(1.0, 0.0, 0.0),
        length=0.24,
        segment_count=8,
        radius=0.01,
    )
    bodies, _ = builder.add_rod(
        rod=rod,
        cfg=builder.default_shape_cfg.copy(),
        stretch_stiffness=1.0e9,
        bend_stiffness=1.0e4,
        wrap_in_articulation=True,
        body_frame_origin="com",
    )
    _pin_body(builder, int(bodies[0]))
    builder.color(balance_colors=False)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    backend = solver._structural_graph_kkt
    test.assertIsNotNone(backend)
    test.assertLess(backend.graph_body_count, model.body_count)
    test.assertEqual(backend.body_matrix.shape[0], backend.graph_body_count)
    test.assertEqual(backend.graph_body_island.shape[0], backend.graph_body_count)
    test.assertFalse(hasattr(solver, "graph_body_island"))
    test.assertFalse(hasattr(solver, "island_all_static_sliding"))
    test.assertIs(backend.body_inverse, backend.body_matrix)
    test.assertIs(backend.body_free, backend.body_rhs)
    test.assertIs(backend.body_scale, backend.body_correction)
    # Contact curvature is read again after factorization to relax the global
    # correction. It must survive both body assembly and path inversion.
    _simulate(model, solver, 1, 1.0 / 600.0)
    np.testing.assert_array_equal(solver.body_dynamic_contact_hessian.numpy(), 0.0)

    bucket = backend.path_buckets[0]
    test.assertEqual(len(bucket.lower), 1)
    test.assertEqual(len(bucket.diagonal), 1)
    test.assertEqual(len(bucket.upper), 1)
    test.assertEqual(len(bucket.rhs), 1)
    test.assertIs(bucket.compliance, bucket.diagonal[0])
    test.assertIs(bucket.residual, bucket.rhs[0])
    test.assertIs(bucket.solution, bucket.rhs)


def _structural_kkt_refreshes_notified_joint_enable_topology(test, device):
    """Rebuild global topology after notified joint enable changes."""
    model, _, _, _ = _build_joint_pair(device, newton.JointType.ROD)
    model.joint_enabled.assign([False])
    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    test.assertIsNone(solver._structural_graph_kkt)

    model.joint_enabled.assign([True])
    solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    test.assertIsNotNone(solver._structural_graph_kkt)
    test.assertEqual(solver._structural_graph_kkt.joint_count, 1)
    state = _simulate(model, solver, 2, 1.0 / 600.0)
    test.assertTrue(np.isfinite(state.body_q.numpy()).all())

    model.joint_enabled.assign([False])
    solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    test.assertIsNone(solver._structural_graph_kkt)
    local_calls = 0
    local_solve = solver._solve_rigid_body_iteration

    def record_local(*args, **kwargs):
        nonlocal local_calls
        local_calls += 1
        return local_solve(*args, **kwargs)

    solver._solve_rigid_body_iteration = record_local
    _simulate(model, solver, 1, 1.0 / 600.0)
    test.assertEqual(local_calls, solver.iterations)


def _structural_kkt_uses_effective_kinematic_mass(test, device):
    """Build topology from effective rather than authored body mass."""
    model, parent, _, _ = _build_joint_pair(device, newton.JointType.FIXED, kinematic_parent=True)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    backend = solver._structural_graph_kkt
    test.assertIsNotNone(backend)
    test.assertNotIn(parent, backend.tree_buckets[0].body_ids_host)
    state = _simulate(model, solver, 2, 1.0 / 600.0)
    test.assertTrue(np.isfinite(state.body_q.numpy()).all())

    flags = model.body_flags.numpy()
    flags[parent] &= ~int(newton.BodyFlags.KINEMATIC)
    model.body_flags.assign(flags)
    solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)
    test.assertIn(parent, solver._structural_graph_kkt.tree_buckets[0].body_ids_host)


def _cable_kkt_preserves_dahl_state_under_capture(test, device):
    """Preserve Dahl state while replaying captured global cable solves."""
    model, _, joints = _build_chain(device, segments=16, dahl=True)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=4,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    backend = solver._structural_graph_kkt
    test.assertIsNotNone(backend)
    test.assertEqual(len(backend.path_buckets), 1)

    state_in = model.state()
    state_out = model.state()
    control = model.control()
    dt = 1.0 / 600.0
    state_in.clear_forces()
    solver.step(state_in, state_out, control, None, dt)
    state_in, state_out = state_out, state_in
    with wp.ScopedCapture(device) as capture:
        state_in.clear_forces()
        solver.step(state_in, state_out, control, None, dt)
        state_out.clear_forces()
        solver.step(state_out, state_in, control, None, dt)
    for _ in range(8):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    sigma = solver.joint_sigma_prev.numpy()[joints]
    tangent = solver.joint_C_fric.numpy()[joints]
    test.assertTrue(np.isfinite(state_in.body_q.numpy()).all())
    test.assertTrue(np.isfinite(sigma).all())
    test.assertGreater(float(np.max(np.abs(sigma))), 1.0e-6)
    test.assertGreater(float(np.max(tangent)), 0.0)


def _cable_kkt_reduces_long_path_error(test, device):
    """Reduce closure error on a long cable path with one global pass."""
    dt = 1.0 / 600.0
    local_model, _, local_joints = _build_chain(device, segments=32)
    local_solver = newton.solvers.SolverVBD(local_model, iterations=1, rigid_compliant_alm=True)
    local_state = _simulate(local_model, local_solver, 40, dt)
    local_gap = _max_joint_gap(local_model, local_state.body_q.numpy(), local_joints)

    kkt_model, _, kkt_joints = _build_chain(device, segments=32)
    kkt_solver = newton.solvers.SolverVBD(
        kkt_model,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    kkt_state = _simulate(kkt_model, kkt_solver, 40, dt)
    kkt_gap = _max_joint_gap(kkt_model, kkt_state.body_q.numpy(), kkt_joints)

    test.assertTrue(np.isfinite(kkt_state.body_q.numpy()).all())
    test.assertLess(
        kkt_gap,
        0.05 * local_gap,
        f"KKT gap {kkt_gap:.3e} did not improve local gap {local_gap:.3e}",
    )


def _cable_kkt_near_hard_capture_is_finite(test, device):
    """Keep near-rigid captured cable solves finite."""
    # Use a very stiff but numerically representable compliant material rather
    # than treating float32's largest value as an exact-constraint sentinel.
    model, _, _ = _build_chain(device, segments=16, stiffness=1.0e12)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=2,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=2,
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    dt = 1.0 / 600.0

    with wp.ScopedCapture(device) as capture:
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_1.clear_forces()
        solver.step(state_1, state_0, control, None, dt)
    for _ in range(4):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    test.assertTrue(np.isfinite(state_0.body_q.numpy()).all())
    test.assertTrue(np.isfinite(state_0.body_qd.numpy()).all())
    test.assertTrue(np.isfinite(solver.joint_lambda_lin.numpy()).all())
    test.assertTrue(np.isfinite(solver.joint_lambda_ang.numpy()).all())


def _cable_kkt_near_hard_closed_cycle_is_finite_under_capture(test, device):
    """Keep the anisotropic near-hard cyclic response solve finite under capture."""
    model, _, joints = _build_loop_with_branch(
        device,
        ring_segments=32,
        stiffness=1.0e12,
        bend_stiffness=1.0e4,
    )
    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    closed_tree = solver._structural_graph_kkt.closed_tree_buckets[0]
    test.assertEqual(closed_tree.use_paired_backbone, device.is_cuda)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    dt = 1.0 / 600.0
    with wp.ScopedCapture(device) as capture:
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_1.clear_forces()
        solver.step(state_1, state_0, control, None, dt)
    for _ in range(40):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    body_q = state_0.body_q.numpy()
    test.assertTrue(np.isfinite(body_q).all())
    test.assertTrue(np.isfinite(state_0.body_qd.numpy()).all())
    test.assertLess(_max_joint_gap(model, body_q, joints), 1.0e-5)


def _cable_kkt_closed_cycle_bounds_multiplier_history(test, device):
    """Bound multiplier-history growth on a redundant captured cycle."""
    model, _, joints = _build_loop_with_branch(
        device,
        ring_segments=8,
        stiffness=1.0e9,
        bend_stiffness=1.0e9,
        add_chord=True,
    )
    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    test.assertEqual(
        solver._structural_graph_kkt.closed_tree_buckets[0].use_paired_backbone,
        device.is_cuda,
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    dt = 1.0 / 600.0
    with wp.ScopedCapture(device) as capture:
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_1.clear_forces()
        solver.step(state_1, state_0, control, None, dt)

    linear_norms = []
    angular_norms = []
    maximum_velocity_component = 0.0
    for _ in range(30):
        for _ in range(50):
            wp.capture_launch(capture.graph)
        linear = solver.joint_lambda_lin.numpy()[joints].astype(np.float64)
        angular = solver.joint_lambda_ang.numpy()[joints].astype(np.float64)
        body_qd = state_0.body_qd.numpy().astype(np.float64)
        test.assertTrue(np.isfinite(linear).all())
        test.assertTrue(np.isfinite(angular).all())
        test.assertTrue(np.isfinite(body_qd).all())
        linear_norms.append(float(np.linalg.norm(linear)))
        angular_norms.append(float(np.linalg.norm(angular)))
        maximum_velocity_component = max(maximum_velocity_component, float(np.max(np.abs(body_qd))))

    # Compare time windows within one material system, avoiding a mixed-unit
    # absolute threshold while still catching a runaway persistent mode.
    message = f"linear={linear_norms}, angular={angular_norms}"
    test.assertGreater(max(linear_norms[:15]), 0.0, message)
    test.assertGreater(max(angular_norms[:15]), 0.0, message)
    test.assertLess(max(linear_norms[15:]), 3.0 * max(linear_norms[:15]), message)
    test.assertLess(max(angular_norms[15:]), 3.0 * max(angular_norms[:15]), message)
    # A median pairwise log slope is insensitive to one multiplier spike. The
    # stationary trace of this redundant cycle spreads over a factor of 5 to 10
    # with either pose update, so ten samples misread it as growth about one
    # time in five. Thirty samples over 2,900 frames, held to the same rate of
    # about 4.5e-4 per frame (1.5 per 900 frames), reject that noise and still
    # catch growth at twice the rate.
    growth_bound = 1.5 ** (29.0 / 9.0)
    test.assertLess(_robust_history_growth_factor(linear_norms), growth_bound, message)
    test.assertLess(_robust_history_growth_factor(angular_norms), growth_bound, message)

    body_q = state_0.body_q.numpy()
    test.assertTrue(np.isfinite(body_q).all())
    test.assertLess(maximum_velocity_component, 1.0e-2)
    test.assertLess(_max_joint_gap(model, body_q, joints), 1.0e-5)


def _structural_kkt_retains_interior_supported_branches(test, device):
    """Prescribed interior supports must not disable the branched global path."""
    for kinematic in (False, True):
        with test.subTest(kinematic=kinematic):
            support_indices = (1, 5, 9)
            model, bodies, joints = _build_y_tree(
                device,
                segments_per_branch=4,
                stiffness=1.0e7,
                interior_pins=() if kinematic else support_indices,
            )
            supports = bodies[list(support_indices)]
            if kinematic:
                flags = model.body_flags.numpy()
                flags[supports] |= int(newton.BodyFlags.KINEMATIC)
                model.body_flags.assign(flags)
            solver = newton.solvers.SolverVBD(
                model,
                iterations=2,
                rigid_compliant_alm=True,
                rigid_joint_global_iterations=1,
            )
            backend = solver._structural_graph_kkt
            test.assertIsNotNone(backend)
            np.testing.assert_array_equal(np.sort(backend.graph_joint_ids.numpy()), np.sort(joints))
            test.assertEqual(len(np.unique(backend.graph_body_ids.numpy())), backend.graph_body_count)
            test.assertFalse(set(supports) & set(backend.graph_body_ids.numpy()))
            initial = model.body_q.numpy()
            state_in, state_out = model.state(), model.state()
            control = model.control()
            dt = 1.0 / 600.0

            def pair(state_in=state_in, state_out=state_out, solver=solver, control=control, dt=dt):
                state_in.clear_forces()
                solver.step(state_in, state_out, control, None, dt)
                state_out.clear_forces()
                solver.step(state_out, state_in, control, None, dt)

            pair()
            if device.is_cuda:
                with wp.ScopedCapture(device) as capture:
                    pair()
                for _ in range(4):
                    wp.capture_launch(capture.graph)
            else:
                for _ in range(4):
                    pair()
            wp.synchronize_device(device)
            pose = state_in.body_q.numpy()
            test.assertTrue(np.isfinite(pose).all())
            fixed = solver.body_inv_mass_effective.numpy() == 0
            np.testing.assert_array_equal(pose[fixed], initial[fixed])
            test.assertLess(_max_joint_gap(model, pose, joints), 1.0e-4)
            if kinematic:
                flags[supports] &= ~int(newton.BodyFlags.KINEMATIC)
                model.body_flags.assign(flags)
                solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)
                test.assertTrue(set(supports) <= set(solver._structural_graph_kkt.graph_body_ids.numpy()))


def _structural_kkt_prescribed_split_matches_float64_oracle(test, device):
    """Check independent boundary-row assembly after a prescribed-body cut."""
    model, _, _ = _build_y_tree(device, segments_per_branch=4, stiffness=1.0e7, interior_pins=(0,))
    metrics = _path_float64_oracle_metrics(device, 1.0e7, model=model)
    message = str(metrics)
    test.assertTrue(all(np.isfinite(value) for value in metrics.values()), message)
    test.assertLess(metrics["formula_matrix_error"], 1.0e-5, message)
    test.assertLess(metrics["formula_rhs_error"], 1.0e-5, message)
    test.assertLess(metrics["residual"], 1.0e-2, message)
    test.assertLess(metrics["multiplier_error"], 1.0e-3, message)
    test.assertLess(metrics["correction_error"], 1.0e-3, message)


def _cable_kkt_tree_handles_stiff_y_junction(test, device):
    """Resolve a stiff branched cable tree with the global solve."""
    dt = 1.0 / 600.0
    local_model, _, local_joints = _build_y_tree(device)
    local_solver = newton.solvers.SolverVBD(local_model, iterations=2, rigid_compliant_alm=True)
    local_state = _simulate(local_model, local_solver, 20, dt)
    local_gap = _max_joint_gap(local_model, local_state.body_q.numpy(), local_joints)

    kkt_model, _, kkt_joints = _build_y_tree(device)
    kkt_solver = newton.solvers.SolverVBD(
        kkt_model,
        iterations=2,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=2,
    )
    backend = kkt_solver._structural_graph_kkt
    test.assertIsNotNone(backend)
    test.assertEqual(len(backend.path_buckets), 0)
    test.assertEqual(len(backend.tree_buckets), 1)
    tree = backend.tree_buckets[0]
    test.assertEqual(tree.matrix_workspace.shape[0], max(2 * tree.row_count, tree.node_count))
    test.assertEqual(tree.diagonal.ptr, tree.jacobian_parent.ptr)
    test.assertEqual(tree.body_slots.ptr, tree.node_body.ptr)
    kkt_state = _simulate(kkt_model, kkt_solver, 20, dt)
    kkt_gap = _max_joint_gap(kkt_model, kkt_state.body_q.numpy(), kkt_joints)

    test.assertTrue(np.isfinite(kkt_state.body_q.numpy()).all())
    test.assertLess(
        kkt_gap,
        0.05 * local_gap,
        f"Tree KKT gap {kkt_gap:.3e} did not improve {local_gap:.3e}",
    )


def _cable_kkt_closes_stiff_loop_with_one_global_pass(test, device):
    """Reduce closure error on a stiff cyclic cable graph."""
    dt = 1.0 / 600.0
    local_model, _, local_joints = _build_loop_with_branch(device, add_chord=True)
    local_solver = newton.solvers.SolverVBD(local_model, iterations=2, rigid_compliant_alm=True)
    local_state = _simulate(local_model, local_solver, 20, dt)
    local_gap = _max_joint_gap(local_model, local_state.body_q.numpy(), local_joints)

    kkt_model, _, kkt_joints = _build_loop_with_branch(device, add_chord=True)
    kkt_solver = newton.solvers.SolverVBD(
        kkt_model,
        iterations=2,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    backend = kkt_solver._structural_graph_kkt
    test.assertIsNotNone(backend)
    test.assertEqual(len(backend.closed_tree_buckets), 1)
    closed_tree = backend.closed_tree_buckets[0]
    test.assertEqual(closed_tree.closure_count, 2)
    test.assertGreater(closed_tree.backbone_node_count, 0)
    test.assertLess(len(closed_tree.backbone_cr_levels), len(closed_tree.tree.levels))
    test.assertEqual(backend.joint_count, len(kkt_joints))
    tree = closed_tree.tree
    test.assertNotEqual(tree.diagonal.ptr, tree.jacobian_parent.ptr)
    test.assertEqual(tree.diagonal.dtype, wp.spatial_matrixd)
    test.assertEqual(tree.jacobian_parent.dtype, wp.spatial_matrix)
    kkt_state = _simulate(kkt_model, kkt_solver, 20, dt)
    kkt_gap = _max_joint_gap(kkt_model, kkt_state.body_q.numpy(), kkt_joints)

    test.assertTrue(np.isfinite(kkt_state.body_q.numpy()).all())
    test.assertLess(
        kkt_gap,
        0.05 * local_gap,
        f"Closed KKT gap {kkt_gap:.3e} did not improve {local_gap:.3e}",
    )

    state_out = kkt_model.state()
    control = kkt_model.control()
    with wp.ScopedCapture(device) as capture:
        kkt_state.clear_forces()
        kkt_solver.step(kkt_state, state_out, control, None, dt)
        state_out.clear_forces()
        kkt_solver.step(state_out, kkt_state, control, None, dt)
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    test.assertTrue(np.isfinite(kkt_state.body_q.numpy()).all())


def _cable_kkt_preserves_local_ground_sliding(test, device):
    """Accelerate a loaded cable without suppressing its Coulomb sliding mode."""
    local = _simulate_ground_drag(device, 0)
    for g in (1, 2):
        with test.subTest(global_iterations=g):
            result = _simulate_ground_drag(device, g)
            expected = 0.5 * (45.0 / result["mass"] - 0.45 * 9.81) * (200.0 / 600.0) ** 2
            test.assertGreater(result["motion"], 0.85 * expected)
            test.assertLess(result["motion"], 1.1 * expected)
            test.assertLess(result["penetration"], 1.0e-3)
            test.assertLess(result["gap"], 0.5 * local["gap"])


def _cable_kkt_substeps_preserve_sliding(test, device):
    """Resolve sliding consistently as the substep budget increases."""
    for substeps, iterations, g in (
        (2, 5, 1),
        (4, 5, 1),
        (8, 5, 1),
        (4, 5, 2),
        (4, 2, 2),
    ):
        with test.subTest(substeps=substeps, iterations=iterations, global_iterations=g):
            result = _simulate_ground_drag(device, g, substeps=substeps, iterations=iterations)
            expected = 0.5 * (45.0 / result["mass"] - 0.45 * 9.81) * (200.0 / 600.0) ** 2
            test.assertGreater(result["motion"], 0.95 * expected)
            test.assertLess(result["motion"], 1.05 * expected)
            test.assertLess(result["penetration"], 2.0e-4)
            test.assertLess(result["gap"], 1.0e-4)


def _cable_kkt_breaks_away_in_first_loaded_steps(test, device):
    """Start sliding at once under a load far above the friction capacity."""
    result = _simulate_ground_drag(device, 1, substeps=4, frames=41)
    # Four implicit steps from rest travel 10 a h^2. A common shift solved against stick-mode contact
    # curvature is below float32 position resolution; lost, it leaves the chain where it was.
    expected = 10.0 * (45.0 / result["mass"] - 0.45 * 9.81) / 2400.0**2
    test.assertGreater(result["motion"], 0.9 * expected)
    test.assertLess(result["motion"], 1.1 * expected)
    test.assertLess(result["penetration"], 2.0e-4)
    test.assertLess(result["gap"], 1.0e-4)


def _cable_kkt_substeps_preserve_static_friction(test, device):
    """Retain static friction under a load inside the Coulomb cone."""
    for substeps in (2, 8):
        with test.subTest(substeps=substeps):
            result = _simulate_ground_drag(device, 1, substeps=substeps, contact_history=True, force=1.0)
            test.assertLess(1.0, 0.45 * 9.81 * result["mass"])
            test.assertLess(abs(result["motion"]), 1.0e-3)
            test.assertLess(result["penetration"], 1.0e-5)
            test.assertLess(result["gap"], 1.0e-5)


def _structural_kkt_preserves_free_translation(test, device):
    """Internal contact majorizers must not change collective momentum."""
    ids = wp.array([0, 1], dtype=int, device=device)
    islands = wp.zeros(2, dtype=int, device=device)
    enabled = wp.ones(1, dtype=int, device=device)
    scale = wp.array([0.5], dtype=float, device=device)
    poses = np.zeros((2, 7), dtype=np.float32)
    poses[:, 6] = 1.0
    targets = poses.copy()
    targets[:, :3] = [[0.3, -0.2, 0.1], [0.1, 0.2, -0.1]]
    mass = np.array([1.0, 2.0], dtype=np.float32)
    forces = np.array([[100.0, 20.0, -10.0], [-100.0, -20.0, 10.0]], dtype=np.float32)
    raw = np.array([[0.03, 0.02, -0.01, 0, 0, 0], [0.04, -0.01, 0.02, 0, 0, 0]], dtype=np.float32)
    correction = wp.array(raw, dtype=wp.spatial_vector, device=device)
    # Equal self blocks model an internal contact's diagonal majorizer. Its
    # exact pair Hessian annihilates a common translation.
    ghost = np.tile(1.0e6 * np.eye(3, dtype=np.float32), (2, 1, 1))
    dynamic = np.zeros((2, 6, 6), dtype=np.float32)
    dynamic[:, :3, :3] = ghost
    system = wp.zeros(1, dtype=wp.mat44d, device=device)
    wp.launch(
        rigid_vbd_kkt.accumulate_free_translation_system,
        2,
        inputs=[
            ids,
            islands,
            ids,
            enabled,
            enabled,
            scale,
            correction,
            1.0,
            wp.array(poses, dtype=wp.transform, device=device),
            wp.array(targets, dtype=wp.transform, device=device),
            wp.zeros(2, dtype=wp.vec3, device=device),
            wp.zeros(2, dtype=wp.vec3, device=device),
            0,
            wp.zeros(2, dtype=wp.vec3, device=device),
            wp.array(mass, dtype=float, device=device),
            wp.array(1.0 / mass, dtype=float, device=device),
            wp.array(ghost, dtype=wp.mat33, device=device),
            wp.zeros(2, dtype=wp.mat33, device=device),
            wp.array(forces, dtype=wp.vec3, device=device),
            wp.array(dynamic, dtype=wp.spatial_matrix, device=device),
            0,
            None,
            None,
            None,
            None,
            0,
            None,
            None,
        ],
        outputs=[system],
        device=device,
    )
    wp.launch(
        rigid_vbd_kkt.correct_free_translation,
        2,
        inputs=[
            islands,
            enabled,
            enabled,
            scale,
            system,
            correction,
            False,
            False,
            True,
            False,
            None,
            ids,
            0,
            wp.zeros(2, dtype=wp.vec3, device=device),
            wp.zeros(2, dtype=wp.transform, device=device),
        ],
        outputs=[
            None,
            None,
            None,
            None,
            wp.zeros(2, dtype=wp.vec3, device=device),
        ],
        device=device,
    )
    accepted = 0.5 * correction.numpy()[:, :3]
    np.testing.assert_allclose(
        (mass[:, None] * accepted).sum(axis=0),
        (mass[:, None] * targets[:, :3]).sum(axis=0),
        atol=5e-8,
    )
    np.testing.assert_allclose(accepted[1] - accepted[0], 0.5 * (raw[1, :3] - raw[0, :3]), atol=2e-8)


def _structural_kkt_external_dynamic_contact_blocks_free_translation(test, device):
    """Do not independently translate islands coupled by a dynamic contact."""
    ids = wp.array([0, 1], dtype=int, device=device)
    enabled = wp.ones(2, dtype=int, device=device)
    scale = wp.ones(2, dtype=float, device=device)
    pose = wp.array(
        [wp.transform_identity(), wp.transform_identity()],
        dtype=wp.transform,
        device=device,
    )
    correction = wp.zeros(2, dtype=wp.spatial_vector, device=device)
    system = wp.zeros(2, dtype=wp.mat44d, device=device)
    wp.launch(
        rigid_vbd_kkt.accumulate_free_translation_system,
        2,
        inputs=[
            ids,
            ids,
            ids,
            enabled,
            enabled,
            scale,
            correction,
            1.0,
            pose,
            pose,
            wp.zeros(2, dtype=wp.vec3, device=device),
            wp.zeros(2, dtype=wp.vec3, device=device),
            0,
            wp.zeros(2, dtype=wp.vec3, device=device),
            wp.ones(2, dtype=float, device=device),
            wp.ones(2, dtype=float, device=device),
            wp.zeros(2, dtype=wp.mat33, device=device),
            wp.zeros(2, dtype=wp.mat33, device=device),
            wp.ones(2, dtype=wp.vec3, device=device),
            wp.zeros(2, dtype=wp.spatial_matrix, device=device),
            1,
            wp.array([1], dtype=int, device=device),
            wp.array([0], dtype=int, device=device),
            wp.array([1], dtype=int, device=device),
            ids,
            1,
            enabled,
            wp.zeros(2, dtype=int, device=device),
        ],
        outputs=[system],
        device=device,
    )
    wp.launch(
        rigid_vbd_kkt.correct_free_translation,
        2,
        inputs=[
            ids,
            enabled,
            enabled,
            scale,
            system,
            correction,
            False,
            False,
            True,
            False,
            None,
            ids,
            0,
            wp.zeros(2, dtype=wp.vec3, device=device),
            wp.zeros(2, dtype=wp.transform, device=device),
        ],
        outputs=[
            None,
            None,
            None,
            None,
            wp.zeros(2, dtype=wp.vec3, device=device),
        ],
        device=device,
    )
    np.testing.assert_array_equal(system.numpy()[:, 3, 3], 1.0)
    np.testing.assert_array_equal(correction.numpy(), 0.0)


def _structural_kkt_translation_stop_keeps_coupled_step(test, device):
    """The stop keeps a step that curvature moves onto a finer axis and drops one stored uphill."""
    ids = wp.zeros(1, dtype=int, device=device)
    enabled = wp.ones(1, dtype=int, device=device)
    scale = wp.ones(1, dtype=float, device=device)
    correction = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    system = wp.zeros(1, dtype=wp.mat44d, device=device)
    support = wp.zeros(1, dtype=rigid_vbd_kkt._TranslationSupport, device=device)
    pending = wp.zeros(1, dtype=int, device=device)
    # Inertia 1e6 N/m. At x = 1 m, 0.2 N in x is within the tolerance (0.238 N). At x = 1024 m the x part
    # of the coupled step (6.7e-6 m) is lost in storage and its y part alone would raise the quadratic.
    for x, force, curvature, step in (
        (1.0, 0.2, 0.0, None),
        (1.0, 0.2, 1.0e6, [2.0**-23, 0.2 / 3.0e6, 0.0]),
        (1024.0, 10.0, 1.0e6, None),
    ):
        start = np.array([[x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        pose = wp.array(start, dtype=wp.transform, device=device)
        hessian = curvature * np.array([[[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]], dtype=np.float32)
        system.zero_()
        pending.zero_()
        wp.launch(
            rigid_vbd_kkt.accumulate_free_translation_system,
            1,
            inputs=[
                ids,
                ids,
                ids,
                enabled,
                enabled,
                scale,
                correction,
                1.0e-3,
                pose,
                pose,
                wp.zeros(1, dtype=wp.vec3, device=device),
                wp.zeros(1, dtype=wp.vec3, device=device),
                0,
                wp.zeros(1, dtype=wp.vec3, device=device),
                wp.ones(1, dtype=float, device=device),
                wp.ones(1, dtype=float, device=device),
                wp.array(hessian, dtype=wp.mat33, device=device),
                wp.zeros(1, dtype=wp.mat33, device=device),
                wp.array([[force, 0.0, 0.0]], dtype=wp.vec3, device=device),
                wp.zeros(1, dtype=wp.spatial_matrix, device=device),
                0,
                None,
                None,
                None,
                None,
                0,
                None,
                None,
            ],
            outputs=[system],
            device=device,
        )
        wp.launch(
            rigid_vbd_kkt.correct_free_translation,
            1,
            inputs=[
                ids,
                enabled,
                enabled,
                scale,
                system,
                correction,
                True,
                True,
                True,
                False,
                support,
                ids,
                0,
                wp.zeros(1, dtype=wp.vec3, device=device),
                wp.zeros(1, dtype=wp.transform, device=device),
            ],
            outputs=[
                pose,
                pending,
                wp.zeros(1, dtype=wp.vec4, device=device),
                wp.zeros(1, dtype=wp.vec4d, device=device),
                wp.zeros(1, dtype=wp.vec3, device=device),
            ],
            device=device,
        )
        moved = pose.numpy()[0, :3] - start[0, :3]
        with test.subTest(x=x, curvature=curvature):
            test.assertEqual(int(pending.numpy()[0]), int(step is not None))
            if step is not None:
                # At x = 1 m: H^-1 r = (1.33e-7, 6.67e-8, 0) m; x keeps one quantum, y is resolved.
                np.testing.assert_allclose(moved, step, rtol=1.0e-5, atol=0.0)
            else:
                np.testing.assert_array_equal(moved, 0.0)


def _cable_kkt_preserves_local_grounded_loop(test, device):
    """Improve stiff-loop closure while maintaining ground support."""
    local = _simulate_grounded_loop_load(device, 0)
    result = _simulate_grounded_loop_load(device, 1)
    test.assertLess(result["gap"], 0.01 * local["gap"])
    test.assertLess(result["penetration"], 1.0e-4)


def _cable_kkt_closed_cycle_preserves_local_contact(test, device):
    """Accelerate cyclic contact islands while preserving support under capture."""

    def simulate(g):
        model, bodies, joints = _build_loop_with_branch(
            device,
            ring_segments=16,
            stiffness=1.0e9,
            bend_stiffness=1.0e4,
            add_chord=True,
            grounded=True,
        )
        enabled = model.joint_enabled.numpy()
        enabled[joints[-1]] = False
        model.joint_enabled.assign(enabled)
        pose = model.body_q.numpy()
        pose[bodies, 2] -= 1.0e-3
        model.body_q.assign(pose)
        pipeline = newton.CollisionPipeline(model, contact_matching="latest")
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverVBD(
            model,
            iterations=2,
            rigid_compliant_alm=True,
            rigid_contact_history=False,
            rigid_joint_global_iterations=g,
            rigid_body_contact_buffer_size=256,
            deterministic=wp.DeterministicMode.RUN_TO_RUN,
        )
        a, b = model.state(), model.state()
        control = model.control()

        def pair():
            for current, output in ((a, b), (b, a)):
                current.clear_forces()
                wp.launch(
                    _apply_body_force,
                    1,
                    inputs=[int(bodies[8]), wp.vec3(-35.0, 0.0, 0.0)],
                    outputs=[current.body_f],
                    device=device,
                )
                pipeline.collide(current, contacts)
                solver.step(current, output, control, contacts, 1.0 / 600.0)

        pair()
        if device.is_cuda:
            with wp.ScopedCapture(device) as capture:
                pair()
            for _ in range(20):
                wp.capture_launch(capture.graph)
        else:
            for _ in range(20):
                pair()
        test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        result = [
            x.numpy().copy()
            for x in (
                a.body_q,
                a.body_qd,
                solver.joint_lambda_lin,
                solver.joint_lambda_ang,
                solver.body_body_contact_lambda,
            )
        ]
        for value in result:
            test.assertTrue(np.isfinite(value).all())
        active_joints = joints[model.joint_enabled.numpy()[joints]]
        return result, _max_joint_gap(model, result[0], active_joints)

    actual, actual_gap = simulate(1)
    _reference, reference_gap = simulate(0)
    test.assertLess(actual_gap, 0.5 * reference_gap)
    test.assertGreater(float(actual[0][:, 2].min()), 0.009)
    for a, b in zip(actual, simulate(1)[0], strict=True):
        np.testing.assert_array_equal(a, b)


def _cable_kkt_contact_capture_is_finite(test, device):
    """Keep captured global cable solves finite under contact."""
    model, pipeline, contacts, solver, _, _, _ = _build_grounded_chain(device, global_iterations=2)
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    dt = 1.0 / 600.0

    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, control, contacts, dt)
    state_in, state_out = state_out, state_in
    with wp.ScopedCapture(device) as capture:
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, dt)
        state_out.clear_forces()
        pipeline.collide(state_out, contacts)
        solver.step(state_out, state_in, control, contacts, dt)
    for _ in range(2):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    test.assertTrue(np.isfinite(state_in.body_q.numpy()).all())
    test.assertTrue(np.isfinite(state_in.body_qd.numpy()).all())


def _structural_kkt_reduces_fixed_chain_error(test, device):
    """Reduce closure error on a long chain of fixed joints."""
    dt = 1.0 / 600.0
    local_model, _, local_joints = _build_fixed_chain(device)
    solver_options = {
        "iterations": 1,
        "rigid_compliant_alm": True,
        "rigid_joint_linear_ke": 1.0e9,
        "rigid_joint_angular_ke": 1.0e9,
    }
    local_solver = newton.solvers.SolverVBD(local_model, **solver_options)
    local_state = _simulate(local_model, local_solver, 40, dt)
    local_gap = _max_joint_gap(local_model, local_state.body_q.numpy(), local_joints)

    kkt_model, _, kkt_joints = _build_fixed_chain(device)
    kkt_solver = newton.solvers.SolverVBD(
        kkt_model,
        **solver_options,
        rigid_joint_global_iterations=1,
    )
    backend = kkt_solver._structural_graph_kkt
    test.assertIsNotNone(backend)
    test.assertEqual(len(backend.path_buckets), 0)
    test.assertEqual(len(backend.tree_buckets), 1)
    kkt_state = _simulate(kkt_model, kkt_solver, 40, dt)
    kkt_gap = _max_joint_gap(kkt_model, kkt_state.body_q.numpy(), kkt_joints)

    test.assertTrue(np.isfinite(kkt_state.body_q.numpy()).all())
    # Finite compliant ALM is already much stronger than the former penalty
    # baseline. Require a material additional reduction and a tight absolute
    # gap instead of preserving the obsolete 100x relative threshold.
    test.assertLess(
        kkt_gap,
        0.25 * local_gap,
        f"Structural KKT gap {kkt_gap:.3e} did not improve {local_gap:.3e}",
    )
    test.assertLess(kkt_gap, 3.0e-5)


def _structural_kkt_batches_replicated_topologies(test, device):
    """Batch equal open and cyclic islands without changing their solve shape."""
    for closed, bucket_name in ((False, "tree_buckets"), (True, "closed_tree_buckets")):
        model = _build_replicated_fixed_topology(device, closed=closed)
        solver = newton.solvers.SolverVBD(
            model,
            iterations=1,
            rigid_compliant_alm=True,
            rigid_joint_global_iterations=1,
        )
        backend = solver._structural_graph_kkt
        test.assertIsNotNone(backend)
        buckets = getattr(backend, bucket_name)
        test.assertEqual(len(buckets), 1)
        test.assertEqual(buckets[0].batch_count, 4)

        state = _simulate(model, solver, 2, 1.0 / 600.0)
        test.assertTrue(np.isfinite(state.body_q.numpy()).all())


def _structural_kkt_memory_preflight_boundary_is_deterministic(test, device):
    """Use exact inclusive arithmetic and immutable capacity instead of free memory."""

    class CapacityOnlyDevice:
        def __init__(self, total_memory):
            self.is_cuda = True
            self.total_memory = total_memory

        @property
        def free_memory(self):
            raise AssertionError("The preflight must not query volatile free memory")

    cap = rigid_vbd_kkt._STRUCTURAL_MEMORY_CAP_BYTES
    test.assertEqual(rigid_vbd_kkt._structural_payload_budget(CapacityOnlyDevice(0)), cap)
    test.assertEqual(rigid_vbd_kkt._structural_payload_budget(CapacityOnlyDevice(4 * cap)), cap // 2)
    test.assertEqual(rigid_vbd_kkt._structural_payload_budget(CapacityOnlyDevice(16 * cap)), 2 * cap)
    initial_budget = rigid_vbd_kkt._structural_payload_budget(device)
    occupied = wp.zeros(1024, dtype=wp.spatial_matrix, device=device)
    test.assertEqual(
        rigid_vbd_kkt._structural_payload_budget(device),
        initial_budget,
    )
    test.assertEqual(occupied.shape[0], 1024)

    boundary = 1 << 100
    test.assertTrue(rigid_vbd_kkt._within_structural_payload_budget(boundary, boundary))
    test.assertFalse(rigid_vbd_kkt._within_structural_payload_budget(boundary + 1, boundary))
    test.assertFalse(rigid_vbd_kkt._within_structural_payload_budget(-1, boundary))
    test.assertEqual(
        rigid_vbd_kkt._array_payload_bytes(boundary, wp.spatial_matrix),
        boundary * wp.types.type_size_in_bytes(wp.spatial_matrix),
    )


def _structural_kkt_estimates_allocation_payloads_exactly(test, device):
    """Match path, tree, and batched closure estimates to intercepted allocations."""
    cases = (
        ("path", lambda: _build_chain(device, segments=8)[0]),
        ("path_unanchored", lambda: _build_chain(device, segments=8, pinned=False)[0]),
        ("path_ragged", lambda: _build_chains(device, (3, 9, 140))),
        ("tree_ragged", lambda: _build_y_trees(device, ((12, 12, 12), (16, 9, 14)))),
        ("tree", lambda: _build_y_tree(device, segments_per_branch=3)[0]),
        (
            "closed_tree",
            lambda: _build_replicated_fixed_topology(
                device,
                closed=True,
                worlds=3,
                link_count=8,
                closure_count=3,
            ),
        ),
        (
            "closed_tree_lane_backsub",
            lambda: _build_replicated_fixed_topology(
                device,
                closed=True,
                worlds=2,
                link_count=19,
                closure_count=17,
            ),
        ),
    )
    for kind, build_model in cases:
        with test.subTest(kind=kind):
            backend, allocated_bytes = _build_structural_backend_with_intercepted_bytes(build_model())
            test.assertTrue(backend.active)
            test.assertEqual(backend._estimated_bytes, allocated_bytes)
            test.assertEqual(backend.body_stress.shape, (2 * backend.joint_count,))
            test.assertEqual(backend.stress_offsets.shape, (backend.graph_body_count + 1,))
            destinations = backend.joint_stress_slot.numpy()
            used = np.sort(destinations[destinations >= 0])
            np.testing.assert_array_equal(used, np.arange(int(backend.stress_offsets.numpy()[-1])))
            test.assertTrue(
                all(bucket._estimated_bytes == bucket._diagnostics.estimated_bytes for bucket in backend.buckets)
            )
            diagnostics = backend._bucket_diagnostics
            test.assertEqual(len(diagnostics), 1)
            expected_kind = (
                "closed_tree"
                if kind.startswith("closed_tree")
                else kind.removesuffix("_ragged").removesuffix("_unanchored")
            )
            test.assertEqual(diagnostics[0].kind, expected_kind)
            test.assertIsNone(diagnostics[0].fallback_reason)
            if kind.startswith("closed_tree"):
                bucket = backend.closed_tree_buckets[0]
                matrix_bytes = wp.types.type_size_in_bytes(bucket.closure_schur.dtype)
                expected_response_bytes = bucket.response_rhs.capacity
                if bucket.paired_response is not None:
                    expected_response_bytes += bucket.paired_response.capacity
                test.assertEqual(
                    diagnostics[0].closure_response_bytes,
                    expected_response_bytes,
                )
                test.assertEqual(
                    diagnostics[0].closure_schur_bytes,
                    bucket.batch_count * bucket.closure_count * bucket.closure_count * matrix_bytes,
                )
                expected_worlds = 2 if kind == "closed_tree_lane_backsub" else 3
                expected_closures = 17 if kind == "closed_tree_lane_backsub" else 3
                test.assertEqual(diagnostics[0].batch_count, expected_worlds)
                test.assertEqual(diagnostics[0].cycle_rank_per_island, expected_closures)
                test.assertEqual(diagnostics[0].closure_count_per_island, expected_closures)
                test.assertEqual(
                    diagnostics[0].closure_count_total,
                    expected_worlds * expected_closures,
                )


def _structural_kkt_oversized_closure_falls_back_before_allocation(test, device):
    """Reject a synthetic high-rank closure bucket before any backend allocation."""
    worlds = 2
    # Keep every added chord unique so this remains a valid synthetic graph,
    # not a collection of parallel-joint duplicates.
    link_count = 67
    closure_count = 64
    model = _build_replicated_fixed_topology(
        device,
        closed=True,
        worlds=worlds,
        link_count=link_count,
        closure_count=closure_count,
    )

    with (
        mock.patch.object(rigid_vbd_kkt, "_structural_payload_budget", return_value=0),
        ExitStack() as allocation_patches,
    ):
        allocation_functions = (
            "array",
            "empty",
            "empty_like",
            "zeros",
            "zeros_like",
            "ones",
        )
        originals = {name: getattr(wp, name) for name in allocation_functions}
        for name, original in originals.items():

            def reject_device_allocation(*args, _original=original, **kwargs):
                result = _original(*args, **kwargs)
                if result.device == model.device:
                    raise AssertionError("An unsupported closed bucket attempted a device allocation")
                return result

            allocation_patches.enter_context(mock.patch.object(wp, name, reject_device_allocation))
        backend = rigid_vbd_kkt.StructuralGraphKKT(model, model.body_inv_mass)

    test.assertFalse(backend.active)
    test.assertEqual(backend._estimated_bytes, 0)
    test.assertEqual(len(backend._bucket_diagnostics), 1)
    diagnostics = backend._bucket_diagnostics[0]
    test.assertEqual(diagnostics.selected_route, "local_vbd")
    test.assertEqual(diagnostics.fallback_reason, "payload_budget")
    test.assertEqual(diagnostics.batch_count, worlds)
    test.assertEqual(diagnostics.row_count_per_island, link_count + closure_count)
    test.assertEqual(diagnostics.row_count_total, worlds * (link_count + closure_count))
    test.assertEqual(diagnostics.cycle_rank_per_island, closure_count)
    test.assertEqual(diagnostics.closure_count_per_island, closure_count)
    test.assertEqual(diagnostics.closure_count_total, worlds * closure_count)
    matrix_dtype = wp.spatial_matrixd
    matrix_bytes = wp.types.type_size_in_bytes(matrix_dtype)
    expected_response_bytes = worlds * (2 * link_count) * closure_count * matrix_bytes
    if diagnostics.planned_route == "closed_tree_paired_backbone_cr":
        expected_response_bytes += (
            worlds * link_count * closure_count * wp.types.type_size_in_bytes(rigid_vbd_kkt._SpatialPairResponseD)
        )
    test.assertEqual(
        diagnostics.closure_response_bytes,
        expected_response_bytes,
    )
    test.assertEqual(
        diagnostics.closure_schur_bytes,
        worlds * closure_count * closure_count * matrix_bytes,
    )


def _structural_kkt_chunks_oversized_equal_topology_batch(test, device):
    """Retain the deterministic prefix of an equal-topology batch that fits."""
    model = _build_replicated_fixed_topology(
        device,
        closed=True,
        worlds=5,
        link_count=20,
        closure_count=8,
    )
    payload_budget = _closed_prefix_payload_budget(model, 2)
    with mock.patch.object(rigid_vbd_kkt, "_structural_payload_budget", return_value=payload_budget):
        backend = rigid_vbd_kkt.StructuralGraphKKT(model, model.body_inv_mass)

    test.assertTrue(backend.active)
    test.assertEqual(backend._estimated_bytes, payload_budget)
    test.assertEqual(len(backend.closed_tree_buckets), 1)
    test.assertEqual(backend.closed_tree_buckets[0].batch_count, 2)
    test.assertEqual(backend.island_count, 2)
    test.assertEqual(len(backend._bucket_diagnostics), 2)
    selected, fallback = backend._bucket_diagnostics
    test.assertEqual(selected.batch_count, 2)
    test.assertEqual(selected.closure_count_per_island, 8)
    test.assertEqual(selected.closure_count_total, 16)
    test.assertIsNone(selected.fallback_reason)
    test.assertEqual(fallback.batch_count, 3)
    test.assertEqual(fallback.closure_count_per_island, 8)
    test.assertEqual(fallback.closure_count_total, 24)
    test.assertEqual(fallback.selected_route, "local_vbd")
    test.assertEqual(fallback.fallback_reason, "payload_budget")
    np.testing.assert_array_equal(
        np.bincount(backend.graph_body_island.numpy(), minlength=2),
        np.asarray((20, 20)),
    )


def _structural_kkt_closed_preflight_skips_unused_contraction(test, device):
    """Do not build a contraction schedule that the closed route never consumes."""
    model = _build_replicated_fixed_topology(
        device,
        closed=True,
        worlds=2,
        link_count=8,
        closure_count=2,
    )
    with mock.patch.object(
        rigid_vbd_kkt,
        "_tree_contraction_schedule",
        side_effect=AssertionError("closed topology requested an unused contraction schedule"),
    ):
        backend = rigid_vbd_kkt.StructuralGraphKKT(model, model.body_inv_mass)
    test.assertTrue(backend.active)
    test.assertEqual(len(backend.closed_tree_buckets), 1)
    test.assertFalse(backend.closed_tree_buckets[0].tree.use_tree_contraction)


def _structural_kkt_mixed_rod_tree_uses_stable_leaf_route(test, device):
    """Keep heterogeneous cable/mechanism trees on the stable rooted pivot order."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    slider = builder.add_link(
        xform=wp.transform(wp.vec3(-0.18, 0.0, 1.0), wp.quat_identity()),
        mass=0.1,
        inertia=wp.mat33(1.0e-3, 0.0, 0.0, 0.0, 1.0e-3, 0.0, 0.0, 0.0, 1.0e-3),
    )
    slider_joint = builder.add_joint_prismatic(parent=-1, child=slider, axis=newton.Axis.X)
    builder.add_articulation([slider_joint])
    positions = [wp.vec3(0.0, 0.0, 1.0)]
    edges = []
    for direction in (
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(-0.5, 0.8660254, 0.0),
        wp.vec3(-0.5, -0.8660254, 0.0),
    ):
        previous = 0
        for segment in range(1, 9):
            positions.append(positions[0] + direction * (0.03 * segment))
            current = len(positions) - 1
            edges.append((previous, current))
            previous = current
    bodies, _ = builder.add_rod(
        rod=newton.Rod(positions, edges=edges, radius=0.01),
        cfg=builder.default_shape_cfg.copy(),
        stretch_stiffness=1.0e9,
        bend_stiffness=1.0e4,
        wrap_in_articulation=True,
        body_frame_origin="com",
    )
    builder.add_joint_ball(
        parent=slider,
        child=int(bodies[0]),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.015), wp.quat_identity()),
    )
    builder.color(balance_colors=False)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=2,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
        deterministic=wp.DeterministicMode.RUN_TO_RUN,
    )
    backend = solver._structural_graph_kkt
    test.assertIsNotNone(backend)
    test.assertEqual(len(backend.tree_buckets), 1)
    bucket = backend.tree_buckets[0]
    test.assertFalse(bucket.use_tree_contraction)
    test.assertEqual(bucket._diagnostics.selected_route, "tree_leaf_rake")
    test.assertEqual(bucket._diagnostics.fill_edge_count_total, 0)
    test.assertEqual(bucket._diagnostics.estimated_bytes, bucket._estimated_bytes)
    test.assertTrue(bucket.use_fused_tree_levels)

    # Preserve the fast route inside its existing homogeneous-ROD coverage.
    homogeneous_model, _, _ = _build_y_tree(device, segments_per_branch=4)
    homogeneous_solver = newton.solvers.SolverVBD(
        homogeneous_model,
        iterations=2,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    test.assertTrue(homogeneous_solver._structural_graph_kkt.tree_buckets[0].use_tree_contraction)


def _structural_kkt_partial_fallback_preserves_healthy_island(test, device):
    """Keep a healthy independent tree bitwise stable beside a rejected cycle."""
    reference_model, reference_healthy_bodies = _build_mixed_fixed_topologies(device)
    solver_options = {
        "iterations": 1,
        "rigid_compliant_alm": True,
        "rigid_joint_global_iterations": 1,
        "deterministic": wp.DeterministicMode.RUN_TO_RUN,
    }
    reference_solver = newton.solvers.SolverVBD(reference_model, **solver_options)
    reference_backend = reference_solver._structural_graph_kkt
    test.assertIsNotNone(reference_backend)
    healthy_diagnostics = next(
        diagnostics for diagnostics in reference_backend._bucket_diagnostics if diagnostics.kind == "tree"
    )
    healthy_payload_budget = healthy_diagnostics.estimated_bytes + rigid_vbd_kkt._shared_structural_payload_bytes(
        reference_model.body_count,
        reference_model.joint_count,
        healthy_diagnostics.body_count_total,
        healthy_diagnostics.batch_count,
        healthy_diagnostics.row_count_total,
        is_cuda=reference_model.device.is_cuda,
    )

    partial_model, partial_healthy_bodies = _build_mixed_fixed_topologies(device)
    with (
        mock.patch.object(
            rigid_vbd_kkt,
            "_structural_payload_budget",
            return_value=healthy_payload_budget,
        ),
        test.assertWarnsRegex(RuntimeWarning, "payload budget"),
    ):
        partial_solver = newton.solvers.SolverVBD(partial_model, **solver_options)
    partial_backend = partial_solver._structural_graph_kkt
    test.assertIsNotNone(partial_backend)
    test.assertEqual(partial_backend._estimated_bytes, healthy_payload_budget)
    test.assertEqual(partial_solver._structural_graph_kkt_estimated_bytes, healthy_payload_budget)
    test.assertEqual(
        partial_solver._structural_graph_kkt_payload_budget_bytes,
        healthy_payload_budget,
    )
    test.assertEqual(len(partial_backend.tree_buckets), 1)
    test.assertEqual(len(partial_backend.closed_tree_buckets), 0)
    test.assertEqual(partial_backend.island_count, 1)
    test.assertEqual(
        [diagnostics.kind for diagnostics in partial_backend._bucket_diagnostics],
        ["closed_tree", "tree"],
    )
    test.assertEqual(partial_backend._bucket_diagnostics[0].selected_route, "local_vbd")
    test.assertIsNone(partial_backend._bucket_diagnostics[1].fallback_reason)
    np.testing.assert_array_equal(partial_backend.graph_body_ids.numpy(), partial_healthy_bodies)
    np.testing.assert_array_equal(
        partial_backend.graph_body_island.numpy(),
        np.zeros(partial_backend.graph_body_count, dtype=np.int32),
    )

    reference_state = _simulate(reference_model, reference_solver, 3, 1.0 / 600.0)
    partial_state = _simulate(partial_model, partial_solver, 3, 1.0 / 600.0)
    np.testing.assert_array_equal(
        reference_state.body_q.numpy()[reference_healthy_bodies],
        partial_state.body_q.numpy()[partial_healthy_bodies],
    )
    np.testing.assert_array_equal(
        reference_state.body_qd.numpy()[reference_healthy_bodies],
        partial_state.body_qd.numpy()[partial_healthy_bodies],
    )


def _structural_kkt_assembles_closure_compliance_for_every_batch(test, device):
    """Place each batched closure compliance on its local Schur diagonal."""
    closure_count = 1
    batch_count = 2
    closure_size = batch_count * closure_count
    node_body = wp.array([-1], dtype=wp.int32, device=device)
    closure_parent_node = wp.full(closure_size, -1, dtype=wp.int32, device=device)
    closure_child_node = wp.full(closure_size, -1, dtype=wp.int32, device=device)
    body_scale = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    row_scale = wp.array(np.ones((closure_size, 6)), dtype=wp.spatial_vector, device=device)
    jacobian_parent = wp.zeros(closure_size, dtype=wp.spatial_matrix, device=device)
    jacobian_child = wp.zeros_like(jacobian_parent)
    compliance_host = np.stack((np.eye(6), 2.0 * np.eye(6))).astype(np.float32)
    compliance = wp.array(compliance_host, dtype=wp.spatial_matrix, device=device)
    response_solution = wp.zeros(1, dtype=wp.spatial_matrix, device=device)
    schur = wp.empty(closure_size, dtype=wp.spatial_matrix, device=device)

    wp.launch(
        assemble_closure_schur,
        closure_size,
        inputs=[
            closure_count,
            node_body,
            closure_parent_node,
            closure_child_node,
            body_scale,
            row_scale,
            jacobian_parent,
            jacobian_child,
            compliance,
            response_solution,
        ],
        outputs=[schur],
        device=device,
    )
    np.testing.assert_allclose(schur.numpy(), compliance_host, atol=0.0, rtol=0.0)


def _structural_kkt_drive_matches_finite_material_equilibrium(test, device):
    """Preserve the finite-stiffness equilibrium of driven joints."""
    stiffness = 1.0e5
    load = 24.0
    body_q, coordinates = _simulate_prismatic_material_chain(
        device,
        kind="drive",
        stiffness=stiffness,
        global_iterations=1,
        iterations=1,
        steps=1200,
    )
    expected = 0.02 + load / stiffness
    test.assertTrue(np.isfinite(body_q).all())
    np.testing.assert_allclose(coordinates, expected, atol=5.0e-6, rtol=0.0)


def _structural_kkt_preserves_near_rigid_joint_limits(test, device):
    """Respect unilateral joint limits across global-pass budgets."""
    for global_iterations in (0, 1, 2):
        body_q, coordinates = _simulate_prismatic_material_chain(
            device,
            kind="limit",
            stiffness=1.0e8,
            global_iterations=global_iterations,
            iterations=5,
            # The compliant limit is a damped transient. Sample after it has
            # settled instead of at the phase-sensitive midpoint used by the
            # former combined drive/limit row.
            steps=1200,
        )
        violation = np.maximum(np.abs(coordinates) - 0.015, 0.0)
        test.assertTrue(np.isfinite(body_q).all())
        test.assertLess(
            float(np.max(violation)),
            5.0e-5,
            f"G={global_iterations} produced excessive near-rigid limit violation",
        )


def _structural_kkt_robust_block_inverse(test, device):
    """Fall back to a robust inverse for ill-scaled spatial blocks."""
    # SPD block with mixed translational/angular scales. The fast 3x3 Schur
    # split has a float32 inverse residual above one, so this exercises the
    # residual-based pivoted fallback used by path CR.
    matrix = np.asarray(
        [
            [2072.3608, 8352.1514, -1317.6613, 1907.2506, 2678.1357, -4.185088],
            [8352.1514, 105578.63, -6395.0137, 11903.0928, 8810.852, 153.98131],
            [-1317.6613, -6395.0137, 898.747, -1294.5392, -1663.8846, -1.9071366],
            [1907.2506, 11903.0928, -1294.5392, 2023.6027, 2359.627, 6.176173],
            [2678.1357, 8810.852, -1663.8846, 2359.627, 3534.656, -11.468282],
            [-4.185088, 153.98131, -1.9071366, 6.176173, -11.468282, 0.56929857],
        ],
        dtype=np.float32,
    )
    matrix_device = wp.array(matrix[None], dtype=wp.spatial_matrix, device=device)
    inverse_device = wp.empty_like(matrix_device)
    wp.launch(
        _invert_spatial_matrix,
        1,
        inputs=[matrix_device],
        outputs=[inverse_device],
        device=device,
    )
    inverse = inverse_device.numpy()[0]
    residual = float(np.max(np.abs(matrix @ inverse - np.eye(6))))

    test.assertTrue(np.isfinite(inverse).all())
    test.assertLess(residual, 2.0e-2)

    # Schur updates are symmetric in exact arithmetic, but their stored
    # float32 blocks need not be. Respect the lower-left block rather than
    # silently replacing it with the transpose of the upper-right block.
    # One rounded entry is enough to expose the old fast inverse without
    # crossing its fallback threshold.
    for direction in (np.inf, -np.inf):
        with test.subTest(asymmetry_direction=direction):
            a = np.eye(3, dtype=np.float32)
            b = 2.0 * a
            c = b.copy()
            c[0, 0] = np.nextafter(c[0, 0], np.float32(direction))
            represented = np.block([[a, b], [c, np.float32(4.0 + 2.0**-10) * a]])
            matrix_device.assign(represented[None])
            wp.launch(
                _invert_spatial_matrix,
                1,
                inputs=[matrix_device],
                outputs=[inverse_device],
                device=device,
            )
            actual = inverse_device.numpy()[0].astype(np.float64)
            matrix64 = represented.astype(np.float64)
            reference = np.linalg.inv(matrix64)
            test.assertLess(np.linalg.norm(actual - reference) / np.linalg.norm(reference), 1.0e-6)
            np.testing.assert_allclose(matrix64 @ actual, np.eye(6), rtol=0.0, atol=1.0e-6)
            np.testing.assert_allclose(actual @ matrix64, np.eye(6), rtol=0.0, atol=1.0e-6)

    # A well-conditioned, finite matrix can still overflow the determinant
    # arithmetic in the fast Schur split. Its representable inverse must use
    # the pivoted fallback rather than silently retiring a global correction.
    for scale in (1.0e-14, 1.0e20):
        with test.subTest(scale=scale):
            scaled = np.eye(6, dtype=np.float32) * scale
            matrix_device.assign(scaled[None])
            wp.launch(
                _invert_spatial_matrix,
                1,
                inputs=[matrix_device],
                outputs=[inverse_device],
                device=device,
            )
            scaled_inverse = inverse_device.numpy()[0]
            test.assertTrue(np.isfinite(scaled_inverse).all())
            np.testing.assert_allclose(scaled @ scaled_inverse, np.eye(6), rtol=0.0, atol=1.0e-6)


def _structural_kkt_path_matches_float64_oracle(test, device):
    """Match path Delassus assembly and CR solutions to float64 oracles on both legacy and native rod models."""
    cases = (
        (16, 1.0e4, False, 1.0e-2, 1.0e-3, 1.0e-3),
        (16, 1.0e9, True, 1.0e-2, 1.0e-3, 1.0e-3),
        (128, 1.0e9, True, 3.0e-2, 2.0e-2, 6.0e-3),
        (16, 1.0e9, False, 1.0e-2, 1.0e-3, 1.0e-3),
        (128, 1.0e9, False, 3.0e-2, 2.0e-2, 6.0e-3),
    )
    for (
        segments,
        stiffness,
        capsule_mass,
        residual_limit,
        multiplier_limit,
        correction_limit,
    ) in cases:
        with test.subTest(segments=segments, stiffness=stiffness, capsule_mass=capsule_mass):
            metrics = _path_float64_oracle_metrics(device, stiffness, segments=segments, capsule_mass=capsule_mass)
            message = f"segments={segments}, K={stiffness:.1e}, capsule_mass={capsule_mass}: {metrics}"
            test.assertTrue(all(np.isfinite(value) for value in metrics.values()), message)
            test.assertLess(metrics["formula_matrix_error"], 1.0e-5, message)
            test.assertLess(metrics["formula_rhs_error"], 1.0e-5, message)
            test.assertLess(metrics["residual"], residual_limit, message)
            test.assertLess(metrics["multiplier_error"], multiplier_limit, message)
            test.assertLess(metrics["correction_error"], correction_limit, message)


def _structural_kkt_tree_matches_float64_oracle(test, device):
    """Match exact tree corrections and reactions to a float64 physical KKT."""
    # The shorter tree uses ordinary leaf elimination; the longer tree uses
    # rake-and-compress. Cover both production schedules against one oracle.
    for segments_per_branch, stiffness in ((2, 1.0e4), (4, 1.0e9)):
        with test.subTest(segments_per_branch=segments_per_branch, stiffness=stiffness):
            metrics = _tree_float64_oracle_metrics(
                device,
                stiffness,
                closed=False,
                tree_segments_per_branch=segments_per_branch,
            )
            message = f"segments_per_branch={segments_per_branch}, K={stiffness:.1e}: {metrics}"
            test.assertTrue(all(np.isfinite(value) for value in metrics.values()), message)
            test.assertLess(metrics["equilibrated_residual"], 2.0e-3, message)
            test.assertLess(metrics["correction_error"], 1.0e-3, message)
            test.assertLess(metrics["multiplier_error"], 1.0e-2, message)
            test.assertLess(metrics["body_reaction_error"], 2.0e-2, message)


def _structural_kkt_dense_closure_matches_float64_oracle(test, device):
    """Keep batched dense closure solves accurate as cycle rank grows."""
    rng = np.random.default_rng(20260924)
    for closures, batches in ((1, 1), (8, 4), (33, 2)):
        with test.subTest(closures=closures, batches=batches):
            size = 6 * closures
            matrix = rng.standard_normal((batches, size, size))
            matrix = matrix @ matrix.transpose(0, 2, 1) + size * np.eye(size)
            rhs = rng.standard_normal((batches, size))
            blocks = matrix.reshape(batches, closures, 6, closures, 6).transpose(0, 1, 3, 2, 4)
            bucket = SimpleNamespace(
                closure_count=closures,
                closure_size=batches * closures,
                use_fused_closure=device.is_cuda and 4 < closures <= rigid_vbd_kkt._FUSED_CLOSURE_MAX_BLOCKS,
                spatial_block_dim=32,
                use_lane_back_substitute=rigid_vbd_kkt._closure_back_substitute_lanes(closures, device),
                closure_back_partial=wp.zeros(batches * closures, dtype=wp.spatial_vectord, device=device),
                batch_count=batches,
                device=device,
                closure_schur=wp.array(blocks.reshape(-1, 6, 6), dtype=wp.spatial_matrixd, device=device),
                closure_multiplier=wp.array(rhs.reshape(-1, 6), dtype=wp.spatial_vectord, device=device),
            )
            rigid_vbd_kkt._ClosedTreeBucket.solve_closure_schur(bucket)
            actual = bucket.closure_multiplier.numpy().reshape(batches, size)
            expected = np.linalg.solve(matrix, rhs[..., None])[..., 0]
            np.testing.assert_allclose(actual, expected, rtol=1.0e-11, atol=1.0e-13)


def _structural_kkt_near_hard_closure_matches_float64_oracle(test, device):
    """Resolve stiff cyclic corrections without losing observable reactions on both legacy and native rod models."""
    for capsule_mass in (True, False):
        for ring_segments in (32, 64, 128):
            with test.subTest(ring_segments=ring_segments, capsule_mass=capsule_mass):
                metrics = _tree_float64_oracle_metrics(
                    device, 1.0e12, closed=True, ring_segments=ring_segments, capsule_mass=capsule_mass
                )
                test.assertTrue(all(np.isfinite(value) for value in metrics.values()), metrics)
                test.assertLess(metrics["equilibrated_residual"], 1.0e-4, metrics)
                test.assertLess(metrics["correction_error"], 1.0e-6, metrics)
                test.assertLess(metrics["internal_reaction_error"], 1.0e-8, metrics)
                test.assertLess(metrics["body_reaction_error"], 1.0e-6, metrics)


def _structural_kkt_closure_matches_float64_oracle(test, device):
    """Match cyclic closure corrections and reactions to a float64 physical KKT."""
    for stiffness in (1.0e4, 1.0e9):
        with test.subTest(stiffness=stiffness):
            metrics = _tree_float64_oracle_metrics(device, stiffness, closed=True)
            message = f"K={stiffness:.1e}: {metrics}"
            test.assertTrue(all(np.isfinite(value) for value in metrics.values()), message)
            test.assertLess(metrics["equilibrated_residual"], 2.0e-3, message)
            test.assertLess(metrics["correction_error"], 1.0e-3, message)
            test.assertLess(metrics["multiplier_error"], 1.0e-2, message)
            test.assertLess(metrics["body_reaction_error"], 2.0e-2, message)
            # Main's closed-form cable Jacobian slightly changes individual
            # float32 Schur entries at near-hard stiffness. Gate their action
            # separately so the physically applied closure response stays tight.
            test.assertLess(metrics["closure_schur_error"], 2.0e-4, message)
            test.assertLess(metrics["closure_schur_action_error"], 1.0e-4, message)
            test.assertLess(metrics["closure_rhs_error"], 1.0e-2, message)


def _structural_kkt_uncovered_routes_match_float64_oracle(test, device):
    """Cover the safe open-tree fallback and non-paired cyclic route."""
    open_metrics = _tree_float64_oracle_metrics(
        device,
        1.0e9,
        closed=False,
        tree_segments_per_branch=8,
        open_shape="spurred",
    )
    # This topology used to select unpaired backbone CR, whose constraint
    # residual exceeded 40 on CUDA. The leaf fallback remains within the
    # expected float32 envelope for this ~3e9-conditioned physical KKT.  CPU
    # and CUDA use different reduction orders; the current CPU route reaches
    # about 1.25%, while CUDA remains near 0.13%.  The independently checked
    # correction and observable-reaction bounds below stay below 1% on both.
    test.assertFalse(open_metrics["uses_backbone_cr"], open_metrics)
    test.assertLess(open_metrics["equilibrated_residual"], 1.5e-2, open_metrics)
    test.assertLess(open_metrics["correction_error"], 1.0e-2, open_metrics)
    test.assertLess(open_metrics["multiplier_error"], 2.0e-3, open_metrics)
    test.assertLess(open_metrics["body_reaction_error"], 2.0e-1, open_metrics)

    closed_metrics = _tree_float64_oracle_metrics(
        device,
        1.0e9,
        closed=True,
        ring_segments=4,
        add_chord=False,
    )
    test.assertFalse(closed_metrics["uses_paired_backbone"], closed_metrics)
    # The raw multiplier is gauge-dependent for this redundant cycle. Check
    # the body-affecting range-space reaction instead of its null component.
    test.assertLess(closed_metrics["equilibrated_residual"], 5.0e-3, closed_metrics)
    test.assertLess(closed_metrics["correction_error"], 5.0e-3, closed_metrics)
    test.assertGreater(closed_metrics["reaction_null_fraction"], 0.99, closed_metrics)
    test.assertLess(closed_metrics["reaction_range_error"], 1.0e-3, closed_metrics)
    test.assertLess(closed_metrics["body_reaction_error"], 1.0e-3, closed_metrics)
    # This deliberately redundant cycle has condition number ~5e12.  Preserve
    # the tighter CUDA action guard while allowing the CPU reduction order's
    # larger null-space-sensitive Schur error; observable range/body reaction
    # errors above remain constrained to 1e-3 on both backends.
    action_limit = 5.0e-1 if device.is_cuda else 8.0e-1
    test.assertLess(closed_metrics["closure_schur_action_error"], action_limit, closed_metrics)
    # The Schur RHS is null-space-sensitive for this deliberately redundant
    # cycle, so an equivalent completion-joint basis can change its relative
    # error without changing the applied correction.  Gate the physical KKT
    # residual, Schur action, and observable range/body reactions above.


def _structural_kkt_closure_separates_reaction_gauge(test, device):
    """Separate redundant reaction gauge from body-affecting cyclic error."""
    for bend_stiffness in (1.0e2, 1.0e4, 1.0e6, 1.0e8, 1.0e9):
        with test.subTest(stretch_stiffness=1.0e9, bend_stiffness=bend_stiffness):
            metrics = _tree_float64_oracle_metrics(
                device,
                1.0e9,
                closed=True,
                bend_stiffness=bend_stiffness,
            )
            message = f"Ks=1.0e9, Kb={bend_stiffness:.1e}: {metrics}"
            test.assertTrue(all(np.isfinite(value) for value in metrics.values()), message)
            test.assertLess(metrics["equilibrated_residual"], 2.0e-3, message)
            test.assertLess(metrics["correction_error"], 1.0e-3, message)
            test.assertLess(metrics["body_reaction_error"], 2.0e-2, message)
            # A null-fraction floor would punish a solver that reduces gauge error;
            # gate only the observable reaction component.
            test.assertLess(metrics["reaction_range_error"], 2.0e-3, message)
            test.assertLess(metrics["closure_rhs_error"], 1.0e-2, message)


def _structural_kkt_path_solve_has_small_residual(test, device):
    """Solve a reduced path system to a small algebraic residual."""
    # Use more rows than the serial terminal so this covers CR reduction,
    # terminal block Thomas, and back substitution as one algebraic solve.
    model, _, _ = _build_chain(device, segments=16, stiffness=1.0e12)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    backend = solver._structural_graph_kkt
    test.assertIsNotNone(backend)
    test.assertEqual(len(backend.path_buckets), 1)
    bucket = backend.path_buckets[0]
    test.assertEqual(len(bucket.cr_schedule.persistent), 1)
    # The production CR factorization deliberately overwrites its four row
    # buffers in place. Capture the assembled operator in test-owned storage
    # immediately before factorization so the residual checks the original
    # system without adding persistent solver memory.
    assembled_lower = wp.empty_like(bucket.lower[0])
    assembled_diagonal = wp.empty_like(bucket.diagonal[0])
    assembled_upper = wp.empty_like(bucket.upper[0])
    assembled_rhs = wp.empty_like(bucket.rhs[0])
    solve_rows = bucket.solve_rows

    def capture_and_solve_rows():
        wp.copy(assembled_lower, bucket.lower[0])
        wp.copy(assembled_diagonal, bucket.diagonal[0])
        wp.copy(assembled_upper, bucket.upper[0])
        wp.copy(assembled_rhs, bucket.rhs[0])
        solve_rows()

    bucket.solve_rows = capture_and_solve_rows
    _simulate(model, solver, 1, 1.0 / 600.0)

    lower = assembled_lower.numpy().astype(np.float64)
    diagonal = assembled_diagonal.numpy().astype(np.float64)
    upper = assembled_upper.numpy().astype(np.float64)
    rhs = assembled_rhs.numpy().astype(np.float64)
    solution = bucket.solution[0].numpy().astype(np.float64)
    residual = np.empty_like(rhs)
    for row in range(bucket.size):
        value = diagonal[row] @ solution[row] - rhs[row]
        if row > 0:
            value += lower[row] @ solution[row - 1]
        if row + 1 < bucket.size:
            value += upper[row] @ solution[row + 1]
        residual[row] = value
    relative_residual = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1.0e-30))

    test.assertTrue(np.isfinite(solution).all())
    test.assertLess(relative_residual, 2.0e-2)


def _structural_kkt_persistent_cr_matches_level_schedule(test, device):
    """Preserve the exact CR correction while collapsing short-path launches."""

    active_threshold = _CR_PERSISTENT_MAX_ROWS

    def simulate(use_persistent_cr):
        # Exercise the largest topology routed through the persistent kernel.
        model, _, _ = _build_chain(device, segments=active_threshold + 1, stiffness=1.0e9)
        solver = newton.solvers.SolverVBD(
            model,
            iterations=2,
            rigid_compliant_alm=True,
            rigid_joint_global_iterations=1,
            deterministic=wp.DeterministicMode.RUN_TO_RUN,
        )
        bucket = solver._structural_graph_kkt.path_buckets[0]
        test.assertEqual(bucket.size, active_threshold)
        test.assertEqual(len(bucket.cr_schedule.persistent), 1)
        bucket.plan_cr(fused=use_persistent_cr)
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        dt = 1.0 / 600.0
        solver.step(state_in, state_out, control, None, dt)
        state_in, state_out = state_out, state_in
        with wp.ScopedCapture(device) as capture:
            solver.step(state_in, state_out, control, None, dt)
            solver.step(state_out, state_in, control, None, dt)
        for _ in range(3):
            wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
        return state_in.body_q.numpy(), state_in.body_qd.numpy()

    level_q, level_qd = simulate(False)
    persistent_q, persistent_qd = simulate(True)
    np.testing.assert_array_equal(persistent_q, level_q)
    np.testing.assert_array_equal(persistent_qd, level_qd)

    larger_model, _, _ = _build_chain(device, segments=active_threshold + 2, stiffness=1.0e9)
    larger_solver = newton.solvers.SolverVBD(
        larger_model,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    larger_bucket = larger_solver._structural_graph_kkt.path_buckets[0]
    test.assertEqual(larger_bucket.size, active_threshold + 1)
    test.assertEqual(len(larger_bucket.cr_schedule.persistent), 0)
    test.assertEqual(len(larger_bucket.cr_schedule.tails), 1)

    cpu = wp.get_device("cpu")
    cpu_model, _, _ = _build_chain(cpu, segments=_CR_PERSISTENT_MAX_ROWS + 1, stiffness=1.0e9)
    cpu_solver = newton.solvers.SolverVBD(
        cpu_model,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    test.assertEqual(len(cpu_solver._structural_graph_kkt.path_buckets[0].cr_schedule.persistent), 0)


def _structural_kkt_cr_tail_preserves_factors(test, device):
    """Check dispatch, independent batch values, and fresh captured CR factors."""
    for count, batches in ((1, 1), (128, 2), (129, 2), (160, 2), (161, 1), (257, 3), (511, 1), (1024, 1)):
        with test.subTest(rows=count, batches=batches):
            paths = [
                rigid_vbd_kkt._Path(
                    joints=list(range(batch * count, (batch + 1) * count)),
                    bodies=list(range(batch * (count + 1), (batch + 1) * (count + 1))),
                    body_rows=[[] for _ in range(count + 1)],
                )
                for batch in range(batches)
            ]
            # Only algebra is exercised here; no body endpoint binding is needed.
            bucket = rigid_vbd_kkt._PathBucket(paths, device)
            baseline = rigid_vbd_kkt._PathBucket(paths, device)
            expected = device.is_cuda and count > _CR_PERSISTENT_MAX_ROWS
            test.assertEqual(len(bucket.cr_schedule.tails), batches if expected else 0)
            for _, rows, stride in bucket.cr_schedule.tails:
                test.assertLessEqual((rows + stride - 1) // stride, rigid_vbd_kkt._CR_TAIL_MAX_ROWS)
            baseline.plan_cr(fused=False)
            buffers = [
                bucket.lower[0],
                bucket.diagonal[0],
                bucket.upper[0],
                bucket.rhs[0],
            ]
            reference_buffers = [
                baseline.lower[0],
                baseline.diagonal[0],
                baseline.upper[0],
                baseline.rhs[0],
            ]
            capture = None
            for seed in (31, 47):
                rng = np.random.default_rng(seed)
                factors = rng.normal(size=(batches, count - 1, 6, 6)) * 0.2
                edge = factors @ factors.swapaxes(-1, -2)
                diagonal = np.broadcast_to(np.diag(np.arange(1.0, 7.0)), (batches, count, 6, 6)).copy()
                lower, upper = np.zeros_like(diagonal), np.zeros_like(diagonal)
                diagonal[:, :-1] += edge
                diagonal[:, 1:] += edge
                lower[:, 1:] = -edge
                upper[:, :-1] = -edge
                # Form b from a known solution of the once-rounded FP32 matrix.
                lower, diagonal, upper = [value.astype(np.float32) for value in (lower, diagonal, upper)]
                known = rng.normal(size=(batches, count, 6))
                rhs = np.einsum("bnij,bnj->bni", diagonal, known)
                rhs[:, 1:] += np.einsum("bnij,bnj->bni", lower[:, 1:], known[:, :-1])
                rhs[:, :-1] += np.einsum("bnij,bnj->bni", upper[:, :-1], known[:, 1:])
                rhs = rhs.astype(np.float32)
                source = [
                    wp.array(
                        value.reshape((-1, *value.shape[2:])),
                        dtype=target.dtype,
                        device=device,
                    )
                    for value, target in zip((lower, diagonal, upper, rhs), buffers, strict=True)
                ]
                for target, ref, value in zip(buffers, reference_buffers, source, strict=True):
                    wp.copy(target, value)
                    wp.copy(ref, value)
                baseline.solve_rows()
                if device.is_cuda:
                    if capture is None:
                        bucket.solve_rows()  # Compile before capture, then restore the fresh operator.
                        for target, value in zip(buffers, source, strict=True):
                            wp.copy(target, value)
                        with wp.ScopedCapture(device) as capture:
                            bucket.solve_rows()
                    wp.capture_launch(capture.graph)
                else:
                    bucket.solve_rows()
                for target, ref in zip(buffers, reference_buffers, strict=True):
                    np.testing.assert_array_equal(target.numpy(), ref.numpy())
                solution = buffers[-1].numpy().reshape(batches, count, 6).astype(np.float64)
                residual = np.einsum("bnij,bnj->bni", diagonal, solution) - rhs
                residual[:, 1:] += np.einsum("bnij,bnj->bni", lower[:, 1:], solution[:, :-1])
                residual[:, :-1] += np.einsum("bnij,bnj->bni", upper[:, :-1], solution[:, 1:])
                test.assertLess(float(np.linalg.norm(residual) / np.linalg.norm(rhs)), 2.0e-6)
                test.assertLess(
                    float(np.linalg.norm(solution - known) / np.linalg.norm(known)),
                    2.0e-6,
                )


def _structural_kkt_ragged_path_bucket_matches_standalone_paths(test, device):
    """Batch paths of different lengths without changing any path's CR arithmetic."""
    model = _build_chains(device, (3, 9, 140))
    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    test.assertEqual(len(solver._structural_graph_kkt.path_buckets), 1)
    test.assertEqual(solver._structural_graph_kkt.path_buckets[0].row_counts, [2, 8, 139])
    state = _simulate(model, solver, 2, 1.0 / 600.0)
    test.assertTrue(np.isfinite(state.body_q.numpy()).all())

    counts = (1, 4, 5, 128, 129, 161, 257, 2)
    paths = []
    for index, count in enumerate(counts):
        first = sum(counts[:index])
        paths.append(
            rigid_vbd_kkt._Path(
                joints=list(range(first, first + count)),
                bodies=list(range(first + index, first + index + count + 1)),
                body_rows=[[] for _ in range(count + 1)],
            )
        )
    rng = np.random.default_rng(19)
    systems = []
    for count in counts:
        factors = rng.normal(size=(count - 1, 6, 6)) * 0.2
        edge = factors @ factors.swapaxes(-1, -2)
        diagonal = np.broadcast_to(np.diag(np.arange(1.0, 7.0)), (count, 6, 6)).copy()
        lower, upper = np.zeros_like(diagonal), np.zeros_like(diagonal)
        diagonal[:-1] += edge
        diagonal[1:] += edge
        lower[1:] = -edge
        upper[:-1] = -edge
        rhs = rng.normal(size=(count, 6))
        systems.append([value.astype(np.float32) for value in (lower, diagonal, upper, rhs)])

    def solve(bucket, parts, fused):
        bucket.plan_cr(fused=fused)
        buffers = [bucket.lower[0], bucket.diagonal[0], bucket.upper[0], bucket.rhs[0]]
        for target, value in zip(buffers, parts, strict=True):
            wp.copy(target, wp.array(value, dtype=target.dtype, device=device))
        bucket.solve_rows()
        return [buffer.numpy() for buffer in buffers]

    ragged = rigid_vbd_kkt._PathBucket(paths, device)
    merged = [np.concatenate([parts[index] for parts in systems]) for index in range(4)]
    starts = np.cumsum([0, *counts])
    for fused in (True, False) if device.is_cuda else (False,):
        with test.subTest(fused=fused):
            result = solve(ragged, merged, fused)
            for index, (path, parts) in enumerate(zip(paths, systems, strict=True)):
                expected = solve(rigid_vbd_kkt._PathBucket([path], device), parts, fused)
                rows = slice(starts[index], starts[index + 1])
                for actual, reference in zip(result, expected, strict=True):
                    np.testing.assert_array_equal(actual[rows], reference)


def _structural_kkt_ragged_tree_bucket_matches_separate_buckets(test, device):
    """Merge contracted trees of different topologies without changing any tree's solve."""

    def simulate(split):
        model = _build_y_trees(device, ((12, 12, 12), (16, 9, 14)))
        solver = newton.solvers.SolverVBD(
            model,
            iterations=4,
            rigid_compliant_alm=True,
            rigid_joint_global_iterations=1,
            deterministic=wp.DeterministicMode.RUN_TO_RUN,
        )
        backend = solver._structural_graph_kkt
        test.assertEqual(len(backend.tree_buckets), 1)
        test.assertEqual(backend.tree_buckets[0].batch_count, 2)
        test.assertTrue(backend.tree_buckets[0].use_tree_contraction)
        if split:
            body_inv_mass = np.asarray(model.body_inv_mass.numpy(), dtype=float)
            paths = rigid_vbd_kkt._build_paths(model, body_inv_mass, ignore_free_completion_joints=True)
            trees = rigid_vbd_kkt._build_trees(
                model,
                {frozenset(path.joints) for path in paths},
                body_inv_mass,
                ignore_free_completion_joints=True,
            )
            for symbolics in (None, rigid_vbd_kkt._tree_bucket_symbolics(trees[0])):
                with test.assertRaisesRegex(ValueError, "per-tree symbolics"):
                    rigid_vbd_kkt._tree_bucket_layout(trees, symbolics)
            body_slot = {int(body): slot for slot, body in enumerate(backend.graph_body_ids.numpy())}
            separate = [
                rigid_vbd_kkt._TreeBucket(tree, body_slot, device)
                for tree in sorted(trees, key=rigid_vbd_kkt._component_order_key)
            ]
            test.assertEqual(len(separate), 2)
            test.assertTrue(all(bucket.use_tree_contraction for bucket in separate))
            backend.tree_buckets = separate
            backend.buckets = [*backend.path_buckets, *separate, *backend.closed_tree_buckets]
        state_in, state_out, control = model.state(), model.state(), model.control()
        dt = 1.0 / 600.0
        solver.step(state_in, state_out, control, None, dt)
        state_in, state_out = state_out, state_in
        if device.is_cuda:
            with wp.ScopedCapture(device) as capture:
                solver.step(state_in, state_out, control, None, dt)
                solver.step(state_out, state_in, control, None, dt)
            for _ in range(3):
                wp.capture_launch(capture.graph)
        else:
            for _ in range(3):
                solver.step(state_in, state_out, control, None, dt)
                solver.step(state_out, state_in, control, None, dt)
        return state_in.body_q.numpy(), state_in.body_qd.numpy()

    merged_q, merged_qd = simulate(False)
    separate_q, separate_qd = simulate(True)
    test.assertTrue(np.isfinite(merged_q).all())
    np.testing.assert_array_equal(merged_q, separate_q)
    np.testing.assert_array_equal(merged_qd, separate_qd)


def _structural_kkt_cr_tail_preserves_captured_trajectory(test, device):
    """Preserve poses and velocities with fresh operators at G1 and G2."""
    model, _, _ = _build_chain(device, segments=258, stiffness=1.0e9)
    for global_iterations in (1, 2):
        results = []
        for use_tail in (False, True):
            solver = newton.solvers.SolverVBD(
                model,
                iterations=4,
                rigid_compliant_alm=True,
                rigid_joint_global_iterations=global_iterations,
                deterministic=wp.DeterministicMode.RUN_TO_RUN,
            )
            bucket = solver._structural_graph_kkt.path_buckets[0]
            test.assertEqual(bucket.size, 257)
            test.assertEqual(len(bucket.cr_schedule.tails), 1)
            bucket.plan_cr(fused=use_tail)
            source, target, control = model.state(), model.state(), model.control()
            solver.step(source, target, control, None, 1.0 / 600.0)
            with wp.ScopedCapture(device) as capture:
                solver.step(target, source, control, None, 1.0 / 600.0)
                solver.step(source, target, control, None, 1.0 / 600.0)
            for _ in range(3):
                wp.capture_launch(capture.graph)
            results.append((target.body_q.numpy(), target.body_qd.numpy()))
        for reference, actual in zip(*results, strict=True):
            np.testing.assert_array_equal(actual, reference)


def _structural_kkt_fused_closure_matches_level_schedule(test, device):
    """Keep closure dispatch bounded and preserve captured pose/velocity updates."""
    for closures, worlds in ((1, 1), (4, 1), (5, 1), (6, 4), (16, 1), (17, 1)):
        model = _build_replicated_fixed_topology(
            device,
            closed=True,
            worlds=worlds,
            link_count=max(10, closures + 2),
            closure_count=closures,
        )

        def simulate(model, closures, use_fused):
            solver = newton.solvers.SolverVBD(
                model,
                iterations=2,
                rigid_compliant_alm=True,
                rigid_joint_global_iterations=1,
            )
            bucket = solver._structural_graph_kkt.closed_tree_buckets[0]
            expected = device.is_cuda and 4 < closures <= rigid_vbd_kkt._FUSED_CLOSURE_MAX_BLOCKS
            test.assertEqual(bucket.use_fused_closure, expected)
            bucket.use_fused_closure = expected and use_fused
            source, target, control = model.state(), model.state(), model.control()
            solver.step(source, target, control, None, 1 / 600)
            if device.is_cuda:
                with wp.ScopedCapture(device) as capture:
                    solver.step(target, source, control, None, 1 / 600)
                    solver.step(source, target, control, None, 1 / 600)
                for _ in range(3):
                    wp.capture_launch(capture.graph)
            else:
                for _ in range(3):
                    solver.step(target, source, control, None, 1 / 600)
                    solver.step(source, target, control, None, 1 / 600)
            return target.body_q.numpy(), target.body_qd.numpy()

        baseline = simulate(model, closures, False)
        candidate = simulate(model, closures, True)
        for actual, expected in zip(candidate, baseline, strict=True):
            np.testing.assert_array_equal(actual, expected)


def _structural_kkt_fused_closure_matches_dense_oracle(test, device):
    """Match optimized closure solves to their serial path and an FP64 oracle."""
    rng = np.random.default_rng(20260912)
    for closures, batches in ((5, 1), (6, 4), (16, 2), (17, 2), (31, 1)):
        n = 6 * closures
        dense = rng.standard_normal((batches, n, n))
        dense = (dense @ dense.transpose(0, 2, 1) + n * np.eye(n)).astype(np.float32)
        rhs = rng.standard_normal((batches, n)).astype(np.float32)
        blocks = dense.reshape(batches, closures, 6, closures, 6).transpose(0, 1, 3, 2, 4).reshape(-1, 6, 6)
        solutions, factors = [], []
        candidate_fused = closures <= rigid_vbd_kkt._FUSED_CLOSURE_MAX_BLOCKS
        candidate_lanes = rigid_vbd_kkt._closure_back_substitute_lanes(closures, device)
        for fused, lanes in ((False, False), (candidate_fused, candidate_lanes)):
            bucket = SimpleNamespace(
                closure_count=closures,
                batch_count=batches,
                closure_size=batches * closures,
                use_fused_closure=fused,
                device=device,
                spatial_block_dim=32,
                use_lane_back_substitute=lanes,
                closure_back_partial=(
                    wp.zeros(batches * closures, dtype=wp.spatial_vector, device=device) if lanes else None
                ),
                closure_schur=wp.array(blocks, dtype=wp.spatial_matrix, device=device),
                closure_multiplier=wp.array(rhs.reshape(-1, 6), dtype=wp.spatial_vector, device=device),
            )
            rigid_vbd_kkt._ClosedTreeBucket.solve_closure_schur(bucket)
            solutions.append(bucket.closure_multiplier.numpy().reshape(batches, n))
            factors.append(bucket.closure_schur.numpy())
        np.testing.assert_array_equal(solutions[0], solutions[1])
        np.testing.assert_array_equal(factors[0], factors[1])
        oracle = np.linalg.solve(dense.astype(np.float64), rhs.astype(np.float64)[..., None])[..., 0]
        np.testing.assert_allclose(solutions[1], oracle, rtol=2e-5, atol=1e-7)


def _structural_kkt_paired_coarse_lanes_match_serial_schedule(test, device):
    """Preserve the paired coarse correction while widening its closure columns."""

    def simulate(model, use_lanes):
        solver = newton.solvers.SolverVBD(
            model,
            iterations=2,
            rigid_compliant_alm=True,
            rigid_joint_global_iterations=1,
            deterministic=wp.DeterministicMode.RUN_TO_RUN,
        )
        bucket = solver._structural_graph_kkt.closed_tree_buckets[0]
        test.assertTrue(bucket.use_paired_backbone)
        test.assertTrue(bucket.use_lane_split_coarse)
        test.assertFalse(bucket.tree.use_fused_tree_levels)
        test.assertGreaterEqual(bucket.closure_count, _PAIRED_COARSE_MIN_CLOSURES)
        # Keep this legacy comparison independent of the new default route.
        bucket.use_cooperative_paired = False
        bucket.paired_cr_levels = []
        stride = 1
        active = bucket.paired_backbone_count
        while active > rigid_vbd_kkt._CR_SERIAL_TERMINAL_SIZE:
            eliminated = len(range(stride, bucket.paired_backbone_count, 2 * stride))
            survivors = len(range(0, bucket.paired_backbone_count, 2 * stride))
            bucket.paired_cr_levels.append((stride, eliminated, survivors))
            stride *= 2
            active = survivors
        bucket.paired_terminal_stride = stride
        bucket.use_lane_split_coarse = use_lanes
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        dt = 1.0 / 600.0
        solver.step(state_in, state_out, control, None, dt)
        state_in, state_out = state_out, state_in
        with wp.ScopedCapture(device) as capture:
            solver.step(state_in, state_out, control, None, dt)
            solver.step(state_out, state_in, control, None, dt)
        for _ in range(3):
            wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
        return state_in.body_q.numpy(), state_in.body_qd.numpy()

    # A single island, then a batch, both above the closure gate.
    for worlds in (1, 4):
        model = _build_replicated_fixed_topology(device, closed=True, worlds=worlds, link_count=10, closure_count=6)
        serial_q, serial_qd = simulate(model, False)
        lane_q, lane_qd = simulate(model, True)
        np.testing.assert_array_equal(lane_q, serial_q)
        np.testing.assert_array_equal(lane_qd, serial_qd)

    # One closure is below the gate: the serial coarse kernel is retained.
    narrow = _build_replicated_fixed_topology(device, closed=True, worlds=1, link_count=10, closure_count=1)
    narrow_solver = newton.solvers.SolverVBD(
        narrow,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    narrow_bucket = narrow_solver._structural_graph_kkt.closed_tree_buckets[0]
    test.assertEqual(narrow_bucket.closure_count, 1)
    test.assertFalse(narrow_bucket.use_lane_split_coarse)

    cpu = wp.get_device("cpu")
    cpu_model = _build_replicated_fixed_topology(cpu, closed=True, worlds=1, link_count=10, closure_count=6)
    cpu_solver = newton.solvers.SolverVBD(
        cpu_model,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    test.assertFalse(cpu_solver._structural_graph_kkt.closed_tree_buckets[0].use_lane_split_coarse)


def _structural_kkt_zero_correction_preserves_pose(test, device):
    """Preserve zero-step poses exactly, including offset COMs and captured trials."""
    rotation = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32)
    host = np.array([[0.25, -0.5, 0.75, *rotation]] * 2 + [[0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)
    original = wp.array(host, dtype=wp.transform, device=device)
    pose = wp.clone(original)
    ids = wp.array([0, 1, 2], dtype=int, device=device)
    state = wp.ones(3, dtype=int, device=device)
    scale = wp.array([1.0, 0.0, 1.0], dtype=float, device=device)
    com = wp.array([[0.43, -0.17, 0.81]] * 2 + [[0, 0, 0]], dtype=wp.vec3, device=device)
    correction = wp.array(
        [[0, 0, 0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [1.0e-24, 0, 0, 0, 0, 0]],
        dtype=wp.spatial_vector,
        device=device,
    )
    pending = wp.zeros(1, dtype=int, device=device)
    no_disp = wp.zeros(3, dtype=wp.vec3, device=device)
    no_disp_ang = wp.zeros(3, dtype=wp.vec3, device=device)

    for trial in (False, True):

        def apply(trial=trial):
            wp.copy(pose, original)
            if trial:
                wp.launch(
                    rigid_vbd_kkt._apply_trial_correction,
                    3,
                    inputs=[ids, ids, correction, scale, com, original, no_disp, no_disp_ang, 0, original],
                    outputs=[pose, pending, no_disp, no_disp_ang],
                    device=device,
                )
            else:
                wp.launch(
                    apply_global_correction,
                    3,
                    inputs=[ids, ids, state, correction, scale, com, 0, original],
                    outputs=[pose, no_disp, no_disp_ang],
                    device=device,
                )

        apply()
        results = [pose.numpy()]
        if device.is_cuda:
            with wp.ScopedCapture(device) as capture:
                apply()
            wp.capture_launch(capture.graph)
            results.append(pose.numpy())
        for result in results:
            np.testing.assert_array_equal(result[:2].view(np.uint32), host[:2].view(np.uint32))
            # A squared-norm zero test would incorrectly discard this normal float32 increment.
            test.assertEqual(result[2, 0], np.float32(1.0e-24))


def _structural_kkt_suppresses_nonfinite_correction(test, device):
    """A non-finite correction must retire its whole island, not corrupt poses.

    The structural correction is a float32 solve whose conditioning is not
    bounded a priori; a closed island with many closures at near-hard stiffness
    can produce a non-finite result. This checks the containment contract
    directly rather than reproducing a precision- and hardware-dependent
    failure: poison one body's correction and require that its island is handed
    back to local VBD intact, while a healthy island in the same launch still
    receives its correction.
    """
    islands = wp.array([0, 0, 1], dtype=wp.int32, device=device)
    body_ids = wp.array([0, 1, 2], dtype=wp.int32, device=device)

    shift = wp.spatial_vector(0.25, 0.0, 0.0, 0.0, 0.0, 0.0)
    poisoned = wp.spatial_vector(float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0)
    correction = wp.array([shift, poisoned, shift], dtype=wp.spatial_vector, device=device)

    state = wp.array([1, 1], dtype=wp.int32, device=device)
    wp.launch(
        suppress_nonfinite_correction,
        3,
        inputs=[islands, correction],
        outputs=[state],
        device=device,
    )
    # Only the island carrying the non-finite value is retired, and it is
    # retired even though that value sat on its *second* body.
    test.assertEqual(state.numpy().tolist(), [-2, 1])

    identity = wp.transform_identity()
    body_q = wp.array([identity, identity, identity], dtype=wp.transform, device=device)
    body_com = wp.array([wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0)], dtype=wp.vec3, device=device)
    step_scale = wp.array([1.0, 1.0], dtype=float, device=device)
    wp.launch(
        apply_global_correction,
        3,
        inputs=[body_ids, islands, state, correction, step_scale, body_com, 0, wp.zeros_like(body_q)],
        outputs=[
            body_q,
            wp.zeros(3, dtype=wp.vec3, device=device),
            wp.zeros(3, dtype=wp.vec3, device=device),
        ],
        device=device,
    )

    poses = body_q.numpy()
    test.assertTrue(np.isfinite(poses).all(), "a retired island must leave finite poses")
    # Both bodies of the retired island keep their local pose exactly.
    test.assertAlmostEqual(float(poses[0][0]), 0.0, places=6)
    test.assertAlmostEqual(float(poses[1][0]), 0.0, places=6)
    # The healthy island is still corrected, so the guard is not a global mute.
    test.assertAlmostEqual(float(poses[2][0]), 0.25, places=6)


def _structural_kkt_fused_tree_levels_match_level_schedule(test, device):
    """Preserve the exact leaf-rake correction while collapsing its launches."""

    def simulate(build, use_fused, *, force_leaf_rake=False):
        model = build(device)[0]
        solver = newton.solvers.SolverVBD(
            model,
            iterations=2,
            rigid_compliant_alm=True,
            rigid_joint_global_iterations=1,
            deterministic=wp.DeterministicMode.RUN_TO_RUN,
        )
        bucket = solver._structural_graph_kkt.tree_buckets[0]
        test.assertTrue(bucket.use_fused_tree_levels)
        if force_leaf_rake:
            # Route this bucket down the leaf-rake schedule the fused kernel
            # replaces, so the comparison is not silently vacuous.
            bucket.use_tree_contraction = False
        shared = [not bool(level[4]) for level in bucket.levels]
        bucket.use_fused_tree_levels = use_fused
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        dt = 1.0 / 600.0
        solver.step(state_in, state_out, control, None, dt)
        state_in, state_out = state_out, state_in
        if device.is_cuda:
            with wp.ScopedCapture(device) as capture:
                solver.step(state_in, state_out, control, None, dt)
                solver.step(state_out, state_in, control, None, dt)
            for _ in range(3):
                wp.capture_launch(capture.graph)
        else:
            for _ in range(3):
                solver.step(state_in, state_out, control, None, dt)
                solver.step(state_out, state_in, control, None, dt)
        return state_in.body_q.numpy(), state_in.body_qd.numpy(), shared

    # Staggered branch depths select leaf rake on their own merits and give
    # every eliminated node a unique recipient, so this comparison cannot be
    # preempted by the path bucket or contraction.
    def staggered_tree(dev):
        builder = newton.ModelBuilder()
        inertia = wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
        root = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
            mass=1.0,
            inertia=inertia,
        )
        joints = [
            builder.add_joint_revolute(
                parent=-1,
                child=root,
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
                child_xform=wp.transform_identity(),
                axis=wp.vec3(0.0, 1.0, 0.0),
            )
        ]
        for branch, segment_count in enumerate((8, 7, 1)):
            previous = root
            for segment in range(segment_count):
                child = builder.add_link(
                    xform=wp.transform(
                        wp.vec3(0.1 * (segment + 1), 0.1 * branch, 1.0),
                        wp.quat_identity(),
                    ),
                    mass=1.0,
                    inertia=inertia,
                )
                joints.append(
                    builder.add_joint_revolute(
                        parent=previous,
                        child=child,
                        parent_xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()),
                        child_xform=wp.transform_identity(),
                        axis=wp.vec3(0.0, 1.0, 0.0),
                    )
                )
                previous = child
        builder.add_articulation(joints)
        builder.color(balance_colors=False)
        return builder.finalize(device=dev), None, None

    level_q, level_qd, unique_shared = simulate(staggered_tree, False)
    fused_q, fused_qd, _ = simulate(staggered_tree, True)
    test.assertFalse(any(unique_shared))
    np.testing.assert_array_equal(fused_q, level_q)
    np.testing.assert_array_equal(fused_qd, level_qd)

    # A branch hub gives one level whose leaves share a recipient, which the
    # fused kernel must handle with the same two-phase message accumulation.
    def short_y_tree(dev):
        return _build_y_tree(dev, segments_per_branch=2)

    shared_level_q, shared_level_qd, shared_flags = simulate(short_y_tree, False)
    shared_fused_q, shared_fused_qd, _ = simulate(short_y_tree, True)
    test.assertTrue(any(shared_flags))
    np.testing.assert_array_equal(shared_fused_q, shared_level_q)
    np.testing.assert_array_equal(shared_fused_qd, shared_level_qd)

    # The host gate declines schedules that cannot repay one synchronized block.
    narrow = [[0], [1], [2], [3]]
    test.assertTrue(_fused_tree_levels_supported(narrow, device))
    test.assertFalse(_fused_tree_levels_supported(narrow[: _FUSED_TREE_MIN_LEVELS - 1], device))
    too_wide = [list(range(_FUSED_TREE_MAX_LEVEL_WIDTH + 1)), *narrow]
    test.assertEqual(_fused_tree_levels_supported(too_wide, device), not device.is_cuda)
    test.assertTrue(_fused_tree_levels_supported(narrow, wp.get_device("cpu")))


def _structural_kkt_backbone_tail_response_is_finalized(test, device):
    """Back-substitute the final CR tail response into a closed tree."""
    """A tail eliminated without a right neighbor still needs its inverse block."""
    row_count = 11
    stride = 2
    eliminated_count = 3
    closure_count = 1
    backbone_nodes = wp.array(np.arange(row_count, dtype=np.int32), dtype=wp.int32, device=device)
    lower_host = np.zeros((row_count, 6, 6), dtype=np.float32)
    upper_host = np.zeros_like(lower_host)
    diagonal_host = np.zeros_like(lower_host)
    response_host = np.zeros_like(lower_host)
    lower_host[10] = 0.25 * np.eye(6, dtype=np.float32)
    diagonal_host[10] = 2.0 * np.eye(6, dtype=np.float32)
    response_host[8] = 0.5 * np.eye(6, dtype=np.float32)
    response_host[10] = np.eye(6, dtype=np.float32)

    lower = wp.array(lower_host, dtype=wp.spatial_matrix, device=device)
    upper = wp.array(upper_host, dtype=wp.spatial_matrix, device=device)
    diagonal = wp.array(diagonal_host, dtype=wp.spatial_matrix, device=device)
    rhs = wp.zeros(row_count, dtype=wp.spatial_vector, device=device)
    response_rhs = wp.array(response_host, dtype=wp.spatial_matrix, device=device)
    wp.launch(
        back_substitute_tree_backbone_cr_in_place,
        eliminated_count,
        inputs=[
            stride,
            row_count,
            eliminated_count,
            closure_count,
            backbone_nodes,
            lower,
            upper,
            diagonal,
            rhs,
            response_rhs,
        ],
        device=device,
    )

    expected = 1.75 * np.eye(6, dtype=np.float32)
    np.testing.assert_allclose(response_rhs.numpy()[10], expected, rtol=0.0, atol=1.0e-6)


def _structural_kkt_angular_damping_matches_local_force(test, device):
    """Match local angular damping for finite, noncommuting endpoint rotations."""
    model, _, _, _ = _build_joint_pair(device, newton.JointType.FIXED)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    dt = 0.02
    _simulate(model, solver, 1, dt)
    solver.joint_material_k.zero_()
    solver.joint_penalty_kd.assign([0.0, 2.0])
    solver.joint_lambda_ang.zero_()
    solver.joint_C0_ang.zero_()
    current = model.body_q.numpy().copy()
    previous = current.copy()
    for body, axis, angle, previous_axis, previous_angle in (
        (0, (1.0, 2.0, -1.0), 0.7, (-1.0, 1.0, 1.0), 0.4),
        (1, (-2.0, 1.0, 1.0), 1.0, (1.0, 1.0, -1.0), 0.6),
    ):
        current[body, 3:] = wp.quat_from_axis_angle(wp.normalize(wp.vec3(*axis)), angle)
        previous[body, 3:] = wp.quat_from_axis_angle(wp.normalize(wp.vec3(*previous_axis)), previous_angle)
    body_q = wp.array(current, dtype=wp.transform, device=device)
    solver.body_q_prev.assign(previous)
    bucket = solver._structural_graph_kkt.buckets[0]
    _linearize_bucket(model, solver, bucket, body_q, dt)
    residual = bucket.residual.numpy()[0].astype(np.float64)
    compliance = bucket.compliance.numpy()[0].astype(np.float64)
    jacobian = bucket.jacobian_child.numpy()[0].astype(np.float64)
    global_torque = -(jacobian.T @ np.linalg.solve(compliance, residual))[3:]
    expected = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        _evaluate_local_angular_damping,
        1,
        inputs=[body_q, solver.body_q_prev, dt],
        outputs=[expected],
        device=device,
    )
    np.testing.assert_allclose(global_torque, expected.numpy()[0], rtol=2.0e-6, atol=2.0e-5)


def _structural_kkt_joint_linearizations_match_finite_difference(test, device):
    """Match global joint Jacobians to finite-difference derivatives."""
    """Check endpoint signs, lever arms, and angular frames for every represented joint."""
    dt = 1.0 / 600.0
    epsilon = 2.0e-3
    represented_types = (
        newton.JointType.ROD,
        newton.JointType.BALL,
        newton.JointType.FIXED,
        newton.JointType.REVOLUTE,
        newton.JointType.PRISMATIC,
        newton.JointType.D6,
    )
    for joint_type in represented_types:
        model, parent, child, joint = _build_joint_pair(device, joint_type)
        solver = newton.solvers.SolverVBD(
            model,
            iterations=1,
            rigid_compliant_alm=True,
            rigid_joint_linear_ke=1.0e6,
            rigid_joint_angular_ke=1.0e6,
            rigid_joint_global_iterations=1,
        )
        state = _simulate(model, solver, 1, dt)
        backend = solver._structural_graph_kkt
        test.assertIsNotNone(backend)
        test.assertEqual(backend.joint_count, 1)
        bucket = backend.buckets[0]

        base_body_q = state.body_q.numpy()
        _linearize_bucket(model, solver, bucket, state.body_q, dt)
        row = int(np.flatnonzero(bucket.joint_ids_host == joint)[0])
        jacobians = (
            bucket.jacobian_parent.numpy()[row].astype(np.float64),
            bucket.jacobian_child.numpy()[row].astype(np.float64),
        )
        # Cable stretch and bend carry the complete finite constitutive defect;
        # both must remain consistent with the assembled row Jacobian.
        checked_components = slice(None)
        for endpoint, body in enumerate((parent, child)):
            for direction in range(6):
                plus = _perturbed_linearization(model, solver, bucket, base_body_q, body, direction, epsilon, dt)[row]
                minus = _perturbed_linearization(model, solver, bucket, base_body_q, body, direction, -epsilon, dt)[row]
                measured = ((plus - minus) / (2.0 * epsilon))[checked_components]
                expected = jacobians[endpoint][checked_components, direction]
                np.testing.assert_allclose(
                    measured,
                    expected,
                    rtol=2.0e-3,
                    atol=2.0e-3,
                    err_msg=f"joint={joint_type}, endpoint={endpoint}, tangent direction={direction}",
                )


def _structural_kkt_fused_stress_clears_disabled_rows(test, device):
    """Omit a fully fused pass without retaining disabled-row curvature."""
    model, bodies, joints = _build_chain(device, segments=8)
    enabled = model.joint_enabled.numpy().copy()
    disabled_joint = int(joints[len(joints) // 2])
    initial_pose = model.body_q.numpy().copy()
    initial_pose[bodies[-1], 2] += 0.02
    results = []
    for fused in (False, True):
        model.joint_enabled.assign(enabled)
        solver = newton.solvers.SolverVBD(
            model,
            iterations=2,
            rigid_compliant_alm=True,
            rigid_joint_global_iterations=1,
            deterministic=wp.DeterministicMode.RUN_TO_RUN,
        )
        # Plan for the step of this test before taking a reference to the backend (the first step plans).
        solver._plan_structural_graph_kkt(1.0 / 600.0)
        backend = solver._structural_graph_kkt
        test.assertTrue(backend._fuse_rod_stress)
        backend._fuse_rod_stress = fused
        state_a, state_b, control = model.state(), model.state(), model.control()
        state_a.body_q.assign(initial_pose)

        def advance(state_a=state_a, state_b=state_b, solver=solver, control=control):
            state_a.clear_forces()
            solver.step(state_a, state_b, control, None, 1.0 / 600.0)
            state_b.clear_forces()
            solver.step(state_b, state_a, control, None, 1.0 / 600.0)

        advance()
        if device.is_cuda:
            with wp.ScopedCapture(device) as capture:
                advance()
        slots = backend.joint_stress_slot.numpy()[disabled_joint]
        test.assertTrue(np.any(slots >= 0))
        phases = []
        for active in (True, False, True, False):
            current_enabled = enabled.copy()
            current_enabled[disabled_joint] = active
            model.joint_enabled.assign(current_enabled)
            # Poison old scratch so this detects stale data even at zero load.
            backend.body_stress.fill_(wp.mat33(7.0))
            if device.is_cuda:
                wp.capture_launch(capture.graph)
            else:
                advance()
            stress = backend.body_stress.numpy().copy()
            if not active:
                np.testing.assert_array_equal(stress[slots[slots >= 0]], 0.0)
            phase = (state_a.body_q.numpy().copy(), state_a.body_qd.numpy().copy(), stress)
            for value in phase:
                test.assertTrue(np.isfinite(value).all())
            phases.append(phase)
        results.append(phases)
    model.joint_enabled.assign(enabled)
    for actual, expected in zip(results[1], results[0], strict=True):
        for a, b in zip(actual, expected, strict=True):
            test.assertEqual(a.tobytes(), b.tobytes())

    # A world-anchor or a general joint still requires the separate pass.
    for other in (_build_fixed_chain(device, link_count=4)[0], _build_loop_with_branch(device, ring_segments=8)[0]):
        solver = newton.solvers.SolverVBD(
            other, iterations=2, rigid_compliant_alm=True, rigid_joint_global_iterations=1
        )
        test.assertFalse(solver._structural_graph_kkt._fuse_rod_stress)


def _structural_kkt_row_assembly_matches_block_equations(test, device):
    """Row-parallel assembly preserves matrix order and in-place buffer reuse."""
    count = 65
    for reversed_endpoints in (False, True):
        rng = np.random.default_rng(4381)
        bounds = np.tile([0, count], (count, 1)).astype(np.int32)
        bounds[:17, 1] = 17
        bounds[17:, 0] = 17
        bodies = np.column_stack((np.arange(count), np.arange(1, count + 1))).astype(np.int32)
        bodies[0, 0] = -1
        bodies[16, 1] = -1
        if reversed_endpoints:
            bodies = bodies[:, ::-1].copy()
        inverse = rng.normal(size=(count + 1, 6, 6)).astype(np.float32)
        inverse = inverse @ inverse.transpose(0, 2, 1) + np.eye(6, dtype=np.float32)
        compliance = np.tile(np.eye(6, dtype=np.float32), (count, 1, 1))
        residual = rng.normal(size=(count, 6)).astype(np.float32)
        with wp.ScopedDevice(device):
            inputs = [
                wp.array(bounds, dtype=wp.vec2i),
                wp.array(bodies, dtype=wp.vec2i),
                wp.array(inverse, dtype=wp.spatial_matrix),
                wp.array(rng.normal(size=(count + 1, 6)), dtype=wp.spatial_vector),
                wp.array(rng.normal(size=(count, 6, 6)), dtype=wp.spatial_matrix),
                wp.array(rng.normal(size=(count, 6, 6)), dtype=wp.spatial_matrix),
                wp.array(compliance, dtype=wp.spatial_matrix),
                wp.array(residual, dtype=wp.spatial_vector),
            ]
            output_types = (wp.spatial_matrix,) * 3 + (wp.spatial_vector,)
            reference = [wp.empty(count, dtype=dtype) for dtype in output_types]
            wp.launch(rigid_vbd_kkt.assemble_joint_path_system, count, inputs=inputs, outputs=reference)
            for alias in (False, True):
                inputs[-2].assign(compliance)
                inputs[-1].assign(residual)
                actual = [wp.empty(count, dtype=dtype) for dtype in output_types]
                if alias:
                    actual[1], actual[3] = inputs[-2], inputs[-1]
                wp.launch(
                    rigid_vbd_kkt.assemble_joint_path_system_rows,
                    (count, 6),
                    inputs=inputs[:2] + [array.view(float) for array in inputs[2:]],
                    outputs=[array.view(float) for array in actual],
                    block_dim=128,
                )
                for expected, result in zip(reference, actual, strict=True):
                    np.testing.assert_array_equal(result.numpy().view(np.uint32), expected.numpy().view(np.uint32))


def _structural_kkt_uses_explicit_iteration_budget(test, device):
    """Distribute the requested global passes over the local iteration budget."""
    model, _, _ = _build_chain(device, segments=4)

    disabled = newton.solvers.SolverVBD(model, iterations=5, rigid_compliant_alm=True)
    test.assertIsNone(disabled._structural_graph_kkt)
    test.assertEqual(disabled._body_contact_scratch.shape, (11, model.body_count, 3))
    explicit_zero = newton.solvers.SolverVBD(model, iterations=5, rigid_joint_global_iterations=0)
    test.assertEqual(explicit_zero._body_contact_scratch.shape, (11, model.body_count, 3))
    # Exercise main's shared scratch reset with nonzero values, not just its layout.
    disabled.body_forces.fill_(wp.vec3(1.0))
    disabled.body_torques.fill_(wp.vec3(1.0))
    disabled.body_hessian_ll.fill_(wp.mat33(1.0))
    disabled.body_hessian_al.fill_(wp.mat33(1.0))
    disabled.body_hessian_aa.fill_(wp.mat33(1.0))
    local_state = _simulate(model, disabled, 2, 1.0 / 600.0)
    zero_state = _simulate(model, explicit_zero, 2, 1.0 / 600.0)
    np.testing.assert_array_equal(local_state.body_q.numpy(), zero_state.body_q.numpy())
    np.testing.assert_array_equal(local_state.body_qd.numpy(), zero_state.body_qd.numpy())

    with test.assertRaisesRegex(ValueError, "rigid_joint_global_iterations"):
        newton.solvers.SolverVBD(
            model,
            iterations=5,
            rigid_compliant_alm=True,
            rigid_joint_global_iterations=6,
        )

    with test.assertRaisesRegex(ValueError, "externally integrated"):
        newton.solvers.SolverVBD(
            model,
            iterations=1,
            rigid_compliant_alm=True,
            rigid_joint_global_iterations=1,
            integrate_with_external_rigid_solver=True,
        )

    mixed_model, _, _ = _build_chain(device, segments=4, with_particle=True)
    with test.assertRaisesRegex(ValueError, "models containing particles"):
        newton.solvers.SolverVBD(
            mixed_model,
            iterations=1,
            rigid_compliant_alm=True,
            rigid_joint_global_iterations=1,
        )

    solver = newton.solvers.SolverVBD(
        model,
        iterations=5,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=2,
    )
    test.assertIsNotNone(solver._structural_graph_kkt)

    events = []
    test.assertEqual(solver._body_contact_scratch.shape, (11, model.body_count, 3))
    local_solve = solver._solve_rigid_body_iteration
    global_solve = solver._solve_structural_graph_kkt

    def record_local(*args, **kwargs):
        events.append("local")
        return local_solve(*args, **kwargs)

    def record_global(*args, **kwargs):
        events.append("global")
        return global_solve(*args, **kwargs)

    solver._solve_rigid_body_iteration = record_local
    solver._solve_structural_graph_kkt = record_global
    _simulate(model, solver, 1, 1.0 / 600.0)

    test.assertEqual(
        events,
        ["global", "local", "local", "global", "local", "local", "local"],
    )

    all_global_solver = newton.solvers.SolverVBD(
        model,
        iterations=2,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=2,
    )
    all_global_events = []
    all_global_local = all_global_solver._solve_rigid_body_iteration
    all_global_solve = all_global_solver._solve_structural_graph_kkt

    def record_all_global_local(*args, **kwargs):
        all_global_events.append("local")
        return all_global_local(*args, **kwargs)

    def record_all_global(*args, **kwargs):
        all_global_events.append("global")
        return all_global_solve(*args, **kwargs)

    all_global_solver._solve_rigid_body_iteration = record_all_global_local
    all_global_solver._solve_structural_graph_kkt = record_all_global
    _simulate(model, all_global_solver, 1, 1.0 / 600.0)
    test.assertEqual(all_global_events, ["global", "local", "global", "local"])


def _structural_kkt_prelocal_stage_bounds_coarse_chain_error(test, device):
    """Reconcile a coarse inertial prediction before a local sweep bends the chain."""
    model, bodies, joints = _build_chain(device, segments=8, stiffness=1.0e7)
    solver = newton.solvers.SolverVBD(model, iterations=1, rigid_compliant_alm=True, rigid_joint_global_iterations=1)
    state = _simulate(model, solver, 30, 1.0 / 60.0)
    pose = state.body_q.numpy()
    test.assertTrue(np.isfinite(pose).all())
    test.assertTrue(np.isfinite(state.body_qd.numpy()).all())
    np.testing.assert_array_equal(pose[bodies[0]], model.body_q.numpy()[bodies[0]])
    # This is a bounded coarse-step regression, not a general accuracy theorem.
    test.assertLess(_max_joint_gap(model, pose, joints), 4.0e-7)


def _structural_kkt_all_memory_fallback_uses_local_schedule(test, device):
    """Keep the ordinary G=0 schedule when every planned bucket falls back."""
    model = _build_replicated_fixed_topology(
        device,
        closed=True,
        worlds=1,
        link_count=20,
        closure_count=16,
    )
    with (
        mock.patch.object(rigid_vbd_kkt, "_structural_payload_budget", return_value=0),
        test.assertWarnsRegex(RuntimeWarning, "payload budget"),
    ):
        solver = newton.solvers.SolverVBD(
            model,
            iterations=2,
            rigid_compliant_alm=True,
            rigid_joint_global_iterations=2,
        )

    test.assertIsNone(solver._structural_graph_kkt)
    test.assertIsNone(solver._rigid_vbd_kkt)
    test.assertEqual(len(solver._structural_graph_kkt_bucket_diagnostics), 1)
    test.assertEqual(solver._structural_graph_kkt_bucket_diagnostics[0].selected_route, "local_vbd")

    events = []
    local_solve = solver._solve_rigid_body_iteration
    global_solve = solver._solve_structural_graph_kkt

    def record_local(*args, **kwargs):
        events.append(("local", bool(kwargs.get("defer_joint_dual", False))))
        return local_solve(*args, **kwargs)

    def record_global(*args, **kwargs):
        events.append(("global", False))
        return global_solve(*args, **kwargs)

    solver._solve_rigid_body_iteration = record_local
    solver._solve_structural_graph_kkt = record_global
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    dt = 1.0 / 600.0
    if device.is_cuda:
        solver.step(state_in, state_out, control, None, dt)
        state_in, state_out = state_out, state_in
        events.clear()
        with wp.ScopedCapture(device) as capture:
            solver.step(state_in, state_out, control, None, dt)
        wp.capture_launch(capture.graph)
    else:
        solver.step(state_in, state_out, control, None, dt)

    test.assertEqual(events, [("local", False), ("local", False)])
    test.assertTrue(np.isfinite(state_out.body_q.numpy()).all())


def _structural_kkt_contact_transitions_are_repeatable(test, device):
    """Keep global contact solves finite and repeatable through contact transitions."""

    def simulate(g):
        model, pipeline, contacts, solver, bodies, _, _ = _build_grounded_chain(
            device,
            global_iterations=g,
            iterations=2,
            segments=8,
            contact_history=True,
            deterministic=wp.DeterministicMode.RUN_TO_RUN,
        )
        q = model.body_q.numpy()
        q[bodies, 2] -= 0.001
        model.body_q.assign(q)
        a, b = model.state(), model.state()
        control = model.control()
        records = []
        for step in range(12):
            a.clear_forces()
            wp.launch(
                _apply_body_force,
                1,
                inputs=[int(bodies[-1]), wp.vec3(10.0, 0.0, 0.0)],
                outputs=[a.body_f],
                device=device,
            )
            pipeline.collide(a, contacts)
            if step in (3, 4):
                contacts.rigid_contact_count.zero_()
            if device.is_cuda and step == 7:
                with wp.ScopedCapture(device) as capture:
                    solver.step(a, b, control, contacts, 1.0 / 600.0)
                wp.capture_launch(capture.graph)
            else:
                solver.step(a, b, control, contacts, 1.0 / 600.0)
            a, b = b, a
            records.append(
                [
                    value.numpy().copy()
                    for value in (
                        a.body_q,
                        a.body_qd,
                        solver.joint_lambda_lin,
                        solver.joint_lambda_ang,
                        solver.body_body_contact_lambda,
                        solver.joint_penalty_k,
                    )
                ]
            )
        return records

    for g in (1, 2):
        reference = simulate(g)
        with test.subTest(global_iterations=g):
            for actual, expected in zip(simulate(g), reference, strict=True):
                for a, b in zip(actual, expected, strict=True):
                    test.assertTrue(np.isfinite(a).all())
                    np.testing.assert_array_equal(a, b)


def _structural_kkt_contacts_preserve_independent_acceleration(test, device):
    """Accelerate both collidable and collision-disabled independent islands."""
    count = 16
    model = _build_replicated_fixed_topology(device, closed=False, worlds=2, link_count=count)
    flags = model.shape_flags.numpy().copy()
    flags[model.shape_body.numpy() >= count] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
    model.shape_flags.assign(flags)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()

    def simulate(g):
        solver = newton.solvers.SolverVBD(
            model,
            iterations=2,
            rigid_compliant_alm=True,
            rigid_joint_global_iterations=g,
            rigid_joint_linear_ke=1.0e9,
            deterministic=wp.DeterministicMode.RUN_TO_RUN,
        )
        a, b = model.state(), model.state()
        control = model.control()
        for _ in range(12):
            a.clear_forces()
            force = np.zeros((model.body_count, 6), dtype=np.float32)
            force[[count - 1, 2 * count - 1], 0] = 10.0
            a.body_f.assign(force)
            solver.step(a, b, control, contacts, 1.0 / 600.0)
            a, b = b, a
        return (
            a.body_q.numpy(),
            a.body_qd.numpy(),
            solver.joint_lambda_lin.numpy(),
            solver,
        )

    reference = simulate(0)
    for g in (1, 2):
        with test.subTest(global_iterations=g):
            actual = simulate(g)
            for joints in (np.arange(count), np.arange(count, 2 * count)):
                test.assertLess(
                    _max_joint_gap(model, actual[0], joints),
                    0.5 * _max_joint_gap(model, reference[0], joints),
                )


def _structural_kkt_contact_history_requires_valid_matching_provenance(test, device):
    """Reject stale match-index storage after matching is disabled."""
    model, latest_pipeline, contacts, solver, bodies, _, _ = _build_grounded_chain(
        device,
        global_iterations=1,
        segments=8,
        contact_history=True,
    )
    model_pose = model.body_q.numpy()
    model_pose[bodies, 2] -= 0.001
    model.body_q.assign(model_pose)
    state_in = model.state()
    state_out = model.state()
    latest_pipeline.collide(state_in, contacts)
    test.assertIsNotNone(contacts.rigid_contact_match_index)

    disabled_pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        rigid_contact_max=contacts.rigid_contact_max,
        contact_matching="disabled",
    )
    disabled_pipeline.collide(state_in, contacts)
    test.assertEqual(contacts.contact_matching_mode, "disabled")
    test.assertIsNotNone(contacts.rigid_contact_match_index)
    with test.assertRaisesRegex(RuntimeError, "valid contact-matching provenance"):
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 600.0)


def _structural_kkt_contact_incidence_overflow_is_bounded(test, device):
    """Bound truncated reads and hand the affected island back to local VBD."""

    def simulate(global_iterations):
        model, pipeline, contacts, solver, bodies, _, _ = _build_grounded_chain(
            device,
            global_iterations=global_iterations,
            iterations=2,
            segments=8,
            contact_history=True,
            contact_buffer_size=1,
            deterministic=wp.DeterministicMode.RUN_TO_RUN,
        )
        model_pose = model.body_q.numpy()
        model_pose[bodies, 2] -= 0.003
        model.body_q.assign(model_pose)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        dt = 1.0 / 600.0
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0
        pipeline.collide(state_0, contacts)
        with wp.ScopedCapture(device) as capture:
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, dt)
            state_1.clear_forces()
            solver.step(state_1, state_0, control, contacts, dt)
        for _ in range(3):
            wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
        return (
            state_0.body_q.numpy(),
            state_0.body_qd.numpy(),
            solver.joint_lambda_lin.numpy(),
            solver.joint_lambda_ang.numpy(),
            solver.body_body_contact_lambda.numpy(),
            solver.body_body_contact_counts.numpy(),
            int(contacts.rigid_contact_count.numpy()[0]),
            (
                solver._structural_graph_kkt.island_contact_state.numpy()
                if solver._structural_graph_kkt is not None
                else np.empty(0, dtype=np.int32)
            ),
        )

    first = simulate(1)
    second = simulate(1)
    local = simulate(0)
    test.assertGreater(first[6], 0)
    test.assertGreater(int(np.max(first[5])), 1)
    np.testing.assert_array_equal(first[7], -2 * np.ones_like(first[7]))
    test.assertTrue(np.isfinite(first[0]).all())
    test.assertTrue(np.isfinite(first[1]).all())
    for first_value, second_value in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_value, second_value)
    # Suppressing the incomplete global objective preserves the exact local
    # pose, velocity, joint duals, and contact duals for the affected island.
    for guarded_value, local_value in zip(first[:5], local[:5], strict=True):
        np.testing.assert_array_equal(guarded_value, local_value)


def _structural_kkt_inherits_solver_deterministic_mode(test, device):
    """Apply the solver-local deterministic mode to the lazily loaded backend."""
    model, _, _ = _build_chain(device, segments=4)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
        deterministic=wp.DeterministicMode.RUN_TO_RUN,
    )
    test.assertIsNotNone(solver._rigid_vbd_kkt)
    options = wp.get_module_options(module=solver._rigid_vbd_kkt)
    test.assertEqual(options["deterministic"], wp.DeterministicMode.RUN_TO_RUN)
    test.assertEqual(options["deterministic_max_records"], 0)
    test.assertFalse(options["enable_backward"])

    # Separate kernel modules must follow the active solver, including after
    # constructing another solver with different deterministic options.
    other = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
        deterministic=wp.DeterministicMode.NOT_GUARANTEED,
    )
    backend = solver._rigid_vbd_kkt
    kernels = (
        backend.assemble_joint_path_system_rows,
        backend.reduce_cr_in_place_rows,
        backend.solve_cr_tail_persistent_rows,
        backend.reduce_cr_in_place_rows_fp64,
        backend.solve_cr_tail_persistent_rows_fp64,
    )
    modules = (backend, *(kernel.module for kernel in kernels))
    for active in (solver, other, solver):
        active._apply_module_options()
        for module in modules:
            options = wp.get_module_options(module=module)
            test.assertEqual(options["deterministic"], active._rigid_module_options["deterministic"])
            test.assertEqual(options["deterministic_max_records"], 0)
        for kernel in kernels:
            effective = wp.get_module_options(module=kernel.module) | kernel.options
            test.assertFalse(effective["enable_backward"])


def _structural_kkt_default_mode_matches_run_to_run(test, device):
    """Sum global stress blocks in a fixed order, so default and run-to-run modes agree bitwise."""
    builders = (
        lambda: _build_y_trees(device, ((12, 12, 12), (16, 9, 14), (20, 7, 11))),
        lambda: _build_chains(device, (3, 9, 40)),
    )
    for index, build in enumerate(builders):
        with test.subTest(model=index):
            results = []
            for deterministic in (None, wp.DeterministicMode.RUN_TO_RUN):
                model = build()
                solver = newton.solvers.SolverVBD(
                    model,
                    iterations=2,
                    rigid_compliant_alm=True,
                    rigid_joint_global_iterations=1,
                    deterministic=deterministic,
                )
                state = _simulate(model, solver, 30, 1.0 / 600.0)
                results.append((state.body_q.numpy(), state.body_qd.numpy()))
            test.assertTrue(np.isfinite(results[0][0]).all())
            np.testing.assert_array_equal(results[0][0], results[1][0])
            np.testing.assert_array_equal(results[0][1], results[1][1])


def _structural_kkt_preflight_is_deterministic_and_capture_safe(test, device):
    """Keep supported cyclic diagnostics and results stable on CPU and in CUDA capture."""

    def simulate():
        model = _build_replicated_fixed_topology(
            device,
            closed=True,
            worlds=2,
            link_count=8,
            closure_count=2,
        )
        model.set_gravity((0.0, 0.0, -9.81))
        solver = newton.solvers.SolverVBD(
            model,
            iterations=1,
            rigid_compliant_alm=True,
            rigid_joint_global_iterations=1,
            deterministic=wp.DeterministicMode.RUN_TO_RUN,
        )
        test.assertIsNotNone(solver._structural_graph_kkt)
        with (
            mock.patch.object(
                wp,
                "synchronize",
                side_effect=AssertionError("diagnostics synchronized the device"),
            ),
            mock.patch.object(
                wp,
                "synchronize_device",
                side_effect=AssertionError("diagnostics synchronized the device"),
            ),
        ):
            diagnostics = tuple(solver._structural_graph_kkt_bucket_diagnostics)
        test.assertEqual(len(diagnostics), 1)
        test.assertEqual(
            diagnostics[0].selected_route,
            "closed_tree_cpu_float64" if device.is_cpu else "closed_tree_paired_backbone_cr",
        )
        test.assertIsNone(diagnostics[0].fallback_reason)

        state_in = model.state()
        state_out = model.state()
        control = model.control()
        dt = 1.0 / 600.0
        solver.step(state_in, state_out, control, None, dt)
        state_in, state_out = state_out, state_in
        if device.is_cuda:
            with wp.ScopedCapture(device) as capture:
                solver.step(state_in, state_out, control, None, dt)
                solver.step(state_out, state_in, control, None, dt)
            for _ in range(2):
                wp.capture_launch(capture.graph)
        else:
            for _ in range(3):
                solver.step(state_in, state_out, control, None, dt)
                solver.step(state_out, state_in, control, None, dt)
        return diagnostics, state_in.body_q.numpy(), state_in.body_qd.numpy()

    first_diagnostics, first_q, first_qd = simulate()
    second_diagnostics, second_q, second_qd = simulate()
    test.assertEqual(first_diagnostics, second_diagnostics)
    test.assertTrue(np.isfinite(first_q).all())
    test.assertTrue(np.isfinite(first_qd).all())
    np.testing.assert_array_equal(first_q, second_q)
    np.testing.assert_array_equal(first_qd, second_qd)


def _structural_kkt_captures_numeric_inertia_refresh(test, device):
    """Numeric coupled inertia updates must not rebuild host topology in capture."""
    model, _, _ = _build_chain(device, segments=8)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=2,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    backend = solver._structural_graph_kkt
    test.assertIsNotNone(backend)

    with wp.ScopedCapture(device) as capture:
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    wp.capture_launch(capture.graph)

    test.assertIs(solver._structural_graph_kkt, backend)


def _structural_kkt_free_root_is_constraint_free(test, device):
    """An explicit FREE root must not disable its rigid descendants' global solve."""
    for stiffness in (1.0e8, 1.0e12):
        results = []
        for explicit_free in (False, True):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            bodies = []
            joints = []
            for index in range(16):
                body = builder.add_link(
                    xform=wp.transform(wp.vec3(index * 0.1, 0.0, 0.0), wp.quat_identity()),
                    mass=1.0,
                )
                builder.add_shape_box(body, hx=0.04, hy=0.02, hz=0.02)
                bodies.append(body)
                if index == 0 and explicit_free:
                    joints.append(builder.add_joint_free(child=body))
                elif index > 0:
                    joints.append(
                        builder.add_joint_fixed(
                            bodies[-2],
                            body,
                            parent_xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()),
                        )
                    )
            builder.add_articulation(joints)
            builder.color(balance_colors=False)
            model = builder.finalize(device=device)
            solver = newton.solvers.SolverVBD(
                model,
                iterations=2,
                rigid_compliant_alm=True,
                rigid_joint_global_iterations=1,
                rigid_joint_linear_ke=stiffness,
                rigid_joint_angular_ke=stiffness,
            )
            test.assertIsNotNone(solver._structural_graph_kkt)
            test.assertEqual(len(solver._structural_graph_kkt.path_buckets), 0)
            test.assertEqual(len(solver._structural_graph_kkt.tree_buckets), 1)
            state_in, state_out = model.state(), model.state()
            pose = state_in.body_q.numpy()
            pose[:, 1] += 0.001 * np.sin(np.arange(16) * 0.4)
            state_in.body_q.assign(pose)

            def pair(solver=solver, state_in=state_in, state_out=state_out):
                solver.step(state_in, state_out, None, None, 1.0 / 600.0)
                solver.step(state_out, state_in, None, None, 1.0 / 600.0)

            pair()
            if device.is_cuda:
                with wp.ScopedCapture(device) as capture:
                    pair()
                for _ in range(59):
                    wp.capture_launch(capture.graph)
            else:
                for _ in range(59):
                    pair()
            q, qd = state_in.body_q.numpy(), state_in.body_qd.numpy()
            test.assertTrue(np.isfinite(q).all())
            test.assertTrue(np.isfinite(qd).all())
            test.assertLess(_max_joint_gap(model, q, joints[int(explicit_free) :]), 1.0e-4)
            results.append((q, qd))
        np.testing.assert_allclose(results[0][0], results[1][0], atol=2.0e-6)
        np.testing.assert_allclose(results[0][1], results[1][1], atol=2.0e-5)


def _structural_kkt_reversing_pull_preserves_support(test, device):
    """Retain support when a sliding cable reverses its applied load on legacy and native rod models."""
    # Travel scales with the inverse of the rod mass. Where rods carry cylinder mass (#5328) the native rod is
    # 13/9 lighter than the capsule rod; on a base without that change the two are one model and the ratio is 1.
    masses = []
    for capsule_mass in (True, False):
        model, _, _, _, bodies, _, _ = _build_grounded_chain(
            device, global_iterations=1, iterations=1, capsule_mass=capsule_mass
        )
        masses.append(float(model.body_mass.numpy()[bodies].sum()))
    native_gain = masses[0] / masses[1]
    for capsule_mass, force in ((True, 45.0), (False, 45.0 / native_gain), (False, 45.0)):
        travel = force / 45.0 * (1.0 if capsule_mass else native_gain)
        for iterations in (1, 5):
            with test.subTest(iterations=iterations, capsule_mass=capsule_mass, force=force):
                result = _simulate_ground_drag(
                    device,
                    1,
                    iterations=iterations,
                    contact_history=True,
                    force=force,
                    reverse_at=140,
                    capsule_mass=capsule_mass,
                )
                test.assertLess(result["penetration"], 5.0e-4)
                test.assertLess(result["rise"], 1.0e-3)
                test.assertLess(result["gap"], 4.0e-4)
                test.assertGreater(result["motion"], 2.0 * travel)
                test.assertLess(result["motion"], 3.0 * travel)


def _structural_kkt_free_self_contact_conserves_momentum(test, device):
    """Free self-colliding rod island preserves collective inertia and angular momentum under Fix B-FP64."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    shape = builder.default_shape_cfg.copy()
    shape.mu = 0.8
    shape.ke = 1.0e6
    shape.kd = 100.0
    # Folded 4-segment rod where segments 0 and 3 overlap with tangential slip in zero gravity
    positions = [
        wp.vec3(0.0, 0.0, 0.0),
        wp.vec3(0.03, 0.0, 0.0),
        wp.vec3(0.03, 0.015, 0.0),
        wp.vec3(0.0, 0.015, 0.0),
        wp.vec3(-0.03, 0.015, 0.0),
    ]
    bodies, _ = builder.add_rod(
        rod=newton.Rod(positions, radius=0.01),
        cfg=shape,
        stretch_stiffness=1.0e7,
        bend_stiffness=1.0e3,
        wrap_in_articulation=True,
        body_frame_origin="com",
    )
    builder.color(balance_colors=False)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, contact_matching="latest")
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=2,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
        rigid_contact_history=True,
    )
    s_in, s_out, ctl = model.state(), model.state(), model.control()
    qd = np.zeros((model.body_count, 6), dtype=np.float32)
    qd[int(bodies[0]), 0] = 0.5
    qd[int(bodies[-1]), 0] = -0.5
    s_in.body_qd.assign(qd)

    def momenta(st):
        q = st.body_q.numpy().astype(np.float64)
        v = st.body_qd.numpy().astype(np.float64)
        m = model.body_mass.numpy().astype(np.float64)
        inert = model.body_inertia.numpy().astype(np.float64)
        lin = (m[:, None] * v[:, :3]).sum(axis=0)
        ang = np.zeros(3, dtype=np.float64)
        for b in range(model.body_count):
            pos = q[b, :3]
            x, y, z, w = q[b, 3], q[b, 4], q[b, 5], q[b, 6]
            rot = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )
            iw = rot @ inert[b] @ rot.T
            ang += np.cross(pos, m[b] * v[b, :3]) + iw @ v[b, 3:6]
        return lin, ang

    p0, l0 = momenta(s_in)
    dt = 1.0 / 600.0
    max_contacts = 0
    for _ in range(10):
        pipeline.collide(s_in, contacts)
        max_contacts = max(max_contacts, int(contacts.rigid_contact_count.numpy()[0]))
        solver.step(s_in, s_out, ctl, contacts, dt)
        s_in, s_out = s_out, s_in
    test.assertGreater(max_contacts, 0)
    p1, l1 = momenta(s_in)
    np.testing.assert_allclose(p1, p0, atol=5.0e-5)
    np.testing.assert_allclose(l1, l0, atol=5.0e-5)
    # The bound above passes with 40 % of the angular momentum lost. Past the tenth step the local
    # sweeps that follow the structural stage used to remove all of it; the six-mode pass after the
    # sweeps keeps the loss at the level of the same rod without contact.
    for _ in range(590):
        pipeline.collide(s_in, contacts)
        solver.step(s_in, s_out, ctl, contacts, dt)
        s_in, s_out = s_out, s_in
    p2, l2 = momenta(s_in)
    np.testing.assert_allclose(p2, p0, atol=5.0e-5)
    test.assertLess(float(np.linalg.norm(l2 - l0)), 0.15 * float(np.linalg.norm(l0)))


def _structural_kkt_axial_pull_converges_without_contacts(test, device):
    """Resolve a stiff axial load with one global and one local iteration."""
    model, _, _, solver, bodies, joints, _ = _build_grounded_chain(
        device,
        global_iterations=1,
        iterations=1,
        contact_history=True,
        deterministic=wp.DeterministicMode.RUN_TO_RUN,
    )
    state_in, state_out, control = model.state(), model.state(), model.control()
    peak_gap = 0.0
    for step in range(240):
        state_in.clear_forces()
        if step >= 40:
            wp.launch(
                _apply_body_force,
                1,
                inputs=[int(bodies[-1]), wp.vec3(45.0, 0.0, 0.0)],
                outputs=[state_in.body_f],
                device=device,
            )
        solver.step(state_in, state_out, control, None, 1.0 / 600.0)
        state_in, state_out = state_out, state_in
        if step % 10 == 9:
            peak_gap = max(peak_gap, _max_joint_gap(model, state_in.body_q.numpy(), joints))
    test.assertLess(peak_gap, 2.0e-6)


def _structural_kkt_translation_resolves_friction_breakaway(test, device):
    """Resolve sliding just outside the aggregate static friction cone."""
    result = _simulate_ground_drag(device, 1, contact_history=True, weight_fraction=1.1 * 0.45)
    expected = 0.5 * (0.1 * 0.45 * 9.81) * (200.0 / 600.0) ** 2
    test.assertGreater(result["motion"], 0.8 * expected)
    test.assertLess(result["motion"], 1.3 * expected)
    test.assertLess(result["penetration"], 5.0e-4)


def _structural_kkt_free_translation_preserves_momentum(test, device):
    """Preserve center-of-mass acceleration after the local joint sweeps."""
    model, bodies, _ = _build_chain(device, segments=16, pinned=False)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=5,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    a, b = model.state(), model.state()
    control = model.control()
    mass = model.body_mass.numpy()[bodies].astype(np.float64)
    initial = (a.body_q.numpy()[bodies, :3] * mass[:, None]).sum(axis=0) / mass.sum()
    dt, steps, force = 1.0 / 600.0, 120, 10.0
    graphs = {}
    for step in range(steps):
        a.clear_forces()
        wp.launch(
            _apply_body_force,
            1,
            inputs=[int(bodies[-1]), wp.vec3(force, 0.0, 0.0)],
            outputs=[a.body_f],
            device=device,
        )
        if step == 0:
            solver.step(a, b, control, None, dt)
        else:
            parity = step % 2
            if parity not in graphs:
                with mock.patch.object(
                    wp.array,
                    "numpy",
                    side_effect=AssertionError("host readback in global solve"),
                ):
                    with wp.ScopedCapture(device) as capture:
                        solver.step(a, b, control, None, dt)
                graphs[parity] = capture.graph
            wp.capture_launch(graphs[parity])
        a, b = b, a
    final = (a.body_q.numpy()[bodies, :3] * mass[:, None]).sum(axis=0) / mass.sum()
    expected = np.array([force / mass.sum(), 0.0, -9.81]) * (dt * dt * steps * (steps + 1) / 2)
    np.testing.assert_allclose(final - initial, expected, atol=1.0e-4, rtol=1.0e-4)


@wp.kernel
def _quartic_directional_slope(
    pose: wp.array[wp.transform],
    correction: wp.array[wp.spatial_vector],
    merit: wp.array[wp.float64],
    evaluations: wp.array[int],
):
    i = wp.tid()
    if i == 0:
        evaluations[0] += 1
    x = wp.transform_get_translation(pose[i])[0]
    slope = x * x * x * correction[i][0]
    merit[i] = wp.float64(slope)


def _structural_kkt_directional_search_rejects_overshoot(test, device):
    """Backtrack a nonlinear overshoot and exactly restore a non-finite island."""
    ids = wp.array([0, 1], dtype=int, device=device)
    original = wp.array(
        [wp.transform((1.0, 0.0, 0.0), wp.quat_identity())] * 2,
        dtype=wp.transform,
        device=device,
    )
    pose = wp.clone(original)
    correction = wp.array(
        [[float("nan"), 0, 0, 0, 0, 0], [-4, 0, 0, 0, 0, 0]],
        dtype=wp.spatial_vector,
        device=device,
    )
    com = wp.zeros(2, dtype=wp.vec3, device=device)
    state = wp.ones(2, dtype=int, device=device)
    enabled = wp.ones(2, dtype=bool, device=device)
    scale = wp.ones(2, dtype=float, device=device)
    merit = wp.zeros(2, dtype=wp.float64, device=device)
    initial_slope = wp.empty(2, dtype=wp.float64, device=device)
    pending = wp.ones(1, dtype=int, device=device)
    evaluations = wp.zeros(1, dtype=int, device=device)
    no_disp = wp.zeros(2, dtype=wp.vec3, device=device)
    no_disp_ang = wp.zeros(2, dtype=wp.vec3, device=device)

    def solve():
        wp.copy(pose, original)
        scale.fill_(1.0)
        pending.fill_(1)
        evaluations.zero_()
        wp.launch(
            _quartic_directional_slope,
            2,
            inputs=[pose, correction],
            outputs=[merit, evaluations],
            device=device,
        )
        wp.launch(
            rigid_vbd_kkt._begin_directional_search,
            2,
            inputs=[state, merit],
            outputs=[enabled, scale, initial_slope, pending],
            device=device,
        )

        def apply_trial(trial):
            wp.launch(
                rigid_vbd_kkt._apply_trial_correction,
                2,
                inputs=[ids, ids, correction, scale, com, original, no_disp, no_disp_ang, 0, original],
                outputs=[pose, pending, no_disp, no_disp_ang],
                device=device,
            )
            wp.launch(
                _quartic_directional_slope,
                2,
                inputs=[pose, correction],
                outputs=[merit, evaluations],
                device=device,
            )
            pending.zero_()
            wp.launch(
                rigid_vbd_kkt._update_directional_search,
                2,
                inputs=[enabled, merit, initial_slope, trial == 4],
                outputs=[scale, pending],
                device=device,
            )

        for trial in range(5):
            if device.is_cpu or (device.is_capturing and wp.is_conditional_graph_supported()):
                wp.capture_if(pending, apply_trial, trial=trial)
            else:
                apply_trial(trial)
        wp.launch(
            rigid_vbd_kkt._apply_trial_correction,
            2,
            inputs=[ids, ids, correction, scale, com, original, no_disp, no_disp_ang, 0, original],
            outputs=[pose, pending, no_disp, no_disp_ang],
            device=device,
        )

    solve()
    with mock.patch.object(wp.array, "numpy", side_effect=AssertionError("host readback in line search")):
        with wp.ScopedCapture(device) as capture:
            solve()
        wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(pose.numpy()[0], original.numpy()[0])
    # The first trial reaches x=-3: slopes -4 and 108 interpolate to 1/28,
    # clamped to 0.1. The re-evaluated x=0.6 has negative slope and is accepted.
    # This is a safeguarded directional search, not fixed-factor halving.
    np.testing.assert_allclose(pose.numpy()[1, 0], 0.6, atol=1.0e-7)
    np.testing.assert_allclose(scale.numpy(), [0.0, 0.1], atol=1.0e-8)
    test.assertLess(float(pose.numpy()[1, 0]) ** 4, 1.0)
    test.assertEqual(
        int(evaluations.numpy()[0]),
        3 if device.is_cpu or wp.is_conditional_graph_supported() else 6,
    )


@wp.kernel
def _count_global_contact_refresh(slot: int, counts: wp.array[int]):
    counts[slot] += 1


def _structural_kkt_captured_contact_shortcut_preserves_state(test, device):
    """Skip completed GPU trials while preserving contact and joint history."""
    if not wp.is_conditional_graph_supported():
        test.skipTest("CUDA conditional graphs are unavailable")
    model, pipeline, contacts, solver, _, _, _ = _build_grounded_chain(
        device,
        global_iterations=1,
        segments=8,
        contact_history=True,
        deterministic=wp.DeterministicMode.RUN_TO_RUN,
    )
    state_in, state_out, control = model.state(), model.state(), model.control()
    backend = solver._structural_graph_kkt
    solver._structural_graph_kkt = None
    dt = 1.0 / 600.0
    for _ in range(15):
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, dt)
        state_in, state_out = state_out, state_in
    solver._structural_graph_kkt = backend
    force = np.zeros((model.body_count, 6), dtype=np.float32)
    force[model.body_count // 2, :3] = (20.0, 5.0, -1.0)
    state_in.body_f.assign(force)
    pipeline.collide(state_in, contacts)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)

    counts = wp.zeros(2, dtype=int, device=device)
    solve = backend.solve

    def counted_solve(*args, **kwargs):
        slot = int(kwargs.get("translation_only", False))
        refresh = kwargs["refresh_contacts"]

        def counted_refresh(*refresh_args, **refresh_kwargs):
            wp.launch(_count_global_contact_refresh, 1, inputs=[slot], outputs=[counts], device=device)
            refresh(*refresh_args, **refresh_kwargs)

        kwargs["refresh_contacts"] = counted_refresh
        return solve(*args, **kwargs)

    backend.solve = counted_solve
    saved = []
    seen = set()
    for owner in [solver, backend, state_in, state_out, control, contacts, *backend.buckets]:
        for value in vars(owner).values():
            if isinstance(value, wp.array) and id(value) not in seen:
                seen.add(id(value))
                saved.append((value, wp.clone(value)))

    def restore():
        for value, original in saved:
            wp.copy(value, original)
        counts.zero_()

    solver.step(state_in, state_out, control, contacts, dt)
    restore()
    outputs = (
        state_out.body_q,
        state_out.body_qd,
        solver.body_body_contact_lambda,
        solver.joint_lambda_lin,
        solver.joint_lambda_ang,
        solver.joint_drive_lambda,
        solver.joint_limit_lambda,
    )
    results = []
    refresh_counts = []
    for conditional in (False, True):
        with ExitStack() as stack:
            if not conditional:
                stack.enter_context(mock.patch.object(wp, "is_conditional_graph_supported", return_value=False))
            stack.enter_context(mock.patch.object(wp.array, "numpy", side_effect=AssertionError("host readback")))
            with wp.ScopedCapture(device) as capture:
                solver.step(state_in, state_out, control, contacts, dt)
        reference = None
        for _ in range(5):
            restore()
            wp.capture_launch(capture.graph)
            current = tuple(value.numpy() for value in outputs)
            if reference is not None:
                for actual, expected in zip(current, reference, strict=True):
                    np.testing.assert_array_equal(actual, expected)
            reference = current
        results.append(reference)
        refresh_counts.append(counts.numpy())
    for actual, expected in zip(results[1], results[0], strict=True):
        np.testing.assert_array_equal(actual, expected)
    test.assertLess(refresh_counts[1][0], refresh_counts[0][0])
    # Translation reconciliation has one initial evaluation and at most
    # fifteen refreshed evaluations, also when represented by a graph loop.
    test.assertLessEqual(refresh_counts[1][1], 15)

    # Default atomic reductions take the same bounded reconciliation loop and the
    # same stopping rule as the reproducible modes.
    # Other directional and collective-mode conditionals remain valid.
    with test.subTest(deterministic=False):
        model, pipeline, contacts, solver, _, _, _ = _build_grounded_chain(
            device,
            global_iterations=1,
            segments=8,
            contact_history=True,
            deterministic=wp.DeterministicMode.NOT_GUARANTEED,
        )
        state_in, state_out, control = model.state(), model.state(), model.control()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, dt)
        with (
            mock.patch.object(wp, "capture_if", wraps=wp.capture_if) as conditional,
            mock.patch.object(wp, "capture_while", wraps=wp.capture_while) as loop,
            mock.patch.object(wp.array, "numpy", side_effect=AssertionError("host readback")),
            wp.ScopedCapture(device) as capture,
        ):
            solver.step(state_in, state_out, control, contacts, dt)
        test.assertNotIn("translate", [call.args[1].__name__ for call in conditional.call_args_list])
        test.assertIn("repeat_translation", [call.args[1].__name__ for call in loop.call_args_list])
        wp.capture_launch(capture.graph)
        test.assertTrue(np.isfinite(state_out.body_q.numpy()).all())


def _structural_kkt_cooperative_paired_matches_serial(test, device):
    """Compare captured physical states, including batches and narrow fallback."""

    def run(model, cooperative):
        solver = newton.solvers.SolverVBD(
            model,
            iterations=2,
            rigid_compliant_alm=True,
            rigid_joint_global_iterations=1,
            deterministic=wp.DeterministicMode.RUN_TO_RUN,
        )
        bucket = solver._structural_graph_kkt.closed_tree_buckets[0]
        selected = (
            bucket.use_paired_backbone
            and device.is_cuda
            and bucket.closure_count >= rigid_vbd_kkt._PAIRED_COOPERATIVE_MIN_CLOSURES
        )
        test.assertEqual(bucket.use_cooperative_paired, selected)
        test.assertEqual(hasattr(bucket, "cooperative_diagonal"), selected)
        test.assertFalse(hasattr(bucket, "cooperative_work"))
        if selected and not cooperative:
            bucket.use_cooperative_paired = False
            bucket.paired_cr_levels = []
            stride = 1
            active = bucket.paired_backbone_count
            while active > rigid_vbd_kkt._CR_SERIAL_TERMINAL_SIZE:
                eliminated = len(range(stride, bucket.paired_backbone_count, 2 * stride))
                survivors = len(range(0, bucket.paired_backbone_count, 2 * stride))
                bucket.paired_cr_levels.append((stride, eliminated, survivors))
                stride *= 2
                active = survivors
            bucket.paired_terminal_stride = stride
        state_in, state_out = model.state(), model.state()
        control = model.control()
        solver.step(state_in, state_out, control, None, 1.0 / 600.0)

        def steps():
            solver.step(state_out, state_in, control, None, 1.0 / 600.0)
            solver.step(state_in, state_out, control, None, 1.0 / 600.0)

        if device.is_cuda:
            with wp.ScopedCapture(device) as capture:
                steps()
            for _ in range(8):
                wp.capture_launch(capture.graph)
        else:
            for _ in range(8):
                steps()
        return state_out.body_q.numpy(), state_out.body_qd.numpy()

    for closures, worlds, length in (
        (1, 1, 2),
        (1, 2, 3),
        (1, 1, 10),
        (2, 1, 10),
        (2, 4, 34),
        (6, 4, 34),
        (11, 2, 34),
        (17, 2, 34),
    ):
        model = _build_replicated_fixed_topology(
            device,
            closed=True,
            worlds=worlds,
            link_count=length,
            closure_count=closures,
        )
        model.set_gravity((0.0, 0.0, -9.81))
        reference = run(model, False)
        candidate = run(model, True)
        repeated = run(model, True)
        for actual, expected, repeat in zip(candidate, reference, repeated, strict=True):
            test.assertTrue(np.isfinite(actual).all())
            np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=1.0e-7)
            np.testing.assert_array_equal(actual.view(np.uint8), repeat.view(np.uint8))


def _structural_kkt_cooperative_paired_reuses_factors(test, device):
    """Frozen factor reuse gives the same RHS/closure solution under capture."""
    model, _, _ = _build_loop_with_branch(
        device,
        ring_segments=32,
        stiffness=1.0e12,
        bend_stiffness=1.0e4,
        add_chord=True,
    )
    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    bucket = solver._structural_graph_kkt.closed_tree_buckets[0]
    test.assertTrue(bucket.use_cooperative_paired)
    original = bucket._solve_paired_backbone_system
    exercised = []

    def compare(factorize, rhs, response):
        test.assertTrue(factorize)
        saved_rhs, saved_response = wp.clone(rhs), wp.clone(response)
        original(True, rhs, response)
        expected_rhs = rhs.numpy().copy()
        expected_response = response.numpy().copy()
        with wp.ScopedCapture(device) as capture:
            wp.copy(rhs, saved_rhs)
            wp.copy(response, saved_response)
            original(False, rhs, response)
        for _ in range(3):
            wp.capture_launch(capture.graph)
            for actual, expected in ((rhs.numpy(), expected_rhs), (response.numpy(), expected_response)):
                # Structured arrays are homogeneous packed FP64; compare every scalar.
                np.testing.assert_allclose(
                    actual.view(np.float64),
                    expected.view(np.float64),
                    rtol=2.0e-11,
                    atol=2.0e-11,
                )
        exercised.append(True)

    bucket._solve_paired_backbone_system = compare
    _simulate(model, solver, 1, 1.0 / 600.0)
    test.assertTrue(exercised)


def _structural_kkt_single_loop_scratch_preserves_coverage(test, device):
    """Block-shared cooperative work must not consume global payload or displace islands."""
    models = [
        _build_replicated_fixed_topology(device, closed=True, worlds=4, link_count=10, closure_count=1),
    ]
    for components in (((20, 1), (6, 0)), ((20, 2), (20, 1), (6, 0))):
        mixed = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        for length, closures in components:
            template, _ = _build_fixed_topology_template(link_count=length, closure_count=closures)
            mixed.add_builder(template, xform=wp.transform((0.0, float(closures), 0.0), wp.quat_identity()))
        mixed.color(balance_colors=False)
        models.append(mixed.finalize(device=device))
    for model in models:
        # This is the established single-loop allocation/schedule, not G0.
        with mock.patch.object(rigid_vbd_kkt, "_PAIRED_COOPERATIVE_MIN_CLOSURES", 2):
            serial, serial_bytes = _build_structural_backend_with_intercepted_bytes(model)
        expected_ids = serial.graph_body_ids.numpy()
        full, full_bytes = _build_structural_backend_with_intercepted_bytes(model)
        test.assertEqual(full_bytes, serial_bytes)
        np.testing.assert_array_equal(full.graph_body_ids.numpy(), expected_ids)
        for budget in sorted({serial_bytes - 1, serial_bytes, full_bytes - 1, full_bytes}):
            with test.subTest(bodies=model.body_count, budget=budget):
                with mock.patch.object(rigid_vbd_kkt, "_structural_payload_budget", return_value=budget):
                    selected, allocated = _build_structural_backend_with_intercepted_bytes(model)
                test.assertEqual(selected._estimated_bytes, allocated)
                test.assertLessEqual(allocated, budget)
                if budget >= serial_bytes:
                    np.testing.assert_array_equal(selected.graph_body_ids.numpy(), expected_ids)
                for bucket in selected.closed_tree_buckets:
                    test.assertEqual(bucket.use_cooperative_paired, bucket._diagnostics.cooperative_paired)
                    test.assertEqual(hasattr(bucket, "cooperative_diagonal"), bucket.use_cooperative_paired)
                    test.assertFalse(hasattr(bucket, "cooperative_work"))
                    test.assertFalse(hasattr(bucket, "cooperative_pivots"))


def _structural_kkt_grouped_rows_preserve_storage_and_mapping(test, device):
    """Stream row scheduling reuses metadata storage and preserves graph incidence."""
    builder, _ = _build_fixed_topology_template(link_count=12, closure_count=3)
    for joint in range(0, builder.joint_count, 2):
        builder.joint_type[joint] = newton.JointType.BALL
    builder.color(balance_colors=False)
    model = builder.finalize(device=device)
    backend, allocated = _build_structural_backend_with_intercepted_bytes(model)
    tree = backend.closed_tree_buckets[0].tree
    arrays = (tree.joint_ids, tree.node_row, tree.body_rows)
    pointers = tuple(array.ptr for array in arrays)
    joint_ids = tree.joint_ids_host.copy()
    node_rows = tree.node_row.numpy().copy()
    body_rows = tree.body_rows.numpy().copy()
    tree.group_rows_by_joint_type(model.joint_type.numpy())
    test.assertEqual(tuple(array.ptr for array in (tree.joint_ids, tree.node_row, tree.body_rows)), pointers)
    np.testing.assert_array_equal(tree.joint_ids_host[tree.body_rows.numpy()], joint_ids[body_rows])
    active = node_rows >= 0
    np.testing.assert_array_equal(tree.joint_ids_host[tree.node_row.numpy()[active]], joint_ids[node_rows[active]])
    test.assertGreater(len(tree.row_slices), 1)
    for ids, count, *_ in tree.row_slices:
        test.assertEqual(len(ids), count)
        test.assertEqual(len(np.unique(model.joint_type.numpy()[ids.numpy()])), 1)
    streamed, streamed_bytes = _build_structural_backend_with_intercepted_bytes(model, enable_stream_concurrency=True)
    test.assertEqual(backend._estimated_bytes, allocated)
    test.assertEqual(streamed._estimated_bytes, streamed_bytes)
    test.assertEqual(allocated, streamed_bytes)


def _run_stiff_ball_tree(test, device, *, depth, stiffness, global_iterations, iterations, bound, steps=30):
    """Hang a binary tree of ball joints from the world and bound its largest joint gap."""
    builder = newton.ModelBuilder()
    link_len, mass, radius = 0.05, 0.05, 0.01
    ixx = 0.5 * mass * radius * radius
    iyy = mass * (3.0 * radius * radius + link_len * link_len) / 12.0
    inertia = wp.mat33(ixx, 0.0, 0.0, 0.0, iyy, 0.0, 0.0, 0.0, iyy)

    def quat_x_to(d):
        x = np.array([1.0, 0.0, 0.0])
        c = float(np.dot(x, d))
        if c > 1.0 - 1.0e-12:
            return wp.quat_identity()
        ax = np.cross(x, d)
        s = float(np.linalg.norm(ax))
        if s < 1.0e-12:
            return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(np.pi))
        return wp.quat_from_axis_angle(wp.vec3(*(ax / s)), float(np.arctan2(s, c)))

    joints = []

    def add_link(direction, origin, parent):
        center = origin + 0.5 * link_len * direction
        q = quat_x_to(direction)
        body = builder.add_link(xform=wp.transform(wp.vec3(*center), q), mass=mass, inertia=inertia)
        pxf = (
            wp.transform(wp.vec3(*origin), q)
            if parent < 0
            else wp.transform(wp.vec3(0.5 * link_len, 0.0, 0.0), wp.quat_identity())
        )
        cxf = wp.transform(wp.vec3(-0.5 * link_len, 0.0, 0.0), wp.quat_identity())
        joints.append(builder.add_joint_ball(parent=parent, child=body, parent_xform=pxf, child_xform=cxf))
        return body, origin + link_len * direction

    d0 = np.array([float(np.cos(0.6)), 0.0, -float(np.sin(0.6))])
    root, end0 = add_link(d0, np.array([0.0, 0.0, 0.25]), -1)
    frontier = [(root, d0, end0, 0)]
    cursor = 0
    while cursor < len(frontier):
        body, d, end, level = frontier[cursor]
        cursor += 1
        if level == depth:
            continue
        u = np.cross(d, [0.0, 1.0, 0.0])
        u = u / np.linalg.norm(u)
        v = np.cross(d, u)
        for branch_idx in range(2):
            phi = np.pi * branch_idx + 0.7 * level
            dc = d + 0.45 * (np.cos(phi) * u + np.sin(phi) * v)
            dc /= np.linalg.norm(dc)
            child, child_end = add_link(dc, end, body)
            frontier.append((child, dc, child_end, level + 1))

    builder.add_articulation(joints)
    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=iterations,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=global_iterations,
        rigid_joint_linear_ke=stiffness,
        rigid_joint_linear_kd=0.0,
        rigid_joint_angular_kd=0.0,
    )
    s_in, s_out, ctl = model.state(), model.state(), model.control()
    jp = model.joint_parent.numpy()
    xp = model.joint_X_p.numpy()[:, :3].astype(float)
    xc = model.joint_X_c.numpy()[:, :3].astype(float)
    max_gap = 0.0
    for _ in range(steps):
        solver.step(s_in, s_out, ctl, None, 1.0 / 600.0)
        s_in, s_out = s_out, s_in
        q = s_in.body_q.numpy().astype(float)
        u_q, w_q = q[:, 3:6], q[:, 6:7]
        t_c = 2.0 * np.cross(u_q, xc)
        pc = q[:, :3] + xc + w_q * t_c + np.cross(u_q, t_c)
        par = np.maximum(jp, 0)
        u_p, w_p = q[par, 3:6], q[par, 6:7]
        t_p = 2.0 * np.cross(u_p, xp)
        pa = q[par, :3] + xp + w_p * t_p + np.cross(u_p, t_p)
        pa[jp < 0] = xp[jp < 0]
        max_gap = max(max_gap, float(np.linalg.norm(pa - pc, axis=1).max()))
    test.assertLess(max_gap, bound)


def _structural_kkt_bushy_tree_avoids_unbacked_row_contraction(test, device):
    """Keep stiff bushy binary trees free of un-backed degree-2 row-pivot penalty blowup."""
    # Un-backed degree-2 row contraction explodes above 0.1 m here; body-backed keeps < 1e-5 m.
    _run_stiff_ball_tree(test, device, depth=6, stiffness=1.0e8, global_iterations=4, iterations=8, bound=1.0e-4)


def _structural_kkt_anchored_tree_survives_near_rigid_rows(test, device):
    """Keep an anchored tree closed at a row stiffness of 1e12.

    Every tree route must eliminate the row to the world after its body. Eliminated first, it is the
    bare pivot -1/K, and float32 then erases the body's inertia: with the row to the world raked first
    this tree opens by 93 um after 60 steps, 0.62 m after 120 and 145 m after 300. Rooting the tree at
    that row keeps it at 0.2 um through 120 steps.
    """
    _run_stiff_ball_tree(
        test, device, depth=5, stiffness=1.0e12, global_iterations=2, iterations=2, bound=1e-05, steps=120
    )


@wp.kernel
def _probe_frame_inv_contact_and_rhs_kernel(
    body_q: wp.array[wp.transform],
    body_inertia_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    inertia: wp.mat33,
    dt: float,
    out_torque_y: wp.array[float],
    out_rhs_fx: wp.array[float],
):
    # 1. Symmetric two-foot contact (+-0.125 m around COM (+u, 0, 0)) at x = 1024.0 m under 1000 N/foot normal load
    u = 3.0517578125e-5  # 2^-15 (0.5 ULP of 1024.0)
    n = wp.vec3(0.0, 0.0, 1.0)
    z3 = wp.vec3(0.0)
    _f0, tau_l, _ll0, _al0, _aa0, _f1, _t1, _ll1, _al1, _aa1 = rigid_vbd_kkt.evaluate_rigid_contact_from_collision(
        0,
        -1,
        body_q,
        body_q_prev,
        body_com,
        wp.vec3(-0.125 - u, 0.0, 0.0),
        z3,
        z3,
        wp.vec3(0.0, 0.0, -0.05),
        z3,
        n,
        1.0e-3,
        1.0e6,
        1.0e6,
        1.0e6,
        0.0,
        z3,
        0.0,
        1.0e-4,
        0,
        1,
        dt,
        z3,
        False,
    )
    _f2, tau_r, _ll2, _al2, _aa2, _f3, _t3, _ll3, _al3, _aa3 = rigid_vbd_kkt.evaluate_rigid_contact_from_collision(
        0,
        -1,
        body_q,
        body_q_prev,
        body_com,
        wp.vec3(0.125 + 3.0 * u, 0.0, 0.0),
        z3,
        z3,
        wp.vec3(0.0, 0.0, -0.05),
        z3,
        n,
        1.0e-3,
        1.0e6,
        1.0e6,
        1.0e6,
        0.0,
        z3,
        0.0,
        1.0e-4,
        0,
        1,
        dt,
        z3,
        False,
    )
    out_torque_y[0] = tau_l[1] + tau_r[1]

    # 2. Temporal COM inertial force in _body_objective_rhs under pure small pitch at x = 1024.0 m
    rhs = rigid_vbd_kkt._body_objective_rhs(
        body_q[0], body_inertia_q[0], 1.0, inertia, body_com[0], z3, z3, 1.0 / (dt * dt)
    )
    out_rhs_fx[0] = rhs[0]


def _structural_kkt_frame_invariance_preserves_off_origin_dynamics(test, device):
    """Body-relative contact levers and temporal COM differences match origin values at x = 1024 m."""
    u = 2.0**-15
    dt = 1.0e-3
    q0_rot = wp.quat_identity()
    q1_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 1.5e-3)
    inertia = wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
    com_arr = wp.array([[u, 0.0, -0.20]], dtype=wp.vec3, device=device)

    results = {}
    for x0 in (0.0, 1024.0):
        q_cur = wp.array([wp.transform(wp.vec3(x0, 0.0, 0.05), q0_rot)], dtype=wp.transform, device=device)
        q_star = wp.array([wp.transform(wp.vec3(x0, 0.0, 0.05), q1_rot)], dtype=wp.transform, device=device)
        tau_y = wp.zeros(1, dtype=float, device=device)
        rhs_fx = wp.zeros(1, dtype=float, device=device)
        wp.launch(
            _probe_frame_inv_contact_and_rhs_kernel,
            1,
            inputs=[q_cur, q_star, q_cur, com_arr, inertia, dt],
            outputs=[tau_y, rhs_fx],
            device=device,
        )
        # Also verify finalize_rigid_bodies COM velocity invariance under rotation from q_cur -> q_star!
        q_prev_buf = wp.clone(q_cur)
        qd_out = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        qd_mir = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        q_out = wp.empty_like(q_star)
        wp.launch(
            update_body_velocity,
            1,
            inputs=[
                dt,
                q_star,
                com_arr,
                wp.ones(1, dtype=float, device=device),
                wp.zeros(1, dtype=wp.vec3, device=device),
                wp.zeros(1, dtype=wp.vec3, device=device),
                0,
            ],
            outputs=[q_prev_buf, qd_out, qd_mir, q_out],
            device=device,
        )
        results[x0] = (float(tau_y.numpy()[0]), float(rhs_fx.numpy()[0]), float(qd_out.numpy()[0, 0]))

    tau0, fx0, vx0 = results[0.0]
    tau_far, fx_far, vx_far = results[1024.0]
    # Naive world-frame levers give |tau_far| ~ 0.083 N*m (vs 0.0), |fx_far - fx0| ~ 30.5 N, |vx_far - vx0| ~ 0.0305 m/s!
    test.assertLess(abs(tau0), 1.0e-6)
    test.assertLess(abs(tau_far), 1.0e-6)
    test.assertLess(abs(fx_far - fx0), 1.0e-3)
    test.assertLess(abs(vx_far - vx0), 1.0e-6)


def _structural_kkt_displacement_form_keeps_free_com(test, device):
    """Keep the centre of mass of a free articulation with zero momentum where it is, 256 m from the origin.

    Taken from rounded float32 poses, a body's velocity random-walks by ulp(x)/h per step and moves the
    centre of mass by about ulp(x) n^1.5. The displacement form takes the velocity from each body's
    displacement from its inertial target and keeps the rounding out of it.
    """

    def drift(offset):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        inertia = wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
        joints = []
        previous = -1
        for index in range(3):
            body = builder.add_link(
                xform=wp.transform(wp.vec3(offset + 0.2 * index, 0.0, 1.0), wp.quat_identity()),
                mass=1.0,
                inertia=inertia,
            )
            if previous < 0:
                joints.append(builder.add_joint_free(child=body))
            else:
                joints.append(
                    builder.add_joint_revolute(
                        parent=previous,
                        child=body,
                        parent_xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()),
                        child_xform=wp.transform(wp.vec3(-0.1, 0.0, 0.0), wp.quat_identity()),
                        axis=wp.vec3(0.0, 0.0, 1.0),
                    )
                )
            previous = body
        builder.add_articulation(joints)
        builder.color()
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverVBD(
            model,
            iterations=1,
            rigid_compliant_alm=True,
            rigid_joint_global_iterations=1,
        )
        state_in, state_out, control = model.state(), model.state(), model.control()
        qd = np.zeros((3, 6), dtype=np.float32)
        qd[0, 5] = 2.0
        qd[1, 5] = -2.0
        qd[2, 5] = 2.0
        state_in.body_qd.assign(qd)
        masses = model.body_mass.numpy().astype(np.float64)
        com0 = (masses[:, None] * state_in.body_q.numpy()[:, :3].astype(np.float64)).sum(0) / masses.sum()
        for _ in range(1000):
            solver.step(state_in, state_out, control, None, 1.0e-3)
            state_in, state_out = state_out, state_in
        com1 = (masses[:, None] * state_in.body_q.numpy()[:, :3].astype(np.float64)).sum(0) / masses.sum()
        momentum = (masses[:, None] * state_in.body_qd.numpy()[:, :3].astype(np.float64)).sum(0)
        return float(np.linalg.norm(com1 - com0)), float(np.linalg.norm(momentum))

    # Poses taken as rounded differences: 71 mm and 0.24 kg m/s after one second here; the stored
    # pose itself is still rounded, so the centre of mass is bounded at a few ulp(256 m) = 3e-5 m.
    com_drift, momentum = drift(256.0)
    test.assertLess(momentum, 1.0e-4)
    # The stored pose still random-walks by its rounding, eps L per step, so n steps move the centre of
    # mass by about eps L sqrt(n) = 0.48 mm here (measured 0.06 mm on CUDA, 1.06 mm on the CPU).
    test.assertLess(com_drift, 5.0 * 2.0**-24 * 256.0 * np.sqrt(1000.0))


def _build_revolute_pendulum(device):
    """Two links on revolute joints about y, hung from the world and released from rest at (0.2, -0.3) rad."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    joints = []
    for link in range(2):
        body = builder.add_link(
            mass=1.0,
            com=wp.vec3(0.0, 0.0, -0.25),
            inertia=wp.mat33(0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.002),
            lock_inertia=True,
        )
        joints.append(
            builder.add_joint_revolute(
                parent=link - 1,
                child=body,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.5 if link == 0 else -0.5)),
                armature=0.0,
                limit_lower=-3.0,
                limit_upper=3.0,
            )
        )
    builder.add_articulation(joints)
    builder.joint_q[:] = [0.2, -0.3]
    builder.color()
    model = builder.finalize(device=device)
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)
    return model


def _swing_revolute_pendulum(device, *, integration, dt, steps, corrections):
    """Return the pendulum's joint angles and body poses after ``steps`` steps of size ``dt``."""
    model = _build_revolute_pendulum(device)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=corrections,
        rigid_joint_global_iterations=corrections,
        rigid_joint_linear_ke=1.0e6,
        rigid_joint_angular_ke=1.0e6,
        rigid_joint_linear_kd=0.0,
        rigid_joint_angular_kd=0.0,
        rigid_time_integration=integration,
    )
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    for _ in range(steps):
        state_in.clear_forces()
        solver.step(state_in, state_out, control, None, dt)
        state_in, state_out = state_out, state_in
    joint_q = wp.zeros_like(model.joint_q)
    joint_qd = wp.zeros_like(model.joint_qd)
    newton.eval_ik(model, state_in, joint_q, joint_qd)
    return model, joint_q.numpy().astype(np.float64), state_in.body_q.numpy()


def _rigid_bdf2_follows_fine_step_motion(test, device):
    """BDF2 is second order: at 2 ms it is far closer to the fine-step motion than backward Euler at 2 ms."""
    duration = 1.0
    _, reference, _ = _swing_revolute_pendulum(
        device, integration="bdf2", dt=2.5e-4, steps=round(duration / 2.5e-4), corrections=2
    )
    errors = {}
    for integration in ("bdf1", "bdf2"):
        _, angles, _ = _swing_revolute_pendulum(
            device, integration=integration, dt=2.0e-3, steps=round(duration / 2.0e-3), corrections=1
        )
        errors[integration] = float(np.linalg.norm(angles - reference))
    test.assertGreater(errors["bdf1"], 2.0e-3)
    test.assertLess(errors["bdf2"], 0.2 * errors["bdf1"])


def _rigid_bdf2_single_correction_holds_at_small_step(test, device):
    """One correction per step keeps the joints closed under BDF2 at a small step.

    A share of each joint force carried by the multiplier of the previous update makes BDF2 unstable below
    ``K h^2 / m`` of about 0.27 (here about 0.1): the gap grows to millimetres within this run. The
    correction's own penalty removes that share.
    """
    model, _, body_q = _swing_revolute_pendulum(device, integration="bdf2", dt=5.0e-4, steps=2000, corrections=1)
    test.assertTrue(np.isfinite(body_q).all())
    test.assertLess(_max_joint_gap(model, body_q, np.arange(model.joint_count)), 2.0e-4)


def _rigid_bdf2_repeated_restart_is_stable(test, device):
    """A body whose state is edited on every step does not blow up under BDF2.

    The start rule of a body without history, repeated on every step, is unstable on rows stiffer than
    ``36 m / dt^2`` (here ``K dt^2 / m = 100``: the joints open and the state goes non-finite within 450
    steps). The rule is therefore not applied on two steps in a row.
    """
    model = _build_revolute_pendulum(device)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_joint_global_iterations=1,
        rigid_joint_linear_ke=1.0e8,
        rigid_joint_angular_ke=1.0e8,
        rigid_joint_linear_kd=0.0,
        rigid_joint_angular_kd=0.0,
        rigid_time_integration="bdf2",
    )
    state_in, state_out, control = model.state(), model.state(), model.control()
    for _ in range(600):
        solver._bdf2_last_q.zero_()  # as if every body had been edited
        state_in.clear_forces()
        solver.step(state_in, state_out, control, None, 1.0e-3)
        state_in, state_out = state_out, state_in
    body_q = state_in.body_q.numpy()
    test.assertTrue(np.isfinite(body_q).all())
    test.assertLess(_max_joint_gap(model, body_q, np.arange(model.joint_count)), 1.0e-4)


def _structural_kkt_revolute_chain64_holds_near_rigidity(test, device):
    """A 64-link revolute chain holds micrometer gaps and tracks motion at K = 1e9 and 1e12."""

    def build_chain64():
        b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        joints = []
        for i in range(64):
            body = b.add_link(
                mass=0.1,
                com=wp.vec3(0.0, 0.0, -0.05),
                inertia=wp.mat33(8.5e-5, 0.0, 0.0, 0.0, 8.5e-5, 0.0, 0.0, 0.0, 5.0e-6),
                lock_inertia=True,
            )
            joints.append(
                b.add_joint_revolute(
                    parent=i - 1,
                    child=body,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, 8.0 if i == 0 else -0.1)),
                    armature=0.0,
                    limit_lower=-3.0,
                    limit_upper=3.0,
                )
            )
        b.add_articulation(joints)
        q0 = [0.0] * 64
        q0[0] = 1.2
        b.joint_q[:] = q0
        b.color()
        m = b.finalize(device=device)
        newton.eval_fk(m, m.joint_q, m.joint_qd, m)
        return m, joints

    steps = 500 if wp.get_device(device).is_cuda else 40
    ref_poses = None
    if wp.get_device(device).is_cuda:
        m_ref, _ = build_chain64()
        s_ref = newton.solvers.SolverVBD(
            m_ref,
            iterations=1,
            rigid_joint_global_iterations=1,
            rigid_joint_linear_ke=1.0e8,
            rigid_joint_angular_ke=1.0e8,
            rigid_joint_linear_kd=0.0,
            rigid_joint_angular_kd=0.0,
            rigid_time_integration="bdf2",
        )
        r_in, r_out, r_ctl = m_ref.state(), m_ref.state(), m_ref.control()
        for _ in range(steps):
            s_ref.step(r_in, r_out, r_ctl, None, 1.0e-3)
            r_in, r_out = r_out, r_in
        ref_poses = r_in.body_q.numpy()[:, :3].copy()

    for ke in (1.0e9, 1.0e12):
        for integration in ("bdf1", "bdf2"):
            model, joints = build_chain64()
            solver = newton.solvers.SolverVBD(
                model,
                iterations=1,
                rigid_joint_global_iterations=1,
                rigid_joint_linear_ke=ke,
                rigid_joint_angular_ke=ke,
                rigid_joint_linear_kd=0.0,
                rigid_joint_angular_kd=0.0,
                rigid_time_integration=integration,
            )
            state_in, state_out, control = model.state(), model.state(), model.control()
            for _ in range(steps):
                solver.step(state_in, state_out, control, None, 1.0e-3)
                state_in, state_out = state_out, state_in
            poses = state_in.body_q.numpy()
            test.assertTrue(np.isfinite(poses).all())
            test.assertLess(_max_joint_gap(model, poses, joints), 2.5e-5)
            if ref_poses is not None and integration == "bdf2":
                max_traj_diff = float(np.linalg.norm(poses[:, :3] - ref_poses, axis=1).max())
                test.assertLess(max_traj_diff, 2.0e-3)


def _structural_correction_penalty_excludes_rod_rows(test, device):
    """Articulated rows solve the exact zero-memory saddle (s_f = 0) and sync clean duals; rod rows keep ALM rho."""
    # 1. Non-rod rows have zero dependence on joint_rho / stale joint_lambda in linearization; rod rows keep rho.
    for joint_type, depends_on_rho in ((newton.JointType.ROD, True), (newton.JointType.REVOLUTE, False)):
        model, _, _, _ = _build_joint_pair(device, joint_type)
        solver = newton.solvers.SolverVBD(
            model,
            iterations=1,
            rigid_compliant_alm=True,
            rigid_joint_linear_ke=1.0e5,
            rigid_joint_angular_ke=1.0e5,
            rigid_joint_global_iterations=1,
        )
        test.assertFalse(hasattr(solver, "_joint_correction_rho"))
        bucket = solver._structural_graph_kkt.buckets[0]
        solver.joint_lambda_lin.fill_(wp.vec3(12.0, -7.0, 5.0))
        solver.joint_rho.fill_(1.0e4)
        _linearize_bucket(model, solver, bucket, model.body_q, 1.0e-3)
        res_lo = bucket.residual.numpy().copy()
        solver.joint_rho.fill_(1.0e9)
        _linearize_bucket(model, solver, bucket, model.body_q, 1.0e-3)
        res_hi = bucket.residual.numpy().copy()
        diff = float(np.max(np.abs(res_lo - res_hi)))
        if depends_on_rho:
            test.assertGreater(diff, 1.0e-5)
        else:
            test.assertEqual(diff, 0.0)

    # 2. Near-rigid K=1e12 ball pendulum syncs exact physical weight O(m g) into joint_lambda without float32 blow-up.
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    body = b.add_link(mass=2.0, inertia=wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1))
    j_ball = b.add_joint_ball(parent=-1, child=body)
    b.add_articulation([j_ball])
    b.color()
    m_ball = b.finalize(device=device)
    s_ball = newton.solvers.SolverVBD(
        m_ball,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_linear_ke=1.0e12,
        rigid_joint_global_iterations=1,
    )
    si, so = m_ball.state(), m_ball.state()
    for _ in range(5):
        s_ball.step(si, so, m_ball.control(), None, 1.0e-3)
        si, so = so, si
    lam_z = float(s_ball.joint_lambda_lin.numpy()[0, 2])
    test.assertAlmostEqual(lam_z, -2.0 * 9.81, places=2)


def _structural_kkt_unit_scale_invariance(test, device):
    """Dimensionless Delassus conditioning selects identical routes and trajectories across 1e-4x..1e+4x mass/K scaling."""

    def run_scaled(scale: float) -> tuple[str, float, np.ndarray]:
        b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        joints = []
        for i in range(64):
            body = b.add_link(
                mass=0.1 * scale,
                com=wp.vec3(0.0, 0.0, -0.05),
                inertia=wp.mat33(8.5e-5 * scale, 0.0, 0.0, 0.0, 8.5e-5 * scale, 0.0, 0.0, 0.0, 5.0e-6 * scale),
                lock_inertia=True,
            )
            joints.append(
                b.add_joint_revolute(
                    parent=i - 1,
                    child=body,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, 8.0 if i == 0 else -0.1)),
                    armature=0.0,
                )
            )
        b.add_articulation(joints)
        q0 = [0.0] * 64
        q0[0] = 1.2
        b.joint_q[:] = q0
        b.color()
        m = b.finalize(device=device)
        newton.eval_fk(m, m.joint_q, m.joint_qd, m)
        stiff = 1.0e9 * scale
        s = newton.solvers.SolverVBD(
            m,
            iterations=1,
            rigid_joint_global_iterations=1,
            rigid_joint_linear_ke=stiff,
            rigid_joint_angular_ke=stiff,
            rigid_joint_linear_kd=0.0,
            rigid_joint_angular_kd=0.0,
            rigid_time_integration="bdf2",
        )
        route = s._structural_graph_kkt.tree_buckets[0]._diagnostics.selected_route
        si, so, ctl = m.state(), m.state(), m.control()
        n_steps = 200 if wp.get_device(device).is_cuda else 30
        for _ in range(n_steps):
            s.step(si, so, ctl, None, 1.0e-3)
            si, so = so, si
        q = si.body_q.numpy()
        return route, _max_joint_gap(m, q, joints), q[:, :3].copy()

    route_ref, gap_ref, pos_ref = run_scaled(1.0)
    test.assertLess(gap_ref, 5.0e-5)
    for scale in (1.0e-4, 1.0e4):
        route_s, gap_s, pos_s = run_scaled(scale)
        test.assertEqual(route_s, route_ref)
        test.assertLess(gap_s, 2.5e-5)
        test.assertLess(float(np.linalg.norm(pos_s - pos_ref, axis=1).max()), 1.0e-3)


def _structural_kkt_clean_saddle_stress_stateless_and_resolve_stable(test, device, stiffness=1.0e12):
    """A float32 tree holds micron gaps at one correction per step and is bit-exact after reset().

    The stress term of a correction comes from the joint forces of the tree's last solve. reset() must clear
    them, or the step after it differs from the first step of a fresh solver.
    """
    builder = newton.ModelBuilder()
    bodies = []
    joints = []
    depth = 6
    count = (1 << depth) - 1
    for idx in range(count):
        lvl = int(np.floor(np.log2(idx + 1)))
        pos_in_lvl = (idx + 1) - (1 << lvl)
        x = 0.04 * (lvl + 1)
        y = 0.03 * (pos_in_lvl - 0.5 * ((1 << lvl) - 1))
        z = 2.0 - 0.04 * lvl
        body = builder.add_link(xform=wp.transform(wp.vec3(x, y, z), wp.quat_identity()), mass=0.05)
        builder.add_shape_sphere(body, radius=0.012)
        bodies.append(body)
        if idx == 0:
            joints.append(
                builder.add_joint_ball(
                    parent=-1,
                    child=body,
                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(-x, -y, 0.0), wp.quat_identity()),
                )
            )
        else:
            p_idx = (idx - 1) // 2
            p_lvl = int(np.floor(np.log2(p_idx + 1)))
            p_pos = (p_idx + 1) - (1 << p_lvl)
            px = 0.04 * (p_lvl + 1)
            py = 0.03 * (p_pos - 0.5 * ((1 << p_lvl) - 1))
            pz = 2.0 - 0.04 * p_lvl
            mid = wp.vec3(0.5 * (px + x), 0.5 * (py + y), 0.5 * (pz + z))
            joints.append(
                builder.add_joint_ball(
                    parent=bodies[p_idx],
                    child=body,
                    parent_xform=wp.transform(mid - wp.vec3(px, py, pz), wp.quat_identity()),
                    child_xform=wp.transform(mid - wp.vec3(x, y, z), wp.quat_identity()),
                )
            )
    builder.add_articulation(joints)
    builder.color(balance_colors=False)
    model = builder.finalize(device=device)
    joint_ids = np.asarray(joints, dtype=np.int32)
    dt = 1.0 / 600.0

    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_linear_ke=stiffness,
        rigid_joint_global_iterations=1,
    )
    s0, s1 = model.state(), model.state()
    ctl = model.control()
    solver.step(s0, s1, ctl, None, dt)
    first_step_q = s1.body_q.numpy().copy()

    for _ in range(120):
        solver.step(s1, s0, ctl, None, dt)
        s1, s0 = s0, s1
    test.assertLess(_max_joint_gap(model, s1.body_q.numpy(), joint_ids), 1.0e-5)

    sr0, sr1 = model.state(), model.state()
    solver.reset(sr0)
    solver.step(sr0, sr1, ctl, None, dt)
    np.testing.assert_array_equal(sr1.body_q.numpy(), first_step_q)


def _structural_kkt_stored_stress_reset_matches_fresh_solver(test, device):
    """Below near rigidity too, the step after reset() is the first step of a fresh solver, bit for bit.

    Fault caught: reset() leaves the joint forces that a tree below K = 1e10 keeps for its stress term.
    """
    _structural_kkt_clean_saddle_stress_stateless_and_resolve_stable(test, device, stiffness=1.0e8)


def _one_and_two_correction_chain_difference(device, *, links=16, steps=600):
    """Largest body-position difference [m] between one and two corrections per step on a hanging revolute chain.

    Thin links (0.1 m, 0.1 kg) on revolute joints about y at K = 1e8, released as a straight line 1.2 rad from
    the vertical, stepped at 1 ms.
    """
    poses = []
    for corrections in (1, 2):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        joints = []
        for link in range(links):
            body = builder.add_link(
                mass=0.1,
                com=wp.vec3(0.0, 0.0, -0.05),
                inertia=wp.mat33(8.5e-5, 0.0, 0.0, 0.0, 8.5e-5, 0.0, 0.0, 0.0, 5.0e-6),
                lock_inertia=True,
            )
            joints.append(
                builder.add_joint_revolute(
                    parent=link - 1,
                    child=body,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 if link == 0 else -0.1)),
                    armature=0.0,
                    limit_lower=-3.0,
                    limit_upper=3.0,
                )
            )
        builder.add_articulation(joints)
        builder.joint_q[0] = 1.2
        builder.color()
        model = builder.finalize(device=device)
        newton.eval_fk(model, model.joint_q, model.joint_qd, model)
        solver = newton.solvers.SolverVBD(
            model,
            iterations=corrections,
            rigid_joint_global_iterations=corrections,
            rigid_joint_linear_ke=1.0e8,
            rigid_joint_angular_ke=1.0e8,
            rigid_joint_linear_kd=0.0,
            rigid_joint_angular_kd=0.0,
        )
        state_in, state_out, control = model.state(), model.state(), model.control()
        for _ in range(steps):
            state_in.clear_forces()
            solver.step(state_in, state_out, control, None, 1.0e-3)
            state_in, state_out = state_out, state_in
        poses.append(state_in.body_q.numpy()[:, :3].astype(np.float64))
    return float(np.abs(poses[0] - poses[1]).max())


def _structural_kkt_first_correction_stress_from_stored_forces(test, device):
    """One correction per step follows two on a hanging revolute chain at K = 1e8.

    Fault caught: the first correction of a step builds its stress term from the row force at the inertial
    target, which is K times the free-fall gap of the step (490 N at K = 1e8 and 1 ms, on a chain whose tension
    is a few newtons). That bound is 29 % of the rotational inertia of these links; the correction is
    over-damped, and one correction per step drifts from two by 9 mm within 0.6 s (2 um with stored forces).
    """
    difference = _one_and_two_correction_chain_difference(device)
    test.assertLess(difference, 1.0e-4, f"one against two corrections: {difference:.3e} m")


def _structural_kkt_stored_stress_reset_respects_world_mask(test, device):
    """reset() of one world zeroes the stored joint forces of that world's rows and no others.

    Fault caught: the reset clears the stored forces of every world, so the next step of a world that was not
    reset loses its stress term.
    """
    template, _ = _build_fixed_topology_template(link_count=6, closure_count=0)
    builder = newton.ModelBuilder()
    for world in range(3):
        builder.add_world(template, xform=wp.transform(wp.vec3(0.0, 0.5 * world, 0.0), wp.quat_identity()))
    builder.color(balance_colors=False)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(model, iterations=1, rigid_compliant_alm=True, rigid_joint_global_iterations=1)
    backend = solver._structural_graph_kkt
    buckets = [bucket for bucket in backend.tree_buckets if id(bucket) in backend.stored_stress_buckets]
    test.assertTrue(buckets)
    state_in, state_out, control = model.state(), model.state(), model.control()
    for _ in range(8):
        state_in.clear_forces()
        solver.step(state_in, state_out, control, None, 1.0 / 600.0)
        state_in, state_out = state_out, state_in
    before = [bucket.solution.numpy().copy() for bucket in buckets]
    mask = np.zeros(model.world_count + 1, dtype=bool)
    mask[1] = True
    solver.reset(state_in, world_mask=wp.array(mask, dtype=wp.bool, device=device), flags=0)
    joint_world = model.joint_world.numpy()
    kept = 0
    for bucket, old in zip(buckets, before, strict=True):
        new = bucket.solution.numpy()
        nodes = bucket.row_node.numpy()
        nodes = np.where(nodes < 0, -2 - nodes, nodes)
        worlds = joint_world[bucket.joint_ids.numpy()]
        np.testing.assert_array_equal(new[nodes[worlds == 1]], 0.0)
        np.testing.assert_array_equal(new[nodes[worlds != 1]], old[nodes[worlds != 1]])
        kept += int(np.count_nonzero(old[nodes[worlds != 1]]))
        test.assertGreater(np.count_nonzero(old[nodes[worlds == 1]]), 0)
    test.assertGreater(kept, 0)


def _structural_kkt_per_island_active_contact_sweep_gating(test, device):
    """Flag islands with active contacts for local sweeps while keeping mid-air islands sweep-free."""
    builder = newton.ModelBuilder()
    cfg = builder.default_shape_cfg.copy()
    cfg.ke = 1.0e5
    cfg.kd = 100.0
    cfg.mu = 0.5
    builder.add_ground_plane(cfg=cfg)

    # Island 0: 2-link chain resting in active ground contact (equilibrium z ~ 0.01998 m, no bounce)
    g0 = builder.add_link(xform=wp.transform(wp.vec3(0.0, -1.0, 0.01998), wp.quat_identity()), mass=0.2)
    builder.add_shape_box(g0, hx=0.04, hy=0.03, hz=0.02, cfg=cfg.copy())
    g1 = builder.add_link(xform=wp.transform(wp.vec3(0.1, -1.0, 0.01998), wp.quat_identity()), mass=0.2)
    builder.add_shape_box(g1, hx=0.04, hy=0.03, hz=0.02, cfg=cfg.copy())
    gj = builder.add_joint_ball(
        parent=g0,
        child=g1,
        parent_xform=wp.transform(wp.vec3(0.05, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.05, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([gj])

    # Island 1: 2-link pendulum swinging in mid-air far above ground
    p0 = builder.add_link(xform=wp.transform(wp.vec3(0.5, 1.0, 3.0), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(p0, hx=0.5, hy=0.04, hz=0.04, cfg=cfg.copy())
    p1 = builder.add_link(xform=wp.transform(wp.vec3(1.5, 1.0, 3.0), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(p1, hx=0.5, hy=0.04, hz=0.04, cfg=cfg.copy())
    pj0 = builder.add_joint_revolute(
        parent=-1,
        child=p0,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 1.0, 3.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
    )
    pj1 = builder.add_joint_revolute(
        parent=p0,
        child=p1,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([pj0, pj1])
    builder.color(balance_colors=False)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    cp = newton.CollisionPipeline(model)
    contacts = cp.contacts()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_linear_ke=1.0e6,
        rigid_joint_angular_ke=1.0e6,
        rigid_joint_global_iterations=1,
    )
    s0, s1 = model.state(), model.state()
    ctl = model.control()
    for _ in range(10):
        cp.collide(s0, contacts)
        solver.step(s0, s1, ctl, contacts, 1.0e-3)
        s0, s1 = s1, s0

    flags = solver._structural_graph_kkt.limit_search_islands.numpy().tolist()
    test.assertEqual(flags, [1, 0])


def _structural_kkt_joint_force_keeps_rod_on_contact_taut(test, device):
    """A joint force must not move the pose a step starts from.

    A 3 g link is pressed on its hinge stop by a constant joint force of 0.04 N m. A rod of 40 segments of 1 mm
    runs from a point 5 mm off the hinge axis over a fixed pin to a 6 g slider, which a joint force of 2 N pulls;
    the rod's moment on the hinge (0.01 N m) is below the preload. Integrated into the starting pose, the two
    joint forces turn the link by 4 deg per step and shift the slider as if both were free; the rod leaves and
    hits the pin, and its tension is 4 N off at 4 iterations and 35 N off at 8. From the prediction without them
    the rod carries 2 N and the link stays on its stop.
    """
    segments, seg, arm, preload, pull, dt = 40, 1.0e-3, 5.0e-3, -0.04, 2.0, 1.0 / 1800.0
    for iterations in (4, 8):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.inertia_absolute_tolerance = 1.0e-20
        builder.inertia_correction = 0.0
        link = builder.add_link(
            xform=wp.transform(wp.vec3(0.0125, 0.0, 0.0), wp.quat_identity()),
            mass=3.3e-3,
            inertia=wp.mat33(9.35e-8, 0.0, 0.0, 0.0, 1.87e-7, 0.0, 0.0, 0.0, 1.73e-7),
        )
        hinge = builder.add_joint_revolute(
            parent=-1,
            child=link,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform(wp.vec3(-0.0125, 0.0, 0.0), wp.quat_identity()),
            limit_lower=0.0,
            limit_upper=1.5,
            limit_ke=300.0,
            limit_kd=0.01,
            target_ke=0.04,
            target_kd=1.5e-4,
        )
        slider = builder.add_link(
            xform=wp.transform(wp.vec3(-seg * segments, arm, 0.0), wp.quat_identity()),
            mass=6.0e-3,
            inertia=wp.mat33(3.6e-8, 0.0, 0.0, 0.0, 3.6e-8, 0.0, 0.0, 0.0, 3.6e-8),
        )
        rod = newton.Rod(
            [wp.vec3(-seg * i, arm, 0.0) for i in range(segments + 1)],
            radius=5.0e-4,
            rest_straight=True,
            stretch_rigidity=2000.0,
            shear_rigidity=2000.0,
            bend_rigidity=1.0e-8,
            twist_rigidity=1.0e-8,
        )
        # 0.79 g/m: a segment weighs about a milligram (the rod's mass rule differs between Newton versions)
        cfg = newton.ModelBuilder.ShapeConfig(
            density=0.79e-3 / (np.pi * 5.0e-4**2), ke=1.0e8, kd=0.0, mu=0.0, gap=5.0e-4
        )
        bodies, rod_joints = builder.add_rod(
            rod=rod,
            cfg=cfg,
            stretch_damping=10.0,
            shear_damping=10.0,
            wrap_in_articulation=False,
        )
        rod_bodies = set(bodies)
        rod_shapes = [shape for shape, body in enumerate(builder.shape_body) if body in rod_bodies]
        for i, shape_a in enumerate(rod_shapes):
            for shape_b in rod_shapes[i + 1 :]:
                builder.add_shape_collision_filter_pair(shape_a, shape_b)
        # A fixed pin under the middle of the rod lifts it by 0.3 mm: the rod lies on it under its tension.
        builder.add_shape_sphere(
            -1,
            xform=wp.transform(wp.vec3(-0.5 * seg * segments, arm - 2.0e-3 - 5.0e-4 + 3.0e-4, 0.0), wp.quat_identity()),
            radius=2.0e-3,
            cfg=cfg,
        )
        first = builder.add_joint_ball(
            parent=link,
            child=bodies[0],
            parent_xform=wp.transform(wp.vec3(-0.0125, arm, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.5 * seg), wp.quat_identity()),
        )
        last = builder.add_joint_ball(
            parent=bodies[-1],
            child=slider,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * seg), wp.quat_identity()),
            child_xform=wp.transform_identity(),
        )
        # One tree; the slider's guide closes the loop through the world and stays outside the articulation.
        builder.add_articulation([hinge, *rod_joints, first, last])
        slide = builder.add_joint_prismatic(
            parent=-1,
            child=slider,
            axis=wp.vec3(-1.0, 0.0, 0.0),
            parent_xform=wp.transform(wp.vec3(-seg * segments, arm, 0.0), wp.quat_identity()),
            child_xform=wp.transform_identity(),
            limit_lower=-0.002,
            limit_upper=0.03,
            limit_ke=2.0e4,
            limit_kd=20.0,
            target_ke=0.0,
            target_kd=20.0,
        )
        builder.color()
        model = builder.finalize(device=device)
        pipeline = newton.CollisionPipeline(model, contact_matching="latest")
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverVBD(
            model,
            iterations=iterations,
            rigid_joint_global_iterations=1,
            rigid_joint_linear_ke=1.0e5,
            rigid_joint_angular_ke=1.0e5,
            rigid_joint_linear_kd=0.0,
            rigid_joint_angular_kd=0.0,
            rigid_contact_history=True,
            rigid_body_contact_buffer_size=64,
        )
        state_0, state_1, control = model.state(), model.state(), model.control()
        qd_start = model.joint_qd_start.numpy()
        joint_f = control.joint_f.numpy()
        joint_f[int(qd_start[hinge])] = preload
        joint_f[int(qd_start[slide])] = pull
        control.joint_f.assign(joint_f)
        joint_q, joint_qd = wp.clone(model.joint_q), wp.clone(model.joint_qd)
        hinge_q = int(model.joint_q_start.numpy()[hinge])
        middle = bodies[segments // 2]
        steps = 300
        angle_error = tension_error = 0.0
        for step in range(steps):
            state_0.clear_forces()
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
            if step < steps // 2:
                continue
            newton.eval_ik(model, state_0, joint_q, joint_qd)
            angle_error = max(angle_error, abs(float(joint_q.numpy()[hinge_q])))
            positions = state_0.body_q.numpy()[:, :3]
            stretch = float(np.linalg.norm(positions[middle + 1] - positions[middle])) - seg
            tension_error = max(tension_error, abs(2000.0 / seg * stretch - pull))
        test.assertLess(tension_error, 0.05, f"rod tension off by {tension_error:.3f} N at {iterations} iterations")
        test.assertLess(np.degrees(angle_error), 0.02, f"link {np.degrees(angle_error):.4f} deg off its stop")


class TestVBDRigidKKT(unittest.TestCase):
    """Validate the optional rigid VBD structural KKT backend."""

    pass


add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_grouped_rows_preserve_storage_and_mapping",
    _structural_kkt_grouped_rows_preserve_storage_and_mapping,
    devices=get_test_devices(),
)


add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_single_loop_scratch_preserves_coverage",
    _structural_kkt_single_loop_scratch_preserves_coverage,
    devices=get_test_devices(),
)


add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_retains_interior_supported_branches",
    _structural_kkt_retains_interior_supported_branches,
    devices=get_test_devices(),
)


add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_prescribed_split_matches_float64_oracle",
    _structural_kkt_prescribed_split_matches_float64_oracle,
    devices=get_test_devices(),
)


add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_zero_correction_preserves_pose",
    _structural_kkt_zero_correction_preserves_pose,
    devices=get_test_devices(),
)


for _test in (
    _structural_kkt_reversing_pull_preserves_support,
    _structural_kkt_axial_pull_converges_without_contacts,
    _structural_kkt_translation_resolves_friction_breakaway,
    _structural_kkt_free_translation_preserves_momentum,
):
    add_function_test(TestVBDRigidKKT, "test" + _test.__name__, _test, devices=get_test_devices())
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_directional_search_rejects_overshoot",
    _structural_kkt_directional_search_rejects_overshoot,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_bounds_material_frame_rod_stress",
    _structural_kkt_bounds_material_frame_rod_stress,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_inactive_limit_adds_curvature_without_force",
    _structural_kkt_inactive_limit_adds_curvature_without_force,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_open_contact_activates_on_penetration",
    _structural_kkt_open_contact_activates_on_penetration,
    devices=get_test_devices(),
)


add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_free_root_is_constraint_free",
    _structural_kkt_free_root_is_constraint_free,
    devices=get_test_devices(),
)


add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_selects_supported_complete_graphs",
    _structural_kkt_selects_supported_complete_graphs,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_classifies_dynamic_contact_topology",
    _structural_kkt_classifies_dynamic_contact_topology,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_relaxes_dynamic_contact_by_curvature",
    _structural_kkt_relaxes_dynamic_contact_by_curvature,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_majorizes_only_represented_dynamic_contact",
    _structural_kkt_majorizes_only_represented_dynamic_contact,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_inactive_trees_do_not_apply_free_body_corrections",
    _structural_kkt_inactive_trees_do_not_apply_free_body_corrections,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_reuses_compact_scratch",
    _structural_kkt_reuses_compact_scratch,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_refreshes_notified_joint_enable_topology",
    _structural_kkt_refreshes_notified_joint_enable_topology,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_uses_effective_kinematic_mass",
    _structural_kkt_uses_effective_kinematic_mass,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_cable_kkt_preserves_dahl_state_under_capture",
    _cable_kkt_preserves_dahl_state_under_capture,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_cable_kkt_reduces_long_path_error",
    _cable_kkt_reduces_long_path_error,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_reduces_fixed_chain_error",
    _structural_kkt_reduces_fixed_chain_error,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_batches_replicated_topologies",
    _structural_kkt_batches_replicated_topologies,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_assembles_closure_compliance_for_every_batch",
    _structural_kkt_assembles_closure_compliance_for_every_batch,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_drive_matches_finite_material_equilibrium",
    _structural_kkt_drive_matches_finite_material_equilibrium,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_preserves_near_rigid_joint_limits",
    _structural_kkt_preserves_near_rigid_joint_limits,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_cable_kkt_tree_handles_stiff_y_junction",
    _cable_kkt_tree_handles_stiff_y_junction,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_cable_kkt_closes_stiff_loop_with_one_global_pass",
    _cable_kkt_closes_stiff_loop_with_one_global_pass,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_cable_kkt_preserves_local_ground_sliding",
    _cable_kkt_preserves_local_ground_sliding,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_cable_kkt_substeps_preserve_sliding",
    _cable_kkt_substeps_preserve_sliding,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_cable_kkt_breaks_away_in_first_loaded_steps",
    _cable_kkt_breaks_away_in_first_loaded_steps,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_cable_kkt_substeps_preserve_static_friction",
    _cable_kkt_substeps_preserve_static_friction,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_preserves_free_translation",
    _structural_kkt_preserves_free_translation,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_external_dynamic_contact_blocks_free_translation",
    _structural_kkt_external_dynamic_contact_blocks_free_translation,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_translation_stop_keeps_coupled_step",
    _structural_kkt_translation_stop_keeps_coupled_step,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_cable_kkt_preserves_local_grounded_loop",
    _cable_kkt_preserves_local_grounded_loop,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_cable_kkt_closed_cycle_preserves_local_contact",
    _cable_kkt_closed_cycle_preserves_local_contact,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_cable_kkt_contact_capture_is_finite",
    _cable_kkt_contact_capture_is_finite,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_cable_kkt_near_hard_capture_is_finite",
    _cable_kkt_near_hard_capture_is_finite,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_cable_kkt_near_hard_closed_cycle_is_finite_under_capture",
    _cable_kkt_near_hard_closed_cycle_is_finite_under_capture,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_cable_kkt_closed_cycle_bounds_multiplier_history",
    _cable_kkt_closed_cycle_bounds_multiplier_history,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_robust_block_inverse",
    _structural_kkt_robust_block_inverse,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_path_matches_float64_oracle",
    _structural_kkt_path_matches_float64_oracle,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_tree_matches_float64_oracle",
    _structural_kkt_tree_matches_float64_oracle,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_closure_matches_float64_oracle",
    _structural_kkt_closure_matches_float64_oracle,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_uncovered_routes_match_float64_oracle",
    _structural_kkt_uncovered_routes_match_float64_oracle,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_closure_separates_reaction_gauge",
    _structural_kkt_closure_separates_reaction_gauge,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_path_solve_has_small_residual",
    _structural_kkt_path_solve_has_small_residual,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_persistent_cr_matches_level_schedule",
    _structural_kkt_persistent_cr_matches_level_schedule,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_cr_tail_preserves_factors",
    _structural_kkt_cr_tail_preserves_factors,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_ragged_path_bucket_matches_standalone_paths",
    _structural_kkt_ragged_path_bucket_matches_standalone_paths,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_ragged_tree_bucket_matches_separate_buckets",
    _structural_kkt_ragged_tree_bucket_matches_separate_buckets,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_cr_tail_preserves_captured_trajectory",
    _structural_kkt_cr_tail_preserves_captured_trajectory,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_fused_closure_matches_level_schedule",
    _structural_kkt_fused_closure_matches_level_schedule,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_fused_closure_matches_dense_oracle",
    _structural_kkt_fused_closure_matches_dense_oracle,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_paired_coarse_lanes_match_serial_schedule",
    _structural_kkt_paired_coarse_lanes_match_serial_schedule,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_suppresses_nonfinite_correction",
    _structural_kkt_suppresses_nonfinite_correction,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_fused_tree_levels_match_level_schedule",
    _structural_kkt_fused_tree_levels_match_level_schedule,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_backbone_tail_response_is_finalized",
    _structural_kkt_backbone_tail_response_is_finalized,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_angular_damping_matches_local_force",
    _structural_kkt_angular_damping_matches_local_force,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_joint_linearizations_match_finite_difference",
    _structural_kkt_joint_linearizations_match_finite_difference,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_prelocal_stage_bounds_coarse_chain_error",
    _structural_kkt_prelocal_stage_bounds_coarse_chain_error,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_fused_stress_clears_disabled_rows",
    _structural_kkt_fused_stress_clears_disabled_rows,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_uses_explicit_iteration_budget",
    _structural_kkt_uses_explicit_iteration_budget,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_row_assembly_matches_block_equations",
    _structural_kkt_row_assembly_matches_block_equations,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_contact_transitions_are_repeatable",
    _structural_kkt_contact_transitions_are_repeatable,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_contacts_preserve_independent_acceleration",
    _structural_kkt_contacts_preserve_independent_acceleration,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_contact_history_requires_valid_matching_provenance",
    _structural_kkt_contact_history_requires_valid_matching_provenance,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_contact_incidence_overflow_is_bounded",
    _structural_kkt_contact_incidence_overflow_is_bounded,
    devices=get_cuda_test_devices(),
    check_output=False,
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_inherits_solver_deterministic_mode",
    _structural_kkt_inherits_solver_deterministic_mode,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_default_mode_matches_run_to_run",
    _structural_kkt_default_mode_matches_run_to_run,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_captures_numeric_inertia_refresh",
    _structural_kkt_captures_numeric_inertia_refresh,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_memory_preflight_boundary_is_deterministic",
    _structural_kkt_memory_preflight_boundary_is_deterministic,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_estimates_allocation_payloads_exactly",
    _structural_kkt_estimates_allocation_payloads_exactly,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_oversized_closure_falls_back_before_allocation",
    _structural_kkt_oversized_closure_falls_back_before_allocation,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_all_memory_fallback_uses_local_schedule",
    _structural_kkt_all_memory_fallback_uses_local_schedule,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_preflight_is_deterministic_and_capture_safe",
    _structural_kkt_preflight_is_deterministic_and_capture_safe,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_chunks_oversized_equal_topology_batch",
    _structural_kkt_chunks_oversized_equal_topology_batch,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_closed_preflight_skips_unused_contraction",
    _structural_kkt_closed_preflight_skips_unused_contraction,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_mixed_rod_tree_uses_stable_leaf_route",
    _structural_kkt_mixed_rod_tree_uses_stable_leaf_route,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_partial_fallback_preserves_healthy_island",
    _structural_kkt_partial_fallback_preserves_healthy_island,
    devices=get_test_devices(),
)


add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_near_hard_closure_matches_float64_oracle",
    _structural_kkt_near_hard_closure_matches_float64_oracle,
    devices=get_test_devices(),
)


add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_dense_closure_matches_float64_oracle",
    _structural_kkt_dense_closure_matches_float64_oracle,
    devices=get_test_devices(),
)


add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_captured_contact_shortcut_preserves_state",
    _structural_kkt_captured_contact_shortcut_preserves_state,
    devices=get_cuda_test_devices(),
)


add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_cooperative_paired_matches_serial",
    _structural_kkt_cooperative_paired_matches_serial,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_cooperative_paired_reuses_factors",
    _structural_kkt_cooperative_paired_reuses_factors,
    devices=get_cuda_test_devices(),
)


def _run_rigid_step_with_trace(model, solver, contacts):
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    with wp.ScopedTimer("rigid-step", print=False, synchronize=True, cuda_filter=wp.TIMING_KERNEL) as timer:
        state_in.clear_forces()
        solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
    wp.synchronize_device(model.device)
    return {
        "q": state_out.body_q.numpy(),
        "qd": state_out.body_qd.numpy(),
        "objectives": [
            solver.body_forces.numpy(),
            solver.body_torques.numpy(),
            solver.body_hessian_ll.numpy(),
            solver.body_hessian_al.numpy(),
            solver.body_hessian_aa.numpy(),
        ],
        "duals": [
            solver.joint_penalty_k.numpy(),
            solver.joint_lambda_lin.numpy(),
            solver.joint_lambda_ang.numpy(),
        ],
        "kernels": [item.name for item in timer.timing_results if item.device == model.device],
    }


def _build_paired_open_chain_model(device, body_count, chain_count=1):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    half = 0.04
    moment = ((2.0 * half) ** 2 + (2.0 * half) ** 2) / 12.0
    inertia = wp.mat33(moment, 0.0, 0.0, 0.0, moment, 0.0, 0.0, 0.0, moment)
    for chain in range(chain_count):
        y = 0.2 * chain
        bodies = [
            builder.add_link(
                xform=wp.transform(wp.vec3(0.1 * index, y, 1.0), wp.quat_identity()),
                mass=1.0,
                inertia=inertia,
                label=f"paired_open_chain_{chain}_body_{index}",
            )
            for index in range(body_count)
        ]
        joints = [
            builder.add_joint_revolute(
                parent=-1,
                child=bodies[0],
                parent_xform=wp.transform(wp.vec3(0.0, y, 1.0), wp.quat_identity()),
                child_xform=wp.transform_identity(),
                axis=wp.vec3(0.0, 0.0, 1.0),
                damping=0.02,
            )
        ]
        for index in range(1, body_count):
            joints.append(
                builder.add_joint_revolute(
                    parent=bodies[index - 1],
                    child=bodies[index],
                    parent_xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()),
                    child_xform=wp.transform_identity(),
                    axis=wp.vec3(0.0, 0.0, 1.0),
                    damping=0.02,
                )
            )
        builder.add_articulation(joints, label=f"paired_open_chain_{chain}")
    builder.color(balance_colors=False)
    return builder.finalize(device=device)


def test_paired_open_chain_global_solve(test, device):
    """The bounded paired route preserves the stock KKT solve and capture path."""

    def planned(*args, **kwargs):
        # The paired route is planned for a step size; plan as the first step of this test (1/240 s) does.
        solver = newton.solvers.SolverVBD(*args, **kwargs)
        solver._plan_structural_graph_kkt(1.0 / 240.0)
        return solver

    model = _build_paired_open_chain_model(device, 64)
    kwargs = {
        "iterations": 4,
        "rigid_joint_global_iterations": 1,
        "rigid_compliant_alm": True,
        "deterministic": wp.DeterministicMode.NOT_GUARANTEED,
    }
    baseline = planned(model, **kwargs)
    candidate = planned(model, **kwargs)
    baseline_bucket = baseline._structural_graph_kkt.tree_buckets[0]
    candidate_bucket = candidate._structural_graph_kkt.tree_buckets[0]
    test.assertEqual(candidate_bucket.use_paired_open_backbone, device.is_cuda)
    baseline_bucket.use_paired_open_backbone = False

    baseline_result = _run_rigid_step_with_trace(model, baseline, None)
    candidate_result = _run_rigid_step_with_trace(model, candidate, None)
    paired_kernel = "initialize_paired_tree_backbone"
    test.assertFalse(any(paired_kernel in name for name in baseline_result["kernels"]))
    test.assertEqual(sum(paired_kernel in name for name in candidate_result["kernels"]), int(device.is_cuda))
    for name in ("q", "qd"):
        np.testing.assert_allclose(candidate_result[name], baseline_result[name], rtol=1.0e-6, atol=1.0e-6)
    for candidate_array, baseline_array in zip(
        candidate_result["objectives"] + candidate_result["duals"],
        baseline_result["objectives"] + baseline_result["duals"],
        strict=True,
    ):
        np.testing.assert_allclose(candidate_array, baseline_array, rtol=1.0e-6, atol=1.0e-6)

    # Identical open chains share one topology bucket. The paired kernels are
    # batch-indexed, so widening from one island must preserve every chain.
    batched_model = _build_paired_open_chain_model(device, 64, chain_count=4)
    batched_baseline = planned(batched_model, **kwargs)
    batched_candidate = planned(batched_model, **kwargs)
    batched_baseline_bucket = batched_baseline._structural_graph_kkt.tree_buckets[0]
    batched_candidate_bucket = batched_candidate._structural_graph_kkt.tree_buckets[0]
    test.assertEqual(batched_candidate_bucket.batch_count, 4)
    test.assertEqual(batched_candidate_bucket.use_paired_open_backbone, device.is_cuda)
    batched_baseline_bucket.use_paired_open_backbone = False
    batched_baseline_result = _run_rigid_step_with_trace(batched_model, batched_baseline, None)
    batched_candidate_result = _run_rigid_step_with_trace(batched_model, batched_candidate, None)
    test.assertEqual(sum(paired_kernel in name for name in batched_candidate_result["kernels"]), int(device.is_cuda))
    for name in ("q", "qd"):
        np.testing.assert_allclose(
            batched_candidate_result[name], batched_baseline_result[name], rtol=1.0e-6, atol=1.0e-6
        )
    for candidate_array, baseline_array in zip(
        batched_candidate_result["objectives"] + batched_candidate_result["duals"],
        batched_baseline_result["objectives"] + batched_baseline_result["duals"],
        strict=True,
    ):
        np.testing.assert_allclose(candidate_array, baseline_array, rtol=1.0e-6, atol=1.0e-6)

    # A Contacts object uses the stock tree solve even when the topology has
    # preplanned paired storage.
    contact_baseline = planned(model, **kwargs)
    contact_candidate = planned(model, **kwargs)
    contact_baseline._structural_graph_kkt.tree_buckets[0].use_paired_open_backbone = False
    contacts = newton.Contacts(0, 0, device=device)
    contact_baseline_result = _run_rigid_step_with_trace(model, contact_baseline, contacts)
    contact_candidate_result = _run_rigid_step_with_trace(model, contact_candidate, contacts)
    test.assertFalse(any(paired_kernel in name for name in contact_candidate_result["kernels"]))
    for name in ("q", "qd"):
        np.testing.assert_array_equal(contact_candidate_result[name], contact_baseline_result[name])
    for candidate_array, baseline_array in zip(
        contact_candidate_result["objectives"] + contact_candidate_result["duals"],
        contact_baseline_result["objectives"] + contact_baseline_result["duals"],
        strict=True,
    ):
        np.testing.assert_array_equal(candidate_array, baseline_array)

    if not device.is_cuda:
        return

    short_model = _build_paired_open_chain_model(device, 63)
    short_solver = planned(short_model, **kwargs)
    test.assertFalse(short_solver._structural_graph_kkt.tree_buckets[0].use_paired_open_backbone)
    refined_model = _build_paired_open_chain_model(device, 256)
    refined_baseline = planned(refined_model, **kwargs)
    refined_candidate = planned(refined_model, **kwargs)
    refined_baseline._structural_graph_kkt.tree_buckets[0].use_paired_open_backbone = False
    refined_bucket = refined_candidate._structural_graph_kkt.tree_buckets[0]
    test.assertIsNotNone(refined_bucket.paired_correction_rhs)
    refined_baseline_result = _run_rigid_step_with_trace(refined_model, refined_baseline, None)
    refined_candidate_result = _run_rigid_step_with_trace(refined_model, refined_candidate, None)
    test.assertEqual(
        sum("compute_paired_tree_backbone_residual" in name for name in refined_candidate_result["kernels"]),
        1,
    )
    test.assertEqual(
        sum("scatter_refined_paired_tree_backbone_solution" in name for name in refined_candidate_result["kernels"]),
        1,
    )
    test.assertFalse(any("add_paired_tree_backbone_correction" in name for name in refined_candidate_result["kernels"]))
    for name in ("q", "qd"):
        np.testing.assert_allclose(
            refined_candidate_result[name], refined_baseline_result[name], rtol=1.0e-6, atol=1.0e-6
        )
    for candidate_array, baseline_array in zip(
        refined_candidate_result["objectives"] + refined_candidate_result["duals"],
        refined_baseline_result["objectives"] + refined_baseline_result["duals"],
        strict=True,
    ):
        np.testing.assert_allclose(candidate_array, baseline_array, rtol=1.0e-6, atol=1.0e-5)

    long_model = _build_paired_open_chain_model(device, 513)
    long_baseline = planned(long_model, **kwargs)
    long_candidate = planned(long_model, **kwargs)
    long_baseline._structural_graph_kkt.tree_buckets[0].use_paired_open_backbone = False
    test.assertTrue(long_candidate._structural_graph_kkt.tree_buckets[0].use_paired_open_backbone)
    long_baseline_result = _run_rigid_step_with_trace(long_model, long_baseline, None)
    long_candidate_result = _run_rigid_step_with_trace(long_model, long_candidate, None)
    for name in ("q", "qd"):
        np.testing.assert_allclose(long_candidate_result[name], long_baseline_result[name], rtol=1.0e-5, atol=1.0e-5)
    deterministic_solver = planned(
        model,
        **{**kwargs, "deterministic": wp.DeterministicMode.RUN_TO_RUN},
    )
    test.assertFalse(deterministic_solver._structural_graph_kkt.tree_buckets[0].use_paired_open_backbone)

    # Compile once before capture, then ensure the natural route records and
    # replays without allocation or host work.
    capture_solver = planned(model, **kwargs)
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    state_in.clear_forces()
    capture_solver.step(state_in, state_out, control, None, 1.0 / 240.0)
    wp.synchronize_device(device)
    with wp.ScopedCapture(device) as capture:
        state_in.clear_forces()
        capture_solver.step(state_in, state_out, control, None, 1.0 / 240.0)
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    test.assertTrue(np.isfinite(state_out.body_q.numpy()).all())


class TestVBDPairedOpenChain(unittest.TestCase):
    pass


add_function_test(
    TestVBDPairedOpenChain,
    "test_paired_open_chain_global_solve",
    test_paired_open_chain_global_solve,
    devices=get_test_devices(),
)


add_function_test(
    TestVBDRigidKKT,
    "test_local_factor_recovery_is_opt_in",
    _local_factor_recovery_is_opt_in,
    devices=get_test_devices(),
)


def _static_contact_modes(test, device):
    """Preserve structural contact outputs across specialized modes and graph replay."""
    with wp.ScopedDevice(device):
        model, pipeline, contacts, solver, _, _, _ = _build_grounded_chain(
            device,
            global_iterations=1,
            segments=4,
            contact_history=True,
            deterministic=wp.DeterministicMode.RUN_TO_RUN,
        )
        state_in, state_out, control = model.state(), model.state(), model.control()
        dt = 1.0 / 600.0
        for _ in range(45):
            state_in.clear_forces()
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in
        test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        backend = solver._structural_graph_kkt
        test.assertIsNotNone(backend)
        backend.island_translation_free.fill_(1)
        names = (
            "body_forces",
            "body_torques",
            "body_hessian_ll",
            "body_hessian_al",
            "body_hessian_aa",
            "body_dynamic_contact_hessian",
        )
        outputs = {}
        for mode in ("full", "forces", "translation"):

            def evaluate(mode=mode):
                solver._reset_structural_solve_state()
                backend.translation_system.zero_()
                backend.translation_support.zero_()
                solver._refresh_structural_contact_objective(
                    state_in,
                    contacts,
                    dt,
                    translation_only=mode == "translation",
                    forces_only=mode == "forces",
                )

            def snapshot():
                result = {name: getattr(solver, name).numpy().copy() for name in names}
                result["translation_system"] = backend.translation_system.numpy().copy()
                result["translation_support"] = backend.translation_support.numpy().copy()
                result["island_contact_state"] = backend.island_contact_state.numpy().copy()
                return result

            evaluate()
            expected = snapshot()
            for _ in range(2):
                evaluate()
                for name, value in snapshot().items():
                    np.testing.assert_array_equal(value, expected[name])
            if device.is_cuda:
                with wp.ScopedCapture(device) as capture:
                    evaluate()
                for _ in range(2):
                    wp.capture_launch(capture.graph)
                    for name, value in snapshot().items():
                        np.testing.assert_array_equal(value, expected[name])
            outputs[mode] = expected

        test.assertGreater(np.linalg.norm(outputs["full"]["body_forces"]), 0.0)
        test.assertGreater(np.linalg.norm(outputs["full"]["body_hessian_ll"]), 0.0)
        for name in ("body_forces", "body_torques"):
            np.testing.assert_array_equal(outputs["full"][name], outputs["forces"][name])
        for name in names[2:]:
            test.assertTrue(np.all(outputs["forces"][name] == 0))
        for name in names:
            test.assertTrue(np.all(outputs["translation"][name] == 0))
        for mode in ("full", "forces"):
            test.assertTrue(np.all(outputs[mode]["translation_system"] == 0))
            test.assertTrue(np.all(outputs[mode]["translation_support"] == 0))
        system = outputs["translation"]["translation_system"]
        # Per island, from the translation evaluation: friction [0:3], Coulomb capacity [3] and the
        # support normal weighted by it [4:7]. All contacts lie on one plane here.
        support = outputs["translation"]["translation_support"]
        test.assertGreater(support[:, 3].min(), 0.0)
        np.testing.assert_allclose(np.linalg.norm(support[:, 4:7], axis=1), support[:, 3], rtol=1.0e-6)
        test.assertTrue(np.all(np.linalg.norm(support[:, 0:3], axis=1) <= (1.0 + 1.0e-6) * support[:, 3]))
        test.assertTrue(np.isfinite(system).all())
        test.assertGreater(np.linalg.norm(system), 0.0)
        # Every contact is against the fixed ground, with COM-centered body frames.
        # Restriction to common translation therefore sums the body linear terms.
        np.testing.assert_array_equal(model.body_com.numpy(), np.zeros((model.body_count, 3)))
        delta = solver.body_inertia_q.numpy()[:, :3] - state_in.body_q.numpy()[:, :3]
        if getattr(solver, "_use_body_displacement", 0):
            # Under the displacement form the inertial term reads the accumulated displacement.
            delta = -solver._body_disp.numpy()
        mass = model.body_mass.numpy().astype(float) / float(np.float32(dt)) ** 2
        expected_rhs = (delta.astype(float) * mass[:, None]).sum(axis=0)
        expected_rhs += outputs["full"]["body_forces"].astype(float).sum(axis=0)
        expected_hessian = outputs["full"]["body_hessian_ll"].astype(float).sum(axis=0)
        expected_hessian += mass.sum() * np.eye(3)
        np.testing.assert_allclose(system[:, :3, 3].sum(axis=0), expected_rhs, rtol=1.0e-6, atol=1.0e-5)
        np.testing.assert_allclose(system[:, :3, :3].sum(axis=0), expected_hessian, rtol=1.0e-6, atol=1.0e-5)

        # Reuse the settled contact manifold to test both damping directions.
        # The local force exporter is independent of the structural gather.
        contact_force = wp.zeros(contacts.rigid_contact_max, dtype=wp.spatial_vector, device=device)
        shape_body = model.shape_body.numpy()
        shape0 = contacts.rigid_contact_shape0.numpy()
        shape1 = contacts.rigid_contact_shape1.numpy()
        count = int(contacts.rigid_contact_count.numpy()[0])
        solver.body_body_contact_material_kd.fill_(10.0)
        for speed in (-0.1, 0.1):
            previous = state_in.body_q.numpy().copy()
            previous[:, 2] -= speed * dt
            solver.body_q_prev.assign(previous)
            damping_outputs = []
            for bidirectional in (False, True):
                with test.subTest(speed=speed, bidirectional=bidirectional):
                    solver.rigid_contact_damping_bidirectional = bidirectional
                    solver._export_body_body_contact_forces(state_in, contacts, dt, contact_force)
                    local_force = np.zeros((model.body_count, 3))
                    for contact, wrench in enumerate(contact_force.numpy()[:count]):
                        for shape, sign in ((shape0[contact], 1.0), (shape1[contact], -1.0)):
                            body = int(shape_body[shape]) if shape >= 0 else -1
                            if body >= 0:
                                local_force[body] += sign * wrench[:3]
                    solver._reset_structural_solve_state()
                    solver._refresh_structural_contact_objective(state_in, contacts, dt)
                    full_force = solver.body_forces.numpy().copy()
                    np.testing.assert_allclose(full_force, local_force, rtol=1.0e-6, atol=1.0e-5)
                    damping_outputs.append(full_force)
                    for mode in ("forces", "translation"):
                        evaluate(mode)
                        if device.is_cuda:
                            with wp.ScopedCapture(device) as capture:
                                evaluate(mode)
                            wp.capture_launch(capture.graph)
                        if mode == "forces":
                            np.testing.assert_array_equal(solver.body_forces.numpy(), full_force)
                        else:
                            rhs = backend.translation_system.numpy()[:, :3, 3].sum(axis=0)
                            expected = (delta.astype(float) * mass[:, None]).sum(axis=0) + local_force.sum(axis=0)
                            np.testing.assert_allclose(rhs, expected, rtol=1.0e-6, atol=1.0e-5)
            if speed < 0.0:
                np.testing.assert_array_equal(*damping_outputs)
            else:
                test.assertGreater(np.linalg.norm(damping_outputs[1] - damping_outputs[0]), 0.01)


add_function_test(
    TestVBDRigidKKT, "test_static_contact_modes", _static_contact_modes, devices=get_test_devices(mode="basic")
)


add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_limit_requests_follow_dynamic_endpoint",
    _structural_kkt_limit_requests_follow_dynamic_endpoint,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_rebuild_accounts_for_global_storage",
    _structural_kkt_rebuild_accounts_for_global_storage,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_limit_capture_fallback_preserves_state",
    _structural_kkt_limit_capture_fallback_preserves_state,
    devices=get_cuda_test_devices(),
)


add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_free_self_contact_conserves_momentum",
    _structural_kkt_free_self_contact_conserves_momentum,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_frame_invariance_preserves_off_origin_dynamics",
    _structural_kkt_frame_invariance_preserves_off_origin_dynamics,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_bushy_tree_avoids_unbacked_row_contraction",
    _structural_kkt_bushy_tree_avoids_unbacked_row_contraction,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_anchored_tree_survives_near_rigid_rows",
    _structural_kkt_anchored_tree_survives_near_rigid_rows,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_displacement_form_keeps_free_com",
    _structural_kkt_displacement_form_keeps_free_com,
    devices=get_test_devices(mode="basic"),
)


def _structural_kkt_multi_anchor_tree_survives_near_rigid_rows(test, device):
    """Keep a multi-world-anchor rigid-angular tree closed at K = 1e12 via fp64 leaf rake."""
    builder = newton.ModelBuilder()
    link_len, mass, radius = 0.05, 0.05, 0.01
    ixx = 0.5 * mass * radius * radius
    iyy = mass * (3.0 * radius * radius + link_len * link_len) / 12.0
    inertia = wp.mat33(ixx, 0.0, 0.0, 0.0, iyy, 0.0, 0.0, 0.0, iyy)

    def quat_x_to(direction):
        x_axis = np.array([1.0, 0.0, 0.0])
        cos_theta = float(np.dot(x_axis, direction))
        if cos_theta > 1.0 - 1.0e-12:
            return wp.quat_identity()
        axis = np.cross(x_axis, direction)
        sin_theta = float(np.linalg.norm(axis))
        if sin_theta < 1.0e-12:
            return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(np.pi))
        return wp.quat_from_axis_angle(wp.vec3(*(axis / sin_theta)), float(np.arctan2(sin_theta, cos_theta)))

    joints = []

    def add_link(direction, origin, parent, use_revolute=False):
        center = origin + 0.5 * link_len * direction
        quat = quat_x_to(direction)
        body = builder.add_link(xform=wp.transform(wp.vec3(*center), quat), mass=mass, inertia=inertia)
        parent_xform = (
            wp.transform(wp.vec3(*origin), quat)
            if parent < 0
            else wp.transform(wp.vec3(0.5 * link_len, 0.0, 0.0), wp.quat_identity())
        )
        child_xform = wp.transform(wp.vec3(-0.5 * link_len, 0.0, 0.0), wp.quat_identity())
        if use_revolute:
            joints.append(
                builder.add_joint_revolute(
                    parent=parent,
                    child=body,
                    parent_xform=parent_xform,
                    child_xform=child_xform,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                )
            )
        else:
            joints.append(
                builder.add_joint_ball(parent=parent, child=body, parent_xform=parent_xform, child_xform=child_xform)
            )
        return body, origin + link_len * direction

    d0 = np.array([float(np.cos(0.6)), 0.0, -float(np.sin(0.6))])
    root, end0 = add_link(d0, np.array([0.0, 0.0, 0.25]), -1, use_revolute=True)
    frontier = [(root, d0, end0, 0)]
    cursor = 0
    last_leaf = -1
    last_end = end0
    while cursor < len(frontier):
        body, direction, end, level = frontier[cursor]
        cursor += 1
        if level == 4:
            last_leaf, last_end = body, end
            continue
        tangent_u = np.cross(direction, [0.0, 1.0, 0.0])
        tangent_u = tangent_u / np.linalg.norm(tangent_u)
        tangent_v = np.cross(direction, tangent_u)
        for branch_idx in range(2):
            phi = np.pi * branch_idx + 0.7 * level
            child_dir = direction + 0.45 * (np.cos(phi) * tangent_u + np.sin(phi) * tangent_v)
            child_dir /= np.linalg.norm(child_dir)
            child, child_end = add_link(child_dir, end, body, use_revolute=(level == 0 and branch_idx == 0))
            frontier.append((child, child_dir, child_end, level + 1))

    builder.add_articulation(joints)
    joints.append(
        builder.add_joint_ball(
            parent=-1,
            child=last_leaf,
            parent_xform=wp.transform(wp.vec3(*last_end), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.5 * link_len, 0.0, 0.0), wp.quat_identity()),
        )
    )
    builder.color()
    model = builder.finalize(device=device)
    joint_ids = np.asarray(joints, dtype=np.int32)

    solver = newton.solvers.SolverVBD(
        model,
        iterations=2,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=2,
        rigid_joint_linear_ke=1.0e12,
        rigid_joint_angular_ke=1.0e12,
        rigid_joint_linear_kd=0.0,
        rigid_joint_angular_kd=0.0,
    )
    bucket = solver._structural_graph_kkt.tree_buckets[0]
    test.assertEqual(bucket._diagnostics.selected_route, "tree_leaf_rake_fp64")
    test.assertEqual(bucket._diagnostics.estimated_bytes, bucket._estimated_bytes)

    state_in, state_out, control = model.state(), model.state(), model.control()
    max_gap = 0.0
    for _ in range(120):
        solver.step(state_in, state_out, control, None, 1.0 / 600.0)
        state_in, state_out = state_out, state_in
        pose = state_in.body_q.numpy()
        test.assertTrue(np.isfinite(pose).all())
        max_gap = max(max_gap, _max_joint_gap(model, pose, joint_ids))
    test.assertLess(max_gap, 1.0e-5)


def _structural_kkt_near_rigid_sled_slides_on_soft_ground(test, device):
    """Preserve soft-contact sliding and unanchored inertia on a K = 1e12 articulated sled."""
    builder = newton.ModelBuilder()
    builder.rigid_gap = 0.002
    cfg = builder.default_shape_cfg.copy()
    cfg.ke = 1.0e4
    cfg.kd = 50.0
    cfg.mu = 0.5
    builder.add_ground_plane(cfg=cfg)
    bodies = []
    joints = []
    spacing = 0.08
    for index in range(8):
        body = builder.add_link(
            xform=wp.transform(wp.vec3(spacing * index, 0.0, 0.02), wp.quat_identity()),
            mass=0.05,
        )
        builder.add_shape_box(body, hx=0.03, hy=0.02, hz=0.02, cfg=cfg.copy())
        bodies.append(body)
        if index > 0:
            joints.append(
                builder.add_joint_fixed(
                    parent=bodies[-2],
                    child=body,
                    parent_xform=wp.transform(wp.vec3(0.5 * spacing, 0.0, 0.0), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(-0.5 * spacing, 0.0, 0.0), wp.quat_identity()),
                )
            )
    builder.add_articulation(joints)
    builder.color(balance_colors=False)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    body_ids = np.asarray(bodies, dtype=np.int32)
    joint_ids = np.asarray(joints, dtype=np.int32)
    total_mass = float(model.body_mass.numpy()[body_ids].sum())
    pull_force = 1.25 * 0.5 * total_mass * 9.81
    dt = 1.0 / 600.0
    slide_steps = 180
    analytic_travel = 0.5 * ((pull_force - 0.5 * total_mass * 9.81) / total_mass) * (slide_steps * dt) ** 2

    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=2,
        rigid_compliant_alm=True,
        rigid_joint_linear_ke=1.0e12,
        rigid_joint_angular_ke=1.0e12,
        rigid_joint_global_iterations=1,
    )
    test.assertEqual(solver._structural_graph_kkt.tree_buckets[0]._diagnostics.selected_route, "tree_leaf_rake_fp64")

    state_in, state_out = model.state(), model.state()
    control = model.control()
    zero_f = np.zeros((model.body_count, 6), dtype=np.float32)
    pull_f = zero_f.copy()
    pull_f[body_ids[0], 0] = pull_force

    for _ in range(90):
        state_in.body_f.assign(zero_f)
        collision_pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, dt)
        state_in, state_out = state_out, state_in
    x_start = float(state_in.body_q.numpy()[body_ids, 0].mean())

    for _ in range(slide_steps):
        state_in.body_f.assign(pull_f)
        collision_pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, dt)
        state_in, state_out = state_out, state_in

    pose = state_in.body_q.numpy()
    test.assertTrue(np.isfinite(pose).all())
    travel = float(pose[body_ids, 0].mean() - x_start)
    test.assertAlmostEqual(travel, analytic_travel, delta=0.05 * analytic_travel)
    test.assertAlmostEqual(float(pose[body_ids, 2].mean()), 0.02, delta=2.0e-3)
    test.assertLess(_max_joint_gap(model, pose, joint_ids), 1.0e-5)


add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_multi_anchor_tree_survives_near_rigid_rows",
    _structural_kkt_multi_anchor_tree_survives_near_rigid_rows,
    devices=get_test_devices(mode="basic"),
)


def _structural_kkt_ball_open_chain_uses_tree_contraction(test, device):
    """Route BALL chains to tree_contraction, well-conditioned REVOLUTE to paired CR, and Woodbury-critical REVOLUTE to leaf rake."""
    if not wp.get_device(device).is_cuda:
        test.skipTest("Paired open-chain CR vs CUDA tree_contraction routing requires CUDA.")
    cases = (
        (False, 1.0e8, "tree_contraction"),
        (True, 1.0e7, "tree_paired_open_backbone_cr"),
        (True, 1.0e8, "tree_leaf_rake"),
    )
    for use_revolute, ke, expected_route in cases:
        builder = newton.ModelBuilder()
        parent = -1
        joints = []
        for idx in range(64):
            body = builder.add_link(
                xform=wp.transform(wp.vec3(0.025 + 0.05 * idx, 0.0, 1.0), wp.quat_identity()),
                mass=0.1,
            )
            builder.add_shape_sphere(body, radius=0.015)
            p_xf = (
                wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity())
                if parent == -1
                else wp.transform(wp.vec3(0.025, 0.0, 0.0), wp.quat_identity())
            )
            c_xf = wp.transform(wp.vec3(-0.025, 0.0, 0.0), wp.quat_identity())
            if use_revolute:
                j = builder.add_joint_revolute(
                    parent=parent, child=body, axis=wp.vec3(0.0, 1.0, 0.0), parent_xform=p_xf, child_xform=c_xf
                )
            else:
                j = builder.add_joint_ball(parent=parent, child=body, parent_xform=p_xf, child_xform=c_xf)
            joints.append(j)
            parent = body
        builder.add_articulation(joints)
        builder.color(balance_colors=False)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverVBD(
            model,
            iterations=1,
            rigid_compliant_alm=True,
            rigid_joint_linear_ke=ke,
            rigid_joint_angular_ke=ke,
            rigid_joint_global_iterations=1,
        )
        # The routes of this table are those of a 1 ms step; the solver plans for a step once it knows one.
        solver._plan_structural_graph_kkt(1.0e-3)
        route = solver._structural_graph_kkt.tree_buckets[0]._diagnostics.selected_route
        test.assertEqual(route, expected_route)
        if use_revolute:
            state_in, state_out = model.state(), model.state()
            control = model.control()
            joint_ids = np.asarray(joints, dtype=np.int32)
            for _ in range(180):
                solver.step(state_in, state_out, control, None, 1.0 / 600.0)
                state_in, state_out = state_out, state_in
            pose = state_in.body_q.numpy()
            test.assertTrue(np.isfinite(pose).all())
            test.assertLess(_max_joint_gap(model, pose, joint_ids), 1.0e-5)


def _structural_kkt_near_rigid_anchored_tree_multistage_stable(test, device):
    """Keep a single-world-anchor near-rigid binary tree sub-10um stable across 2 stages."""
    builder = newton.ModelBuilder()
    bodies = []
    joints = []
    depth = 6
    count = (1 << depth) - 1
    for idx in range(count):
        lvl = int(np.floor(np.log2(idx + 1)))
        pos_in_lvl = (idx + 1) - (1 << lvl)
        x = 0.08 * (idx + 1) * 0.15
        y = 0.03 * (pos_in_lvl - 0.5 * ((1 << lvl) - 1))
        body = builder.add_link(
            xform=wp.transform(wp.vec3(x, y, 2.0 - 0.04 * lvl), wp.quat_identity()),
            mass=0.05,
        )
        builder.add_shape_sphere(body, radius=0.012)
        bodies.append(body)
        if idx == 0:
            joints.append(
                builder.add_joint_ball(
                    parent=-1,
                    child=body,
                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(-x, -y, 0.0), wp.quat_identity()),
                )
            )
        else:
            p_idx = (idx - 1) // 2
            p_lvl = int(np.floor(np.log2(p_idx + 1)))
            p_pos = (p_idx + 1) - (1 << p_lvl)
            px = 0.08 * (p_idx + 1) * 0.15
            py = 0.03 * (p_pos - 0.5 * ((1 << p_lvl) - 1))
            pz = 2.0 - 0.04 * p_lvl
            cz = 2.0 - 0.04 * lvl
            mid = wp.vec3(0.5 * (px + x), 0.5 * (py + y), 0.5 * (pz + cz))
            joints.append(
                builder.add_joint_ball(
                    parent=bodies[p_idx],
                    child=body,
                    parent_xform=wp.transform(mid - wp.vec3(px, py, pz), wp.quat_identity()),
                    child_xform=wp.transform(mid - wp.vec3(x, y, cz), wp.quat_identity()),
                )
            )
    builder.add_articulation(joints)
    builder.color(balance_colors=False)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=2,
        rigid_compliant_alm=True,
        rigid_joint_linear_ke=1.0e12,
        rigid_joint_angular_ke=1.0e12,
        rigid_joint_global_iterations=2,
    )
    test.assertEqual(solver._structural_graph_kkt.tree_buckets[0]._diagnostics.selected_route, "tree_leaf_rake")
    state_in, state_out = model.state(), model.state()
    control = model.control()
    joint_ids = np.asarray(joints, dtype=np.int32)
    for _ in range(220):
        solver.step(state_in, state_out, control, None, 1.0 / 600.0)
        state_in, state_out = state_out, state_in
    pose = state_in.body_q.numpy()
    test.assertTrue(np.isfinite(pose).all())
    test.assertLess(_max_joint_gap(model, pose, joint_ids), 5.0e-5)


def _structural_kkt_sqp_saddle_chord_reduces_pendulum_step_error(test, device):
    """Verify that a second solve on the stage's factorization tightens one-stage backward Euler."""

    def _run_pendulum(*, global_iters: int, iters: int, disable_sqp: bool = False, resolve: bool = False) -> np.ndarray:
        builder = newton.ModelBuilder()
        b0 = builder.add_link(xform=wp.transform(wp.vec3(0.5, 0.0, 2.0), wp.quat_identity()), mass=1.0)
        builder.add_shape_box(b0, hx=0.5, hy=0.04, hz=0.04)
        b1 = builder.add_link(xform=wp.transform(wp.vec3(1.5, 0.0, 2.0), wp.quat_identity()), mass=1.0)
        builder.add_shape_box(b1, hx=0.5, hy=0.04, hz=0.04)
        j0 = builder.add_joint_revolute(
            parent=-1,
            child=b0,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
        )
        j1 = builder.add_joint_revolute(
            parent=b0,
            child=b1,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([j0, j1])
        builder.color(balance_colors=False)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverVBD(
            model,
            iterations=iters,
            rigid_compliant_alm=True,
            rigid_joint_linear_ke=1.0e8,
            rigid_joint_angular_ke=1.0e8,
            rigid_joint_global_iterations=global_iters,
            rigid_joint_global_resolve=resolve,
        )
        if disable_sqp:
            solver._structural_graph_kkt.has_resolvable_trees = False
        s0, s1 = model.state(), model.state()
        ctl = model.control()
        traj = []
        for step in range(1200):
            solver.step(s0, s1, ctl, None, 1.0e-3)
            s0, s1 = s1, s0
            if (step + 1) % 20 == 0:
                traj.append(s0.body_q.numpy()[:, :3].copy())
        return np.asarray(traj, dtype=np.float64)

    ref = _run_pendulum(global_iters=8, iters=8)
    sl = _run_pendulum(global_iters=1, iters=1, disable_sqp=True)
    o8 = _run_pendulum(global_iters=1, iters=1, resolve=True)
    err_sl = float(np.sqrt(np.mean((sl - ref) ** 2)))
    err_o8 = float(np.sqrt(np.mean((o8 - ref) ** 2)))
    test.assertLess(err_o8, 0.25 * err_sl)


add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_near_rigid_sled_slides_on_soft_ground",
    _structural_kkt_near_rigid_sled_slides_on_soft_ground,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_ball_open_chain_uses_tree_contraction",
    _structural_kkt_ball_open_chain_uses_tree_contraction,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_near_rigid_anchored_tree_multistage_stable",
    _structural_kkt_near_rigid_anchored_tree_multistage_stable,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_sqp_saddle_chord_reduces_pendulum_step_error",
    _structural_kkt_sqp_saddle_chord_reduces_pendulum_step_error,
    devices=get_test_devices(mode="basic"),
)

add_function_test(
    TestVBDRigidKKT,
    "test_rigid_bdf2_follows_fine_step_motion",
    _rigid_bdf2_follows_fine_step_motion,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestVBDRigidKKT,
    "test_rigid_bdf2_single_correction_holds_at_small_step",
    _rigid_bdf2_single_correction_holds_at_small_step,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestVBDRigidKKT,
    "test_rigid_bdf2_repeated_restart_is_stable",
    _rigid_bdf2_repeated_restart_is_stable,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_correction_penalty_excludes_rod_rows",
    _structural_correction_penalty_excludes_rod_rows,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_clean_saddle_stress_stateless_and_resolve_stable",
    _structural_kkt_clean_saddle_stress_stateless_and_resolve_stable,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_stored_stress_reset_matches_fresh_solver",
    _structural_kkt_stored_stress_reset_matches_fresh_solver,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_first_correction_stress_from_stored_forces",
    _structural_kkt_first_correction_stress_from_stored_forces,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_stored_stress_reset_respects_world_mask",
    _structural_kkt_stored_stress_reset_respects_world_mask,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_per_island_active_contact_sweep_gating",
    _structural_kkt_per_island_active_contact_sweep_gating,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_revolute_chain64_holds_near_rigidity",
    _structural_kkt_revolute_chain64_holds_near_rigidity,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_unit_scale_invariance",
    _structural_kkt_unit_scale_invariance,
    devices=get_test_devices(mode="basic"),
)


def _build_branched_ball_tree(device, *, gravity, anchored=True):
    """Three links on ball joints, branched at the first link; optionally hung from the world at its free end."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    mass, length, radius = 0.05, 0.05, 0.01
    ixx = 0.5 * mass * radius * radius
    iyy = mass * (3.0 * radius * radius + length * length) / 12.0
    inertia = wp.mat33(ixx, 0.0, 0.0, 0.0, iyy, 0.0, 0.0, 0.0, iyy)
    half = wp.vec3(0.5 * length, 0.0, 0.0)
    root = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.25), wp.quat_identity()), mass=mass, inertia=inertia)
    joints = []
    if anchored:
        joints.append(
            builder.add_joint_ball(
                parent=-1,
                child=root,
                parent_xform=wp.transform(wp.vec3(-0.5 * length, 0.0, 0.25), wp.quat_identity()),
                child_xform=wp.transform(-half, wp.quat_identity()),
            )
        )
    for side in (1.0, -1.0):
        angle = 0.6 * side
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)
        center = wp.vec3(0.5 * length * (1.0 + np.cos(angle)), 0.5 * length * np.sin(angle), 0.25)
        body = builder.add_link(xform=wp.transform(center, q), mass=mass, inertia=inertia)
        joints.append(
            builder.add_joint_ball(
                parent=root,
                child=body,
                parent_xform=wp.transform(half, q),
                child_xform=wp.transform(-half, wp.quat_identity()),
            )
        )
    builder.add_articulation(joints)
    builder.color()
    return builder.finalize(device=device), np.asarray(joints, dtype=np.int32)


def _structural_kkt_second_stage_follows_correction(test, device):
    """The automatic second stage is decided by the rotation of the first stage's own correction.

    A branched three-link tree is released from rest. A step of 0.5 ms turns its links too little for the
    linearized rows to be wrong in single precision, at any stiffness: no second stage, also at K = 1e12. A step
    of 16 ms turns them by several hundredths of a radian: a second stage, also at K = 1e6, and it closes the
    joints (0.16 um; 45 um without the stage). The same tree in time units scaled by 4 and by 1/4 decides alike.
    Catches a decision taken from the structure (stiffness and depth), which takes the stage at K = 1e12 and
    omits it at K = 1e6.
    """

    def run(stiffness, step, time_unit):
        model, joints = _build_branched_ball_tree(device, gravity=-9.81 / time_unit**2)
        solver = newton.solvers.SolverVBD(
            model,
            iterations=1,
            rigid_joint_global_iterations=1,
            rigid_joint_linear_ke=stiffness / time_unit**2,
            rigid_joint_angular_ke=stiffness / time_unit**2,
            rigid_joint_linear_kd=0.0,
            rigid_joint_angular_kd=0.0,
        )
        state_in, state_out, control = model.state(), model.state(), model.control()
        solver.step(state_in, state_out, control, None, step * time_unit)
        needed = int(solver._structural_graph_kkt.second_stage_needed.numpy()[0])
        return needed, _max_joint_gap(model, state_out.body_q.numpy(), joints)

    for time_unit in (1.0, 4.0, 0.25):
        needed, gap = run(1.0e12, 5.0e-4, time_unit)
        test.assertEqual(needed, 0)
        test.assertLess(gap, 1.0e-7)
        needed, gap = run(1.0e6, 1.6e-2, time_unit)
        test.assertEqual(needed, 1)
        test.assertLess(gap, 2.0e-6)


def _structural_kkt_plan_follows_step(test, device):
    """Factor precisions are planned for the step that is solved, not for a nominal one.

    A free three-link tree at K = 1e11 takes float64 factors once omega^2 h^2 reaches 2^24, at a step h_c of
    0.79 ms that the plan reports. Before any step the plan is the safe one (float64). A first step of 2 h_c
    keeps float64 factors, and a later step of h_c / 2 keeps that plan. A first step of h_c / 2 plans float32
    factors, and a later step of 2 h_c replans float64. A first step of h_c / 2 inside a captured graph keeps
    the safe plan. The same holds in time units scaled by 4 and by 1/4. Catches a plan that keeps the decisions
    of a nominal step (O14: 1 ms, whatever the step and the time unit; o16e: when the first step is captured).
    """
    for time_unit in (1.0, 4.0, 0.25):

        def make(time_unit=time_unit):
            model, _ = _build_branched_ball_tree(device, gravity=-9.81 / time_unit**2, anchored=False)
            solver = newton.solvers.SolverVBD(
                model,
                iterations=1,
                rigid_joint_global_iterations=1,
                rigid_joint_linear_ke=1.0e11 / time_unit**2,
                rigid_joint_angular_ke=1.0e11 / time_unit**2,
                rigid_joint_linear_kd=0.0,
                rigid_joint_angular_kd=0.0,
            )
            return model, solver

        model, solver = make()
        rigidities = solver._structural_graph_kkt.plan_rigidities
        test.assertEqual(len(rigidities), 1)
        critical_step = float(np.sqrt(float(1 << 24) / rigidities[0]))
        test.assertAlmostEqual(critical_step / time_unit, 7.883e-4, delta=0.02 * 7.883e-4)
        test.assertEqual(solver._structural_graph_kkt.tree_buckets[0].factor_dtype, wp.float64)

        state_in, state_out, control = model.state(), model.state(), model.control()
        solver.step(state_in, state_out, control, None, 2.0 * critical_step)
        backend = solver._structural_graph_kkt
        test.assertEqual(backend.tree_buckets[0].factor_dtype, wp.float64)
        solver.step(state_out, state_in, control, None, 0.5 * critical_step)
        test.assertIs(solver._structural_graph_kkt, backend)
        test.assertTrue(np.isfinite(state_in.body_q.numpy()).all())

        model, solver = make()
        state_in, state_out, control = model.state(), model.state(), model.control()
        solver.step(state_in, state_out, control, None, 0.5 * critical_step)
        test.assertEqual(solver._structural_graph_kkt.tree_buckets[0].factor_dtype, wp.float32)
        test.assertTrue(np.isfinite(state_out.body_q.numpy()).all())
        solver.step(state_out, state_in, control, None, 2.0 * critical_step)
        test.assertEqual(solver._structural_graph_kkt.tree_buckets[0].factor_dtype, wp.float64)
        test.assertTrue(np.isfinite(state_in.body_q.numpy()).all())

        if device.is_cuda:
            model, solver = make()
            state_in, state_out, control = model.state(), model.state(), model.control()
            backend = solver._structural_graph_kkt
            with wp.ScopedCapture(device=device) as capture:
                solver.step(state_in, state_out, control, None, 0.5 * critical_step)
            wp.capture_launch(capture.graph)
            test.assertIs(solver._structural_graph_kkt, backend)
            test.assertEqual(backend.tree_buckets[0].factor_dtype, wp.float64)
            test.assertTrue(np.isfinite(state_out.body_q.numpy()).all())


def _structural_kkt_contact_adds_second_solve(test, device):
    """One requested structural correction is followed by a second one when a rigid contact is active.

    Friction that the sweeps find reaches a stiff structure only through a correction. Catches the loss of that
    rule (with one correction and five sweeps a dragged chain of 12 boxes at K = 1e8 is 38 % off the analytic
    travel) and a second correction that runs although nothing touches (63 % of the step on idle chains).
    """

    def corrections(iterations, steps, with_buffer=True):
        model, pipeline, contacts, solver, _, _, _ = _build_grounded_chain(
            device, global_iterations=1, iterations=iterations, segments=8
        )
        count = []
        solve = solver._solve_structural_graph_kkt

        def counted(*args, **kwargs):
            if not kwargs.get("translation_only", False) and not kwargs.get("resolve_only", False):
                count.append(1)
            return solve(*args, **kwargs)

        solver._solve_structural_graph_kkt = counted
        state_in, state_out, control = model.state(), model.state(), model.control()
        for _ in range(steps):
            del count[:]
            state_in.clear_forces()
            if with_buffer:
                pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts if with_buffer else None, 1.0 / 240.0)
            state_in, state_out = state_out, state_in
        test.assertTrue(np.isfinite(state_in.body_q.numpy()).all())
        return len(count)

    # The chain starts 1 mm above the ground: candidates inside the detection gap, nothing touches yet.
    test.assertEqual(corrections(4, 1), 1)
    # At rest on the ground the contacts are active and the second correction runs.
    test.assertEqual(corrections(4, 30), 2)
    test.assertEqual(corrections(4, 30, with_buffer=False), 1)
    test.assertEqual(corrections(1, 30), 1)


def _structural_kkt_long_chain_large_step(test, device):
    """A 256-link revolute chain at K = 1e8 stays closed at a step of 4 ms under BDF2.

    Catches the paired cyclic reduction kept up to omega^2 h^2 < 2^24: the chain opens after 150 steps and is
    apart by 1.8 m at step 200 (O13, O14, O15); with leaf rake its largest joint gap is 5 um.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    joints = []
    for i in range(256):
        body = builder.add_link(
            mass=0.1,
            com=wp.vec3(0.0, 0.0, -0.05),
            inertia=wp.mat33(8.5e-5, 0.0, 0.0, 0.0, 8.5e-5, 0.0, 0.0, 0.0, 5.0e-6),
            lock_inertia=True,
        )
        joints.append(
            builder.add_joint_revolute(
                parent=i - 1,
                child=body,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 30.0 if i == 0 else -0.1)),
                armature=0.0,
            )
        )
    builder.add_articulation(joints)
    q0 = [0.0] * 256
    q0[0] = 1.2
    builder.joint_q[:] = q0
    builder.color()
    model = builder.finalize(device=device)
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_joint_global_iterations=1,
        rigid_joint_linear_ke=1.0e8,
        rigid_joint_angular_ke=1.0e8,
        rigid_joint_linear_kd=0.0,
        rigid_joint_angular_kd=0.0,
        rigid_time_integration="bdf2",
    )
    state_in, state_out, control = model.state(), model.state(), model.control()
    for _ in range(200):
        solver.step(state_in, state_out, control, None, 4.0e-3)
        state_in, state_out = state_out, state_in
    poses = state_in.body_q.numpy()
    test.assertTrue(np.isfinite(poses).all())
    test.assertLess(_max_joint_gap(model, poses, np.asarray(joints, dtype=np.int32)), 1.0e-4)


add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_second_stage_follows_correction",
    _structural_kkt_second_stage_follows_correction,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_plan_follows_step",
    _structural_kkt_plan_follows_step,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_contact_adds_second_solve",
    _structural_kkt_contact_adds_second_solve,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_long_chain_large_step",
    _structural_kkt_long_chain_large_step,
    devices=get_cuda_test_devices(),
)


def _structural_kkt_arbitrary_length_paired_cr_and_budget_warning(test, device):
    """Chains beyond 256 links use logarithmic paired CR while chi < 2^24, and budget fallbacks warn."""
    model = _build_paired_open_chain_model(device, 512, chain_count=2)
    with (
        mock.patch.object(rigid_vbd_kkt, "_structural_payload_budget", return_value=1),
        test.assertWarnsRegex(RuntimeWarning, r"2 island\(s\) \(1024 bodies\) fell back to local VBD sweeps"),
    ):
        warned_solver = newton.solvers.SolverVBD(
            model,
            iterations=1,
            rigid_joint_global_iterations=1,
            rigid_compliant_alm=True,
        )
    test.assertIsNone(warned_solver._structural_graph_kkt)

    if not device.is_cuda:
        return

    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_joint_global_iterations=1,
        rigid_compliant_alm=True,
        deterministic=wp.DeterministicMode.NOT_GUARANTEED,
    )
    state_in, state_out, control = model.state(), model.state(), model.control()
    solver.step(state_in, state_out, control, None, 1.0e-3)
    bucket = solver._structural_graph_kkt.tree_buckets[0]
    test.assertTrue(bucket.use_paired_open_backbone)
    test.assertEqual(bucket._diagnostics.selected_route, "tree_paired_open_backbone_cr")
    test.assertTrue(np.isfinite(state_out.body_q.numpy()).all())


add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_arbitrary_length_paired_cr_and_budget_warning",
    _structural_kkt_arbitrary_length_paired_cr_and_budget_warning,
    devices=get_test_devices(mode="basic"),
)

add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_joint_force_keeps_rod_on_contact_taut",
    _structural_kkt_joint_force_keeps_rod_on_contact_taut,
    devices=get_test_devices(),
)


def _build_chain_condensation_scene(device, *, pin="world", ke=1.0e8, iterations=4, deterministic=False):
    segments, seg, arm, preload, pull, dt = 24, 1.0e-3, 5.0e-3, -0.04, 2.0, 1.0 / 1800.0
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.inertia_absolute_tolerance = 1.0e-20
    builder.inertia_correction = 0.0
    link = builder.add_link(
        xform=wp.transform(wp.vec3(0.0125, 0.0, 0.0), wp.quat_identity()),
        mass=3.3e-3,
        inertia=wp.mat33(9.35e-8, 0.0, 0.0, 0.0, 1.87e-7, 0.0, 0.0, 0.0, 1.73e-7),
    )
    hinge = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(-0.0125, 0.0, 0.0), wp.quat_identity()),
        limit_lower=0.0,
        limit_upper=1.5,
        limit_ke=300.0,
        limit_kd=0.01,
        target_ke=0.04,
        target_kd=1.5e-4,
    )
    slider = builder.add_link(
        xform=wp.transform(wp.vec3(-seg * segments, arm, 0.0), wp.quat_identity()),
        mass=6.0e-3,
        inertia=wp.mat33(3.6e-8, 0.0, 0.0, 0.0, 3.6e-8, 0.0, 0.0, 0.0, 3.6e-8),
    )
    rod = newton.Rod(
        [wp.vec3(-seg * i, arm, 0.0) for i in range(segments + 1)],
        radius=5.0e-4,
        rest_straight=True,
        stretch_rigidity=2000.0,
        shear_rigidity=2000.0,
        bend_rigidity=1.0e-8,
        twist_rigidity=1.0e-8,
    )
    cfg = newton.ModelBuilder.ShapeConfig(density=0.79e-3 / (np.pi * 5.0e-4**2), ke=ke, kd=0.0, mu=0.0, gap=5.0e-4)
    bodies, rod_joints = builder.add_rod(
        rod=rod,
        cfg=cfg,
        stretch_damping=10.0,
        shear_damping=10.0,
        wrap_in_articulation=False,
    )
    rod_bodies = set(bodies)
    rod_shapes = [shape for shape, body in enumerate(builder.shape_body) if body in rod_bodies]
    for i, shape_a in enumerate(rod_shapes):
        for shape_b in rod_shapes[i + 1 :]:
            builder.add_shape_collision_filter_pair(shape_a, shape_b)
    pin_y = arm - 2.0e-3 - 5.0e-4 + 3.0e-4
    pin_x = -0.5 * seg * segments
    if pin == "world":
        builder.add_shape_sphere(
            -1, xform=wp.transform(wp.vec3(pin_x, pin_y, 0.0), wp.quat_identity()), radius=2.0e-3, cfg=cfg
        )
    else:
        builder.add_shape_sphere(
            link,
            xform=wp.transform(wp.vec3(pin_x - 0.0125, pin_y, 0.0), wp.quat_identity()),
            radius=2.0e-3,
            cfg=cfg,
        )
    first = builder.add_joint_ball(
        parent=link,
        child=bodies[0],
        parent_xform=wp.transform(wp.vec3(-0.0125, arm, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.5 * seg), wp.quat_identity()),
    )
    last = builder.add_joint_ball(
        parent=bodies[-1],
        child=slider,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * seg), wp.quat_identity()),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([hinge, *rod_joints, first, last])
    slide = builder.add_joint_prismatic(
        parent=-1,
        child=slider,
        axis=wp.vec3(-1.0, 0.0, 0.0),
        parent_xform=wp.transform(wp.vec3(-seg * segments, arm, 0.0), wp.quat_identity()),
        child_xform=wp.transform_identity(),
        limit_lower=-0.002,
        limit_upper=0.03,
        limit_ke=2.0e4,
        limit_kd=20.0,
        target_ke=0.0,
        target_kd=20.0,
    )
    builder.color()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, contact_matching="latest")
    contacts = pipeline.contacts()
    det_mode = wp.DeterministicMode.RUN_TO_RUN if deterministic else wp.DeterministicMode.NOT_GUARANTEED
    solver = newton.solvers.SolverVBD(
        model,
        iterations=iterations,
        rigid_joint_global_iterations=iterations,
        rigid_joint_linear_ke=1.0e5,
        rigid_joint_angular_ke=1.0e5,
        rigid_joint_linear_kd=0.0,
        rigid_joint_angular_kd=0.0,
        rigid_contact_history=True,
        rigid_body_contact_buffer_size=64,
        deterministic=det_mode,
    )
    control = model.control()
    qd_start = model.joint_qd_start.numpy()
    joint_f = control.joint_f.numpy()
    joint_f[int(qd_start[hinge])] = preload
    joint_f[int(qd_start[slide])] = pull
    control.joint_f.assign(joint_f)
    return SimpleNamespace(
        model=model,
        solver=solver,
        pipeline=pipeline,
        contacts=contacts,
        control=control,
        link=link,
        slider=slider,
        bodies=bodies,
        hinge=hinge,
        dt=dt,
    )


def _chain_condensation_dense_reference(scene):
    """Build exact dense float64 body-space KKT reference from assembled chain-route blocks."""
    model = scene.model
    backend = scene.solver._structural_graph_kkt
    route = backend.chain_route
    slot_of = backend.body_slot_by_id.numpy()
    count = backend.graph_body_count
    matrix = np.zeros((6 * count, 6 * count), dtype=np.float64)
    rhs = np.zeros(6 * count, dtype=np.float64)
    assembly = route.assembly

    hub_start = route.island_hub_start.numpy()
    for e in range(route.island_count):
        h0, h1 = int(hub_start[e]), int(hub_start[e + 1])
        for i in range(h1 - h0):
            hub_body = int(route.hub_body_host[h0 + i])
            slot = int(slot_of[hub_body])
            matrix[6 * slot : 6 * slot + 6, 6 * slot : 6 * slot + 6] = assembly["hub_base"][e, i]
            rhs[6 * slot : 6 * slot + 6] = assembly["hub_rhs"][e, i]

    bucket = backend.tree_buckets[0]
    jacobian_parent = bucket.jacobian_parent.numpy().astype(np.float64)
    jacobian_child = bucket.jacobian_child.numpy().astype(np.float64)
    compliance = bucket.compliance.numpy().astype(np.float64)
    residual = bucket.residual.numpy().astype(np.float64)
    active = bucket.row_active.numpy()
    parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
    inv_mass = model.body_inv_mass.numpy()
    seg_set = {int(b) for b in route.segment_body_host}
    for row, joint in enumerate(bucket.joint_ids_host):
        if not active[row] or int(model.joint_type.numpy()[joint]) == int(newton.JointType.FREE):
            continue
        p_b, c_b = int(parent[joint]), int(child[joint])
        ends = [
            (int(slot_of[body]), jacobian)
            for body, jacobian in ((p_b, jacobian_parent[row]), (c_b, jacobian_child[row]))
            if body >= 0 and inv_mass[body] > 0.0 and body not in seg_set
        ]
        if not ends:
            continue
        weight = np.linalg.inv(compliance[row])
        for slot_a, jac_a in ends:
            rhs[6 * slot_a : 6 * slot_a + 6] -= jac_a.T @ weight @ residual[row]
            for slot_b, jac_b in ends:
                matrix[6 * slot_a : 6 * slot_a + 6, 6 * slot_b : 6 * slot_b + 6] += jac_a.T @ weight @ jac_b

    island_of = route.segment_island.numpy()
    for s, body in enumerate(route.segment_body_host):
        slot = int(slot_of[body])
        block = slice(6 * slot, 6 * slot + 6)
        matrix[block, block] = assembly["diagonal"][s]
        rhs[block] = assembly["rhs"][s]
        idx = int(route.segment_index.numpy()[s])
        length = int(route.segment_chain_length.numpy()[s])
        if idx > 0:
            prev_slot = int(slot_of[route.segment_body_host[s - 1]])
            matrix[block, 6 * prev_slot : 6 * prev_slot + 6] = assembly["left"][s]
        if idx + 1 < length:
            next_slot = int(slot_of[route.segment_body_host[s + 1]])
            matrix[block, 6 * next_slot : 6 * next_slot + 6] = assembly["right"][s]
        island = int(island_of[s])
        for hub in range(int(hub_start[island + 1] - hub_start[island])):
            hub_slot = int(slot_of[route.hub_body_host[hub_start[island] + hub]])
            hub_slice = slice(6 * hub_slot, 6 * hub_slot + 6)
            matrix[block, hub_slice] = assembly["border"][s, hub]
            matrix[hub_slice, block] = assembly["border"][s, hub].T
            matrix[hub_slice, hub_slice] += assembly["hub_diagonal"][s, hub]
            rhs[hub_slice] += assembly["hub_wrench"][s, hub]
    return matrix, rhs


def _chain_condensation_matches_dense_solve(test, device):
    """Pure-float64 atomic-free ChainCondensation matches dense float64 solve, stays SPD at high ke, and is deterministic."""
    for pin, ke in (("world", 1.0e8), ("link", 1.0e8), ("link", 1.0e10)):
        scene = _build_chain_condensation_scene(device, pin=pin, ke=ke, iterations=3)
        model, solver, contacts = scene.model, scene.solver, scene.contacts
        test.assertIsNotNone(solver._structural_graph_kkt.chain_route)
        state_0, state_1 = model.state(), model.state()
        with mock.patch.object(rigid_vbd_kkt_chains.ChainCondensation, "keep_assembly", True):
            for _step in range(25):
                state_0.clear_forces()
                scene.pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, scene.control, contacts, scene.dt)
                state_0, state_1 = state_1, state_0
        route = solver._structural_graph_kkt.chain_route
        test.assertEqual((route.island_count, route.chain_count, route.hub_count), (1, 1, 2))
        matrix, rhs = _chain_condensation_dense_reference(scene)
        sym = 0.5 * (matrix + matrix.T)
        test.assertLess(float(np.abs(matrix - matrix.T).max()), 1.0e-9 * float(np.abs(matrix).max()))
        min_eig = float(np.linalg.eigvalsh(sym)[0])
        test.assertGreater(min_eig, 0.0, f"pin={pin} ke={ke}: condensed system lost SPD (min_eig={min_eig})")
        reference = np.linalg.solve(sym, rhs)
        correction = route.assembly["correction"].astype(np.float64).reshape(-1)
        scale = float(np.abs(reference).max())
        test.assertGreater(scale, 0.0)
        test.assertLess(float(np.abs(correction - reference).max()), 1.0e-5 * scale)
        touching = int(
            np.count_nonzero(np.abs(route.assembly["hub_diagonal"]).reshape(route.segment_count, -1).max(axis=1))
        )
        if pin == "world":
            test.assertEqual(touching, 0)
        else:
            test.assertGreater(touching, 0)
            if ke == 1.0e8:
                scene2 = _build_chain_condensation_scene(device, pin=pin, ke=ke, iterations=3)
                s2_0, s2_1 = scene2.model.state(), scene2.model.state()
                with mock.patch.object(rigid_vbd_kkt_chains.ChainCondensation, "keep_assembly", True):
                    for _step in range(25):
                        s2_0.clear_forces()
                        scene2.pipeline.collide(s2_0, scene2.contacts)
                        scene2.solver.step(s2_0, s2_1, scene2.control, scene2.contacts, scene2.dt)
                        s2_0, s2_1 = s2_1, s2_0
                test.assertTrue(
                    np.array_equal(
                        route.assembly["correction"],
                        scene2.solver._structural_graph_kkt.chain_route.assembly["correction"],
                    )
                )


def _chain_condensation_scope(test, device):
    """ChainCondensation activates on rod+hub islands and coexists cleanly with pure-rod islands."""
    scene = _build_chain_condensation_scene(device, pin="link", iterations=2)
    backend = scene.solver._structural_graph_kkt
    test.assertIsNotNone(backend.chain_route)
    test.assertEqual(backend.tree_buckets[0]._diagnostics.selected_route, "chain_condensation")

    with mock.patch.object(rigid_vbd_kkt_chains, "ENABLED", False):
        disabled_scene = _build_chain_condensation_scene(device, pin="link", iterations=2)
    test.assertIsNone(disabled_scene.solver._structural_graph_kkt.chain_route)

    model_y, _b, _j = _build_y_tree(device)
    solver_y = newton.solvers.SolverVBD(model_y, iterations=2, rigid_joint_global_iterations=1)
    test.assertIsNone(solver_y._structural_graph_kkt.chain_route)


def _heterogeneous_fused_trees_and_near_rigid_fp64_paired_cr(test, device):
    """Heterogeneous leaf-rake trees merge into 1 fused CTA bucket, and near-rigid 512-link chains use fp64 paired CR."""
    if not device.is_cuda:
        return
    builder = newton.ModelBuilder(gravity=(0.0, -9.81, 0.0))
    for depth_idx, length in enumerate((4, 6, 5)):
        prev = -1
        for k in range(length):
            b = builder.add_link(
                xform=wp.transform(wp.vec3(float(depth_idx), -0.1 * float(k + 1), 0.0), wp.quat_identity()),
                mass=1.0,
                inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01),
            )
            j = builder.add_joint_revolute(
                parent=prev,
                child=b,
                axis=wp.vec3(0.0, 0.0, 1.0),
                parent_xform=wp.transform(
                    wp.vec3(float(depth_idx), 0.0, 0.0) if prev < 0 else wp.vec3(0.0, -0.05, 0.0),
                    wp.quat_identity(),
                ),
                child_xform=wp.transform(wp.vec3(0.0, 0.05, 0.0), wp.quat_identity()),
            )
            if k == 0:
                joints = [j]
            else:
                joints.append(j)
            prev = b
        # Add a side branch on the middle island so topologies are genuinely heterogeneous
        if depth_idx == 1:
            side = builder.add_link(
                xform=wp.transform(wp.vec3(1.2, -0.2, 0.0), wp.quat_identity()),
                mass=0.5,
                inertia=wp.mat33(0.005, 0.0, 0.0, 0.0, 0.005, 0.0, 0.0, 0.0, 0.005),
            )
            joints.append(
                builder.add_joint_revolute(
                    parent=prev - 2,
                    child=side,
                    axis=wp.vec3(1.0, 0.0, 0.0),
                    parent_xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(-0.1, 0.0, 0.0), wp.quat_identity()),
                )
            )
        builder.add_articulation(joints)
    builder.color()
    het_model = builder.finalize(device=device)
    het_solver = newton.solvers.SolverVBD(het_model, iterations=2, rigid_joint_global_iterations=2)
    s0, s1, ctrl = het_model.state(), het_model.state(), het_model.control()
    het_solver.step(s0, s1, ctrl, None, 1.0e-3)
    het_buckets = het_solver._structural_graph_kkt.tree_buckets
    test.assertEqual(len(het_buckets), 1, "three heterogeneous leaf-rake trees should merge into 1 multi-CTA bucket")
    test.assertTrue(het_buckets[0].use_fused_tree_levels)
    test.assertEqual(het_buckets[0].batch_count, 3)
    test.assertTrue(np.isfinite(s1.body_q.numpy()).all())

    # Near-rigid (ke=1e10) 512-link open chain routes to fp64 paired CR (O(log N) depth + float64 precision)
    long_model = _build_paired_open_chain_model(device, 512, chain_count=1)
    long_solver = newton.solvers.SolverVBD(
        long_model,
        iterations=1,
        rigid_joint_global_iterations=1,
        rigid_joint_linear_ke=1.0e10,
        rigid_joint_angular_ke=1.0e10,
        rigid_compliant_alm=True,
        deterministic=wp.DeterministicMode.NOT_GUARANTEED,
    )
    ls0, ls1, lctrl = long_model.state(), long_model.state(), long_model.control()
    long_solver.step(ls0, ls1, lctrl, None, 2.0e-3)
    lbucket = long_solver._structural_graph_kkt.tree_buckets[0]
    test.assertTrue(lbucket.use_paired_open_backbone)
    test.assertEqual(lbucket.factor_dtype, wp.float64)
    test.assertEqual(lbucket._diagnostics.selected_route, "tree_paired_open_backbone_cr_fp64")
    test.assertTrue(np.isfinite(ls1.body_q.numpy()).all())


@wp.kernel
def _eval_unilateral_active_set_probe(
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    penetration: float,
    lam_n: float,
    active_set: int,
    out_fn: wp.array[float],
    out_knn: wp.array[float],
):
    n = wp.vec3(0.0, 1.0, 0.0)
    f0, _t0, h_ll0, _hal0, _haa0, _f1, _t1, _hll1, _hal1, _haa1 = evaluate_rigid_contact_from_collision(
        0,
        -1,
        body_q,
        body_q,
        body_com,
        wp.vec3(0.0),
        wp.vec3(0.0),
        wp.vec3(0.0),
        wp.vec3(0.0),
        wp.vec3(0.0),
        n,
        penetration,
        1.0e6,
        1.0e6,
        1.0e5,
        0.0,
        n * lam_n,
        0.5,
        1.0e-4,
        0,
        1,
        1.0e-3,
        wp.vec3(0.0),
        False,
        active_set,
    )
    out_fn[0] = wp.dot(f0, n)
    out_knn[0] = wp.dot(n, h_ll0 * n)


def _rod_kirchhoff_proximal_and_unilateral_active_set_hessian(test, device):
    """Unilateral active-set Hessian keeps K_n while lam_n > 0 with f_n >= 0; Kirchhoff proximal only acts under contact."""
    bq = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
    bcom = wp.zeros(1, dtype=wp.vec3, device=device)
    out_fn = wp.zeros(1, dtype=float, device=device)
    out_knn = wp.zeros(1, dtype=float, device=device)

    # Slight momentary separation (penetration = -2e-6 m) while ALM multiplier lam_n = 0.5 N > 0:
    # primal trial force k*C + lam_n = 5e5 * (-2e-6) + 0.5 = -0.5 -> clamped to f_n = 0.0 (strictly unilateral, zero glue!),
    # while active_set=1 retains K_n = 5e5 N/m (active_set=0 drops K_n to 0.0).
    wp.launch(
        _eval_unilateral_active_set_probe,
        1,
        inputs=[bq, bcom, -2.0e-6, 0.5, 0],
        outputs=[out_fn, out_knn],
        device=device,
    )
    test.assertEqual(float(out_fn.numpy()[0]), 0.0)
    test.assertEqual(float(out_knn.numpy()[0]), 0.0)

    wp.launch(
        _eval_unilateral_active_set_probe,
        1,
        inputs=[bq, bcom, -2.0e-6, 0.5, 1],
        outputs=[out_fn, out_knn],
        device=device,
    )
    test.assertEqual(float(out_fn.numpy()[0]), 0.0)
    test.assertGreater(float(out_knn.numpy()[0]), 4.0e5)

    # Verify Kirchhoff director proximal uses each rod body's own half-arm r_b and excludes non-rod hubs.
    builder = newton.ModelBuilder()
    hub = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
    builder.add_shape_box(hub, hx=0.04, hy=0.01, hz=0.01)
    j_hub = builder.add_joint_revolute(
        parent=-1,
        child=hub,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    seg0 = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0305), wp.quat_identity()))
    seg1 = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0315), wp.quat_identity()))
    for sb in (seg0, seg1):
        builder.add_shape_capsule(sb, radius=2.0e-4, half_height=5.0e-4)
    j_rod0 = builder.add_joint_rod(
        parent=hub,
        child=seg0,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.03), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, -5.0e-4), wp.quat_identity()),
        bend_stiffness=1.0,
        stretch_stiffness=2.0e6,
    )
    j_rod1 = builder.add_joint_rod(
        parent=seg0,
        child=seg1,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 5.0e-4), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, -5.0e-4), wp.quat_identity()),
        bend_stiffness=1.0,
        stretch_stiffness=2.0e6,
    )
    builder.add_articulation([j_hub, j_rod0, j_rod1])
    builder.color()
    rod_model = builder.finalize(device=device)
    rod_solver = newton.solvers.SolverVBD(
        rod_model,
        iterations=2,
        rigid_joint_global_iterations=2,
        rigid_joint_linear_ke=2.0e6,
        rigid_joint_angular_ke=1.0e5,
        rigid_compliant_alm=True,
    )
    rod_solver._plan_structural_graph_kkt(1.0e-3)
    rod_ids = rod_solver._rod_kirchhoff_body_ids.numpy().tolist()
    rod_prox = dict(zip(rod_ids, rod_solver._rod_kirchhoff_proximal.numpy().tolist(), strict=True))
    test.assertNotIn(hub, rod_prox)
    test.assertAlmostEqual(rod_prox[seg0], 2.0 * (5.0e-4**2) * 2.0e6, places=6)
    test.assertAlmostEqual(rod_prox[seg1], (5.0e-4**2) * 2.0e6, places=6)


add_function_test(
    TestVBDRigidKKT,
    "test_chain_condensation_matches_dense_solve",
    _chain_condensation_matches_dense_solve,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestVBDRigidKKT,
    "test_chain_condensation_scope",
    _chain_condensation_scope,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestVBDRigidKKT,
    "test_heterogeneous_fused_trees_and_near_rigid_fp64_paired_cr",
    _heterogeneous_fused_trees_and_near_rigid_fp64_paired_cr,
    devices=get_test_devices(mode="basic"),
)
add_function_test(
    TestVBDRigidKKT,
    "test_rod_kirchhoff_proximal_and_unilateral_active_set_hessian",
    _rod_kirchhoff_proximal_and_unilateral_active_set_hessian,
    devices=get_test_devices(mode="basic"),
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
