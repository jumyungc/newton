# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the private VBD structural compliance-KKT backend."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd import rigid_vbd_kkt
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    evaluate_angular_constraint_force_hessian,
)
from newton._src.solvers.vbd.rigid_vbd_kkt import (
    _CR_PERSISTENT_MAX_ROWS,
    _FUSED_TREE_MAX_LEVEL_WIDTH,
    _FUSED_TREE_MIN_LEVELS,
    _PAIRED_COARSE_MIN_CLOSURES,
    _corrected_pose,
    _cr_persistent_max_rows,
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
    torque, _hessian, _kappa, _jacobian = evaluate_angular_constraint_force_hessian(
        wp.transform_get_rotation(body_q[0]),
        wp.transform_get_rotation(body_q[1]),
        wp.quat_identity(),
        wp.quat_identity(),
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


def _build_chain(
    device,
    *,
    segments=16,
    stiffness=1.0e7,
    dahl=False,
    pinned=True,
    with_particle=False,
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


def _build_y_tree(device, *, segments_per_branch=6, stiffness=1.0e9):
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
    bodies, joints = builder.add_rod_graph(
        node_positions=positions,
        edges=edges,
        radius=0.01,
        cfg=builder.default_shape_cfg.copy(),
        stretch_stiffness=stiffness,
        bend_stiffness=1.0e4,
        bend_damping=1.0e3,
        wrap_in_articulation=True,
        body_frame_origin="com",
    )
    _pin_body(builder, int(bodies[segments_per_branch - 1]))
    builder.color(balance_colors=False)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))
    return model, np.asarray(bodies, dtype=np.int32), np.asarray(joints, dtype=np.int32)


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
    bodies, joints = builder.add_rod_graph(
        node_positions=positions,
        edges=edges,
        radius=0.01,
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
    bodies, joints = builder.add_rod_graph(
        node_positions=positions,
        edges=edges,
        radius=cable_radius,
        cfg=shape,
        stretch_stiffness=stiffness,
        bend_stiffness=bend_stiffness,
        bend_damping=1.0e3,
        wrap_in_articulation=True,
        body_frame_origin="com",
    )
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


def _build_structural_backend_with_intercepted_bytes(model):
    """Construct one backend while summing unique Warp allocation payloads."""
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
        backend = rigid_vbd_kkt.StructuralGraphKKT(
            model,
            model.body_inv_mass,
            ignore_free_completion_joints=True,
        )
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
        len(body_ids),
        count,
        len(joint_ids),
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


def _path_float64_oracle_metrics(device, stiffness, *, segments=16):
    """Compare one production path Schur solve with independent float64 assembly."""
    model, _, _ = _build_chain(device, segments=segments, stiffness=stiffness)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
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
    _simulate(model, solver, 1, 1.0 / 600.0)

    row_count = bucket.row_count
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
    production_correction = backend.body_correction.numpy().astype(np.float64)

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
):
    """Compare a production open/cyclic tree solve with a float64 physical KKT."""
    if closed:
        model, _, _ = _build_loop_with_branch(
            device,
            ring_segments=ring_segments,
            stiffness=stiffness,
            bend_stiffness=bend_stiffness,
            add_chord=add_chord,
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


def _linearize_bucket(model, solver, bucket, body_q, dt):
    control = model.control()
    wp.launch(
        linearize_joint_path_rows,
        bucket.size,
        inputs=[
            bucket.joint_ids,
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
            solver.joint_drive_limit_support,
            solver.joint_drive_lambda,
            solver.joint_limit_lambda,
            solver.rigid_joint_alpha,
            body_q,
            solver.body_q_prev,
            model.body_q,
            model.body_com,
            dt,
        ],
        outputs=[
            bucket.jacobian_parent,
            bucket.jacobian_child,
            bucket.compliance,
            bucket.residual,
            bucket.row_active,
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
):
    model, pipeline, contacts, solver, bodies, joints, radius = _build_grounded_chain(
        device,
        global_iterations=global_iterations,
        iterations=iterations,
        contact_history=contact_history,
        deterministic=wp.DeterministicMode.RUN_TO_RUN,
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
    for step in range(240 * substeps):
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
        ],
        outputs=[active_q],
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
            matrix = wp.zeros(2, dtype=wp.spatial_matrix, device=device)
            inputs = [
                ids,
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
                        outputs=[matrix],
                        device=device,
                    )
                wp.capture_launch(capture.graph)
            else:
                wp.launch(
                    add_joint_stress_majorizer,
                    1,
                    inputs=inputs,
                    outputs=[matrix],
                    device=device,
                )
            value = matrix.numpy().astype(np.float64)
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
            matrix.zero_()
            before = (jp.numpy(), jc.numpy(), C.numpy())
            wp.launch(
                add_joint_stress_majorizer,
                1,
                inputs=inputs,
                outputs=[matrix],
                device=device,
            )
            np.testing.assert_array_equal(matrix.numpy(), 0.0)
            for expected, actual in zip(before, (jp.numpy(), jc.numpy(), C.numpy()), strict=True):
                np.testing.assert_array_equal(actual, expected)


def _structural_kkt_inactive_limit_adds_curvature_without_force(test, device):
    """A finite, currently open limit adds a metric, never a ghost force."""
    model, _, _, _ = _build_joint_pair(device, newton.JointType.REVOLUTE, finite_limit=True)
    solver = newton.solvers.SolverVBD(model, iterations=1, rigid_compliant_alm=True, rigid_joint_global_iterations=1)
    _simulate(model, solver, 1, 1.0 / 600.0)
    solver.joint_drive_limit_support.fill_(100.0)
    solver.joint_drive_lambda.zero_()
    solver.joint_limit_lambda.zero_()
    bucket = solver._structural_graph_kkt.buckets[0]
    _linearize_bucket(model, solver, bucket, model.body_q, 1.0 / 600.0)
    jacobian = bucket.jacobian_child.numpy()[0]
    test.assertGreater(abs(float(jacobian[5, 5])), 0.99)
    np.testing.assert_array_equal(bucket.residual.numpy()[0, 5], 0.0)
    test.assertGreater(float(bucket.compliance.numpy()[0, 5, 5]), 0.0)


def _structural_kkt_open_contact_adds_curvature_without_force(test, device):
    """An open collision candidate bounds activation but exerts no force."""
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
    test.assertGreater(float(np.max(solver.body_hessian_ll.numpy()[bodies, 2, 2])), 0.0)
    np.testing.assert_array_equal(solver.body_forces.numpy()[bodies], 0.0)
    np.testing.assert_array_equal(solver.body_torques.numpy()[bodies], 0.0)
    np.testing.assert_array_equal(solver._structural_graph_kkt.island_contact_state.numpy(), 1)


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
    for _ in range(10):
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
    test.assertGreater(max(linear_norms[:5]), 0.0, message)
    test.assertGreater(max(angular_norms[:5]), 0.0, message)
    test.assertLess(max(linear_norms[5:]), 3.0 * max(linear_norms[:5]), message)
    test.assertLess(max(angular_norms[5:]), 3.0 * max(angular_norms[:5]), message)
    # A median pairwise log slope is insensitive to one multiplier spike. Over
    # this 900-frame sampled span, 1.5 rejects sustained growth above roughly
    # 4.5e-4 per frame without requiring a noisy stationary trace to be flat.
    test.assertLess(_robust_history_growth_factor(linear_norms), 1.5, message)
    test.assertLess(_robust_history_growth_factor(angular_norms), 1.5, message)

    body_q = state_0.body_q.numpy()
    test.assertTrue(np.isfinite(body_q).all())
    test.assertLess(maximum_velocity_component, 1.0e-2)
    test.assertLess(_max_joint_gap(model, body_q, joints), 1.0e-5)


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
        inputs=[islands, enabled, enabled, scale, system],
        outputs=[correction],
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
        inputs=[ids, enabled, enabled, scale, system],
        outputs=[correction],
        device=device,
    )
    np.testing.assert_array_equal(system.numpy()[:, 3, 3], 1.0)
    np.testing.assert_array_equal(correction.numpy(), 0.0)


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
    test.assertEqual(rigid_vbd_kkt._structural_payload_budget(CapacityOnlyDevice(16 * cap)), cap)
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
            test.assertTrue(
                all(bucket._estimated_bytes == bucket._diagnostics.estimated_bytes for bucket in backend.buckets)
            )
            diagnostics = backend._bucket_diagnostics
            test.assertEqual(len(diagnostics), 1)
            expected_kind = "closed_tree" if kind.startswith("closed_tree") else kind
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
    bodies, _ = builder.add_rod_graph(
        node_positions=positions,
        edges=edges,
        radius=0.01,
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
        healthy_diagnostics.body_count_total,
        healthy_diagnostics.batch_count,
        healthy_diagnostics.row_count_total,
    )

    partial_model, partial_healthy_bodies = _build_mixed_fixed_topologies(device)
    with mock.patch.object(
        rigid_vbd_kkt,
        "_structural_payload_budget",
        return_value=healthy_payload_budget,
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
    """Match path Delassus assembly and CR solutions to float64 oracles."""
    cases = (
        (16, 1.0e4, 1.0e-2, 1.0e-3, 1.0e-3),
        (16, 1.0e9, 1.0e-2, 1.0e-3, 1.0e-3),
        # Exercise every persistent-CR level at a condition number above 5e8.
        # The looser bounds are explicit float32 forward-error limits, not a
        # relaxation of the independently checked Delassus identity.
        (128, 1.0e9, 3.0e-2, 2.0e-2, 6.0e-3),
    )
    for (
        segments,
        stiffness,
        residual_limit,
        multiplier_limit,
        correction_limit,
    ) in cases:
        with test.subTest(segments=segments, stiffness=stiffness):
            metrics = _path_float64_oracle_metrics(device, stiffness, segments=segments)
            message = f"segments={segments}, K={stiffness:.1e}: {metrics}"
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
    """Resolve stiff cyclic corrections without losing observable reactions."""
    for ring_segments in (32, 64, 128):
        with test.subTest(ring_segments=ring_segments):
            metrics = _tree_float64_oracle_metrics(device, 1.0e12, closed=True, ring_segments=ring_segments)
            test.assertTrue(all(np.isfinite(value) for value in metrics.values()), metrics)
            test.assertLess(metrics["equilibrated_residual"], 1.0e-4, metrics)
            test.assertLess(metrics["correction_error"], 1.0e-6, metrics)
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
    test.assertTrue(bucket.use_persistent_cr)
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
    for row in range(bucket.row_count):
        value = diagonal[row] @ solution[row] - rhs[row]
        if row > 0:
            value += lower[row] @ solution[row - 1]
        if row + 1 < bucket.row_count:
            value += upper[row] @ solution[row + 1]
        residual[row] = value
    relative_residual = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1.0e-30))

    test.assertTrue(np.isfinite(solution).all())
    test.assertLess(relative_residual, 2.0e-2)


def _structural_kkt_persistent_cr_matches_level_schedule(test, device):
    """Preserve the exact CR correction while collapsing short-path launches."""

    active_threshold = _cr_persistent_max_rows(device)
    expected_threshold = 160 if device.is_cuda and device.arch == 120 else _CR_PERSISTENT_MAX_ROWS
    test.assertEqual(active_threshold, expected_threshold)

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
        test.assertEqual(bucket.row_count, active_threshold)
        test.assertTrue(bucket.use_persistent_cr)
        bucket.use_persistent_cr = use_persistent_cr
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
    test.assertEqual(larger_bucket.row_count, active_threshold + 1)
    test.assertFalse(larger_bucket.use_persistent_cr)

    cpu = wp.get_device("cpu")
    cpu_model, _, _ = _build_chain(cpu, segments=_cr_persistent_max_rows(cpu) + 1, stiffness=1.0e9)
    cpu_solver = newton.solvers.SolverVBD(
        cpu_model,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    test.assertFalse(cpu_solver._structural_graph_kkt.path_buckets[0].use_persistent_cr)


def _structural_kkt_cr_tail_preserves_factors(test, device):
    """Check dispatch, independent batch values, and fresh captured CR factors."""
    for count, batches in ((1, 1), (160, 2), (161, 1), (257, 3), (511, 1), (1024, 1)):
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
            expected = device.is_cuda and device.arch == 120 and count > _cr_persistent_max_rows(device)
            test.assertEqual(bucket.use_fused_cr_tail, expected)
            if expected:
                test.assertLessEqual(
                    (count + bucket.cr_tail_stride - 1) // bucket.cr_tail_stride,
                    rigid_vbd_kkt._CR_TAIL_MAX_ROWS,
                )
            baseline.use_persistent_cr = False
            baseline.use_fused_cr_tail = False
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


def _structural_kkt_cr_tail_preserves_captured_trajectory(test, device):
    """Preserve poses and velocities with fresh operators at G1 and G2."""
    if device.arch != 120:
        test.skipTest("Fused CR tail is enabled only on the measured SM120 architecture")
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
            test.assertEqual(bucket.row_count, 257)
            test.assertTrue(bucket.use_fused_cr_tail)
            bucket.use_fused_cr_tail = use_tail
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
        inputs=[body_ids, islands, state, correction, step_scale, body_com],
        outputs=[body_q],
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


def _structural_kkt_uses_explicit_iteration_budget(test, device):
    """Distribute the requested global passes over the local iteration budget."""
    model, _, _ = _build_chain(device, segments=4)

    disabled = newton.solvers.SolverVBD(model, iterations=5, rigid_compliant_alm=True)
    test.assertIsNone(disabled._structural_graph_kkt)

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
        ["local", "global", "local", "local", "global", "local", "local"],
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
    test.assertEqual(all_global_events, ["local", "global", "local", "global", "local"])


def _structural_kkt_all_memory_fallback_uses_local_schedule(test, device):
    """Keep the ordinary G=0 schedule when every planned bucket falls back."""
    model = _build_replicated_fixed_topology(
        device,
        closed=True,
        worlds=1,
        link_count=20,
        closure_count=16,
    )
    with mock.patch.object(rigid_vbd_kkt, "_structural_payload_budget", return_value=0):
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
    """Retain support when a sliding cable reverses its applied load."""
    for iterations in (1, 5):
        with test.subTest(iterations=iterations):
            result = _simulate_ground_drag(device, 1, iterations=iterations, contact_history=True, reverse_at=140)
            test.assertLess(result["penetration"], 5.0e-4)
            test.assertLess(result["rise"], 1.0e-2)
            test.assertLess(result["gap"], 4.0e-4)
            test.assertGreater(result["motion"], 2.0)
            test.assertLess(result["motion"], 3.0)


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
    pending = wp.ones(1, dtype=int, device=device)
    evaluations = wp.zeros(1, dtype=int, device=device)

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
            outputs=[enabled, scale],
            device=device,
        )

        def apply_trial(trial):
            wp.launch(
                rigid_vbd_kkt._apply_trial_correction,
                2,
                inputs=[ids, ids, correction, scale, com, original],
                outputs=[pose],
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
                inputs=[enabled, merit, trial == 4],
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
            inputs=[ids, ids, correction, scale, com, original],
            outputs=[pose],
            device=device,
        )

    solve()
    with mock.patch.object(wp.array, "numpy", side_effect=AssertionError("host readback in line search")):
        with wp.ScopedCapture(device) as capture:
            solve()
        wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(pose.numpy()[0], original.numpy()[0])
    test.assertEqual(float(pose.numpy()[1, 0]), 0.0)
    np.testing.assert_array_equal(scale.numpy(), [0.0, 0.25])
    test.assertEqual(
        int(evaluations.numpy()[0]),
        4 if device.is_cpu or wp.is_conditional_graph_supported() else 6,
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

        def counted_refresh():
            wp.launch(_count_global_contact_refresh, 1, inputs=[slot], outputs=[counts], device=device)
            refresh()

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

    # Default atomic reductions must retain fixed scheduling: skipping a
    # numerically different repeated evaluation changed stiff toy hold motion.
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
            mock.patch.object(wp, "capture_if", side_effect=AssertionError("nondeterministic shortcut")),
            mock.patch.object(wp.array, "numpy", side_effect=AssertionError("host readback")),
            wp.ScopedCapture(device) as capture,
        ):
            solver.step(state_in, state_out, control, contacts, dt)
        wp.capture_launch(capture.graph)
        test.assertTrue(np.isfinite(state_out.body_q.numpy()).all())


class TestVBDRigidKKT(unittest.TestCase):
    """Validate the optional rigid VBD structural KKT backend."""

    pass


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
    "test_structural_kkt_open_contact_adds_curvature_without_force",
    _structural_kkt_open_contact_adds_curvature_without_force,
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
    "test_structural_kkt_uses_explicit_iteration_budget",
    _structural_kkt_uses_explicit_iteration_budget,
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
