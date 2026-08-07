# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the private VBD structural compliance-KKT backend."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.rigid_vbd_kkt import (
    _corrected_pose,
    _inverse_spatial_robust,
    back_substitute_tree_backbone_cr_in_place,
    classify_global_contact_islands,
    limit_dynamic_contact_jacobi_step,
    linearize_joint_path_rows,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


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


def _pin_body(builder: newton.ModelBuilder, body: int) -> None:
    builder.body_mass[body] = 0.0
    builder.body_inv_mass[body] = 0.0
    builder.body_inertia[body] = wp.mat33(0.0)
    builder.body_inv_inertia[body] = wp.mat33(0.0)


def _build_chain(device, *, segments=16, stiffness=1.0e7, dahl=False, pinned=True, with_particle=False):
    builder = newton.ModelBuilder()
    if dahl:
        newton.solvers.SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)
    points = newton.utils.create_straight_cable_points(
        start=wp.vec3(-0.5 * segments * 0.03, 0.0, 1.0),
        direction=wp.vec3(1.0, 0.0, 0.0),
        length=segments * 0.03,
        num_segments=segments,
    )
    quaternions = newton.utils.create_parallel_transport_cable_quaternions(points)
    bodies, joints = builder.add_rod(
        positions=points,
        quaternions=quaternions,
        radius=0.01,
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


def _build_loop_with_branch(device, *, ring_segments=8, stiffness=1.0e9, add_chord=False, grounded=False):
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
        bend_stiffness=1.0e4,
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
        closure = builder.add_joint_cable(
            parent=int(bodies[previous_edge]),
            child=int(bodies[next_edge]),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * previous_length), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.5 * next_length), wp.quat_identity()),
            stretch_stiffness=stiffness,
            bend_stiffness=1.0e4,
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
        chord = builder.add_joint_cable(
            parent=parent_body,
            child=child_body,
            parent_xform=wp.transform(parent_anchor, wp.quat_identity()),
            child_xform=wp.transform(child_anchor, child_anchor_rotation),
            stretch_stiffness=stiffness,
            bend_stiffness=1.0e4,
            bend_damping=1.0e3,
        )
        joints.append(chord)
    branch_length = float(wp.length(positions[-1] - positions[-2]))
    attachment = builder.add_joint_cable(
        parent=-1,
        child=int(bodies[-1]),
        parent_xform=wp.transform(positions[-1], wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * branch_length), wp.quat_identity()),
        stretch_stiffness=stiffness,
        bend_stiffness=1.0e4,
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


def _build_replicated_fixed_topology(device, *, closed: bool, worlds=4, link_count=6):
    """Build equal-shape rigid islands to exercise topology batching."""
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
    if closed:
        template.add_joint_fixed(
            parent=bodies[-1],
            child=bodies[0],
            parent_xform=wp.transform((-spacing * (link_count - 1), 0.0, 0.0), wp.quat_identity()),
        )

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for world in range(worlds):
        builder.add_builder(template, xform=wp.transform((0.0, 0.2 * world, 0.0), wp.quat_identity()))
    builder.color(balance_colors=False)
    return builder.finalize(device=device)


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


def _build_joint_pair(device, joint_type, *, finite_limit=False, kinematic_parent=False):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent = builder.add_link(xform=wp.transform_identity(), mass=1.0, is_kinematic=kinematic_parent)
    child = builder.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(parent, hx=0.04, hy=0.03, hz=0.02)
    builder.add_shape_box(child, hx=0.04, hy=0.03, hz=0.02)
    parent_xform = wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity())
    child_xform = wp.transform_identity()
    if joint_type == newton.JointType.CABLE:
        joint = builder.add_joint_cable(
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
    builder.add_articulation([joint])
    builder.color(balance_colors=False)
    return builder.finalize(device=device), parent, child, joint


def _build_grounded_chain(device, *, global_iterations):
    builder = newton.ModelBuilder()
    shape = builder.default_shape_cfg.copy()
    shape.mu = 0.45
    shape.ke = 1.0e5
    shape.kd = 100.0
    radius = 0.01
    segments = 32
    points = newton.utils.create_straight_cable_points(
        start=wp.vec3(-0.5 * segments * 0.03, 0.0, radius + 0.001),
        direction=wp.vec3(1.0, 0.0, 0.0),
        length=segments * 0.03,
        num_segments=segments,
    )
    bodies, joints = builder.add_rod(
        positions=points,
        quaternions=newton.utils.create_parallel_transport_cable_quaternions(points),
        radius=radius,
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
        iterations=5,
        rigid_compliant_alm=True,
        rigid_contact_history=False,
        rigid_joint_global_iterations=global_iterations,
        rigid_body_contact_buffer_size=128,
    )
    return model, pipeline, contacts, solver, np.asarray(bodies, dtype=np.int32), np.asarray(joints), radius


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
            solver.joint_cable_rest_kb_local,
            solver.joint_cable_rest_twist,
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


def _simulate_ground_drag(device, global_iterations):
    model, pipeline, contacts, solver, bodies, joints, radius = _build_grounded_chain(
        device, global_iterations=global_iterations
    )
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    dt = 1.0 / 600.0
    initial_center = 0.0
    minimum_z = np.inf
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
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, dt)
        state_in, state_out = state_out, state_in
        if step == 39:
            initial_center = float(state_in.body_q.numpy()[bodies, 0].mean())
        elif step >= 40:
            minimum_z = min(minimum_z, float(state_in.body_q.numpy()[bodies, 2].min()))
    body_q = state_in.body_q.numpy()
    return {
        "motion": float(body_q[bodies, 0].mean()) - initial_center,
        "penetration": max(0.0, radius - minimum_z),
        "gap": _max_joint_gap(model, body_q, joints),
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
    test.assertFalse(elastic_solver._structural_graph_kkt.has_joint_limits)

    dahl, _, _ = _build_chain(device, dahl=True)
    dahl_solver = newton.solvers.SolverVBD(
        dahl,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    test.assertIsNotNone(dahl_solver._structural_graph_kkt)
    test.assertFalse(dahl_solver._structural_graph_kkt.has_joint_limits)

    limited, _, _, _ = _build_joint_pair(device, newton.JointType.REVOLUTE, finite_limit=True)
    limited_solver = newton.solvers.SolverVBD(
        limited,
        iterations=1,
        rigid_compliant_alm=True,
        rigid_joint_global_iterations=1,
    )
    test.assertIsNotNone(limited_solver._structural_graph_kkt)
    test.assertTrue(limited_solver._structural_graph_kkt.has_joint_limits)
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
    contact_hessian_ll = wp.array(
        [wp.mat33(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
        dtype=wp.mat33,
        device=device,
    )
    contact_hessian_al = wp.zeros(1, dtype=wp.mat33, device=device)
    contact_hessian_aa = wp.zeros(1, dtype=wp.mat33, device=device)
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
            contact_hessian_ll,
            contact_hessian_al,
            contact_hessian_aa,
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
            contact_hessian_ll,
            contact_hessian_al,
            contact_hessian_aa,
            1.0,
            body_q,
            body_mass,
            body_inertia,
        ],
        outputs=[step_scale],
        device=device,
    )
    np.testing.assert_array_equal(step_scale.numpy(), [1.0])


def _structural_kkt_reuses_compact_scratch(test, device):
    """Reuse compact graph-local workspaces across non-overlapping lifetimes."""
    builder = newton.ModelBuilder()
    for index in range(8):
        builder.add_body(
            xform=wp.transform(wp.vec3(float(index), 0.0, 2.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
    points = newton.utils.create_straight_cable_points(
        start=wp.vec3(0.0, 0.0, 1.0),
        direction=wp.vec3(1.0, 0.0, 0.0),
        length=0.24,
        num_segments=8,
    )
    bodies, _ = builder.add_rod(
        positions=points,
        quaternions=newton.utils.create_parallel_transport_cable_quaternions(points),
        radius=0.01,
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
    test.assertIs(solver.body_dynamic_contact_hessian, backend.body_matrix)
    test.assertFalse(backend.has_joint_limits)

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
    model, _, _, _ = _build_joint_pair(device, newton.JointType.CABLE)
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
    test.assertLess(kkt_gap, 0.05 * local_gap, f"KKT gap {kkt_gap:.3e} did not improve local gap {local_gap:.3e}")


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
    test.assertLess(kkt_gap, 0.05 * local_gap, f"Tree KKT gap {kkt_gap:.3e} did not improve {local_gap:.3e}")


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
    test.assertEqual(tree.diagonal.ptr, tree.jacobian_parent.ptr)
    kkt_state = _simulate(kkt_model, kkt_solver, 20, dt)
    kkt_gap = _max_joint_gap(kkt_model, kkt_state.body_q.numpy(), kkt_joints)

    test.assertTrue(np.isfinite(kkt_state.body_q.numpy()).all())
    test.assertLess(kkt_gap, 0.05 * local_gap, f"Closed KKT gap {kkt_gap:.3e} did not improve {local_gap:.3e}")

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


def _cable_kkt_preserves_certified_ground_sliding(test, device):
    """Improve cable closure without increasing certified ground penetration."""
    local = _simulate_ground_drag(device, 0)
    kkt = _simulate_ground_drag(device, 2)

    test.assertGreater(local["motion"], 1.0)
    test.assertGreater(kkt["motion"], 0.8 * local["motion"])
    test.assertLess(kkt["penetration"], 0.01 * local["penetration"])
    test.assertLess(kkt["gap"], 0.01 * local["gap"])


def _cable_kkt_closes_stiff_loop_on_static_ground(test, device):
    """Improve a grounded loop while preserving static contact support."""
    local = _simulate_grounded_loop_load(device, 0)
    kkt = _simulate_grounded_loop_load(device, 1)

    test.assertLess(kkt["gap"], 0.01 * local["gap"])
    test.assertLess(kkt["penetration"], 1.0e-4)


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
    test.assertLess(kkt_gap, 0.25 * local_gap, f"Structural KKT gap {kkt_gap:.3e} did not improve {local_gap:.3e}")
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
    wp.launch(_invert_spatial_matrix, 1, inputs=[matrix_device], outputs=[inverse_device], device=device)
    inverse = inverse_device.numpy()[0]
    residual = float(np.max(np.abs(matrix @ inverse - np.eye(6))))

    test.assertTrue(np.isfinite(inverse).all())
    test.assertLess(residual, 2.0e-2)


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

    def simulate(use_persistent_cr):
        # Exercise the largest topology routed through the persistent kernel.
        model, _, _ = _build_chain(device, segments=128, stiffness=1.0e9)
        solver = newton.solvers.SolverVBD(
            model,
            iterations=2,
            rigid_compliant_alm=True,
            rigid_joint_global_iterations=1,
        )
        bucket = solver._structural_graph_kkt.path_buckets[0]
        test.assertTrue(bucket.use_persistent_cr)
        bucket.use_persistent_cr = use_persistent_cr
        state = _simulate(model, solver, 8, 1.0 / 600.0)
        return state.body_q.numpy(), state.body_qd.numpy()

    level_q, level_qd = simulate(False)
    persistent_q, persistent_qd = simulate(True)
    np.testing.assert_array_equal(persistent_q, level_q)
    np.testing.assert_array_equal(persistent_qd, level_qd)


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


def _structural_kkt_joint_linearizations_match_finite_difference(test, device):
    """Match global joint Jacobians to finite-difference derivatives."""
    """Check endpoint signs, lever arms, and angular frames for every represented joint."""
    dt = 1.0 / 600.0
    epsilon = 2.0e-3
    represented_types = (
        newton.JointType.CABLE,
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


class TestVBDRigidKKT(unittest.TestCase):
    """Validate the optional rigid VBD structural KKT backend."""

    pass


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
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_reduces_fixed_chain_error",
    _structural_kkt_reduces_fixed_chain_error,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_batches_replicated_topologies",
    _structural_kkt_batches_replicated_topologies,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_structural_kkt_drive_matches_finite_material_equilibrium",
    _structural_kkt_drive_matches_finite_material_equilibrium,
    devices=get_cuda_test_devices(),
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
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_cable_kkt_closes_stiff_loop_with_one_global_pass",
    _cable_kkt_closes_stiff_loop_with_one_global_pass,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_cable_kkt_preserves_certified_ground_sliding",
    _cable_kkt_preserves_certified_ground_sliding,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestVBDRigidKKT,
    "test_cable_kkt_closes_stiff_loop_on_static_ground",
    _cable_kkt_closes_stiff_loop_on_static_ground,
    devices=get_cuda_test_devices(),
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
    "test_structural_kkt_robust_block_inverse",
    _structural_kkt_robust_block_inverse,
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
    "test_structural_kkt_backbone_tail_response_is_finalized",
    _structural_kkt_backbone_tail_response_is_finalized,
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
    "test_structural_kkt_captures_numeric_inertia_refresh",
    _structural_kkt_captures_numeric_inertia_refresh,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
