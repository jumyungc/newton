# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal compliance-KKT accelerator for persistent VBD joint graphs.

Accelerates coupled joint structures with a global correction that communicates
across a connected island. Local VBD keeps ownership of inertia, contact,
friction, history, and multipliers; this module accelerates structural coupling.

The system is written in compliance form,

    (C + J B^-1 J^T) lambda = c + J B^-1 f,

and avoids explicitly forming the stiff primal product ``J^T K J``. This does
not bound conditioning: redundant constraints, anisotropy, and finite-precision
elimination can still make near-rigid systems inaccurate or non-finite.

Contact candidates participate in the global objective. A device-side
directional line search shortens steps that cross a force-equilibrium point
and rejects non-descent or non-finite directions. Contact history and duals
remain owned by local VBD. This check assumes valid collision witnesses; it
cannot repair incorrect normals or separations from collision detection.

Each eligible connected island is factored by the exact scheme its topology allows.
Serial, branched, and cyclic graphs are private factorization schedules of the
same system, not user-visible solver modes. Existing VBD contact Hessians enter
as a majorizer -- stiffening supported directions without adding contact-graph
edges. Before committing a correction, an island-wide block-Jacobi relaxation
accounts for omitted dynamic-contact overlap. A following local sweep
relinearizes nonlinear contact before the timestep is finalized.
History-bearing rows otherwise retain the step-frozen values owned by local VBD.

This module is private; the only public control is the global-iteration count G.
"""

from __future__ import annotations

import contextlib
from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Any

import numpy as np
import warp as wp

from newton._src.core.types import MAXVAL
from newton._src.math import quat_velocity
from newton._src.sim import JointType
from newton._src.solvers.rod_kernels import (
    RodBendTwistMeasure,
    _finite_curvature_binormal_derivative,
    _transported_twist_angle_jacobian_from_measure,
    assemble_geometric_rod_kappa_z,
    measure_rod_bend_twist_z,
    rod_bend_twist_delta,
)

from .rigid_vbd_kernels import (
    _NUM_CONTACT_THREADS_PER_BODY,
    _SMALL_ANGLE_EPS,
    _SMALL_LENGTH_EPS,
    _compliant_alm_coefficients,
    _evaluate_drive_axis,
    _evaluate_limit_axis,
    _global_limit_support,
    _material_force_terms,
    _pose_com_diff,
    _reset_world_selected,
    _resolve_active_drive_row,
    _unwrap_hinge_angle,
    build_joint_projectors,
    compute_joint_kappa_and_jacobian,
    compute_kappa_dot,
    contact_surface_separation,
    evaluate_rigid_contact_angular_friction,
    evaluate_rigid_contact_normal_sliding,
    pose_from_displacement,
    rotation_vector_compose,
)
from .rigid_vbd_kkt_chains import ChainCondensation
from .rigid_vbd_kkt_chains import island_in_scope as _chain_island_in_scope

wp.set_module_options({"enable_backward": False})

# Per island, from the first evaluation of a translation pass, for contacts against static or kinematic bodies:
# the friction part of the island's right-hand side [0:3], their Coulomb capacity, the sum of mu * f_n [3],
# and their support normal weighted by that capacity [4:7].
_TranslationSupport = wp.types.vector(length=7, dtype=wp.float64)


def _stream_scope(stream):
    """Launch on ``stream`` when given, else on the current stream."""
    return wp.ScopedStream(stream, sync_enter=False) if stream is not None else contextlib.nullcontext()


@wp.func
def _rod_bend_jacobian_column_z_directional(
    q_wp: wp.quat,
    measure: RodBendTwistMeasure,
    omega_world: wp.vec3,
    is_parent: bool,
) -> wp.vec3:
    """Evaluate one bend-Jacobian column in the global solver's float32 path."""
    dt0 = wp.vec3(0.0)
    dt1 = wp.vec3(0.0)
    if is_parent:
        dt0 = wp.cross(omega_world, measure.t0)
    else:
        dt1 = wp.cross(omega_world, measure.t1)
    dkb_world = _finite_curvature_binormal_derivative(measure.t0, measure.t1, dt0, dt1)
    if is_parent:
        dkb_world = dkb_world - wp.cross(omega_world, measure.kb_world)
    return wp.quat_rotate(wp.quat_inverse(q_wp), dkb_world)


@wp.func
def _rod_bend_twist_jacobian_z_global(
    q_wp: wp.quat,
    measure: RodBendTwistMeasure,
    is_parent: bool,
) -> wp.mat33:
    """Assemble the global cable Jacobian with stable directional bend columns.

    The factored compliance solve is sensitive to float32 linearization drift at
    the contact stick/slip boundary. Preserve the directional bend arithmetic
    used by the validated reference derivative while retaining the optimized
    closed-form twist row.
    """
    e0 = wp.vec3(1.0, 0.0, 0.0)
    e1 = wp.vec3(0.0, 1.0, 0.0)
    e2 = wp.vec3(0.0, 0.0, 1.0)
    bend0 = _rod_bend_jacobian_column_z_directional(q_wp, measure, e0, is_parent)
    bend1 = _rod_bend_jacobian_column_z_directional(q_wp, measure, e1, is_parent)
    bend2 = _rod_bend_jacobian_column_z_directional(q_wp, measure, e2, is_parent)
    twist_jacobian = _transported_twist_angle_jacobian_from_measure(measure, is_parent)
    return wp.matrix_from_cols(
        wp.vec3(bend0[0], bend0[1], twist_jacobian[0]),
        wp.vec3(bend1[0], bend1[1], twist_jacobian[1]),
        wp.vec3(bend2[0], bend2[1], twist_jacobian[2]),
    )


_CR_SERIAL_TERMINAL_SIZE = 4
"""Maximum coarse row count finished by one batched serial kernel after CR."""

_CR_PERSISTENT_MAX_ROWS = 128
"""Conservative path limit for the synchronized single-block CUDA solve."""

_CR_TAIL_MAX_ROWS = 64
"""Maximum active row count fused after parallel path reduction."""


_PAIRED_OPEN_CHAIN_MIN_BODIES = 64
"""Smallest open chain admitted to the paired body-row solve."""

_SPATIAL_GPU_BLOCK_DIM = 32
"""One-warp blocks for register-heavy spatial-matrix kernels."""

_ROW_PASS_GPU_BLOCK_DIM = 8
"""Blocks for the atomic-free joint row passes, whose rows are visited grouped by joint type.

One thread per row is latency-bound: narrower blocks put more independent warps on more
multiprocessors, and type grouping keeps rod and general rows in different warps.
"""

_FUSED_CLOSURE_MAX_BLOCKS = 16
"""Largest measured dense closure solve owned by one synchronized CUDA block."""

_CLOSURE_BACK_SUBSTITUTE_MIN_BLOCKS = _FUSED_CLOSURE_MAX_BLOCKS + 1
"""Smallest dense closure system whose back-substitution earns a lane split.

Smaller CUDA systems are already solved by the fused persistent kernel, so
allocating a second path's scratch for them would be dead weight. CPU systems
retain their serial solve.
"""


def _closure_back_substitute_lanes(closure_count: int, device) -> bool:
    """Return whether the dense back-substitution should be lane-parallel."""
    return bool(device.is_cuda) and closure_count >= _CLOSURE_BACK_SUBSTITUTE_MIN_BLOCKS


_PAIRED_COOPERATIVE_MIN_CLOSURES = 1
"""Use cooperative FP64 pivots for every CUDA paired backbone.

One 64-thread block owns each survivor. Lanes 0-35 form the four 6x6
products, and another lane applies the physical right-hand side over the
same interval. A single closure used to stay on the serial paired schedule.
"""

_PAIRED_COOPERATIVE_BLOCK_DIM = 64
"""Threads launched for each cooperative survivor or eliminated pair."""

_PAIRED_COOPERATIVE_MATRIX_LANES = wp.constant(36)
"""Lanes that each own one entry of a 6x6 block product."""

_PAIRED_COOPERATIVE_RHS_LANE = wp.constant(36)
"""Lane that updates the physical right-hand side during a cooperative reduction."""

_PAIRED_COOPERATIVE_PHYSICAL_LANE = wp.constant(32)
"""Second-warp lane that applies the physical right-hand side in the triangular kernels.

Response columns 0-31 stay in the first warp, so this lane's spatial products
overlap them. The lane also owns response column 32 when that column exists;
those writes touch a different buffer.
"""

_PAIRED_COARSE_MIN_CLOSURES = 5
"""Smallest closure count whose coarse solve repays a lane-split block.

The closure columns are the only parallel dimension in the paired coarse
block-Thomas tail. Below this the split adds barriers without recovering
enough width, and a single-closure loop measurably regresses.
"""

_FUSED_TREE_MIN_LEVELS = 2
"""Smallest leaf-rake schedule worth collapsing into one synchronized block.

Below this the per-level launches already fit in the dependent-launch budget
and the fused kernel only trades grid width for loop overhead.
"""

_FUSED_TREE_MAX_LEVEL_WIDTH = 8 * _SPATIAL_GPU_BLOCK_DIM
"""Widest per-tree elimination level admitted to the single-block schedule.

One block replaces the whole schedule, so a level wider than a few block
strides would serialize work that the per-level launches spread over several
multiprocessors. Wider trees keep the exact per-level fallback.
"""

_STRUCTURAL_MEMORY_CAPACITY_DIVISOR = 8
"""Maximum physical-device fraction reserved by the optional global backend."""

_STRUCTURAL_MEMORY_CAP_BYTES = 1 << 30
"""Deterministic 1 GiB fallback budget when device capacity is unavailable (CPU or unknown).

The preflight uses immutable ``Device.total_memory``, never volatile free
memory. Planned persistent accelerator payload is bounded to one eighth of a
known CUDA device capacity, or to this 1 GiB fallback when capacity is absent.
This is not a memory reservation and cannot guarantee allocation when other
processes or existing model state have already occupied the device.
"""

_INVERSE_RESIDUAL_TOLERANCE = wp.constant(1.0e-2)
"""Maximum absolute entry of the left residual accepted from a fast float32 block inverse."""


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def _synchronize_cr_block(): ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__shared__ double words[360];
return wp::array_t<wp::float64>(words, 360);
#else
return wp::array_t<wp::float64>();
#endif
""")
def _block_shared_words() -> wp.array[wp.float64]:
    """Return this block's FP64 cooperative scratch (CUDA only)."""
    ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__shared__ int pivot[1];
return wp::array_t<wp::int32>(pivot, 1);
#else
return wp::array_t<wp::int32>();
#endif
""")
def _block_shared_pivot() -> wp.array[wp.int32]:
    """Return this block's cooperative pivot slot (CUDA only)."""
    ...


@wp.func
def _normalized_cable_row(
    error: float,
    previous_error: float,
    material_k: float,
    rho: float,
    damping: float,
    multiplier: float,
    history_force: float,
    history_tangent: float,
    dt: float,
):
    """Return one cable row's tangent and normalized local force defect."""
    s, k_eff, _a = _compliant_alm_coefficients(material_k, rho)
    damping_tangent = damping / dt
    tangent = k_eff + damping_tangent + wp.max(history_tangent, 0.0)
    if tangent <= 0.0:
        return 0.0, 0.0
    force = k_eff * error + s * multiplier + history_force + damping_tangent * (error - previous_error)
    return tangent, force / tangent


@wp.func
def _inverse_spatial(matrix: wp.spatial_matrix):
    """Invert a 6x6 matrix by a 3x3 Schur split when both pivots are invertible."""
    a = wp.mat33(0.0)
    b = wp.mat33(0.0)
    c = wp.mat33(0.0)
    d = wp.mat33(0.0)
    # Rounded Schur updates need not be symmetric. Invert the represented
    # lower-left block rather than substituting the upper-right transpose.
    for row in range(3):
        for column in range(3):
            a[row, column] = matrix[row, column]
            b[row, column] = matrix[row, column + 3]
            c[row, column] = matrix[row + 3, column]
            d[row, column] = matrix[row + 3, column + 3]

    a_inverse = wp.inverse(a)
    a_inverse_b = a_inverse * b
    c_a_inverse = c * a_inverse
    schur = d - c_a_inverse * b
    schur_inverse = wp.inverse(schur)
    upper_right = -(a_inverse_b * schur_inverse)
    lower_left = -(schur_inverse * c_a_inverse)
    upper_left = a_inverse - upper_right * c_a_inverse

    result = wp.spatial_matrix(0.0)
    for row in range(3):
        for column in range(3):
            result[row, column] = upper_left[row, column]
            result[row, column + 3] = upper_right[row, column]
            result[row + 3, column] = lower_left[row, column]
            result[row + 3, column + 3] = schur_inverse[row, column]
    return result


@wp.func
def _inverse_spatial_pivoted(matrix: wp.spatial_matrix):
    """Invert a general 6x6 frontal block with row partial pivoting."""
    value = matrix
    inverse = wp.identity(6, float)
    for column in range(6):
        pivot = column
        pivot_magnitude = wp.abs(value[column, column])
        for row in range(column + 1, 6):
            magnitude = wp.abs(value[row, column])
            if magnitude > pivot_magnitude:
                pivot = row
                pivot_magnitude = magnitude
        if pivot != column:
            for entry in range(6):
                temporary = value[column, entry]
                value[column, entry] = value[pivot, entry]
                value[pivot, entry] = temporary
                temporary = inverse[column, entry]
                inverse[column, entry] = inverse[pivot, entry]
                inverse[pivot, entry] = temporary

        reciprocal = 1.0 / value[column, column]
        for entry in range(6):
            value[column, entry] = reciprocal * value[column, entry]
            inverse[column, entry] = reciprocal * inverse[column, entry]
        for row in range(6):
            if row != column:
                factor = value[row, column]
                for entry in range(6):
                    value[row, entry] = value[row, entry] - factor * value[column, entry]
                    inverse[row, entry] = inverse[row, entry] - factor * inverse[column, entry]
    return inverse


@wp.func
def _inverse_spatial_pivoted(matrix: wp.spatial_matrixd):
    """Invert a general 6x6 frontal block with row partial pivoting."""
    value = matrix
    inverse = wp.identity(6, wp.float64)
    for column in range(6):
        pivot = column
        pivot_magnitude = wp.abs(value[column, column])
        for row in range(column + 1, 6):
            magnitude = wp.abs(value[row, column])
            if magnitude > pivot_magnitude:
                pivot = row
                pivot_magnitude = magnitude
        if pivot != column:
            for entry in range(6):
                temporary = value[column, entry]
                value[column, entry] = value[pivot, entry]
                value[pivot, entry] = temporary
                temporary = inverse[column, entry]
                inverse[column, entry] = inverse[pivot, entry]
                inverse[pivot, entry] = temporary

        reciprocal = wp.float64(1.0) / value[column, column]
        for entry in range(6):
            value[column, entry] = reciprocal * value[column, entry]
            inverse[column, entry] = reciprocal * inverse[column, entry]
        for row in range(6):
            if row != column:
                factor = value[row, column]
                for entry in range(6):
                    value[row, entry] = value[row, entry] - factor * value[column, entry]
                    inverse[row, entry] = inverse[row, entry] - factor * inverse[column, entry]
    return inverse


@wp.func
def _inverse_spatial_robust(matrix: wp.spatial_matrix):
    """Use the fast block inverse unless its computed inverse fails a residual check."""
    inverse = _inverse_spatial(matrix)
    maximum_error = 0.0
    finite = True
    for row in range(6):
        for column in range(6):
            value = 0.0
            for entry in range(6):
                value = value + matrix[row, entry] * inverse[entry, column]
            target = 1.0 if row == column else 0.0
            # wp.max follows fmax semantics and ignores NaN operands. An
            # overflowed fast inverse must still select the pivoted solve.
            finite = finite and wp.isfinite(value)
            maximum_error = wp.max(maximum_error, wp.abs(value - target))
    if not finite or maximum_error > _INVERSE_RESIDUAL_TOLERANCE:
        inverse = _inverse_spatial_pivoted(matrix)
    return inverse


@wp.func_native("""
#if defined(__CUDA_ARCH__)
return fma(a, b, c);
#else
return a * b + c;
#endif
""")
def _fma_f64(a: wp.float64, b: wp.float64, c: wp.float64) -> wp.float64: ...


@wp.func
def _inverse_spatial(matrix: wp.spatial_matrixd):
    """Invert a 6x6 float64 matrix by a register-resident 3x3 Schur split."""
    a = wp.mat33d(0.0)
    b = wp.mat33d(0.0)
    c = wp.mat33d(0.0)
    d = wp.mat33d(0.0)
    for row in range(3):
        for column in range(3):
            a[row, column] = matrix[row, column]
            b[row, column] = matrix[row, column + 3]
            c[row, column] = matrix[row + 3, column]
            d[row, column] = matrix[row + 3, column + 3]

    a_inverse = wp.inverse(a)
    a_inverse_b = a_inverse * b
    c_a_inverse = c * a_inverse
    schur = d - c_a_inverse * b
    schur_inverse = wp.inverse(schur)
    upper_right = -(a_inverse_b * schur_inverse)
    lower_left = -(schur_inverse * c_a_inverse)
    upper_left = a_inverse - upper_right * c_a_inverse

    result = wp.spatial_matrixd(0.0)
    for row in range(3):
        for column in range(3):
            result[row, column] = upper_left[row, column]
            result[row, column + 3] = upper_right[row, column]
            result[row + 3, column] = lower_left[row, column]
            result[row + 3, column + 3] = schur_inverse[row, column]
    return result


@wp.func
def _inverse_spatial_robust(matrix: wp.spatial_matrixd):
    """Use the fast 3x3 Schur inverse unless its float64 residual exceeds tolerance."""
    inverse = _inverse_spatial(matrix)
    maximum_error = wp.float64(0.0)
    finite = True
    for row in range(6):
        for column in range(6):
            value = wp.float64(0.0)
            for entry in range(6):
                value = _fma_f64(matrix[row, entry], inverse[entry, column], value)
            target = wp.float64(1.0) if row == column else wp.float64(0.0)
            finite = finite and wp.isfinite(value)
            maximum_error = wp.max(maximum_error, wp.abs(value - target))
    if not finite or maximum_error > wp.float64(1.0e-9):
        inverse = _inverse_spatial_pivoted(matrix)
    return inverse


@wp.func
def _set_spatial_block(matrix: wp.spatial_matrix, row_block: int, column_block: int, value: wp.mat33):
    for row in range(3):
        for column in range(3):
            matrix[row + 3 * row_block, column + 3 * column_block] = value[row, column]
    return matrix


@wp.func
def _spatial_block(matrix: wp.spatial_matrix, row_block: int, column_block: int):
    value = wp.mat33(0.0)
    for row in range(3):
        for column in range(3):
            value[row, column] = matrix[row + 3 * row_block, column + 3 * column_block]
    return value


@wp.func
def _spatial_vector_nonzero(value: wp.spatial_vector) -> bool:
    return (
        value[0] != 0.0 or value[1] != 0.0 or value[2] != 0.0 or value[3] != 0.0 or value[4] != 0.0 or value[5] != 0.0
    )


@wp.func
def _scale_spatial_vector(scale: wp.spatial_vector, value: wp.spatial_vector):
    result = wp.spatial_vector()
    for row in range(6):
        result[row] = scale[row] * value[row]
    return result


@wp.func
def _scale_spatial_vector(scale: wp.spatial_vector, value: wp.spatial_vectord):
    result = wp.spatial_vector()
    for row in range(6):
        result[row] = float(wp.float64(scale[row]) * value[row])
    return result


@wp.func
def _scale_spatial_matrix(
    left_scale: wp.spatial_vector,
    matrix: wp.spatial_matrix,
    right_scale: wp.spatial_vector,
):
    result = wp.spatial_matrix(0.0)
    for row in range(6):
        for column in range(6):
            result[row, column] = left_scale[row] * matrix[row, column] * right_scale[column]
    return result


@wp.kernel
def clear_structural_contact_forces(
    body_ids: wp.array[wp.int32],
    body_forces: wp.array[wp.vec3],
    body_torques: wp.array[wp.vec3],
):
    """Clear only the contact scratch consumed by the directional search."""
    body = body_ids[wp.tid()]
    body_forces[body] = wp.vec3(0.0)
    body_torques[body] = wp.vec3(0.0)


@wp.kernel
def clear_structural_contact_objective(
    body_ids: wp.array[wp.int32],
    body_forces: wp.array[wp.vec3],
    body_torques: wp.array[wp.vec3],
    body_hessian_ll: wp.array[wp.mat33],
    body_hessian_al: wp.array[wp.mat33],
    body_hessian_aa: wp.array[wp.mat33],
    body_dynamic_contact_hessian: wp.array[wp.spatial_matrix],
):
    """Clear contact scratch only for bodies represented by the global graph."""
    slot = wp.tid()
    body = body_ids[slot]
    body_forces[body] = wp.vec3(0.0)
    body_torques[body] = wp.vec3(0.0)
    body_hessian_ll[body] = wp.mat33(0.0)
    body_hessian_al[body] = wp.mat33(0.0)
    body_hessian_aa[body] = wp.mat33(0.0)
    body_dynamic_contact_hessian[slot] = wp.spatial_matrix(0.0)


@wp.kernel
def reset_translation_solve_state(
    island_contact_state: wp.array[wp.int32],
    island_step_scale: wp.array[float],
    translation_system: wp.array[wp.mat44d],
    translation_support: wp.array[_TranslationSupport],
):
    """Reset island-level translation scratch in one lightweight launch."""
    island = wp.tid()
    island_contact_state[island] = 1
    island_step_scale[island] = 1.0
    translation_system[island] = wp.mat44d(0.0)
    translation_support[island] = _TranslationSupport()


@wp.kernel
def reset_structural_solve_state(
    island_count: int,
    has_contacts: int,
    clear_limit_islands: int,
    body_ids: wp.array[wp.int32],
    body_forces: wp.array[wp.vec3],
    body_torques: wp.array[wp.vec3],
    body_hessian_ll: wp.array[wp.mat33],
    body_hessian_al: wp.array[wp.mat33],
    body_hessian_aa: wp.array[wp.mat33],
    body_dynamic_contact_hessian: wp.array[wp.spatial_matrix],
    body_correction: wp.array[wp.spatial_vector],
    island_contact_state: wp.array[wp.int32],
    island_step_scale: wp.array[float],
    limit_search_islands: wp.array[wp.int32],
):
    """Reset all per-pass structural scratch in one launch."""
    slot = wp.tid()
    if has_contacts != 0:
        body = body_ids[slot]
        body_forces[body] = wp.vec3(0.0)
        body_torques[body] = wp.vec3(0.0)
        body_hessian_ll[body] = wp.mat33(0.0)
        body_hessian_al[body] = wp.mat33(0.0)
        body_hessian_aa[body] = wp.mat33(0.0)
        body_dynamic_contact_hessian[slot] = wp.spatial_matrix(0.0)
    body_correction[slot] = wp.spatial_vector()
    if slot < island_count:
        island_contact_state[slot] = 1
        island_step_scale[slot] = 1.0
        if clear_limit_islands != 0:
            limit_search_islands[slot] = 0


@wp.kernel(module="unique")
def add_contact_island_rod_kirchhoff_proximal(
    rod_body_ids: wp.array[wp.int32],
    rod_body_islands: wp.array[wp.int32],
    rod_kirchhoff_proximal: wp.array[float],
    island_contact_state: wp.array[wp.int32],
    body_hessian_aa: wp.array[wp.mat33],
):
    """Add exact zero-bias Cosserat director shear-bending proximal metric on contact-active rod islands.

    Coupled Schur elimination across Cosserat ``ROD`` chains annihilates shear stiffness along the
    zero-shear Kirchhoff transverse manifold, leaving micro-inertia segments free to whip across
    open pulley/conduit clearances within one iteration. Retaining ``c_rod * L^2 * K_shear`` on
    ``body_hessian_aa`` (zero RHS bias) bounds single-step director rotation on contact-bearing
    islands while leaving contact-free rods (``island_contact_state != -1``) and all equilibria unchanged.
    """
    idx = wp.tid()
    if island_contact_state[rod_body_islands[idx]] != -1:
        return
    body = rod_body_ids[idx]
    prox = rod_kirchhoff_proximal[idx]
    h_aa = body_hessian_aa[body]
    h_aa[0, 0] = h_aa[0, 0] + prox
    h_aa[1, 1] = h_aa[1, 1] + prox
    h_aa[2, 2] = h_aa[2, 2] + prox
    body_hessian_aa[body] = h_aa


def _make_accumulate_structural_contacts(translation_only: bool, forces_only: bool):
    """Specialize atomic destinations without changing the contact work budget.

    The deterministic runtime plans scatter storage from compiled destinations.
    Static modes avoid sorting and polling destinations unused by this pass.
    """

    def accumulate_structural_body_body_contacts(
        dt: float,
        body_ids: wp.array[wp.int32],
        body_q_prev: wp.array[wp.transform],
        body_q: wp.array[wp.transform],
        body_com: wp.array[wp.vec3],
        body_inv_mass: wp.array[float],
        contact_damping_bidirectional: bool,
        contact_normal_rho: wp.array[float],
        contact_material_ke: wp.array[float],
        contact_material_kd: wp.array[float],
        contact_material_mu: wp.array[wp.vec3],
        contact_tangent_rho: wp.array[wp.mat22],
        contact_torsional_rho: wp.array[float],
        contact_rolling_rho: wp.array[wp.mat22],
        contact_lambda: wp.array[wp.vec3],
        contact_lambda_angular: wp.array[wp.vec3],
        contact_C0: wp.array[wp.vec3],
        stab_alpha: float,
        rigid_contact_count: wp.array[int],
        rigid_contact_shape0: wp.array[int],
        rigid_contact_shape1: wp.array[int],
        rigid_contact_point0: wp.array[wp.vec3],
        rigid_contact_point1: wp.array[wp.vec3],
        rigid_contact_surface_velocity: wp.array[wp.vec3],
        rigid_contact_offset0: wp.array[wp.vec3],
        rigid_contact_offset1: wp.array[wp.vec3],
        rigid_contact_normal: wp.array[wp.vec3],
        rigid_contact_margin0: wp.array[float],
        rigid_contact_margin1: wp.array[float],
        shape_body: wp.array[wp.int32],
        body_slot_by_id: wp.array[wp.int32],
        graph_body_island: wp.array[wp.int32],
        classify_islands: int,
        body_inertia_q: wp.array[wp.transform],
        body_disp: wp.array[wp.vec3],
        body_disp_ang: wp.array[wp.vec3],
        use_displacement: int,
        body_mass: wp.array[float],
        island_translation_free: wp.array[int],
        translation_system: wp.array[wp.mat44d],
        translation_carry: int,
        translation_remainder: wp.array[wp.vec4],
        translation_support: wp.array[_TranslationSupport],
        body_contact_buffer_pre_alloc: int,
        body_contact_counts: wp.array[wp.int32],
        body_contact_indices: wp.array[wp.int32],
        body_forces: wp.array[wp.vec3],
        body_torques: wp.array[wp.vec3],
        body_hessian_ll: wp.array[wp.mat33],
        body_hessian_al: wp.array[wp.mat33],
        body_hessian_aa: wp.array[wp.mat33],
        body_dynamic_contact_hessian: wp.array[wp.spatial_matrix],
        island_contact_state: wp.array[wp.int32],
    ):
        """Accumulate only the selected objective and classify structural islands."""
        tid = wp.tid()
        body_slot = tid // _NUM_CONTACT_THREADS_PER_BODY
        contact_thread = tid % _NUM_CONTACT_THREADS_PER_BODY
        if body_slot >= body_ids.shape[0]:
            return

        body = body_ids[body_slot]
        if body_inv_mass[body] <= 0.0:
            return

        body_contact_count = body_contact_counts[body]
        island = graph_body_island[body_slot]
        if classify_islands == 1 and contact_thread == 0 and body_contact_count > body_contact_buffer_pre_alloc:
            wp.atomic_min(island_contact_state, island, -2)
        num_contacts = wp.min(body_contact_count, body_contact_buffer_pre_alloc)
        if contact_thread >= num_contacts and not (wp.static(translation_only) and contact_thread == 0):
            # Translation lane zero also owns the body's inertial contribution.
            return
        if wp.static(translation_only) and island_translation_free[island] == 0:
            return
        if wp.static(translation_only):
            # The part of this pass's common shifts, tangential to the island's support plane, that the
            # body's float32 position could not store. It enters the slip and the inertial term below.
            carried = wp.vec3(0.0)
            shifted = bool(False)
            if translation_carry != 0:
                kept = translation_remainder[body_slot]
                carried = wp.vec3(kept[0], kept[1], kept[2])
                shifted = wp.length_sq(carried) > 0.0
            friction_acc = wp.vec3(0.0)
            capacity_acc = float(0.0)
            support_acc = wp.vec3(0.0)
        contact_count = rigid_contact_count[0]
        force_acc = wp.vec3(0.0)
        torque_acc = wp.vec3(0.0)
        h_ll_acc = wp.mat33(0.0)
        h_al_acc = wp.mat33(0.0)
        h_aa_acc = wp.mat33(0.0)
        dynamic_h_ll_acc = wp.mat33(0.0)
        dynamic_h_al_acc = wp.mat33(0.0)
        dynamic_h_aa_acc = wp.mat33(0.0)

        i = contact_thread
        while i < num_contacts:
            contact = body_contact_indices[body * body_contact_buffer_pre_alloc + i]
            if contact >= contact_count:
                i += _NUM_CONTACT_THREADS_PER_BODY
                continue

            shape0 = rigid_contact_shape0[contact]
            shape1 = rigid_contact_shape1[contact]
            body0 = shape_body[shape0] if shape0 >= 0 else -1
            body1 = shape_body[shape1] if shape1 >= 0 else -1
            if body0 != body and body1 != body:
                i += _NUM_CONTACT_THREADS_PER_BODY
                continue

            point0_local = rigid_contact_point0[contact]
            point1_local = rigid_contact_point1[contact]
            point0_offset = rigid_contact_offset0[contact]
            point1_offset = rigid_contact_offset1[contact]
            normal = rigid_contact_normal[contact]
            point0 = wp.transform_point(body_q[body0], point0_local) if body0 >= 0 else point0_local
            point1 = wp.transform_point(body_q[body1], point1_local) if body1 >= 0 else point1_local
            normal_error = -contact_surface_separation(
                point0,
                point1,
                normal,
                rigid_contact_margin0[contact],
                rigid_contact_margin1[contact],
            )
            if wp.static(translation_only):
                side = float(1.0)
                if body != body0:
                    side = float(-1.0)

            other_body = body1 if body == body0 else body0
            dynamic_pair = other_body >= 0 and body_inv_mass[other_body] > 0.0 and body_slot_by_id[other_body] >= 0
            if wp.static(translation_only):
                if other_body >= 0 and body_inv_mass[other_body] > 0.0:
                    other_slot = body_slot_by_id[other_body]
                    if other_slot < 0 or graph_body_island[other_slot] != island:
                        # A dynamic contact crossing islands couples their translations.
                        blocked = wp.mat44d(0.0)
                        blocked[3, 3] = wp.float64(1.0)
                        wp.atomic_add(translation_system, island, blocked)
                        return
            if (
                wp.static(translation_only)
                and dynamic_pair
                and graph_body_island[body_slot_by_id[other_body]] == island
            ):
                # A common translation does no work on an internal pair. Omit
                # both endpoint forces instead of cancelling rounded sums.
                i += _NUM_CONTACT_THREADS_PER_BODY
                continue
            if classify_islands == 1 and dynamic_pair:
                separation = (
                    wp.dot(normal, point1 - point0) - rigid_contact_margin0[contact] - rigid_contact_margin1[contact]
                )
                if not (separation > 0.0 and wp.dot(contact_lambda[contact], normal) <= 0.0):
                    wp.atomic_min(island_contact_state, island, -1)

            multiplier = contact_lambda[contact]
            multiplier_normal = wp.dot(multiplier, normal)
            C0 = contact_C0[contact]
            C0_normal = wp.dot(normal, C0)
            stabilized_error = normal_error - C0_normal
            tangent_C0 = (1.0 - stab_alpha) * (C0 - normal * C0_normal)
            normal_weight = contact_normal_rho[contact]
            material_k = contact_material_ke[contact]

            normal_primal_k, multiplier_normal_eff = _material_force_terms(
                normal_weight,
                material_k,
                multiplier_normal,
            )
            inactive = multiplier_normal <= 0.0 and (
                normal_error <= _SMALL_LENGTH_EPS or normal_primal_k * stabilized_error + multiplier_normal_eff <= 0.0
            )
            if inactive:
                # Use the local active-set metric: a zero-reaction, open contact
                # adds neither force nor curvature. Trial contact refresh still
                # detects activation along a simultaneous structural correction.
                i += _NUM_CONTACT_THREADS_PER_BODY
                continue

            surface_velocity = wp.vec3(0.0)
            if rigid_contact_surface_velocity:
                surface_velocity = rigid_contact_surface_velocity[contact]
            if wp.static(translation_only):
                if shifted:
                    surface_velocity -= (side / dt) * carried

            friction_mu = contact_material_mu[contact]
            (
                force0,
                torque0,
                h_ll0,
                h_al0,
                h_aa0,
                force1,
                torque1,
                h_ll1,
                h_al1,
                h_aa1,
            ) = evaluate_rigid_contact_normal_sliding(
                body0,
                body1,
                body_q,
                body_q_prev,
                body_com,
                point0_local,
                point1_local,
                surface_velocity,
                point0_offset,
                point1_offset,
                normal,
                stabilized_error,
                normal_weight,
                material_k,
                contact_tangent_rho[contact],
                contact_material_kd[contact],
                multiplier,
                friction_mu[0],
                dt,
                tangent_C0,
                contact_damping_bidirectional,
                0 if wp.static(forces_only) else 1,
            )
            if not wp.static(translation_only):
                mu_torsional = friction_mu[1]
                mu_rolling = friction_mu[2]
                normal_load = wp.max(normal_primal_k * stabilized_error + multiplier_normal_eff, 0.0)
                if normal_load > 0.0 and (mu_torsional > 0.0 or mu_rolling > 0.0):
                    pure_torque_b, k_angular = evaluate_rigid_contact_angular_friction(
                        body0,
                        body1,
                        body_q,
                        body_q_prev,
                        normal,
                        normal_load,
                        mu_torsional,
                        mu_rolling,
                        contact_torsional_rho[contact],
                        contact_rolling_rho[contact],
                        contact_lambda_angular[contact],
                    )
                    torque0 -= pure_torque_b
                    torque1 += pure_torque_b
                    if not wp.static(forces_only):
                        h_aa0 += k_angular
                        h_aa1 += k_angular

            if body == body0:
                force_acc += force0
                torque_acc += torque0
                h_ll_acc += h_ll0
                h_al_acc += h_al0
                h_aa_acc += h_aa0
                if dynamic_pair:
                    dynamic_h_ll_acc += h_ll0
                    dynamic_h_al_acc += h_al0
                    dynamic_h_aa_acc += h_aa0
            else:
                force_acc += force1
                torque_acc += torque1
                h_ll_acc += h_ll1
                h_al_acc += h_al1
                h_aa_acc += h_aa1
                if dynamic_pair:
                    dynamic_h_ll_acc += h_ll1
                    dynamic_h_al_acc += h_al1
                    dynamic_h_aa_acc += h_aa1
            if wp.static(translation_only):
                if translation_carry == 0:
                    # Friction is the tangential part of the contact force; mu * f_n bounds it.
                    applied = force1
                    if body == body0:
                        applied = force0
                    friction_acc += applied - wp.dot(normal, applied) * normal
                    capacity = friction_mu[0] * wp.max(normal_primal_k * stabilized_error + multiplier_normal_eff, 0.0)
                    capacity_acc += capacity
                    support_acc += (side * capacity) * normal
            i += _NUM_CONTACT_THREADS_PER_BODY

        if wp.static(translation_only):
            # Restrict directly to the three common-translation coordinates.
            # No rotational scratch or body-matrix gather is consumed here.
            contribution = wp.mat44d(0.0)
            com_world = wp.transform_point(body_q[body], body_com[body])
            inertial_delta = _pose_com_diff(body_inertia_q[body], body_q[body], body_com[body]) - carried
            if use_displacement != 0:
                # The accumulated displacement holds every solved shift, stored or not.
                inertial_delta = -body_disp[body]
            mass = wp.float64(0.0)
            if contact_thread == 0:
                mass = wp.float64(body_mass[body]) / (wp.float64(dt) * wp.float64(dt))
                for col in range(3):
                    contribution[3, col] = mass * wp.float64(wp.abs(com_world[col]))
            for row in range(3):
                contribution[row, 3] = mass * wp.float64(inertial_delta[row]) + wp.float64(force_acc[row])
                for column in range(3):
                    value = wp.float64(h_ll_acc[row, column])
                    if row == column:
                        value += mass
                    contribution[row, column] = value
            wp.atomic_add(translation_system, island, contribution)
            if capacity_acc > 0.0:
                supported = _TranslationSupport()
                for axis in range(3):
                    supported[axis] = wp.float64(friction_acc[axis])
                    supported[4 + axis] = wp.float64(support_acc[axis])
                supported[3] = wp.float64(capacity_acc)
                wp.atomic_add(translation_support, island, supported)
            return

        else:
            wp.atomic_add(body_forces, body, force_acc)
            if wp.static(forces_only):
                # The line search reads force/torque, never these trial Hessians.
                wp.atomic_add(body_torques, body, torque_acc)
                return
            else:
                wp.atomic_add(body_hessian_ll, body, h_ll_acc)
                if wp.static(not translation_only):
                    wp.atomic_add(body_torques, body, torque_acc)
                    wp.atomic_add(body_hessian_al, body, h_al_acc)
                    wp.atomic_add(body_hessian_aa, body, h_aa_acc)
                dynamic_hessian = wp.spatial_matrix(0.0)
                for row in range(3):
                    for column in range(3):
                        dynamic_hessian[row, column] = dynamic_h_ll_acc[row, column]
                        if wp.static(not translation_only):
                            dynamic_hessian[row, column + 3] = dynamic_h_al_acc[column, row]
                            dynamic_hessian[row + 3, column] = dynamic_h_al_acc[row, column]
                            dynamic_hessian[row + 3, column + 3] = dynamic_h_aa_acc[row, column]
                wp.atomic_add(body_dynamic_contact_hessian, body_slot, dynamic_hessian)

    return wp.kernel(accumulate_structural_body_body_contacts)


accumulate_structural_body_body_contacts = _make_accumulate_structural_contacts(False, False)
_accumulate_structural_contact_forces = _make_accumulate_structural_contacts(False, True)
_accumulate_structural_translation_contacts = _make_accumulate_structural_contacts(True, False)


@wp.kernel
def classify_global_contact_islands(
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[int],
    rigid_contact_shape1: wp.array[int],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_margin0: wp.array[float],
    rigid_contact_margin1: wp.array[float],
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_inv_mass: wp.array[float],
    contact_lambda: wp.array[wp.vec3],
    body_slot_by_id: wp.array[wp.int32],
    graph_body_island: wp.array[wp.int32],
    body_contact_buffer_pre_alloc: int,
    body_contact_counts: wp.array[wp.int32],
    island_contact_state: wp.array[wp.int32],
):
    """Classify dynamic contact and truncated incidence by structural island."""
    contact = wp.tid()
    if contact >= rigid_contact_count[0]:
        return
    shape0 = rigid_contact_shape0[contact]
    shape1 = rigid_contact_shape1[contact]
    body0 = shape_body[shape0] if shape0 >= 0 else -1
    body1 = shape_body[shape1] if shape1 >= 0 else -1
    dynamic0 = body0 >= 0 and body_inv_mass[body0] > 0.0
    dynamic1 = body1 >= 0 and body_inv_mass[body1] > 0.0
    if not dynamic0 and not dynamic1:
        return

    slot0 = body_slot_by_id[body0] if dynamic0 else -1
    slot1 = body_slot_by_id[body1] if dynamic1 else -1
    island0 = graph_body_island[slot0] if slot0 >= 0 else -1
    island1 = graph_body_island[slot1] if slot1 >= 0 else -1
    # -2 dominates the -1 active-pair state and suppresses the global pose
    # correction for an island whose body-centric contact objective is incomplete.
    if slot0 >= 0 and body_contact_counts[body0] > body_contact_buffer_pre_alloc:
        wp.atomic_min(island_contact_state, island0, -2)
    if slot1 >= 0 and body_contact_counts[body1] > body_contact_buffer_pre_alloc:
        wp.atomic_min(island_contact_state, island1, -2)
    normal = rigid_contact_normal[contact]
    point0 = (
        wp.transform_point(body_q[body0], rigid_contact_point0[contact])
        if body0 >= 0
        else rigid_contact_point0[contact]
    )
    point1 = (
        wp.transform_point(body_q[body1], rigid_contact_point1[contact])
        if body1 >= 0
        else rigid_contact_point1[contact]
    )
    separation = wp.dot(normal, point1 - point0) - rigid_contact_margin0[contact] - rigid_contact_margin1[contact]
    if separation > 0.0 and wp.dot(contact_lambda[contact], normal) <= 0.0:
        return
    if dynamic0 and dynamic1 and slot0 >= 0 and slot1 >= 0:
        wp.atomic_min(island_contact_state, island0, -1)
        wp.atomic_min(island_contact_state, island1, -1)


@wp.kernel
def flag_active_contact_islands(
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[int],
    rigid_contact_shape1: wp.array[int],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_margin0: wp.array[float],
    rigid_contact_margin1: wp.array[float],
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_inv_mass: wp.array[float],
    contact_normal_rho: wp.array[float],
    contact_material_ke: wp.array[float],
    contact_lambda: wp.array[wp.vec3],
    contact_C0: wp.array[wp.vec3],
    body_slot_by_id: wp.array[wp.int32],
    graph_body_island: wp.array[wp.int32],
    body_contact_buffer_pre_alloc: int,
    body_contact_counts: wp.array[wp.int32],
    active_islands: wp.array[wp.int32],
):
    """Mark structural islands that own at least one active rigid contact at body_q."""
    contact = wp.tid()
    if contact >= rigid_contact_count[0]:
        return
    shape0 = rigid_contact_shape0[contact]
    shape1 = rigid_contact_shape1[contact]
    body0 = shape_body[shape0] if shape0 >= 0 else -1
    body1 = shape_body[shape1] if shape1 >= 0 else -1
    dynamic0 = body0 >= 0 and body_inv_mass[body0] > 0.0
    dynamic1 = body1 >= 0 and body_inv_mass[body1] > 0.0
    if not dynamic0 and not dynamic1:
        return
    slot0 = body_slot_by_id[body0] if dynamic0 else -1
    slot1 = body_slot_by_id[body1] if dynamic1 else -1
    island0 = graph_body_island[slot0] if slot0 >= 0 else -1
    island1 = graph_body_island[slot1] if slot1 >= 0 else -1
    if island0 < 0 and island1 < 0:
        return
    if (slot0 >= 0 and body_contact_counts[body0] > body_contact_buffer_pre_alloc) or (
        slot1 >= 0 and body_contact_counts[body1] > body_contact_buffer_pre_alloc
    ):
        if island0 >= 0:
            active_islands[island0] = 1
        if island1 >= 0:
            active_islands[island1] = 1
        return
    normal = rigid_contact_normal[contact]
    point0 = (
        wp.transform_point(body_q[body0], rigid_contact_point0[contact])
        if body0 >= 0
        else rigid_contact_point0[contact]
    )
    point1 = (
        wp.transform_point(body_q[body1], rigid_contact_point1[contact])
        if body1 >= 0
        else rigid_contact_point1[contact]
    )
    normal_error = -contact_surface_separation(
        point0,
        point1,
        normal,
        rigid_contact_margin0[contact],
        rigid_contact_margin1[contact],
    )
    normal_weight = contact_normal_rho[contact]
    material_k = contact_material_ke[contact]
    multiplier_normal = wp.dot(contact_lambda[contact], normal)
    C0_normal = wp.dot(normal, contact_C0[contact])
    stabilized_error = normal_error - C0_normal
    normal_primal_k, multiplier_normal_eff = _material_force_terms(
        normal_weight,
        material_k,
        multiplier_normal,
    )
    inactive = multiplier_normal <= 0.0 and (
        normal_error <= _SMALL_LENGTH_EPS or normal_primal_k * stabilized_error + multiplier_normal_eff <= 0.0
    )
    if inactive:
        return
    if island0 >= 0:
        active_islands[island0] = 1
    if island1 >= 0:
        active_islands[island1] = 1


@wp.func
def _body_objective_rhs(
    pose: wp.transform,
    inertial_pose: wp.transform,
    mass: float,
    inertia: wp.mat33,
    com_local: wp.vec3,
    contact_force: wp.vec3,
    contact_torque: wp.vec3,
    dt_inv_sq: float,
):
    rotation = wp.transform_get_rotation(pose)
    inertial_rotation = wp.transform_get_rotation(inertial_pose)
    force = (mass * dt_inv_sq) * _pose_com_diff(inertial_pose, pose, com_local) + contact_force
    rotation_delta = wp.quat_inverse(rotation) * inertial_rotation
    if rotation_delta[3] < 0.0:
        rotation_delta = wp.quat(
            -rotation_delta[0],
            -rotation_delta[1],
            -rotation_delta[2],
            -rotation_delta[3],
        )
    axis, angle = wp.quat_to_axis_angle(rotation_delta)
    torque = wp.quat_rotate(rotation, inertia * (axis * angle * dt_inv_sq)) + contact_torque
    return wp.spatial_vector(force, torque)


@wp.func
def _body_objective_rhs_displacement(
    pose: wp.transform,
    mass: float,
    inertia: wp.mat33,
    contact_force: wp.vec3,
    contact_torque: wp.vec3,
    dt_inv_sq: float,
    displacement: wp.vec3,
    rotation_displacement: wp.vec3,
):
    """:func:`_body_objective_rhs` from the displacement accumulated since the inertial target.

    The accumulated displacement keeps the digits that the difference of two stored poses
    rounds away.
    """
    rotation = wp.transform_get_rotation(pose)
    angular = wp.quat_rotate(rotation, inertia * wp.quat_rotate_inv(rotation, rotation_displacement))
    return wp.spatial_vector(contact_force - (mass * dt_inv_sq) * displacement, contact_torque - dt_inv_sq * angular)


@wp.func
def _body_surrogate_matrix(
    rotation: wp.quat,
    mass: float,
    inertia: wp.mat33,
    contact_ll: wp.mat33,
    contact_al: wp.mat33,
    contact_aa: wp.mat33,
    dynamic: wp.spatial_matrix,
    dt_inv_sq: float,
):
    """Share the unstressed body metric between structural and rigid-mode solves."""
    # Off-topology dynamic pairs require an extra self block, not doubled
    # curvature for unrelated static/kinematic support.
    h_ll = mass * dt_inv_sq * wp.identity(3, float) + contact_ll + _spatial_block(dynamic, 0, 0)
    rotation_matrix = wp.quat_to_matrix(rotation)
    h_aa = (
        dt_inv_sq * rotation_matrix * inertia * wp.transpose(rotation_matrix)
        + contact_aa
        + _spatial_block(dynamic, 1, 1)
    )
    h_al = contact_al + _spatial_block(dynamic, 1, 0)
    # Match local VBD regularization and the existing structural body solve.
    eps_l = 1.0e-9 * (wp.trace(h_ll) / 3.0 + 1.0)
    eps_a = 1.0e-9 * (wp.trace(h_aa) / 3.0 + 1.0)
    for axis in range(3):
        h_ll[axis, axis] = h_ll[axis, axis] + eps_l
        h_aa[axis, axis] = h_aa[axis, axis] + eps_a
    matrix = wp.spatial_matrix(0.0)
    matrix = _set_spatial_block(matrix, 0, 0, h_ll)
    matrix = _set_spatial_block(matrix, 0, 1, wp.transpose(h_al))
    matrix = _set_spatial_block(matrix, 1, 0, h_al)
    matrix = _set_spatial_block(matrix, 1, 1, h_aa)
    return matrix


@wp.kernel
def build_body_surrogate(
    body_ids: wp.array[wp.int32],
    dt: float,
    body_q: wp.array[wp.transform],
    body_inertia_q: wp.array[wp.transform],
    body_disp: wp.array[wp.vec3],
    body_disp_ang: wp.array[wp.vec3],
    use_displacement: int,
    body_mass: wp.array[float],
    body_inv_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    body_com: wp.array[wp.vec3],
    contact_hessian_ll: wp.array[wp.mat33],
    contact_hessian_al: wp.array[wp.mat33],
    contact_hessian_aa: wp.array[wp.mat33],
    contact_forces: wp.array[wp.vec3],
    contact_torques: wp.array[wp.vec3],
    dynamic_contact_hessian: wp.array[wp.spatial_matrix],
    has_contacts: int,
    include_stress: int,
    stress_offsets: wp.array[wp.int32],
    body_stress: wp.array[wp.mat33],
    body_path_inverse: wp.array[wp.int32],
    body_matrix_out: wp.array[wp.spatial_matrix],
    body_rhs_out: wp.array[wp.spatial_vector],
):
    """Build the contact-proximal body metric for a structural correction.

    With ``include_stress``, add the angular stress-majorizer blocks of the
    body's incident rows, stored contiguously in ascending joint order at
    ``body_stress[stress_offsets[slot]:stress_offsets[slot + 1]]``. Entries are
    added one at a time and only where nonzero, which reproduces a sequential
    per-row sum.

    Bodies of path components (``body_path_inverse``) store the inverse metric
    and free motion that path assembly consumes; tree bodies keep the metric.
    """
    slot = wp.tid()
    body = body_ids[slot]
    if body_inv_mass[body] <= 0.0:
        body_matrix_out[slot] = wp.spatial_matrix(0.0)
        body_rhs_out[slot] = wp.spatial_vector()
        return

    dt_inv_sq = 1.0 / (dt * dt)
    pose = body_q[body]
    inertial_pose = body_inertia_q[body]
    rotation = wp.transform_get_rotation(pose)
    com_local = body_com[body]

    zero3 = wp.mat33(0.0)
    matrix = _body_surrogate_matrix(
        rotation,
        body_mass[body],
        body_inertia[body],
        contact_hessian_ll[body] if has_contacts != 0 else zero3,
        contact_hessian_al[body] if has_contacts != 0 else zero3,
        contact_hessian_aa[body] if has_contacts != 0 else zero3,
        dynamic_contact_hessian[slot] if has_contacts != 0 else wp.spatial_matrix(0.0),
        dt_inv_sq,
    )
    if include_stress != 0:
        first = stress_offsets[slot]
        last = stress_offsets[slot + 1]
        if last > first:
            # Sum into the angular block only; the loop then carries 9 values, not 36.
            angular = _spatial_block(matrix, 1, 1)
            for entry in range(first, last):
                stress = body_stress[entry]
                for row in range(3):
                    for column in range(3):
                        value = stress[row, column]
                        if value != 0.0:
                            angular[row, column] = angular[row, column] + value
            matrix = _set_spatial_block(matrix, 1, 1, angular)
    zero_v = wp.vec3(0.0)
    rhs = _body_objective_rhs(
        pose,
        inertial_pose,
        body_mass[body],
        body_inertia[body],
        com_local,
        contact_forces[body] if has_contacts != 0 else zero_v,
        contact_torques[body] if has_contacts != 0 else zero_v,
        dt_inv_sq,
    )
    if use_displacement != 0:
        rhs = _body_objective_rhs_displacement(
            pose,
            body_mass[body],
            body_inertia[body],
            contact_forces[body] if has_contacts != 0 else zero_v,
            contact_torques[body] if has_contacts != 0 else zero_v,
            dt_inv_sq,
            body_disp[body],
            body_disp_ang[body],
        )
    if (body_path_inverse[slot] & 1) != 0:
        inverse = _inverse_spatial_robust(matrix)
        body_matrix_out[slot] = inverse
        body_rhs_out[slot] = inverse * rhs
    else:
        body_matrix_out[slot] = matrix
        body_rhs_out[slot] = rhs


@wp.func
def _stress_majorized_row(
    joint: int,
    joint_stress_slot: wp.array[wp.vec2i],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_slot: wp.array[int],
    jacobian_parent: wp.spatial_matrix,
    jacobian_child: wp.spatial_matrix,
    compliance: wp.spatial_matrix,
    residual: wp.spatial_vector,
    body_stress: wp.array[wp.mat33],
):
    """Apply a parented rod row's stress metric and write its uniquely owned body-stress slots.

    Graphs whose rows are all parented rods call this before storing rows; the arithmetic is
    :func:`add_joint_stress_majorizer`'s rod branch. Whitening preserves the current force in
    exact arithmetic. The local curvature bound is not a descent proof.
    """
    out_jp = jacobian_parent
    out_jc = jacobian_child
    out_compliance = compliance
    out_residual = residual
    parent, child = joint_parent[joint], joint_child[joint]
    destinations = joint_stress_slot[joint]
    # Each destination is written exactly once below, zero where the row adds no stress.
    stress_parent = wp.mat33(0.0)
    stress_child = wp.mat33(0.0)
    reaction = _inverse_spatial_robust(compliance) * residual
    anchor = wp.transform_point(body_q[child], wp.transform_get_translation(joint_X_c[joint]))
    rp = anchor - wp.transform_point(body_q[parent], body_com[parent])
    rc = anchor - wp.transform_point(body_q[child], body_com[child])
    lp, lc = wp.length(rp), wp.length(rc)
    length = lp + lc
    force_norm = wp.length(wp.spatial_top(wp.transpose(jacobian_parent) * reaction))
    if length > 0.0 and force_norm > 0.0:
        # Let d = vc-vp-Wp*rp+Wc*rc be the relative anchor increment.
        # Material-frame stress is f dot (-Wp^2*rp - 2*Wp*d + Wc^2*rc).
        # Since f dot (Wp*d) ignores d parallel to f, Young's inequality
        # needs only |f|/length * |(I-n*n^T)*d|^2, n = f/|f|, plus
        # rotational self blocks. This preserves common translation and
        # avoids fictitious axial stiffness in a loaded straight rod.
        added = force_norm / length
        metric_compliance = wp.spatial_matrix(compliance)
        metric_residual = residual + wp.spatial_vector()
        jp = wp.spatial_matrix(jacobian_parent)
        jc = wp.spatial_matrix(jacobian_child)
        material_from_world = wp.quat_to_matrix(
            wp.quat_inverse(wp.transform_get_rotation(body_q[parent] * joint_X_p[joint]))
        )
        normal = wp.normalize(wp.vec3d(wp.spatial_top(reaction)))
        metric = wp.mat33d(0.0)
        for axis in range(3):
            enabled = wp.length_sq(wp.vec3(jc[axis, 0], jc[axis, 1], jc[axis, 2])) > 0.0
            tangent = 1.0 / metric_compliance[axis, axis] if enabled else 0.0
            for column in range(3):
                metric[axis, column] = -wp.float64(added) * wp.float64(normal[axis]) * wp.float64(normal[column])
            metric[axis, axis] += wp.float64(tangent) + wp.float64(added)
        # Whiten the three linear rows: M = L*L^T, J_new = L^T*J,
        # c_new = L^-1*f, C_new = I. Thus J_new^T*c_new = J^T*f.
        # A dense float32 inverse compliance loses this force identity
        # when material and stress eigenvalues differ substantially.
        lower = wp.mat33d(0.0)
        normalized_force = wp.vec3d(0.0)
        for axis in range(3):
            for column in range(axis + 1):
                value = metric[axis, column]
                for k in range(column):
                    value -= lower[axis, k] * lower[column, k]
                lower[axis, column] = wp.sqrt(value) if column == axis else value / lower[column, column]
            value = wp.float64(reaction[axis])
            for column in range(axis):
                value -= lower[axis, column] * normalized_force[column]
            normalized_force[axis] = value / lower[axis, axis]
            metric_residual[axis] = float(normalized_force[axis])
            for column in range(3):
                metric_compliance[axis, column] = 1.0 if axis == column else 0.0
        weighted_frame = wp.mat33(wp.transpose(lower)) * material_from_world
        jp = _set_spatial_block(jp, 0, 0, -weighted_frame)
        jp = _set_spatial_block(jp, 0, 1, weighted_frame * wp.skew(rp))
        jc = _set_spatial_block(jc, 0, 0, weighted_frame)
        jc = _set_spatial_block(jc, 0, 1, -weighted_frame * wp.skew(rc))
        out_jp = jp
        out_jc = jc
        out_compliance = metric_compliance
        out_residual = metric_residual
        for side in range(2):
            body = parent if side == 0 else child
            slot = body_slot[body]
            if slot >= 0 and body_inv_mass[body] > 0.0:
                if side == 0:
                    stress_parent = (force_norm * (lp + length)) * wp.identity(3, float)
                else:
                    stress_child = (force_norm * lc) * wp.identity(3, float)
    if destinations[0] >= 0:
        body_stress[destinations[0]] = stress_parent
    if destinations[1] >= 0:
        body_stress[destinations[1]] = stress_child
    return out_jp, out_jc, out_compliance, out_residual


@wp.kernel
def add_joint_stress_majorizer(
    joint_stress_slot: wp.array[wp.vec2i],
    joint_ids: wp.array[int],
    row_order: wp.array[wp.int32],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_slot: wp.array[int],
    jacobian_parent: wp.array[wp.spatial_matrix],
    jacobian_child: wp.array[wp.spatial_matrix],
    compliance: wp.array[wp.spatial_matrix],
    residual: wp.array[wp.spatial_vector],
    body_stress: wp.array[wp.mat33],
):
    """Bound positional joint stress curvature omitted by Gauss--Newton.

    This changes only the solve metric, not forces, materials, or ALM state.
    The bound is local to the current pose, not a finite-step descent proof.

    Each row writes its parent/child angular blocks, zero where it adds none, to the slots
    ``joint_stress_slot[joint]`` of its dynamic endpoint bodies (``-1`` otherwise). Every body owns a
    contiguous range of slots in ascending joint order, which ``build_body_surrogate`` sums. The body
    metric therefore depends neither on thread scheduling nor on how components are grouped into
    buckets, and needs no atomics.

    Graphs whose rows are all parented rods apply the same rod arithmetic in
    :func:`_stress_majorized_row` during linearization and skip this pass; a test
    requires both paths to produce identical bytes.
    """
    row = row_order[wp.tid()]
    joint = joint_ids[row]
    parent, child = joint_parent[joint], joint_child[joint]
    destinations = joint_stress_slot[joint]
    # Each destination is written exactly once below, zero where the row adds no stress.
    stress_parent = wp.mat33(0.0)
    stress_child = wp.mat33(0.0)
    reaction = _inverse_spatial_robust(compliance[row]) * residual[row]
    if joint_type[joint] == JointType.ROD and parent >= 0:
        anchor = wp.transform_point(body_q[child], wp.transform_get_translation(joint_X_c[joint]))
        rp = anchor - wp.transform_point(body_q[parent], body_com[parent])
        rc = anchor - wp.transform_point(body_q[child], body_com[child])
        lp, lc = wp.length(rp), wp.length(rc)
        length = lp + lc
        force_norm = wp.length(wp.spatial_top(wp.transpose(jacobian_parent[row]) * reaction))
        if length > 0.0 and force_norm > 0.0:
            # Let d = vc-vp-Wp*rp+Wc*rc be the relative anchor increment.
            # Material-frame stress is f dot (-Wp^2*rp - 2*Wp*d + Wc^2*rc).
            # Since f dot (Wp*d) ignores d parallel to f, Young's inequality
            # needs only |f|/length * |(I-n*n^T)*d|^2, n = f/|f|, plus
            # rotational self blocks. This preserves common translation and
            # avoids fictitious axial stiffness in a loaded straight rod.
            added = force_norm / length
            metric_compliance = wp.spatial_matrix(compliance[row])
            metric_residual = residual[row] + wp.spatial_vector()
            jp = wp.spatial_matrix(jacobian_parent[row])
            jc = wp.spatial_matrix(jacobian_child[row])
            material_from_world = wp.quat_to_matrix(
                wp.quat_inverse(wp.transform_get_rotation(body_q[parent] * joint_X_p[joint]))
            )
            normal = wp.normalize(wp.vec3d(wp.spatial_top(reaction)))
            metric = wp.mat33d(0.0)
            for axis in range(3):
                enabled = wp.length_sq(wp.vec3(jc[axis, 0], jc[axis, 1], jc[axis, 2])) > 0.0
                tangent = 1.0 / metric_compliance[axis, axis] if enabled else 0.0
                for column in range(3):
                    metric[axis, column] = -wp.float64(added) * wp.float64(normal[axis]) * wp.float64(normal[column])
                metric[axis, axis] += wp.float64(tangent) + wp.float64(added)
            # Whiten the three linear rows: M = L*L^T, J_new = L^T*J,
            # c_new = L^-1*f, C_new = I. Thus J_new^T*c_new = J^T*f.
            # A dense float32 inverse compliance loses this force identity
            # when material and stress eigenvalues differ substantially.
            lower = wp.mat33d(0.0)
            normalized_force = wp.vec3d(0.0)
            for axis in range(3):
                for column in range(axis + 1):
                    value = metric[axis, column]
                    for k in range(column):
                        value -= lower[axis, k] * lower[column, k]
                    lower[axis, column] = wp.sqrt(value) if column == axis else value / lower[column, column]
                value = wp.float64(reaction[axis])
                for column in range(axis):
                    value -= lower[axis, column] * normalized_force[column]
                normalized_force[axis] = value / lower[axis, axis]
                metric_residual[axis] = float(normalized_force[axis])
                for column in range(3):
                    metric_compliance[axis, column] = 1.0 if axis == column else 0.0
            weighted_frame = wp.mat33(wp.transpose(lower)) * material_from_world
            jp = _set_spatial_block(jp, 0, 0, -weighted_frame)
            jp = _set_spatial_block(jp, 0, 1, weighted_frame * wp.skew(rp))
            jc = _set_spatial_block(jc, 0, 0, weighted_frame)
            jc = _set_spatial_block(jc, 0, 1, -weighted_frame * wp.skew(rc))
            jacobian_parent[row] = jp
            jacobian_child[row] = jc
            compliance[row] = metric_compliance
            residual[row] = metric_residual
            for side in range(2):
                body = parent if side == 0 else child
                slot = body_slot[body]
                if slot >= 0 and body_inv_mass[body] > 0.0:
                    if side == 0:
                        stress_parent = (force_norm * (lp + length)) * wp.identity(3, float)
                    else:
                        stress_child = (force_norm * lc) * wp.identity(3, float)
        if destinations[0] >= 0:
            body_stress[destinations[0]] = stress_parent
        if destinations[1] >= 0:
            body_stress[destinations[1]] = stress_child
        return

    # Ordinary world-frame anchor errors have only rotational self stress.
    for side in range(2):
        body = parent if side == 0 else child
        if body >= 0:
            slot = body_slot[body]
            if slot >= 0 and body_inv_mass[body] > 0.0:
                frame = joint_X_p[joint] if side == 0 else joint_X_c[joint]
                lever = wp.quat_rotate(
                    wp.transform_get_rotation(body_q[body]),
                    wp.transform_get_translation(frame) - body_com[body],
                )
                jacobian = jacobian_parent[row] if side == 0 else jacobian_child[row]
                force = -wp.spatial_top(wp.transpose(jacobian) * reaction)
                curvature = wp.dot(lever, force) * wp.identity(3, float) - 0.5 * (
                    wp.outer(lever, force) + wp.outer(force, lever)
                )
                eigenvectors, eigenvalues = wp.eig3(curvature)
                positive = wp.mat33(0.0)
                for axis in range(3):
                    positive[axis, axis] = wp.max(eigenvalues[axis], 0.0)
                positive = eigenvectors * positive * wp.transpose(eigenvectors)
                if side == 0:
                    stress_parent = positive
                else:
                    stress_child = positive
    if destinations[0] >= 0:
        body_stress[destinations[0]] = stress_parent
    if destinations[1] >= 0:
        body_stress[destinations[1]] = stress_child


@wp.kernel
def clear_stored_row_multipliers(
    world_mask: wp.array[wp.bool],
    reset_all: bool,
    world_count: int,
    joint_world: wp.array[wp.int32],
    joint_ids: wp.array[wp.int32],
    row_node: wp.array[wp.int32],
    solution: wp.array[wp.spatial_vector],
):
    """Zero the stored multipliers of the rows of reset worlds: their next stage builds no stress from them."""
    row = wp.tid()
    if _reset_world_selected(joint_world[joint_ids[row]], world_mask, reset_all, world_count):
        node = row_node[row]
        if node < 0:
            node = -2 - node
        solution[node] = wp.spatial_vector()


@wp.kernel
def add_clean_saddle_stress_majorizer(
    joint_stress_slot: wp.array[wp.vec2i],
    joint_ids: wp.array[wp.int32],
    row_order: wp.array[wp.int32],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_slot: wp.array[wp.int32],
    jacobian_parent: wp.array[wp.spatial_matrix],
    jacobian_child: wp.array[wp.spatial_matrix],
    row_node: wp.array[wp.int32],
    row_scale: wp.array[wp.spatial_vector],
    row_active: wp.array[wp.int32],
    prev_solution: wp.array[wp.spatial_vector],
    body_stress: wp.array[wp.mat33],
):
    """Populate the PSD rotational stress majorizer of a non-rod float32 tree from the multiplier
    mu = row_scale * prev_solution[r_node] that the bucket's last solve stored: the preceding stage
    of the step or, at the first stage, the last solve of the preceding step (zero after
    construction and after reset). The row force at the pose a stage starts from, C_0^-1 * c(x),
    is K times the free-fall gap of the step at the first stage and K times float32 pose rounding
    near rigidity; mu is bounded by subtree inertia uniformly as K -> inf. A row that is inactive
    in this stage adds no stress.
    """
    row = row_order[wp.tid()]
    joint = joint_ids[row]
    parent, child = joint_parent[joint], joint_child[joint]
    destinations = joint_stress_slot[joint]
    stress_parent = wp.mat33(0.0)
    stress_child = wp.mat33(0.0)
    if joint_type[joint] == JointType.ROD and parent >= 0:
        if destinations[0] >= 0:
            body_stress[destinations[0]] = stress_parent
        if destinations[1] >= 0:
            body_stress[destinations[1]] = stress_child
        return
    r_node = row_node[row]
    if r_node < 0:
        r_node = -2 - r_node
    reaction = wp.spatial_vector()
    if row_active[row] != 0:
        reaction = _scale_spatial_vector(row_scale[row], prev_solution[r_node])
    for side in range(2):
        body = parent if side == 0 else child
        if body >= 0:
            slot = body_slot[body]
            if slot >= 0 and body_inv_mass[body] > 0.0:
                frame = joint_X_p[joint] if side == 0 else joint_X_c[joint]
                lever = wp.quat_rotate(
                    wp.transform_get_rotation(body_q[body]),
                    wp.transform_get_translation(frame) - body_com[body],
                )
                jacobian = jacobian_parent[row] if side == 0 else jacobian_child[row]
                force = -wp.spatial_top(wp.transpose(jacobian) * reaction)
                curvature = wp.dot(lever, force) * wp.identity(3, float) - 0.5 * (
                    wp.outer(lever, force) + wp.outer(force, lever)
                )
                eigenvectors, eigenvalues = wp.eig3(curvature)
                positive = wp.mat33(0.0)
                for axis in range(3):
                    positive[axis, axis] = wp.max(eigenvalues[axis], 0.0)
                positive = eigenvectors * positive * wp.transpose(eigenvectors)
                if side == 0:
                    stress_parent = positive
                else:
                    stress_child = positive
    if destinations[0] >= 0:
        body_stress[destinations[0]] = stress_parent
    if destinations[1] >= 0:
        body_stress[destinations[1]] = stress_child


@wp.kernel
def sync_tree_saddle_to_alm_duals(
    joint_ids: wp.array[wp.int32],
    row_node: wp.array[wp.int32],
    row_scale: wp.array[wp.spatial_vector],
    row_active: wp.array[wp.int32],
    solution: wp.array[Any],
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_qd_start: wp.array[wp.int32],
    joint_dof_dim: wp.array2d[wp.int32],
    joint_constraint_start: wp.array[wp.int32],
    joint_material_k: wp.array[float],
    joint_penalty_kd: wp.array[float],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_q_rest: wp.array[wp.transform],
    body_slot_by_id: wp.array[wp.int32],
    graph_body_island: wp.array[wp.int32],
    island_step_scale: wp.array[float],
    limit_search_islands: wp.array[wp.int32],
    only_unflagged_islands: int,
    dt: float,
    joint_lambda_lin: wp.array[wp.vec3],
    joint_lambda_ang: wp.array[wp.vec3],
):
    """Synchronize the elastic KKT saddle reaction ``mu_S`` into Compliant ALM ``joint_lambda``.

    At the zero-memory saddle point ``mu_S = K c(x^+) + (d/dt) dc(x^+)`` is solved directly in
    tangent displacement space without subtracting rounded ``float32`` world poses, remaining
    uniformly bounded by subtree weight ``O(M g)`` as ``K -> inf``. Subtracting only the bounded
    viscous share ``(d/dt) dc(x^+)`` yields the exact physical Lagrange multiplier ``K c(x^+)``.
    """
    row = wp.tid()
    if row_active[row] == 0:
        return
    j = joint_ids[row]
    jt = joint_type[j]
    if jt == JointType.ROD:
        return
    child = joint_child[j]
    if child < 0:
        return
    parent = joint_parent[j]
    slot = body_slot_by_id[child]
    if slot < 0 and parent >= 0:
        slot = body_slot_by_id[parent]
    alpha = 1.0
    if slot >= 0:
        isl = graph_body_island[slot]
        if isl >= 0:
            if only_unflagged_islands != 0 and limit_search_islands[isl] != 0:
                return
            alpha = wp.clamp(island_step_scale[isl], 0.0, 1.0)
    if alpha <= 0.0:
        return

    r_node = row_node[row]
    if r_node < 0:
        r_node = -2 - r_node
    reaction = _scale_spatial_vector(row_scale[row], solution[r_node])
    f_top = wp.spatial_top(reaction)
    f_bot = wp.spatial_bottom(reaction)
    c_start = joint_constraint_start[j]

    X_wp = body_q[parent] * joint_X_p[j] if parent >= 0 else joint_X_p[j]
    X_wc = body_q[child] * joint_X_c[j]
    q_wp = wp.transform_get_rotation(X_wp)
    lin_count = 0
    ang_count = 0
    qd_start = joint_qd_start[j]
    if jt == JointType.PRISMATIC:
        lin_count = 1
    elif jt == JointType.REVOLUTE:
        ang_count = 1
    elif jt == JointType.D6:
        lin_count = joint_dof_dim[j, 0]
        ang_count = joint_dof_dim[j, 1]
    p_lin, p_ang = build_joint_projectors(jt, joint_axis, qd_start, lin_count, ang_count, q_wp)

    sk = wp.max(joint_material_k[c_start], 0.0)
    sd = wp.max(joint_penalty_kd[c_start], 0.0) / dt
    mu_lin = wp.vec3(0.0)
    if sk > 0.0:
        mu_lin = p_lin * f_top
        if sd > 0.0:
            X_wp_prev = body_q_prev[parent] * joint_X_p[j] if parent >= 0 else joint_X_p[j]
            X_wc_prev = body_q_prev[child] * joint_X_c[j]
            dx_cur = wp.transform_get_translation(X_wc) - wp.transform_get_translation(X_wp)
            dx_prev = wp.transform_get_translation(X_wc_prev) - wp.transform_get_translation(X_wp_prev)
            mu_lin = mu_lin - sd * (p_lin * (dx_cur - dx_prev))
    joint_lambda_lin[j] = (1.0 - alpha) * joint_lambda_lin[j] + alpha * mu_lin

    if jt != JointType.BALL:
        bk = wp.max(joint_material_k[c_start + 1], 0.0)
        bd = wp.max(joint_penalty_kd[c_start + 1], 0.0) / dt
        mu_ang = wp.vec3(0.0)
        if bk > 0.0:
            mu_ang = p_ang * f_bot
            if bd > 0.0:
                X_wp_prev = body_q_prev[parent] * joint_X_p[j] if parent >= 0 else joint_X_p[j]
                X_wc_prev = body_q_prev[child] * joint_X_c[j]
                X_wp_rest = body_q_rest[parent] * joint_X_p[j] if parent >= 0 else joint_X_p[j]
                X_wc_rest = body_q_rest[child] * joint_X_c[j]
                q_wc = wp.transform_get_rotation(X_wc)
                q_wp_prev = wp.transform_get_rotation(X_wp_prev)
                q_wc_prev = wp.transform_get_rotation(X_wc_prev)
                q_wp_rest = wp.transform_get_rotation(X_wp_rest)
                q_wc_rest = wp.transform_get_rotation(X_wc_rest)
                _kap, j_ang_w = compute_joint_kappa_and_jacobian(
                    joint_axis, qd_start + lin_count, ang_count, q_wp, q_wc, q_wp_rest, q_wc_rest
                )
                om_p = quat_velocity(q_wp, q_wp_prev, dt)
                om_c = quat_velocity(q_wc, q_wc_prev, dt)
                ang_rate = dt * (p_ang * compute_kappa_dot(j_ang_w, om_p, om_c))
                mu_ang = mu_ang - bd * ang_rate
        joint_lambda_ang[j] = (1.0 - alpha) * joint_lambda_ang[j] + alpha * mu_ang


@wp.func
def _spatial_diagonal(matrix: wp.spatial_matrix):
    return wp.spatial_vector(matrix[0, 0], matrix[1, 1], matrix[2, 2], matrix[3, 3], matrix[4, 4], matrix[5, 5])


@wp.func
def _inactive_limit_majorizer(
    has_limit: bool,
    limit_tangent: float,
    limit_lambda: float,
    step_stiffness: float,
    support: float,
) -> float:
    if has_limit and limit_tangent == 0.0 and limit_lambda != 0.0:
        _, majorizer, _ = _compliant_alm_coefficients(step_stiffness, support)
        return majorizer
    return limit_tangent


def _make_linearize_joint_path_rows(fuse_rod_stress: bool):
    """Build the general or the all-parented-rod linearization kernel.

    The variants share one source; ``wp.static`` removes the fused stress code from the
    general kernel and the unreachable non-rod code from the all-rod kernel, so neither
    pays registers for the other's work. ``limit_flag`` may be ``None``; when given, it is
    raised by a full linearization that finds a limit row loaded or within reach.
    """

    def linearize_joint_path_rows(
        joint_ids: wp.array[wp.int32],
        row_order: wp.array[wp.int32],
        joint_type: wp.array[int],
        joint_enabled: wp.array[bool],
        joint_parent: wp.array[int],
        joint_child: wp.array[int],
        joint_X_p: wp.array[wp.transform],
        joint_X_c: wp.array[wp.transform],
        joint_axis: wp.array[wp.vec3],
        joint_rod_rest_kb_local: wp.array[wp.vec3],
        joint_rod_rest_twist: wp.array[float],
        joint_qd_start: wp.array[int],
        joint_target_q_start: wp.array[int],
        joint_dof_dim: wp.array2d[int],
        joint_constraint_start: wp.array[int],
        joint_material_k: wp.array[float],
        joint_rho: wp.array[float],
        joint_penalty_kd: wp.array[float],
        joint_lambda_lin: wp.array[wp.vec3],
        joint_lambda_ang: wp.array[wp.vec3],
        joint_C0_lin: wp.array[wp.vec3],
        joint_C0_ang: wp.array[wp.vec3],
        joint_sigma_start: wp.array[wp.vec3],
        joint_C_fric: wp.array[wp.vec3],
        joint_target_ke: wp.array[float],
        joint_target_kd: wp.array[float],
        joint_target_q: wp.array[float],
        joint_target_qd: wp.array[float],
        joint_limit_lower: wp.array[float],
        joint_limit_upper: wp.array[float],
        joint_limit_ke: wp.array[float],
        joint_limit_kd: wp.array[float],
        joint_rest_angle: wp.array[float],
        joint_angle_prev: wp.array[float],
        joint_drive_limit_support: wp.array[float],
        joint_drive_lambda: wp.array[float],
        joint_limit_lambda: wp.array[float],
        stab_alpha: float,
        body_q: wp.array[wp.transform],
        body_q_prev: wp.array[wp.transform],
        body_q_rest: wp.array[wp.transform],
        body_com: wp.array[wp.vec3],
        dt: float,
        slope_only: bool,
        body_slot: wp.array[int],
        body_island: wp.array[int],
        correction: wp.array[wp.spatial_vector],
        merit: wp.array[wp.float64],
        jacobian_parent: wp.array[wp.spatial_matrix],
        jacobian_child: wp.array[wp.spatial_matrix],
        compliance: wp.array[wp.spatial_matrix],
        residual: wp.array[wp.spatial_vector],
        row_active: wp.array[wp.int32],
        build_majorizer: bool,
        joint_stress_slot: wp.array[wp.vec2i],
        body_inv_mass: wp.array[float],
        body_stress: wp.array[wp.mat33],
        store_slope: bool,
        slope_offset: int,
        slope_jacobian_parent: wp.array[wp.spatial_matrix],
        slope_jacobian_child: wp.array[wp.spatial_matrix],
        slope_compliance: wp.array[wp.spatial_vector],
        slope_residual: wp.array[wp.spatial_vector],
        slope_objective: wp.array[wp.spatial_vector],
        slope_mode: wp.array[wp.int32],
        limit_flag: wp.array[int],
        limit_island: wp.array[int],
    ):
        """Linearize a joint, or store only its slope inputs on a line-search trial.

        With ``store_slope`` or ``slope_only``, the pass stores the unmajorized slope inputs of each row at
        ``slope_offset + row`` (mode 0: no slope term, 1: diagonal rows, 2: objective force);
        :func:`accumulate_stored_joint_slope` then adds them in original row order. A slope-only
        pass stores nothing else.

        In the all-rod variant, ``build_majorizer`` applies :func:`_stress_majorized_row`
        to each parented rod row before it is stored, and the caller omits
        :func:`add_joint_stress_majorizer`. The general variant contains no fused code;
        its graphs use that separate pass.
        """
        row = row_order[wp.tid()]
        joint = joint_ids[row]
        zero_matrix = wp.spatial_matrix(0.0)
        jt = joint_type[joint]
        supported = (
            jt == JointType.ROD
            or jt == JointType.BALL
            or jt == JointType.FIXED
            or jt == JointType.REVOLUTE
            or jt == JointType.PRISMATIC
            or jt == JointType.D6
        )
        if not supported or not joint_enabled[joint]:
            if store_slope or slope_only:
                slope_mode[slope_offset + row] = 0
            if slope_only:
                return
            if wp.static(fuse_rod_stress):
                if build_majorizer:
                    # A disabled row must not retain the preceding step's stress.
                    destinations = joint_stress_slot[joint]
                    for side in range(2):
                        if destinations[side] >= 0:
                            body_stress[destinations[side]] = wp.mat33(0.0)
            jacobian_parent[row] = zero_matrix
            jacobian_child[row] = zero_matrix
            compliance[row] = wp.identity(6, float)
            residual[row] = wp.spatial_vector()
            row_active[row] = 0
            return

        parent = joint_parent[joint]
        child = joint_child[joint]
        parent_pose = wp.transform_identity()
        parent_pose_prev = parent_pose
        parent_pose_rest = parent_pose
        if parent >= 0:
            parent_pose = body_q[parent]
            parent_pose_prev = body_q_prev[parent]
            parent_pose_rest = body_q_rest[parent]
        child_pose = body_q[child]
        child_pose_prev = body_q_prev[child]
        child_pose_rest = body_q_rest[child]

        X_wp = parent_pose * joint_X_p[joint]
        X_wc = child_pose * joint_X_c[joint]
        X_wp_prev = parent_pose_prev * joint_X_p[joint]
        X_wc_prev = child_pose_prev * joint_X_c[joint]
        X_wp_rest = parent_pose_rest * joint_X_p[joint]
        X_wc_rest = child_pose_rest * joint_X_c[joint]

        x_p = wp.transform_get_translation(X_wp)
        x_c = wp.transform_get_translation(X_wc)
        x_p_prev = wp.transform_get_translation(X_wp_prev)
        x_c_prev = wp.transform_get_translation(X_wc_prev)
        q_wp = wp.transform_get_rotation(X_wp)
        q_wc = wp.transform_get_rotation(X_wc)
        q_wp_prev = wp.transform_get_rotation(X_wp_prev)
        q_wc_prev = wp.transform_get_rotation(X_wc_prev)
        q_wp_rest = wp.transform_get_rotation(X_wp_rest)
        q_wc_rest = wp.transform_get_rotation(X_wc_rest)

        c_start = joint_constraint_start[joint]
        if jt == JointType.ROD:
            # Rod rows live in their material frame:
            # [shear_x, shear_y, stretch_z, bend_x, bend_y, twist_z].
            stretch_idx = c_start
            shear_idx = c_start + 1
            bend_idx = c_start + 2
            twist_idx = c_start + 3

            q_wp_inv = wp.quat_inverse(q_wp)
            material_from_world = wp.quat_to_matrix(q_wp_inv)
            u = wp.quat_rotate(q_wp_inv, x_c - x_p)
            u_prev = wp.quat_rotate(wp.quat_inverse(q_wp_prev), x_c_prev - x_p_prev)
            C0_lin = joint_C0_lin[joint]
            lambda_lin = joint_lambda_lin[joint]

            shear_tangent, shear_defect_x = _normalized_cable_row(
                u[0] - stab_alpha * C0_lin[0],
                u_prev[0] - stab_alpha * C0_lin[0],
                joint_material_k[shear_idx],
                joint_rho[shear_idx],
                joint_penalty_kd[shear_idx],
                lambda_lin[0],
                0.0,
                0.0,
                dt,
            )
            shear_tangent_y, shear_defect_y = _normalized_cable_row(
                u[1] - stab_alpha * C0_lin[1],
                u_prev[1] - stab_alpha * C0_lin[1],
                joint_material_k[shear_idx],
                joint_rho[shear_idx],
                joint_penalty_kd[shear_idx],
                lambda_lin[1],
                0.0,
                0.0,
                dt,
            )
            stretch_tangent, stretch_defect = _normalized_cable_row(
                u[2] - stab_alpha * C0_lin[2],
                u_prev[2] - stab_alpha * C0_lin[2],
                joint_material_k[stretch_idx],
                joint_rho[stretch_idx],
                joint_penalty_kd[stretch_idx],
                lambda_lin[2],
                0.0,
                0.0,
                dt,
            )
            linear_mask = wp.mat33(0.0)
            linear_compliance = wp.mat33(0.0)
            if shear_tangent > 0.0:
                linear_mask[0, 0] = 1.0
                linear_compliance[0, 0] = 1.0 / shear_tangent
            else:
                linear_compliance[0, 0] = 1.0
            if shear_tangent_y > 0.0:
                linear_mask[1, 1] = 1.0
                linear_compliance[1, 1] = 1.0 / shear_tangent_y
            else:
                linear_compliance[1, 1] = 1.0
            if stretch_tangent > 0.0:
                linear_mask[2, 2] = 1.0
                linear_compliance[2, 2] = 1.0 / stretch_tangent
            else:
                linear_compliance[2, 2] = 1.0

            jp = wp.spatial_matrix(0.0)
            jc = wp.spatial_matrix(0.0)
            linear_jacobian = linear_mask * material_from_world
            parent_com_world = wp.vec3(0.0)
            if parent >= 0:
                parent_com_world = wp.transform_point(parent_pose, body_com[parent])
            child_com_world = wp.transform_point(child_pose, body_com[child])
            # Parent material rotation makes its effective lever arm end at the
            # child anchor, matching the local cable Gauss-Newton Hessian.
            jp = _set_spatial_block(jp, 0, 0, -linear_jacobian)
            jp = _set_spatial_block(jp, 0, 1, linear_jacobian * wp.skew(x_c - parent_com_world))
            jc = _set_spatial_block(jc, 0, 0, linear_jacobian)
            jc = _set_spatial_block(jc, 0, 1, -(linear_jacobian * wp.skew(x_c - child_com_world)))

            measure = measure_rod_bend_twist_z(q_wp, q_wc)
            kappa = assemble_geometric_rod_kappa_z(
                q_wp,
                measure.kb_world,
                measure.twist,
                joint_rod_rest_kb_local[joint],
                joint_rod_rest_twist[joint],
            )
            previous_measure = measure_rod_bend_twist_z(q_wp_prev, q_wc_prev)
            kappa_prev = assemble_geometric_rod_kappa_z(
                q_wp_prev,
                previous_measure.kb_world,
                previous_measure.twist,
                joint_rod_rest_kb_local[joint],
                joint_rod_rest_twist[joint],
            )
            delta_kappa = rod_bend_twist_delta(kappa, kappa_prev)
            C0_ang = joint_C0_ang[joint]
            lambda_ang = joint_lambda_ang[joint]
            sigma = joint_sigma_start[joint]
            friction_tangent = joint_C_fric[joint]

            bend_error_x = kappa[0] - stab_alpha * C0_ang[0]
            bend_error_y = kappa[1] - stab_alpha * C0_ang[1]
            twist_error = kappa[2] - stab_alpha * C0_ang[2]
            bend_tangent, bend_defect_x = _normalized_cable_row(
                bend_error_x,
                bend_error_x - delta_kappa[0],
                joint_material_k[bend_idx],
                joint_rho[bend_idx],
                joint_penalty_kd[bend_idx],
                lambda_ang[0],
                sigma[0],
                friction_tangent[0],
                dt,
            )
            bend_tangent_y, bend_defect_y = _normalized_cable_row(
                bend_error_y,
                bend_error_y - delta_kappa[1],
                joint_material_k[bend_idx],
                joint_rho[bend_idx],
                joint_penalty_kd[bend_idx],
                lambda_ang[1],
                sigma[1],
                friction_tangent[1],
                dt,
            )
            twist_tangent, twist_defect = _normalized_cable_row(
                twist_error,
                twist_error - delta_kappa[2],
                joint_material_k[twist_idx],
                joint_rho[twist_idx],
                joint_penalty_kd[twist_idx],
                lambda_ang[2],
                sigma[2],
                friction_tangent[2],
                dt,
            )
            angular_mask = wp.mat33(0.0)
            angular_compliance = wp.mat33(0.0)
            if bend_tangent > 0.0:
                angular_mask[0, 0] = 1.0
                angular_compliance[0, 0] = 1.0 / bend_tangent
            else:
                angular_compliance[0, 0] = 1.0
            if bend_tangent_y > 0.0:
                angular_mask[1, 1] = 1.0
                angular_compliance[1, 1] = 1.0 / bend_tangent_y
            else:
                angular_compliance[1, 1] = 1.0
            if twist_tangent > 0.0:
                angular_mask[2, 2] = 1.0
                angular_compliance[2, 2] = 1.0 / twist_tangent
            else:
                angular_compliance[2, 2] = 1.0

            parent_angular_jacobian = _rod_bend_twist_jacobian_z_global(q_wp, measure, True)
            child_angular_jacobian = _rod_bend_twist_jacobian_z_global(q_wp, measure, False)
            jp = _set_spatial_block(jp, 1, 1, angular_mask * parent_angular_jacobian)
            jc = _set_spatial_block(jc, 1, 1, angular_mask * child_angular_jacobian)

            material_compliance = wp.spatial_matrix(0.0)
            material_compliance = _set_spatial_block(material_compliance, 0, 0, linear_compliance)
            material_compliance = _set_spatial_block(material_compliance, 1, 1, angular_compliance)
            defect = wp.spatial_vector(
                wp.vec3(shear_defect_x, shear_defect_y, stretch_defect),
                wp.vec3(bend_defect_x, bend_defect_y, twist_defect),
            )
            if store_slope or slope_only:
                slope_row = slope_offset + row
                slope_jacobian_parent[slope_row] = jp
                slope_jacobian_child[slope_row] = jc
                slope_compliance[slope_row] = _spatial_diagonal(material_compliance)
                slope_residual[slope_row] = defect
                slope_mode[slope_row] = 1
            if slope_only:
                return
            if wp.static(fuse_rod_stress):
                if build_majorizer and parent >= 0:
                    jp, jc, material_compliance, defect = _stress_majorized_row(
                        joint,
                        joint_stress_slot,
                        joint_parent,
                        joint_child,
                        joint_X_p,
                        joint_X_c,
                        body_q,
                        body_com,
                        body_inv_mass,
                        body_slot,
                        jp,
                        jc,
                        material_compliance,
                        defect,
                        body_stress,
                    )
            jacobian_parent[row] = jp
            jacobian_child[row] = jc
            compliance[row] = material_compliance
            residual[row] = defect
            row_active[row] = (
                1
                if shear_tangent > 0.0
                or shear_tangent_y > 0.0
                or stretch_tangent > 0.0
                or bend_tangent > 0.0
                or bend_tangent_y > 0.0
                or twist_tangent > 0.0
                else 0
            )
            return

        if wp.static(fuse_rod_stress):
            # Fused graphs contain only parented rods, which returned above.
            return

        lin_count = 0
        ang_count = 0
        qd_start = joint_qd_start[joint]
        if jt == JointType.PRISMATIC:
            lin_count = 1
        elif jt == JointType.REVOLUTE:
            ang_count = 1
        elif jt == JointType.D6:
            lin_count = joint_dof_dim[joint, 0]
            ang_count = joint_dof_dim[joint, 1]
        kappa, angular_jacobian_world = compute_joint_kappa_and_jacobian(
            joint_axis, qd_start + lin_count, ang_count, q_wp, q_wc, q_wp_rest, q_wc_rest
        )
        p_linear, p_angular = build_joint_projectors(jt, joint_axis, qd_start, lin_count, ang_count, q_wp)
        if jt == JointType.BALL:
            p_angular = wp.mat33(0.0)

        stretch_k = wp.max(joint_material_k[c_start], 0.0)
        # Articulated constraints solve the exact zero-memory saddle compliance (s_f = 0,
        # C_S = (K + D/h)^-1): zero stale-multiplier mixing, unconditional BDF1/BDF2 A-stability,
        # and no auxiliary scaled-rho buffer.
        stretch_s = 0.0
        stretch_primal_k = stretch_k
        stretch_d = wp.max(joint_penalty_kd[c_start], 0.0) / dt
        bend_k = 0.0
        bend_s = 0.0
        bend_primal_k = 0.0
        bend_d = 0.0
        if jt != JointType.BALL:
            bend_k = wp.max(joint_material_k[c_start + 1], 0.0)
            bend_primal_k = bend_k
            bend_d = wp.max(joint_penalty_kd[c_start + 1], 0.0) / dt
        stretch_scale = stretch_primal_k + stretch_d
        bend_offset = bend_s * (p_angular * joint_lambda_ang[joint])
        stretch_enabled = stretch_scale > 0.0 and wp.trace(p_linear) > 1.0e-6
        bend_enabled = wp.trace(p_angular) > 1.0e-6 and bend_primal_k + bend_d > 0.0
        jp = wp.spatial_matrix(0.0)
        jc = wp.spatial_matrix(0.0)
        material_compliance = wp.spatial_matrix(0.0)
        c_linear = wp.vec3(0.0)
        c_angular = wp.vec3(0.0)

        if stretch_enabled:
            r_parent = (
                wp.quat_rotate(
                    wp.transform_get_rotation(parent_pose),
                    wp.transform_get_translation(joint_X_p[joint]) - body_com[parent],
                )
                if parent >= 0
                else x_p
            )
            r_child = wp.quat_rotate(
                wp.transform_get_rotation(child_pose),
                wp.transform_get_translation(joint_X_c[joint]) - body_com[child],
            )

            jp = _set_spatial_block(jp, 0, 0, -p_linear)
            jp = _set_spatial_block(jp, 0, 1, p_linear * wp.skew(r_parent))
            jc = _set_spatial_block(jc, 0, 0, p_linear)
            jc = _set_spatial_block(jc, 0, 1, -(p_linear * wp.skew(r_child)))
            stretch_compliance = 1.0 / stretch_scale
            linear_compliance = stretch_compliance * p_linear + (wp.identity(3, float) - p_linear)
            material_compliance = _set_spatial_block(material_compliance, 0, 0, linear_compliance)
            linear_error = p_linear * (x_c - x_p - stab_alpha * joint_C0_lin[joint])
            linear_previous = p_linear * (x_c_prev - x_p_prev)
            # Form the normalized compliant-ALM defect by bounded ratios. This is
            # exactly the force/tangent row used by local VBD, including damping
            # and the current multiplier.
            c_linear = (
                (stretch_primal_k / stretch_scale) * linear_error
                + (stretch_s / stretch_scale) * (p_linear * joint_lambda_lin[joint])
                + (stretch_d / stretch_scale) * (p_linear * (x_c - x_p) - linear_previous)
            )
        else:
            for axis in range(3):
                material_compliance[axis, axis] = 1.0

        if bend_enabled:
            angular_jacobian = p_angular * wp.transpose(angular_jacobian_world)
            jp = _set_spatial_block(jp, 1, 1, -angular_jacobian)
            jc = _set_spatial_block(jc, 1, 1, angular_jacobian)
            angular_error = p_angular * (kappa - stab_alpha * joint_C0_ang[joint])
            # Match local VBD's current-Jacobian angular-velocity damping.
            omega_parent = quat_velocity(q_wp, q_wp_prev, dt)
            omega_child = quat_velocity(q_wc, q_wc_prev, dt)
            angular_rate = dt * (p_angular * compute_kappa_dot(angular_jacobian_world, omega_parent, omega_child))
            bend_scale = bend_primal_k + bend_d
            angular_compliance = (1.0 / bend_scale) * p_angular + (wp.identity(3, float) - p_angular)
            material_compliance = _set_spatial_block(material_compliance, 1, 1, angular_compliance)
            c_angular = (
                (bend_primal_k / bend_scale) * angular_error
                + (bend_d / bend_scale) * angular_rate
                + bend_offset / bend_scale
            )
        else:
            for axis in range(3):
                material_compliance[axis + 3, axis + 3] = 1.0

        # Recover the projected force before adding free-axis drive/limit rows.
        # These separate coefficients avoid inverting a rotated FP32 compliance
        # whose small eigenvalues can be lost at large stiffness ratios.
        objective_linear = stretch_scale * c_linear
        objective_angular = (bend_primal_k + bend_d) * c_angular

        # Free-axis compliance is filled directly below.  Keeping the temporary
        # unit eigenvalue and later applying ``I + (1/tangent - 1) aa^T`` loses
        # ``1/tangent`` to float32 cancellation for near-rigid drives.  Start from
        # only the constrained projector so every active free-axis eigenvalue is
        # written as a positive quantity without subtraction.
        if lin_count > 0:
            constrained_linear_compliance = p_linear
            if stretch_enabled:
                constrained_linear_compliance = (1.0 / stretch_scale) * p_linear
            material_compliance = _set_spatial_block(material_compliance, 0, 0, constrained_linear_compliance)
        if ang_count > 0:
            constrained_angular_compliance = p_angular
            if bend_enabled:
                constrained_angular_compliance = (1.0 / (bend_primal_k + bend_d)) * p_angular
            material_compliance = _set_spatial_block(material_compliance, 1, 1, constrained_angular_compliance)

        # The six-dimensional joint row already has one component for every free
        # joint axis.  Populate those otherwise-neutral components with finite
        # drive or active-limit rows.  This supplies their cross-body tangent to
        # the same KKT solve without adding graph rows or a second solver path.
        target_q_base = joint_target_q_start[joint]
        r_parent = (
            wp.quat_rotate(
                wp.transform_get_rotation(parent_pose),
                wp.transform_get_translation(joint_X_p[joint]) - body_com[parent],
            )
            if parent >= 0
            else x_p
        )
        r_child = wp.quat_rotate(
            wp.transform_get_rotation(child_pose),
            wp.transform_get_translation(joint_X_c[joint]) - body_com[child],
        )
        relative_linear = x_c - x_p
        relative_linear_previous = x_c_prev - x_p_prev
        drive_limit_enabled = False

        for free_linear_axis in range(3):
            if free_linear_axis < lin_count:
                dof = qd_start + free_linear_axis
                target_index = target_q_base + free_linear_axis
                axis_world = wp.normalize(wp.quat_rotate(q_wp, joint_axis[dof]))
                coordinate = wp.dot(relative_linear, axis_world)
                coordinate_previous = wp.dot(relative_linear_previous, axis_world)
                target_position = joint_target_q[target_index]
                target_velocity = joint_target_qd[dof]
                drive_k = joint_target_ke[dof]
                drive_d = joint_target_kd[dof]
                lower = joint_limit_lower[dof]
                upper = joint_limit_upper[dof]
                has_drive = drive_k > 0.0 or drive_d > 0.0
                has_limit = joint_limit_ke[dof] > 0.0 and (lower > -MAXVAL or upper < MAXVAL)
                mode, drive_error = _resolve_active_drive_row(
                    coordinate,
                    target_position,
                    lower,
                    upper,
                    has_drive,
                    has_limit,
                )
                inv_dt = 1.0 / dt
                rate = (coordinate - coordinate_previous) * inv_dt
                force, tangent = _evaluate_drive_axis(
                    drive_error,
                    rate,
                    target_velocity,
                    mode,
                    drive_k,
                    drive_d,
                    joint_drive_limit_support[dof],
                    joint_drive_lambda[dof],
                    inv_dt,
                )
                lin_lim_step_k = joint_limit_ke[dof] + joint_limit_kd[dof] * inv_dt
                lin_lim_sup = _global_limit_support(joint_drive_limit_support[dof], lin_lim_step_k, True)
                limit_force, limit_tangent = _evaluate_limit_axis(
                    coordinate,
                    lower,
                    upper,
                    has_limit,
                    joint_limit_ke[dof],
                    joint_limit_kd[dof],
                    rate,
                    lin_lim_sup,
                    joint_limit_lambda[dof],
                    inv_dt,
                )
                limit_tangent = _inactive_limit_majorizer(
                    has_limit,
                    limit_tangent,
                    joint_limit_lambda[dof],
                    lin_lim_step_k,
                    lin_lim_sup,
                )
                if limit_flag:
                    if (
                        not slope_only
                        and has_limit
                        and (
                            limit_force != 0.0
                            or joint_limit_lambda[dof] != 0.0
                            or wp.min(coordinate - lower, upper - coordinate) <= 2.0 * wp.abs(rate) * dt
                        )
                    ):
                        limit_slot = body_slot[child]
                        if parent >= 0:
                            limit_slot = wp.max(limit_slot, body_slot[parent])
                        if limit_slot >= 0:
                            wp.atomic_max(limit_flag, 0, 1)
                            wp.atomic_max(limit_island, body_island[limit_slot], 1)
                force = force + limit_force
                tangent = tangent + limit_tangent
                normalized_defect = force / tangent if tangent > 0.0 else 0.0
                if tangent > 0.0:
                    drive_limit_enabled = True
                    axis_projector = wp.outer(axis_world, axis_world)
                    jp = _set_spatial_block(jp, 0, 0, _spatial_block(jp, 0, 0) - axis_projector)
                    jp = _set_spatial_block(
                        jp,
                        0,
                        1,
                        _spatial_block(jp, 0, 1) + axis_projector * wp.skew(r_parent),
                    )
                    jc = _set_spatial_block(jc, 0, 0, _spatial_block(jc, 0, 0) + axis_projector)
                    jc = _set_spatial_block(
                        jc,
                        0,
                        1,
                        _spatial_block(jc, 0, 1) - axis_projector * wp.skew(r_child),
                    )
                    linear_compliance = _spatial_block(material_compliance, 0, 0)
                    linear_compliance = linear_compliance + (1.0 / tangent) * axis_projector
                    material_compliance = _set_spatial_block(material_compliance, 0, 0, linear_compliance)
                    c_linear = c_linear + normalized_defect * axis_world
                    objective_linear = objective_linear + force * axis_world
                else:
                    linear_compliance = _spatial_block(material_compliance, 0, 0)
                    material_compliance = _set_spatial_block(
                        material_compliance,
                        0,
                        0,
                        linear_compliance + wp.outer(axis_world, axis_world),
                    )

        if ang_count > 0:
            omega_parent = wp.vec3(0.0)
            if parent >= 0:
                omega_parent = quat_velocity(q_wp, q_wp_prev, dt)
            omega_child = quat_velocity(q_wc, q_wc_prev, dt)
            kappa_rate = compute_kappa_dot(angular_jacobian_world, omega_parent, omega_child)
            for free_angular_axis in range(3):
                if free_angular_axis < ang_count:
                    dof = qd_start + lin_count + free_angular_axis
                    target_index = target_q_base + lin_count + free_angular_axis
                    axis_local = wp.normalize(joint_axis[dof])
                    coordinate = wp.dot(kappa, axis_local) + joint_rest_angle[dof]
                    if jt == JointType.REVOLUTE or (jt == JointType.D6 and ang_count == 1):
                        coordinate = _unwrap_hinge_angle(coordinate, joint_angle_prev[dof])
                    target_position = joint_target_q[target_index]
                    target_velocity = joint_target_qd[dof]
                    drive_k = joint_target_ke[dof]
                    drive_d = joint_target_kd[dof]
                    lower = joint_limit_lower[dof]
                    upper = joint_limit_upper[dof]
                    has_drive = drive_k > 0.0 or drive_d > 0.0
                    has_limit = joint_limit_ke[dof] > 0.0 and (lower > -MAXVAL or upper < MAXVAL)
                    mode, drive_error = _resolve_active_drive_row(
                        coordinate,
                        target_position,
                        lower,
                        upper,
                        has_drive,
                        has_limit,
                    )
                    inv_dt = 1.0 / dt
                    rate = wp.dot(kappa_rate, axis_local)
                    force, tangent = _evaluate_drive_axis(
                        drive_error,
                        rate,
                        target_velocity,
                        mode,
                        drive_k,
                        drive_d,
                        joint_drive_limit_support[dof],
                        joint_drive_lambda[dof],
                        inv_dt,
                    )
                    ang_lim_step_k = joint_limit_ke[dof] + joint_limit_kd[dof] * inv_dt
                    ang_lim_sup = _global_limit_support(joint_drive_limit_support[dof], ang_lim_step_k, True)
                    limit_force, limit_tangent = _evaluate_limit_axis(
                        coordinate,
                        lower,
                        upper,
                        has_limit,
                        joint_limit_ke[dof],
                        joint_limit_kd[dof],
                        rate,
                        ang_lim_sup,
                        joint_limit_lambda[dof],
                        inv_dt,
                    )
                    limit_tangent = _inactive_limit_majorizer(
                        has_limit,
                        limit_tangent,
                        joint_limit_lambda[dof],
                        ang_lim_step_k,
                        ang_lim_sup,
                    )
                    if limit_flag:
                        if (
                            not slope_only
                            and has_limit
                            and (
                                limit_force != 0.0
                                or joint_limit_lambda[dof] != 0.0
                                or wp.min(coordinate - lower, upper - coordinate) <= 2.0 * wp.abs(rate) * dt
                            )
                        ):
                            limit_slot = body_slot[child]
                            if parent >= 0:
                                limit_slot = wp.max(limit_slot, body_slot[parent])
                            if limit_slot >= 0:
                                wp.atomic_max(limit_flag, 0, 1)
                                wp.atomic_max(limit_island, body_island[limit_slot], 1)
                    force = force + limit_force
                    tangent = tangent + limit_tangent
                    normalized_defect = force / tangent if tangent > 0.0 else 0.0
                    if tangent > 0.0:
                        drive_limit_enabled = True
                        axis_projector = wp.outer(axis_local, axis_local)
                        angular_jacobian = axis_projector * wp.transpose(angular_jacobian_world)
                        jp = _set_spatial_block(jp, 1, 1, _spatial_block(jp, 1, 1) - angular_jacobian)
                        jc = _set_spatial_block(jc, 1, 1, _spatial_block(jc, 1, 1) + angular_jacobian)
                        angular_compliance = _spatial_block(material_compliance, 1, 1)
                        angular_compliance = angular_compliance + (1.0 / tangent) * axis_projector
                        material_compliance = _set_spatial_block(material_compliance, 1, 1, angular_compliance)
                        c_angular = c_angular + normalized_defect * axis_local
                        objective_angular = objective_angular + force * axis_local
                    else:
                        angular_compliance = _spatial_block(material_compliance, 1, 1)
                        material_compliance = _set_spatial_block(
                            material_compliance,
                            1,
                            1,
                            angular_compliance + wp.outer(axis_local, axis_local),
                        )

        if store_slope or slope_only:
            slope_row = slope_offset + row
            slope_jacobian_parent[slope_row] = jp
            slope_jacobian_child[slope_row] = jc
            slope_compliance[slope_row] = _spatial_diagonal(material_compliance)
            slope_residual[slope_row] = wp.spatial_vector(c_linear, c_angular)
            slope_objective[slope_row] = wp.spatial_vector(objective_linear, objective_angular)
            slope_mode[slope_row] = 1 if jt == JointType.BALL or jt == JointType.FIXED else 2
        if slope_only:
            return
        jacobian_parent[row] = jp
        jacobian_child[row] = jc
        compliance[row] = material_compliance
        residual[row] = wp.spatial_vector(c_linear, c_angular)
        row_active[row] = 1 if stretch_enabled or bend_enabled or drive_limit_enabled else 0

    return wp.kernel(linearize_joint_path_rows)


linearize_joint_path_rows = _make_linearize_joint_path_rows(False)
_linearize_fused_rod_rows = _make_linearize_joint_path_rows(True)


@wp.kernel
def accumulate_stored_joint_slope(
    joint_ids: wp.array[wp.int32],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    slope_offset: int,
    slope_jacobian_parent: wp.array[wp.spatial_matrix],
    slope_jacobian_child: wp.array[wp.spatial_matrix],
    slope_compliance: wp.array[wp.spatial_vector],
    slope_residual: wp.array[wp.spatial_vector],
    slope_objective: wp.array[wp.spatial_vector],
    slope_mode: wp.array[wp.int32],
    body_slot: wp.array[int],
    body_island: wp.array[int],
    correction: wp.array[wp.spatial_vector],
    merit: wp.array[wp.float64],
):
    """Line-search slope from the row inputs stored by the linearization at the same pose.

    One thread per row in original row order, so the FP64 atomic order does not depend on how the
    linearization grouped its rows.
    """
    row = wp.tid()
    slope_row = slope_offset + row
    mode = slope_mode[slope_row]
    if mode == 0:
        return
    joint = joint_ids[row]
    parent = joint_parent[joint]
    child = joint_child[joint]
    p, c = int(-1), int(-1)
    if parent >= 0:
        p = body_slot[parent]
    if child >= 0:
        c = body_slot[child]
    slot = wp.max(p, c)
    if slot < 0:
        return
    jp = slope_jacobian_parent[slope_row]
    jc = slope_jacobian_child[slope_row]
    compliance = slope_compliance[slope_row]
    residual = slope_residual[slope_row]
    objective_force = slope_objective[slope_row]
    linear = wp.float64(0.0)
    for axis in range(6):
        value = wp.float64(0.0)
        for column in range(6):
            if p >= 0:
                value += wp.float64(jp[axis, column]) * wp.float64(correction[p][column])
            if c >= 0:
                value += wp.float64(jc[axis, column]) * wp.float64(correction[c][column])
        if mode == 1:
            inverse_compliance = wp.float64(1.0) / wp.float64(compliance[axis])
            linear += value * wp.float64(residual[axis]) * inverse_compliance
        else:
            linear += value * wp.float64(objective_force[axis])
    wp.atomic_add(merit, body_island[slot], linear)


_SpatialRowMatrix = wp.types.matrix(shape=(1, 6), dtype=float)


@wp.func_native("return fmaf(a, b, c);")
def _path_matrix_fma(a: float, b: float, c: float) -> float:
    pass


@wp.func
def _path_matrix_vector_row(J: wp.array3d[float], row: int, r: int, free: wp.array2d[float], body: int):
    matrix = _SpatialRowMatrix(0.0)
    for c in range(6):
        matrix[0, c] = J[row, r, c]
    value = wp.spatial_vector(free[body, 0], free[body, 1], free[body, 2], free[body, 3], free[body, 4], free[body, 5])
    return (matrix * value)[0]


@wp.func
def _path_row_product(J: wp.array3d[float], row: int, r: int, M: wp.array3d[float], body: int):
    result = wp.spatial_vector()
    for c in range(6):
        value = float(0.0)
        for t in range(6):
            value = _path_matrix_fma(J[row, r, t], M[body, t, c], value)
        result[c] = value
    return result


@wp.func
def _path_row_dot(v: wp.spatial_vector, J: wp.array3d[float], row: int, c: int):
    value = float(0.0)
    for t in range(6):
        value = _path_matrix_fma(v[t], J[row, c, t], value)
    return value


@wp.kernel(module="unique", enable_backward=False)
def assemble_joint_path_system_rows(
    bounds: wp.array[wp.vec2i],
    bodies: wp.array[wp.vec2i],
    inverse: wp.array3d[float],
    free: wp.array2d[float],
    jp: wp.array3d[float],
    jc: wp.array3d[float],
    compliance: wp.array3d[float],
    residual: wp.array2d[float],
    lower: wp.array3d[float],
    diagonal: wp.array3d[float],
    upper: wp.array3d[float],
    rhs: wp.array2d[float],
):
    """Assemble one scalar row per thread without changing the block equations.

    Native matrix-vector evaluation and explicit matrix FMA match the reference
    arithmetic order. Each thread owns disjoint output entries, including when
    compliance/diagonal and residual/rhs reuse the same storage.
    """
    row, r = wp.tid()
    p, c = bodies[row][0], bodies[row][1]
    ap, ac = wp.spatial_vector(), wp.spatial_vector()
    value = residual[row, r]
    if p >= 0:
        ap = _path_row_product(jp, row, r, inverse, p)
        f = _path_matrix_vector_row(jp, row, r, free, p)
        value += f
    if c >= 0:
        ac = _path_row_product(jc, row, r, inverse, c)
        f = _path_matrix_vector_row(jc, row, r, free, c)
        value += f
    rhs[row, r] = value
    for col in range(6):
        d = compliance[row, r, col]
        if p >= 0:
            d += _path_row_dot(ap, jp, row, col)
        if c >= 0:
            d += _path_row_dot(ac, jc, row, col)
        diagonal[row, r, col] = d
        lo, up = float(0.0), float(0.0)
        if row > bounds[row][0]:
            previous = bodies[row - 1]
            shared = p
            if shared < 0 or (shared != previous[0] and shared != previous[1]):
                shared = c
            a = ap if p == shared else ac
            if previous[0] == shared:
                lo = _path_row_dot(a, jp, row - 1, col)
            else:
                lo = _path_row_dot(a, jc, row - 1, col)
        if row + 1 < bounds[row][1]:
            following = bodies[row + 1]
            shared = p
            if shared < 0 or (shared != following[0] and shared != following[1]):
                shared = c
            a = ap if p == shared else ac
            if following[0] == shared:
                up = _path_row_dot(a, jp, row + 1, col)
            else:
                up = _path_row_dot(a, jc, row + 1, col)
        lower[row, r, col] = lo
        upper[row, r, col] = up


@wp.kernel
def assemble_joint_path_system(
    row_bounds: wp.array[wp.vec2i],
    row_body: wp.array[wp.vec2i],
    body_inverse: wp.array[wp.spatial_matrix],
    body_free: wp.array[wp.spatial_vector],
    jacobian_parent: wp.array[wp.spatial_matrix],
    jacobian_child: wp.array[wp.spatial_matrix],
    compliance: wp.array[wp.spatial_matrix],
    residual: wp.array[wp.spatial_vector],
    lower: wp.array[wp.spatial_matrix],
    diagonal: wp.array[wp.spatial_matrix],
    upper: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
):
    """Assemble one block-tridiagonal constraint-space system per path.

    Each row owns one structural joint and stores its path's ``[start, end)``
    row range. Adjacent path rows are coupled through their shared body's
    inverse surrogate; non-neighbor couplings are absent by construction of an
    eligible path batch.
    """
    row = wp.tid()
    bounds = row_bounds[row]
    endpoints = row_body[row]
    parent = endpoints[0]
    child = endpoints[1]
    jp = jacobian_parent[row]
    jc = jacobian_child[row]

    diagonal_value = compliance[row]
    rhs_value = residual[row]
    if parent >= 0:
        diagonal_value = diagonal_value + jp * body_inverse[parent] * wp.transpose(jp)
        rhs_value = rhs_value + jp * body_free[parent]
    if child >= 0:
        diagonal_value = diagonal_value + jc * body_inverse[child] * wp.transpose(jc)
        rhs_value = rhs_value + jc * body_free[child]

    lower_value = wp.spatial_matrix(0.0)
    if row > bounds[0]:
        previous = row - 1
        previous_endpoints = row_body[previous]
        shared = parent
        if shared < 0 or (shared != previous_endpoints[0] and shared != previous_endpoints[1]):
            shared = child
        current_j = jp if parent == shared else jc
        previous_j = jacobian_parent[previous] if previous_endpoints[0] == shared else jacobian_child[previous]
        lower_value = current_j * body_inverse[shared] * wp.transpose(previous_j)

    upper_value = wp.spatial_matrix(0.0)
    if row + 1 < bounds[1]:
        following = row + 1
        following_endpoints = row_body[following]
        shared = parent
        if shared < 0 or (shared != following_endpoints[0] and shared != following_endpoints[1]):
            shared = child
        current_j = jp if parent == shared else jc
        following_j = jacobian_parent[following] if following_endpoints[0] == shared else jacobian_child[following]
        upper_value = current_j * body_inverse[shared] * wp.transpose(following_j)

    lower[row] = lower_value
    diagonal[row] = diagonal_value
    upper[row] = upper_value
    rhs[row] = rhs_value


@wp.kernel
def _cast_path_rows_to_fp64(
    lower32: wp.array[wp.spatial_matrix],
    diagonal32: wp.array[wp.spatial_matrix],
    upper32: wp.array[wp.spatial_matrix],
    rhs32: wp.array[wp.spatial_vector],
    lower64: wp.array[wp.spatial_matrixd],
    diagonal64: wp.array[wp.spatial_matrixd],
    upper64: wp.array[wp.spatial_matrixd],
    rhs64: wp.array[wp.spatial_vectord],
):
    row = wp.tid()
    lower64[row] = wp.spatial_matrixd(lower32[row])
    diagonal64[row] = wp.spatial_matrixd(diagonal32[row])
    upper64[row] = wp.spatial_matrixd(upper32[row])
    rhs64[row] = wp.spatial_vectord(rhs32[row])


@wp.kernel
def _cast_path_solution_to_fp32(
    rhs64: wp.array[wp.spatial_vectord],
    rhs32: wp.array[wp.spatial_vector],
):
    row = wp.tid()
    val = rhs64[row]
    out = wp.spatial_vector()
    for axis in range(6):
        out[axis] = float(val[axis])
    rhs32[row] = out


@wp.kernel
def compute_path_correction_fp64(
    body_ids: wp.array[wp.int32],
    body_slot_by_id: wp.array[wp.int32],
    body_incident_rows: wp.array[wp.vec2i],
    row_active: wp.array[wp.int32],
    jacobian_parent: wp.array[wp.spatial_matrix],
    jacobian_child: wp.array[wp.spatial_matrix],
    multiplier: wp.array[wp.spatial_vectord],
    body_inverse: wp.array[wp.spatial_matrix],
    body_free: wp.array[wp.spatial_vector],
    body_inv_mass: wp.array[float],
    body_correction: wp.array[wp.spatial_vector],
):
    index = wp.tid()
    body = body_ids[index]
    slot = body_slot_by_id[body]
    if body_inv_mass[body] <= 0.0:
        return
    reaction = wp.spatial_vectord(0.0)
    active = int(0)
    incident_rows = body_incident_rows[index]
    for incident in range(2):
        code = incident_rows[incident]
        if code >= 0:
            row = code // 2
            jacobian = jacobian_parent[row] if code % 2 == 0 else jacobian_child[row]
            reaction = reaction + wp.transpose(wp.spatial_matrixd(jacobian)) * multiplier[row]
            active = active + row_active[row]
    if active == 0:
        body_correction[slot] = wp.spatial_vector()
        return
    corr_d = wp.spatial_vectord(body_free[slot]) - wp.spatial_matrixd(body_inverse[slot]) * reaction
    out = wp.spatial_vector()
    for axis in range(6):
        out[axis] = float(corr_d[axis])
    body_correction[slot] = out


@wp.kernel
def invert_cr_eliminated_in_place(
    eliminated_rows: wp.array[wp.vec3i],
    diagonal: wp.array[Any],
):
    """Replace eliminated diagonal blocks by their inverses in dead slots.

    Each entry stores the path's first row, the local row, and the path row count.
    """
    entry = eliminated_rows[wp.tid()]
    row = entry[0] + entry[1]
    diagonal[row] = _inverse_spatial_robust(diagonal[row])


@wp.func
def _reduce_cr_row(
    base: int,
    local_row: int,
    stride: int,
    row_count: int,
    invert_limit: int,
    lower: wp.array[Any],
    diagonal: wp.array[Any],
    upper: wp.array[Any],
    rhs: wp.array[Any],
):
    """Reduce one retained CR row after its neighboring pivots are inverted."""
    row = base + local_row
    lower_value = lower.dtype(0.0)
    diagonal_value = diagonal[row]
    upper_value = upper.dtype(0.0)
    rhs_value = rhs[row]
    if local_row >= stride:
        eliminated = row - stride
        factor = lower[row] * diagonal[eliminated]
        lower_value = -(factor * lower[eliminated])
        diagonal_value = diagonal_value - factor * upper[eliminated]
        rhs_value = rhs_value - factor * rhs[eliminated]
    if local_row + stride < row_count:
        eliminated = row + stride
        factor = upper[row] * diagonal[eliminated]
        diagonal_value = diagonal_value - factor * lower[eliminated]
        rhs_value = rhs_value - factor * rhs[eliminated]
        if local_row + 2 * stride < row_count:
            upper_value = -(factor * upper[eliminated])

    next_stride = 2 * stride
    if invert_limit > 0 and (row_count + next_stride - 1) // next_stride > invert_limit:
        if ((local_row // next_stride) & 1) == 1:
            diagonal_value = _inverse_spatial_robust(diagonal_value)

    lower[row] = lower_value
    diagonal[row] = diagonal_value
    upper[row] = upper_value
    rhs[row] = rhs_value


@wp.kernel
def reduce_cr_in_place(
    stride: int,
    invert_limit: int,
    survivor_rows: wp.array[wp.vec3i],
    lower: wp.array[Any],
    diagonal: wp.array[Any],
    upper: wp.array[Any],
    rhs: wp.array[Any],
):
    """Reduce retained CR rows without allocating a second hierarchy."""
    entry = survivor_rows[wp.tid()]
    _reduce_cr_row(entry[0], entry[1], stride, entry[2], invert_limit, lower, diagonal, upper, rhs)


@wp.func
def _solve_strided_cr_coarse_impl(
    base: int,
    stride: int,
    row_count: int,
    lower: wp.array[Any],
    diagonal: wp.array[Any],
    upper: wp.array[Any],
    rhs: wp.array[Any],
):
    """Finish one small retained CR system with block Thomas."""
    active_count = (row_count + stride - 1) // stride
    for local in range(1, active_count):
        previous = base + (local - 1) * stride
        row = base + local * stride
        diagonal[previous] = _inverse_spatial_robust(diagonal[previous])
        multiplier = lower[row] * diagonal[previous]
        diagonal[row] = diagonal[row] - multiplier * upper[previous]
        rhs[row] = rhs[row] - multiplier * rhs[previous]

    last = base + (active_count - 1) * stride
    diagonal[last] = _inverse_spatial_robust(diagonal[last])
    rhs[last] = diagonal[last] * rhs[last]
    for reverse_local in range(1, active_count):
        row = base + (active_count - 1 - reverse_local) * stride
        rhs[row] = diagonal[row] * (rhs[row] - upper[row] * rhs[row + stride])


@wp.kernel
def solve_strided_cr_coarse(
    paths: wp.array[wp.vec3i],
    lower: wp.array[Any],
    diagonal: wp.array[Any],
    upper: wp.array[Any],
    rhs: wp.array[Any],
):
    """Solve each path's small retained system in place with block Thomas.

    Each path entry stores its first row, row count, and terminal stride.
    """
    path = paths[wp.tid()]
    _solve_strided_cr_coarse_impl(path[0], path[2], path[1], lower, diagonal, upper, rhs)


@wp.func
def _back_substitute_cr_row(
    base: int,
    local_row: int,
    stride: int,
    row_count: int,
    lower: wp.array[Any],
    diagonal: wp.array[Any],
    upper: wp.array[Any],
    rhs: wp.array[Any],
):
    """Recover one eliminated CR row from its retained neighbors."""
    row = base + local_row
    value = rhs[row] - lower[row] * rhs[row - stride]
    if local_row + stride < row_count:
        value = value - upper[row] * rhs[row + stride]
    rhs[row] = diagonal[row] * value


@wp.kernel
def back_substitute_cr_in_place(
    stride: int,
    eliminated_rows: wp.array[wp.vec3i],
    lower: wp.array[Any],
    diagonal: wp.array[Any],
    upper: wp.array[Any],
    rhs: wp.array[Any],
):
    """Recover eliminated rows into rhs, which now stores the solution."""
    entry = eliminated_rows[wp.tid()]
    _back_substitute_cr_row(entry[0], entry[1], stride, entry[2], lower, diagonal, upper, rhs)


# Scalar-row long-path reduction adapted from Argon's 11 November review.
@wp.func
def _mul_row6_mat66(row_vec: wp.spatial_vector, mat: wp.array3d[float], slot: int) -> wp.spatial_vector:
    out = wp.spatial_vector()
    for c in range(6):
        s = float(0.0)
        for t in range(6):
            s = _path_matrix_fma(row_vec[t], mat[slot, t, c], s)
        out[c] = s
    return out


@wp.func
def _dot_row6_vec6(row_vec: wp.spatial_vector, vec: wp.array2d[float], slot: int) -> float:
    m = _SpatialRowMatrix(0.0)
    for c in range(6):
        m[0, c] = row_vec[c]
    v = wp.spatial_vector(vec[slot, 0], vec[slot, 1], vec[slot, 2], vec[slot, 3], vec[slot, 4], vec[slot, 5])
    return (m * v)[0]


@wp.func
def _reduce_cr_scalar_row(
    base: int,
    local_row: int,
    r: int,
    stride: int,
    row_count: int,
    lower: wp.array3d[float],
    diagonal: wp.array3d[float],
    upper: wp.array3d[float],
    rhs: wp.array2d[float],
):
    row = base + local_row
    lower_r = wp.spatial_vector()
    diag_r = wp.spatial_vector(
        diagonal[row, r, 0],
        diagonal[row, r, 1],
        diagonal[row, r, 2],
        diagonal[row, r, 3],
        diagonal[row, r, 4],
        diagonal[row, r, 5],
    )
    upper_r = wp.spatial_vector()
    rhs_r = rhs[row, r]

    if local_row >= stride:
        eliminated = row - stride
        lo_in = wp.spatial_vector(
            lower[row, r, 0],
            lower[row, r, 1],
            lower[row, r, 2],
            lower[row, r, 3],
            lower[row, r, 4],
            lower[row, r, 5],
        )
        factor = _mul_row6_mat66(lo_in, diagonal, eliminated)
        prod_lo = _mul_row6_mat66(factor, lower, eliminated)
        prod_up = _mul_row6_mat66(factor, upper, eliminated)
        lower_r = -prod_lo
        diag_r = diag_r - prod_up
        rhs_r = rhs_r - _dot_row6_vec6(factor, rhs, eliminated)

    if local_row + stride < row_count:
        eliminated = row + stride
        up_in = wp.spatial_vector(
            upper[row, r, 0],
            upper[row, r, 1],
            upper[row, r, 2],
            upper[row, r, 3],
            upper[row, r, 4],
            upper[row, r, 5],
        )
        factor = _mul_row6_mat66(up_in, diagonal, eliminated)
        prod_lo = _mul_row6_mat66(factor, lower, eliminated)
        diag_r = diag_r - prod_lo
        rhs_r = rhs_r - _dot_row6_vec6(factor, rhs, eliminated)
        if local_row + 2 * stride < row_count:
            prod_up = _mul_row6_mat66(factor, upper, eliminated)
            upper_r = -prod_up

    for c in range(6):
        lower[row, r, c] = lower_r[c]
        diagonal[row, r, c] = diag_r[c]
        upper[row, r, c] = upper_r[c]
    rhs[row, r] = rhs_r


@wp.kernel(module="unique", enable_backward=False)
def reduce_cr_in_place_rows(
    stride: int,
    survivor_rows: wp.array[wp.vec3i],
    lower_f: wp.array3d[float],
    diagonal_f: wp.array3d[float],
    upper_f: wp.array3d[float],
    rhs_f: wp.array2d[float],
):
    """Reduce retained CR rows with one scalar row per thread."""
    idx, r = wp.tid()
    entry = survivor_rows[idx]
    _reduce_cr_scalar_row(entry[0], entry[1], r, stride, entry[2], lower_f, diagonal_f, upper_f, rhs_f)


@wp.kernel(module="unique", enable_backward=False)
def solve_cr_tail_persistent_rows(
    paths: wp.array[wp.vec3i],
    lower: wp.array[wp.spatial_matrix],
    diagonal: wp.array[wp.spatial_matrix],
    upper: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
    lower_f: wp.array3d[float],
    diagonal_f: wp.array3d[float],
    upper_f: wp.array3d[float],
    rhs_f: wp.array2d[float],
):
    """Finish a strided CR tail in place with scalar-row CR reductions."""
    tid = wp.tid()
    width = wp.block_dim()
    lane = tid % width
    path = paths[tid // width]
    base = path[0]
    row_count = path[1]
    initial_stride = path[2]
    stride = int(initial_stride)
    groups = width // 6
    group = lane // 6
    r = lane % 6
    active_count = (row_count + stride - 1) // stride
    while active_count > _CR_SERIAL_TERMINAL_SIZE:
        row = stride + 2 * stride * lane
        while row < row_count:
            diagonal[base + row] = _inverse_spatial_robust(diagonal[base + row])
            row += 2 * stride * width
        _synchronize_cr_block()
        if group < groups:
            row = 2 * stride * group
            while row < row_count:
                _reduce_cr_scalar_row(base, row, r, stride, row_count, lower_f, diagonal_f, upper_f, rhs_f)
                row += 2 * stride * groups
        _synchronize_cr_block()
        stride *= 2
        active_count = (row_count + stride - 1) // stride
    if lane == 0:
        _solve_strided_cr_coarse_impl(base, stride, row_count, lower, diagonal, upper, rhs)
    _synchronize_cr_block()
    stride //= 2
    while stride >= initial_stride:
        row = stride + 2 * stride * lane
        while row < row_count:
            _back_substitute_cr_row(base, row, stride, row_count, lower, diagonal, upper, rhs)
            row += 2 * stride * width
        _synchronize_cr_block()
        stride //= 2


@wp.func
def _mul_row6_mat66_f64(row_vec: wp.spatial_vectord, mat: wp.array3d[wp.float64], slot: int) -> wp.spatial_vectord:
    out = wp.spatial_vectord()
    for c in range(6):
        s = wp.float64(0.0)
        for t in range(6):
            s = _fma_f64(row_vec[t], mat[slot, t, c], s)
        out[c] = s
    return out


@wp.func
def _dot_row6_vec6_f64(row_vec: wp.spatial_vectord, vec: wp.array2d[wp.float64], slot: int) -> wp.float64:
    s = wp.float64(0.0)
    for c in range(6):
        s = _fma_f64(row_vec[c], vec[slot, c], s)
    return s


@wp.func
def _reduce_cr_scalar_row_f64(
    base: int,
    local_row: int,
    r: int,
    stride: int,
    row_count: int,
    lower: wp.array3d[wp.float64],
    diagonal: wp.array3d[wp.float64],
    upper: wp.array3d[wp.float64],
    rhs: wp.array2d[wp.float64],
):
    row = base + local_row
    lower_r = wp.spatial_vectord()
    diag_r = wp.spatial_vectord(
        diagonal[row, r, 0],
        diagonal[row, r, 1],
        diagonal[row, r, 2],
        diagonal[row, r, 3],
        diagonal[row, r, 4],
        diagonal[row, r, 5],
    )
    upper_r = wp.spatial_vectord()
    rhs_r = rhs[row, r]

    if local_row >= stride:
        eliminated = row - stride
        lo_in = wp.spatial_vectord(
            lower[row, r, 0],
            lower[row, r, 1],
            lower[row, r, 2],
            lower[row, r, 3],
            lower[row, r, 4],
            lower[row, r, 5],
        )
        factor = _mul_row6_mat66_f64(lo_in, diagonal, eliminated)
        prod_lo = _mul_row6_mat66_f64(factor, lower, eliminated)
        prod_up = _mul_row6_mat66_f64(factor, upper, eliminated)
        lower_r = -prod_lo
        diag_r = diag_r - prod_up
        rhs_r = rhs_r - _dot_row6_vec6_f64(factor, rhs, eliminated)

    if local_row + stride < row_count:
        eliminated = row + stride
        up_in = wp.spatial_vectord(
            upper[row, r, 0],
            upper[row, r, 1],
            upper[row, r, 2],
            upper[row, r, 3],
            upper[row, r, 4],
            upper[row, r, 5],
        )
        factor = _mul_row6_mat66_f64(up_in, diagonal, eliminated)
        prod_lo = _mul_row6_mat66_f64(factor, lower, eliminated)
        diag_r = diag_r - prod_lo
        rhs_r = rhs_r - _dot_row6_vec6_f64(factor, rhs, eliminated)
        if local_row + 2 * stride < row_count:
            prod_up = _mul_row6_mat66_f64(factor, upper, eliminated)
            upper_r = -prod_up

    for c in range(6):
        lower[row, r, c] = lower_r[c]
        diagonal[row, r, c] = diag_r[c]
        upper[row, r, c] = upper_r[c]
    rhs[row, r] = rhs_r


@wp.kernel(module="unique", enable_backward=False)
def reduce_cr_in_place_rows_fp64(
    stride: int,
    survivor_rows: wp.array[wp.vec3i],
    lower_d: wp.array3d[wp.float64],
    diagonal_d: wp.array3d[wp.float64],
    upper_d: wp.array3d[wp.float64],
    rhs_d: wp.array2d[wp.float64],
):
    """Reduce retained FP64 CR rows with one scalar row per thread."""
    idx, r = wp.tid()
    entry = survivor_rows[idx]
    _reduce_cr_scalar_row_f64(entry[0], entry[1], r, stride, entry[2], lower_d, diagonal_d, upper_d, rhs_d)


@wp.kernel(module="unique", enable_backward=False)
def solve_cr_tail_persistent_rows_fp64(
    paths: wp.array[wp.vec3i],
    lower: wp.array[wp.spatial_matrixd],
    diagonal: wp.array[wp.spatial_matrixd],
    upper: wp.array[wp.spatial_matrixd],
    rhs: wp.array[wp.spatial_vectord],
    lower_d: wp.array3d[wp.float64],
    diagonal_d: wp.array3d[wp.float64],
    upper_d: wp.array3d[wp.float64],
    rhs_d: wp.array2d[wp.float64],
):
    """Finish a strided FP64 CR tail in place with scalar-row FP64 CR reductions."""
    tid = wp.tid()
    width = wp.block_dim()
    lane = tid % width
    path = paths[tid // width]
    base = path[0]
    row_count = path[1]
    initial_stride = path[2]
    stride = int(initial_stride)
    groups = width // 6
    group = lane // 6
    r = lane % 6
    active_count = (row_count + stride - 1) // stride
    while active_count > _CR_SERIAL_TERMINAL_SIZE:
        row = stride + 2 * stride * lane
        while row < row_count:
            diagonal[base + row] = _inverse_spatial_robust(diagonal[base + row])
            row += 2 * stride * width
        _synchronize_cr_block()
        if group < groups:
            row = 2 * stride * group
            while row < row_count:
                _reduce_cr_scalar_row_f64(base, row, r, stride, row_count, lower_d, diagonal_d, upper_d, rhs_d)
                row += 2 * stride * groups
        _synchronize_cr_block()
        stride *= 2
        active_count = (row_count + stride - 1) // stride
    if lane == 0:
        _solve_strided_cr_coarse_impl(base, stride, row_count, lower, diagonal, upper, rhs)
    _synchronize_cr_block()
    stride //= 2
    while stride >= initial_stride:
        row = stride + 2 * stride * lane
        while row < row_count:
            _back_substitute_cr_row(base, row, stride, row_count, lower, diagonal, upper, rhs)
            row += 2 * stride * width
        _synchronize_cr_block()
        stride //= 2


@wp.kernel
def solve_cr_persistent_fp64_io(
    paths: wp.array[wp.vec2i],
    terminal_size: int,
    lower32: wp.array[wp.spatial_matrix],
    diagonal32: wp.array[wp.spatial_matrix],
    upper32: wp.array[wp.spatial_matrix],
    rhs32: wp.array[wp.spatial_vector],
    lower: wp.array[wp.spatial_matrixd],
    diagonal: wp.array[wp.spatial_matrixd],
    upper: wp.array[wp.spatial_matrixd],
    rhs: wp.array[wp.spatial_vectord],
):
    """Fused FP32->FP64 load + persistent FP64 CR solve + FP64->FP32 store in 1 launch."""
    tid = wp.tid()
    block_dim = wp.block_dim()
    lane = tid % block_dim
    path = paths[tid // block_dim]
    base = path[0]
    row_count = path[1]

    local = lane
    while local < row_count:
        slot = base + local
        lower[slot] = wp.spatial_matrixd(lower32[slot])
        diagonal[slot] = wp.spatial_matrixd(diagonal32[slot])
        upper[slot] = wp.spatial_matrixd(upper32[slot])
        rhs[slot] = wp.spatial_vectord(rhs32[slot])
        local += block_dim
    _synchronize_cr_block()

    stride = int(1)
    active_count = int(row_count)
    while active_count > terminal_size:
        eliminated_count = (row_count - stride + 2 * stride - 1) // (2 * stride)
        local = lane
        while local < eliminated_count:
            row = base + stride + 2 * stride * local
            diagonal[row] = _inverse_spatial_robust(diagonal[row])
            local += block_dim
        _synchronize_cr_block()

        survivor_count = (row_count + 2 * stride - 1) // (2 * stride)
        local = lane
        while local < survivor_count:
            _reduce_cr_row(base, 2 * stride * local, stride, row_count, 0, lower, diagonal, upper, rhs)
            local += block_dim
        _synchronize_cr_block()
        stride *= 2
        active_count = survivor_count

    if lane == 0:
        _solve_strided_cr_coarse_impl(base, stride, row_count, lower, diagonal, upper, rhs)
    _synchronize_cr_block()

    stride //= 2
    while stride >= 1:
        eliminated_count = (row_count - stride + 2 * stride - 1) // (2 * stride)
        local = lane
        while local < eliminated_count:
            _back_substitute_cr_row(
                base,
                stride + 2 * stride * local,
                stride,
                row_count,
                lower,
                diagonal,
                upper,
                rhs,
            )
            local += block_dim
        _synchronize_cr_block()
        stride //= 2

    local = lane
    while local < row_count:
        slot = base + local
        val = rhs[slot]
        out = wp.spatial_vector()
        for axis in range(6):
            out[axis] = float(val[axis])
        rhs32[slot] = out
        local += block_dim


@wp.kernel
def solve_cr_persistent(
    paths: wp.array[wp.vec2i],
    terminal_size: int,
    lower: wp.array[Any],
    diagonal: wp.array[Any],
    upper: wp.array[Any],
    rhs: wp.array[Any],
):
    """Solve one short path per CUDA block without dependent level launches.

    Each path entry stores its first row and row count.
    """
    tid = wp.tid()
    block_dim = wp.block_dim()
    lane = tid % block_dim
    path = paths[tid // block_dim]
    base = path[0]
    row_count = path[1]

    # Explicit casts make these loop-carried values dynamic in Warp codegen.
    stride = int(1)
    active_count = int(row_count)
    while active_count > terminal_size:
        eliminated_count = (row_count - stride + 2 * stride - 1) // (2 * stride)
        local = lane
        while local < eliminated_count:
            row = base + stride + 2 * stride * local
            diagonal[row] = _inverse_spatial_robust(diagonal[row])
            local += block_dim
        _synchronize_cr_block()

        survivor_count = (row_count + 2 * stride - 1) // (2 * stride)
        local = lane
        while local < survivor_count:
            _reduce_cr_row(base, 2 * stride * local, stride, row_count, 0, lower, diagonal, upper, rhs)
            local += block_dim
        _synchronize_cr_block()
        stride *= 2
        active_count = survivor_count

    if lane == 0:
        _solve_strided_cr_coarse_impl(base, stride, row_count, lower, diagonal, upper, rhs)
    _synchronize_cr_block()

    stride //= 2
    while stride >= 1:
        eliminated_count = (row_count - stride + 2 * stride - 1) // (2 * stride)
        local = lane
        while local < eliminated_count:
            _back_substitute_cr_row(
                base,
                stride + 2 * stride * local,
                stride,
                row_count,
                lower,
                diagonal,
                upper,
                rhs,
            )
            local += block_dim
        _synchronize_cr_block()
        stride //= 2


@wp.kernel
def solve_cr_tail_persistent(
    paths: wp.array[wp.vec3i],
    lower: wp.array[Any],
    diagonal: wp.array[Any],
    upper: wp.array[Any],
    rhs: wp.array[Any],
):
    """Finish a strided CR tail in place, preserving factors for outer recovery.

    Each path entry stores its first row, row count, and tail stride. Keep the
    short-path kernel separate so its register/launch configuration is
    unchanged. All lanes of each independent path reach every barrier.
    """
    tid = wp.tid()
    width = wp.block_dim()
    lane = tid % width
    path = paths[tid // width]
    base = path[0]
    row_count = path[1]
    initial_stride = path[2]
    stride = int(initial_stride)
    active_count = (row_count + stride - 1) // stride
    while active_count > _CR_SERIAL_TERMINAL_SIZE:
        row = stride + 2 * stride * lane
        while row < row_count:
            diagonal[base + row] = _inverse_spatial_robust(diagonal[base + row])
            row += 2 * stride * width
        _synchronize_cr_block()
        row = 2 * stride * lane
        while row < row_count:
            _reduce_cr_row(base, row, stride, row_count, 0, lower, diagonal, upper, rhs)
            row += 2 * stride * width
        _synchronize_cr_block()
        stride *= 2
        active_count = (row_count + stride - 1) // stride
    if lane == 0:
        _solve_strided_cr_coarse_impl(base, stride, row_count, lower, diagonal, upper, rhs)
    _synchronize_cr_block()
    stride //= 2
    while stride >= initial_stride:
        row = stride + 2 * stride * lane
        while row < row_count:
            _back_substitute_cr_row(base, row, stride, row_count, lower, diagonal, upper, rhs)
            row += 2 * stride * width
        _synchronize_cr_block()
        stride //= 2


@wp.struct
class _SpatialPairMatrix:
    """A 2x2 matrix whose entries are 6x6 spatial blocks."""

    m00: wp.spatial_matrix
    m01: wp.spatial_matrix
    m10: wp.spatial_matrix
    m11: wp.spatial_matrix


@wp.struct
class _SpatialPairVector:
    """A two-entry vector of spatial vectors."""

    v0: wp.spatial_vector
    v1: wp.spatial_vector


@wp.struct
class _SpatialPairResponse:
    """A two-entry vector of 6-column spatial response blocks."""

    v0: wp.spatial_matrix
    v1: wp.spatial_matrix


@wp.struct
class _SpatialPairEliminationFactor:
    """The two sparse multiplier rows adjacent to one eliminated pair."""

    to_left0: wp.spatial_matrix
    to_left1: wp.spatial_matrix
    to_right0: wp.spatial_matrix
    to_right1: wp.spatial_matrix


@wp.func
def _zero_spatial_pair_matrix(matrix: Any):
    return type(matrix)()


@wp.func
def _spatial_pair_matrix_vector_multiply(matrix: Any, vector: Any):
    result = type(vector)()
    result.v0 = matrix.m00 * vector.v0 + matrix.m01 * vector.v1
    result.v1 = matrix.m10 * vector.v0 + matrix.m11 * vector.v1
    return result


@wp.func
def _spatial_pair_matrix_response_multiply(matrix: Any, response: Any):
    result = type(response)()
    result.v0 = matrix.m00 * response.v0 + matrix.m01 * response.v1
    result.v1 = matrix.m10 * response.v0 + matrix.m11 * response.v1
    return result


@wp.func
def _inverse_spatial_pair_body_first(matrix: Any):
    """Invert one symbolic body-row saddle pair through its body block."""
    inverse_body = _inverse_spatial_pivoted(matrix.m00)
    inverse_body_coupling = inverse_body * matrix.m01
    schur = matrix.m11 - matrix.m10 * inverse_body_coupling
    inverse_schur = _inverse_spatial_pivoted(schur)

    result = type(matrix)()
    result.m01 = -(inverse_body_coupling * inverse_schur)
    result.m10 = -(inverse_schur * matrix.m10 * inverse_body)
    result.m00 = inverse_body + inverse_body_coupling * inverse_schur * matrix.m10 * inverse_body
    result.m11 = inverse_schur
    return result


@wp.kernel
def initialize_tree_backbone_edges(
    row_count: int,
    backbone_nodes: wp.array[wp.int32],
    parent_node: wp.array[wp.int32],
    coupling: wp.array[Any],
    lower: wp.array[Any],
    upper: wp.array[Any],
):
    """Gather the two oriented edge blocks of a tree backbone."""
    index = wp.tid()
    batch = index // row_count
    row = index - batch * row_count
    base = batch * row_count
    node = backbone_nodes[index]
    lower_value = coupling.dtype(0.0)
    if row > 0:
        previous = backbone_nodes[base + row - 1]
        if parent_node[node] == previous:
            lower_value = coupling[node]
        else:
            lower_value = wp.transpose(coupling[previous])

    upper_value = coupling.dtype(0.0)
    if row + 1 < row_count:
        following = backbone_nodes[base + row + 1]
        if parent_node[node] == following:
            upper_value = coupling[node]
        else:
            upper_value = wp.transpose(coupling[following])

    lower[index] = lower_value
    upper[index] = upper_value


@wp.kernel
def initialize_paired_tree_backbone(
    pack_rhs: int,
    pair_count: int,
    backbone_node_count: int,
    backbone_nodes: wp.array[wp.int32],
    backbone_lower: wp.array[Any],
    backbone_upper: wp.array[Any],
    diagonal: wp.array[Any],
    rhs: wp.array[Any],
    pair_lower: wp.array[Any],
    pair_diagonal: wp.array[Any],
    pair_upper: wp.array[Any],
    pair_rhs: wp.array[Any],
):
    """Pack an alternating body-row backbone into symbolic saddle pairs."""
    index = wp.tid()
    batch = index // pair_count
    local_pair = index - batch * pair_count
    backbone_base = batch * backbone_node_count
    first_row = 2 * local_pair
    first_node = backbone_nodes[backbone_base + first_row]
    has_second = first_row + 1 < backbone_node_count

    diagonal_value = _zero_spatial_pair_matrix(pair_lower[0])
    diagonal_value.m00 = diagonal[first_node]
    rhs_value = pair_rhs.dtype()
    rhs_value.v0 = rhs[first_node]
    rhs_value.v1 = rhs.dtype()
    if has_second:
        second_node = backbone_nodes[backbone_base + first_row + 1]
        diagonal_value.m01 = backbone_upper[backbone_base + first_row]
        diagonal_value.m10 = backbone_lower[backbone_base + first_row + 1]
        diagonal_value.m11 = diagonal[second_node]
        rhs_value.v1 = rhs[second_node]
    else:
        # Pad an odd body-ended backbone with an uncoupled identity block so
        # every CR unknown has the same symbolic 12x12 layout.
        diagonal_value.m11 = wp.identity(6, backbone_lower.dtype.dtype)

    lower_value = _zero_spatial_pair_matrix(pair_lower[0])
    if local_pair > 0:
        # Current body (component 0) couples to the previous row (component 1).
        lower_value.m01 = backbone_lower[backbone_base + first_row]

    upper_value = _zero_spatial_pair_matrix(pair_lower[0])
    if has_second and local_pair + 1 < pair_count:
        # Current row (component 1) couples to the following body (component 0).
        upper_value.m10 = backbone_upper[backbone_base + first_row + 1]

    pair_lower[index] = lower_value
    pair_diagonal[index] = diagonal_value
    pair_upper[index] = upper_value
    if pack_rhs != 0:
        pair_rhs[index] = rhs_value


@wp.kernel
def initialize_paired_tree_backbone_response(
    pair_count: int,
    backbone_node_count: int,
    closure_count: int,
    backbone_nodes: wp.array[wp.int32],
    response_rhs: wp.array[Any],
    pair_response: wp.array[Any],
):
    """Pack all closure-response columns into the paired backbone layout."""
    index = wp.tid()
    pair_index = index // closure_count
    closure = index - pair_index * closure_count
    batch = pair_index // pair_count
    local_pair = pair_index - batch * pair_count
    backbone_base = batch * backbone_node_count
    first_row = 2 * local_pair
    first_node = backbone_nodes[backbone_base + first_row]

    value = pair_response.dtype()
    value.v0 = response_rhs[first_node * closure_count + closure]
    value.v1 = response_rhs.dtype(0.0)
    if first_row + 1 < backbone_node_count:
        second_node = backbone_nodes[backbone_base + first_row + 1]
        value.v1 = response_rhs[second_node * closure_count + closure]
    pair_response[index] = value


@wp.kernel
def invert_paired_tree_backbone_cr_eliminated(
    stride: int,
    pair_count: int,
    eliminated_count: int,
    lower: wp.array[Any],
    diagonal: wp.array[Any],
    upper: wp.array[Any],
    elimination_factor: wp.array[Any],
):
    """Invert one CR level and retain its sparse forward multipliers."""
    index = wp.tid()
    batch = index // eliminated_count
    local = index - batch * eliminated_count
    row = stride + 2 * stride * local
    if local < eliminated_count and row < pair_count:
        base = batch * pair_count
        pair = base + row
        inverse = _inverse_spatial_pair_body_first(diagonal[pair])
        diagonal[pair] = inverse

        zero = type(lower[0].m00)(0.0)
        factor = elimination_factor.dtype()
        factor.to_left0 = zero
        factor.to_left1 = zero
        factor.to_right0 = zero
        factor.to_right1 = zero
        if row >= stride:
            # The left survivor's upper block occupies pair row 1.
            coupling = upper[base + row - stride].m10
            factor.to_left0 = coupling * inverse.m00
            factor.to_left1 = coupling * inverse.m01
        if row + stride < pair_count:
            # The right survivor's lower block occupies pair row 0.
            coupling = lower[base + row + stride].m01
            factor.to_right0 = coupling * inverse.m10
            factor.to_right1 = coupling * inverse.m11
        elimination_factor[pair] = factor


@wp.kernel
def reduce_paired_tree_backbone_cr_in_place(
    stride: int,
    pair_count: int,
    survivor_count: int,
    closure_count: int,
    factorize: int,
    lower: wp.array[Any],
    diagonal: wp.array[Any],
    upper: wp.array[Any],
    elimination_factor: wp.array[Any],
    rhs: wp.array[Any],
    response_rhs: wp.array[Any],
):
    """Reduce paired right-hand sides, optionally building this CR level."""
    index = wp.tid()
    batch = index // survivor_count
    local = index - batch * survivor_count
    base = batch * pair_count
    local_row = 2 * stride * local
    if local >= survivor_count or local_row >= pair_count:
        return
    row = base + local_row

    lower_value = _zero_spatial_pair_matrix(lower[0])
    diagonal_value = diagonal[row]
    upper_value = _zero_spatial_pair_matrix(lower[0])
    rhs_value = rhs[row]
    if local_row >= stride:
        eliminated = base + local_row - stride
        factor = elimination_factor[eliminated]
        if factorize != 0:
            lower_value.m01 = -(factor.to_right0 * lower[eliminated].m01)
            diagonal_value.m00 = diagonal_value.m00 - factor.to_right1 * upper[eliminated].m10
        eliminated_rhs = rhs[eliminated]
        rhs_value.v0 = rhs_value.v0 - (factor.to_right0 * eliminated_rhs.v0 + factor.to_right1 * eliminated_rhs.v1)
        for closure in range(closure_count):
            response_index = row * closure_count + closure
            eliminated_response_index = eliminated * closure_count + closure
            response_value = response_rhs[response_index]
            eliminated_response = response_rhs[eliminated_response_index]
            response_value.v0 = response_value.v0 - (
                factor.to_right0 * eliminated_response.v0 + factor.to_right1 * eliminated_response.v1
            )
            response_rhs[response_index] = response_value
    if local_row + stride < pair_count:
        eliminated = base + local_row + stride
        factor = elimination_factor[eliminated]
        if factorize != 0:
            diagonal_value.m11 = diagonal_value.m11 - factor.to_left0 * lower[eliminated].m01
        eliminated_rhs = rhs[eliminated]
        rhs_value.v1 = rhs_value.v1 - (factor.to_left0 * eliminated_rhs.v0 + factor.to_left1 * eliminated_rhs.v1)
        for closure in range(closure_count):
            response_index = row * closure_count + closure
            eliminated_response_index = eliminated * closure_count + closure
            response_value = response_rhs[response_index]
            eliminated_response = response_rhs[eliminated_response_index]
            response_value.v1 = response_value.v1 - (
                factor.to_left0 * eliminated_response.v0 + factor.to_left1 * eliminated_response.v1
            )
            response_rhs[response_index] = response_value
        if factorize != 0 and local_row + 2 * stride < pair_count:
            upper_value.m10 = -(factor.to_left1 * upper[eliminated].m10)

    if factorize != 0:
        lower[row] = lower_value
        diagonal[row] = diagonal_value
        upper[row] = upper_value
    rhs[row] = rhs_value


@wp.kernel
def reduce_paired_tree_backbone_cr_lanes(
    stride: int,
    pair_count: int,
    survivor_count: int,
    closure_count: int,
    factorize: int,
    lower: wp.array[Any],
    diagonal: wp.array[Any],
    upper: wp.array[Any],
    elimination_factor: wp.array[Any],
    rhs: wp.array[Any],
    response_rhs: wp.array[Any],
):
    """Reduce one paired CR level with its closure columns split across lanes.

    The serial form gives each surviving row one thread, which then walks every
    closure column in turn. On a high-closure island that is the dominant cost
    of the whole global pass: one thread performing `closure_count` dependent
    12x12 block updates while the rest of the machine idles.

    The closure columns are independent -- each touches its own
    ``response_rhs[row * closure_count + closure]`` entry, with multipliers that
    do not depend on the column -- and `elimination_factor`, `lower` and `upper`
    are read-only here. Survivors and eliminated rows are disjoint sets at every
    level, so no barrier is needed: one block per row, lanes stride the closure
    dimension, and only lane zero performs the row-level writes. Consecutive
    lanes then read consecutive response entries, which also coalesces an access
    that was previously one thread walking the whole column.

    Arithmetic per column is unchanged and no accumulation is reordered, so the
    result is bit-identical to the serial kernel.
    """
    thread = wp.tid()
    width = wp.block_dim()
    lane = thread % width
    index = thread // width
    batch = index // survivor_count
    local = index - batch * survivor_count
    base = batch * pair_count
    local_row = 2 * stride * local
    if local >= survivor_count or local_row >= pair_count:
        return
    row = base + local_row

    lower_value = _zero_spatial_pair_matrix(lower[0])
    diagonal_value = diagonal[row]
    upper_value = _zero_spatial_pair_matrix(lower[0])
    rhs_value = rhs[row]
    if local_row >= stride:
        eliminated = base + local_row - stride
        factor = elimination_factor[eliminated]
        if factorize != 0:
            lower_value.m01 = -(factor.to_right0 * lower[eliminated].m01)
            diagonal_value.m00 = diagonal_value.m00 - factor.to_right1 * upper[eliminated].m10
        eliminated_rhs = rhs[eliminated]
        rhs_value.v0 = rhs_value.v0 - (factor.to_right0 * eliminated_rhs.v0 + factor.to_right1 * eliminated_rhs.v1)
        closure = int(lane)
        while closure < closure_count:
            response_index = row * closure_count + closure
            eliminated_response_index = eliminated * closure_count + closure
            response_value = response_rhs[response_index]
            eliminated_response = response_rhs[eliminated_response_index]
            response_value.v0 = response_value.v0 - (
                factor.to_right0 * eliminated_response.v0 + factor.to_right1 * eliminated_response.v1
            )
            response_rhs[response_index] = response_value
            closure += width
    if local_row + stride < pair_count:
        eliminated = base + local_row + stride
        factor = elimination_factor[eliminated]
        if factorize != 0:
            diagonal_value.m11 = diagonal_value.m11 - factor.to_left0 * lower[eliminated].m01
        eliminated_rhs = rhs[eliminated]
        rhs_value.v1 = rhs_value.v1 - (factor.to_left0 * eliminated_rhs.v0 + factor.to_left1 * eliminated_rhs.v1)
        closure = int(lane)
        while closure < closure_count:
            response_index = row * closure_count + closure
            eliminated_response_index = eliminated * closure_count + closure
            response_value = response_rhs[response_index]
            eliminated_response = response_rhs[eliminated_response_index]
            response_value.v1 = response_value.v1 - (
                factor.to_left0 * eliminated_response.v0 + factor.to_left1 * eliminated_response.v1
            )
            response_rhs[response_index] = response_value
            closure += width
        if factorize != 0 and local_row + 2 * stride < pair_count:
            upper_value.m10 = -(factor.to_left1 * upper[eliminated].m10)

    if lane == 0:
        if factorize != 0:
            lower[row] = lower_value
            diagonal[row] = diagonal_value
            upper[row] = upper_value
        rhs[row] = rhs_value


@wp.kernel
def solve_paired_tree_backbone_cr_coarse(
    stride: int,
    pair_count: int,
    closure_count: int,
    factorize: int,
    lower: wp.array[Any],
    diagonal: wp.array[Any],
    upper: wp.array[Any],
    rhs: wp.array[Any],
    response_rhs: wp.array[Any],
):
    """Solve the block-Thomas tail, optionally building its factor."""
    batch = wp.tid()
    base = batch * pair_count
    active_count = (pair_count + stride - 1) // stride
    for local in range(1, active_count):
        previous = base + (local - 1) * stride
        row = base + local * stride
        multiplier0 = lower[row].m00
        multiplier1 = lower[row].m01
        if factorize != 0:
            inverse_previous = _inverse_spatial_pair_body_first(diagonal[previous])
            diagonal[previous] = inverse_previous
            multiplier0 = lower[row].m01 * inverse_previous.m10
            multiplier1 = lower[row].m01 * inverse_previous.m11
            # Coarse rows are never CR-eliminated, so after this forward step
            # their lower coupling is dead. Retain the multiplier in its place.
            factor_value = lower[row]
            factor_value.m00 = multiplier0
            factor_value.m01 = multiplier1
            lower[row] = factor_value
            diagonal_value = diagonal[row]
            diagonal_value.m00 = diagonal_value.m00 - multiplier1 * upper[previous].m10
            diagonal[row] = diagonal_value
        previous_rhs = rhs[previous]
        row_rhs = rhs[row]
        row_rhs.v0 = row_rhs.v0 - (multiplier0 * previous_rhs.v0 + multiplier1 * previous_rhs.v1)
        rhs[row] = row_rhs
        for closure in range(closure_count):
            response_index = row * closure_count + closure
            previous_response_index = previous * closure_count + closure
            response_value = response_rhs[response_index]
            previous_response = response_rhs[previous_response_index]
            response_value.v0 = response_value.v0 - (
                multiplier0 * previous_response.v0 + multiplier1 * previous_response.v1
            )
            response_rhs[response_index] = response_value

    last = base + (active_count - 1) * stride
    if factorize != 0:
        diagonal[last] = _inverse_spatial_pair_body_first(diagonal[last])
    rhs[last] = _spatial_pair_matrix_vector_multiply(diagonal[last], rhs[last])
    for closure in range(closure_count):
        response_index = last * closure_count + closure
        response_rhs[response_index] = _spatial_pair_matrix_response_multiply(
            diagonal[last], response_rhs[response_index]
        )

    for reverse_local in range(1, active_count):
        row = base + (active_count - 1 - reverse_local) * stride
        following = row + stride
        rhs_value = rhs[row]
        rhs_value.v1 = rhs_value.v1 - upper[row].m10 * rhs[following].v0
        rhs[row] = _spatial_pair_matrix_vector_multiply(diagonal[row], rhs_value)
        for closure in range(closure_count):
            response_index = row * closure_count + closure
            following_response_index = following * closure_count + closure
            response_value = response_rhs[response_index]
            response_value.v1 = response_value.v1 - upper[row].m10 * response_rhs[following_response_index].v0
            response_rhs[response_index] = _spatial_pair_matrix_response_multiply(diagonal[row], response_value)


@wp.kernel
def solve_paired_tree_backbone_cr_coarse_lanes(
    stride: int,
    pair_count: int,
    closure_count: int,
    factorize: int,
    lower: wp.array[Any],
    diagonal: wp.array[Any],
    upper: wp.array[Any],
    rhs: wp.array[Any],
    response_rhs: wp.array[Any],
):
    """Solve the same block-Thomas tail with its closure columns lane-split.

    ``solve_paired_tree_backbone_cr_coarse`` runs one thread per island and
    walks every closure column serially inside a serial pair recurrence. The
    recurrence itself is irreducibly sequential, but the closure columns are
    not: each iterates over disjoint ``response_rhs`` entries with multipliers
    that do not depend on the column.

    This variant gives each island one block. Lane zero computes each pair
    inverse and stores it in the existing factor arrays before a block-wide
    barrier. All lanes then read those factors, with closure columns strided
    across lanes. The arithmetic per column and its order are unchanged; the
    factors are computed once rather than redundantly by every lane.
    """
    thread = wp.tid()
    block = wp.block_dim()
    lane = thread % block
    batch = thread // block
    base = batch * pair_count
    active_count = (pair_count + stride - 1) // stride

    for local in range(1, active_count):
        previous = base + (local - 1) * stride
        row = base + local * stride
        if lane == 0:
            multiplier0 = lower[row].m00
            multiplier1 = lower[row].m01
            if factorize != 0:
                inverse_previous = _inverse_spatial_pair_body_first(diagonal[previous])
                diagonal[previous] = inverse_previous
                multiplier0 = lower[row].m01 * inverse_previous.m10
                multiplier1 = lower[row].m01 * inverse_previous.m11
                factor_value = lower[row]
                factor_value.m00 = multiplier0
                factor_value.m01 = multiplier1
                lower[row] = factor_value
                diagonal_value = diagonal[row]
                diagonal_value.m00 = diagonal_value.m00 - multiplier1 * upper[previous].m10
                diagonal[row] = diagonal_value
            previous_rhs = rhs[previous]
            row_rhs = rhs[row]
            row_rhs.v0 = row_rhs.v0 - (multiplier0 * previous_rhs.v0 + multiplier1 * previous_rhs.v1)
            rhs[row] = row_rhs
        _synchronize_cr_block()
        multiplier0 = lower[row].m00
        multiplier1 = lower[row].m01
        closure = int(lane)
        while closure < closure_count:
            response_index = row * closure_count + closure
            previous_response_index = previous * closure_count + closure
            response_value = response_rhs[response_index]
            previous_response = response_rhs[previous_response_index]
            response_value.v0 = response_value.v0 - (
                multiplier0 * previous_response.v0 + multiplier1 * previous_response.v1
            )
            response_rhs[response_index] = response_value
            closure += block
        _synchronize_cr_block()

    last = base + (active_count - 1) * stride
    if lane == 0:
        diagonal_last = diagonal[last]
        if factorize != 0:
            diagonal_last = _inverse_spatial_pair_body_first(diagonal[last])
            diagonal[last] = diagonal_last
        rhs[last] = _spatial_pair_matrix_vector_multiply(diagonal_last, rhs[last])
    _synchronize_cr_block()
    diagonal_last = diagonal[last]
    closure = int(lane)
    while closure < closure_count:
        response_index = last * closure_count + closure
        response_rhs[response_index] = _spatial_pair_matrix_response_multiply(
            diagonal_last, response_rhs[response_index]
        )
        closure += block
    _synchronize_cr_block()

    for reverse_local in range(1, active_count):
        row = base + (active_count - 1 - reverse_local) * stride
        following = row + stride
        diagonal_row = diagonal[row]
        rhs_value = rhs[row]
        rhs_value.v1 = rhs_value.v1 - upper[row].m10 * rhs[following].v0
        solved_rhs = _spatial_pair_matrix_vector_multiply(diagonal_row, rhs_value)
        _synchronize_cr_block()
        if lane == 0:
            rhs[row] = solved_rhs
        closure = int(lane)
        while closure < closure_count:
            response_index = row * closure_count + closure
            following_response_index = following * closure_count + closure
            response_value = response_rhs[response_index]
            response_value.v1 = response_value.v1 - upper[row].m10 * response_rhs[following_response_index].v0
            response_rhs[response_index] = _spatial_pair_matrix_response_multiply(diagonal_row, response_value)
            closure += block
        _synchronize_cr_block()


@wp.kernel
def back_substitute_paired_tree_backbone_cr_in_place(
    stride: int,
    pair_count: int,
    eliminated_count: int,
    closure_count: int,
    lower: wp.array[Any],
    diagonal: wp.array[Any],
    upper: wp.array[Any],
    rhs: wp.array[Any],
    response_rhs: wp.array[Any],
):
    """Recover one eliminated paired-backbone level for every right-hand side."""
    index = wp.tid()
    batch = index // eliminated_count
    local = index - batch * eliminated_count
    base = batch * pair_count
    local_row = stride + 2 * stride * local
    if local >= eliminated_count or local_row >= pair_count:
        return
    row = base + local_row
    previous = row - stride
    rhs_value = rhs[row]
    rhs_value.v0 = rhs_value.v0 - lower[row].m01 * rhs[previous].v1
    if local_row + stride < pair_count:
        following = row + stride
        rhs_value.v1 = rhs_value.v1 - upper[row].m10 * rhs[following].v0
    rhs[row] = _spatial_pair_matrix_vector_multiply(diagonal[row], rhs_value)

    for closure in range(closure_count):
        response_index = row * closure_count + closure
        previous_response_index = previous * closure_count + closure
        response_value = response_rhs[response_index]
        response_value.v0 = response_value.v0 - lower[row].m01 * response_rhs[previous_response_index].v1
        if local_row + stride < pair_count:
            following = row + stride
            following_response_index = following * closure_count + closure
            response_value.v1 = response_value.v1 - upper[row].m10 * response_rhs[following_response_index].v0
        response_rhs[response_index] = _spatial_pair_matrix_response_multiply(diagonal[row], response_value)


@wp.kernel
def back_substitute_paired_tree_backbone_cr_lanes(
    stride: int,
    pair_count: int,
    eliminated_count: int,
    closure_count: int,
    lower: wp.array[Any],
    diagonal: wp.array[Any],
    upper: wp.array[Any],
    rhs: wp.array[Any],
    response_rhs: wp.array[Any],
):
    """Recover one paired level with its closure columns split across lanes.

    Same argument as ``reduce_paired_tree_backbone_cr_lanes``: the row factors
    are read-only, eliminated and neighbouring rows are disjoint, and each
    closure column is independent, so lanes may stride the closure dimension
    with no barrier and only lane zero writes the row right-hand side. The
    arithmetic per column is unchanged, so the result is bit-identical.
    """
    thread = wp.tid()
    width = wp.block_dim()
    lane = thread % width
    index = thread // width
    batch = index // eliminated_count
    local = index - batch * eliminated_count
    base = batch * pair_count
    local_row = stride + 2 * stride * local
    if local >= eliminated_count or local_row >= pair_count:
        return
    row = base + local_row
    previous = row - stride
    rhs_value = rhs[row]
    rhs_value.v0 = rhs_value.v0 - lower[row].m01 * rhs[previous].v1
    if local_row + stride < pair_count:
        following = row + stride
        rhs_value.v1 = rhs_value.v1 - upper[row].m10 * rhs[following].v0

    closure = int(lane)
    while closure < closure_count:
        response_index = row * closure_count + closure
        previous_response_index = previous * closure_count + closure
        response_value = response_rhs[response_index]
        response_value.v0 = response_value.v0 - lower[row].m01 * response_rhs[previous_response_index].v1
        if local_row + stride < pair_count:
            following = row + stride
            following_response_index = following * closure_count + closure
            response_value.v1 = response_value.v1 - upper[row].m10 * response_rhs[following_response_index].v0
        response_rhs[response_index] = _spatial_pair_matrix_response_multiply(diagonal[row], response_value)
        closure += width

    if lane == 0:
        rhs[row] = _spatial_pair_matrix_vector_multiply(diagonal[row], rhs_value)


@wp.kernel
def scatter_paired_tree_backbone_solution(
    pair_count: int,
    backbone_node_count: int,
    closure_count: int,
    backbone_nodes: wp.array[int],
    pair_rhs: wp.array[Any],
    pair_response: wp.array[Any],
    solution: wp.array[Any],
    response: wp.array[Any],
):
    index = wp.tid()
    pair = index // (closure_count + 1)
    column = index % (closure_count + 1)
    batch = pair // pair_count
    first = 2 * (pair % pair_count)
    node = backbone_nodes[batch * backbone_node_count + first]
    if column == closure_count:
        solution[node] = pair_rhs[pair].v0
        if first + 1 < backbone_node_count:
            second = backbone_nodes[batch * backbone_node_count + first + 1]
            solution[second] = pair_rhs[pair].v1
    else:
        response[node * closure_count + column] = pair_response[pair * closure_count + column].v0
        if first + 1 < backbone_node_count:
            second = backbone_nodes[batch * backbone_node_count + first + 1]
            response[second * closure_count + column] = pair_response[pair * closure_count + column].v1


@wp.kernel
def compute_paired_tree_backbone_residual(
    pair_count: int,
    backbone_node_count: int,
    closure_count: int,
    backbone_nodes: wp.array[wp.int32],
    backbone_lower: wp.array[Any],
    backbone_upper: wp.array[Any],
    diagonal: wp.array[Any],
    original_rhs: wp.array[Any],
    original_response: wp.array[Any],
    pair_solution: wp.array[Any],
    pair_response_solution: wp.array[Any],
    residual: wp.array[Any],
    response_residual: wp.array[Any],
):
    """Evaluate one paired-factor defect against the unmodified backbone."""
    index = wp.tid()
    batch = index // pair_count
    local_pair = index - batch * pair_count
    backbone_base = batch * backbone_node_count
    first_row = 2 * local_pair
    first_index = backbone_base + first_row
    first_node = backbone_nodes[first_index]
    solution = pair_solution[index]
    first_solution = solution.v0

    value = pair_solution.dtype()
    value.v0 = original_rhs[first_node] - diagonal[first_node] * first_solution
    value.v1 = original_rhs.dtype()
    if first_row > 0:
        value.v0 = value.v0 - backbone_lower[first_index] * pair_solution[index - 1].v1

    has_second = first_row + 1 < backbone_node_count
    if has_second:
        second_index = first_index + 1
        second_node = backbone_nodes[second_index]
        second_solution = solution.v1
        value.v0 = value.v0 - backbone_upper[first_index] * second_solution
        value.v1 = original_rhs[second_node] - diagonal[second_node] * second_solution
        value.v1 = value.v1 - backbone_lower[second_index] * first_solution
        if first_row + 2 < backbone_node_count:
            value.v1 = value.v1 - backbone_upper[second_index] * pair_solution[index + 1].v0
    residual[index] = value

    for closure in range(closure_count):
        response_index = index * closure_count + closure
        pair_response = pair_response_solution[response_index]
        first_response = pair_response.v0
        response_value = pair_response_solution.dtype()
        response_value.v0 = (
            original_response[first_node * closure_count + closure] - diagonal[first_node] * first_response
        )
        response_value.v1 = backbone_lower.dtype(0.0)
        if first_row > 0:
            previous_response = pair_response_solution[(index - 1) * closure_count + closure].v1
            response_value.v0 = response_value.v0 - backbone_lower[first_index] * previous_response
        if has_second:
            second_index = first_index + 1
            second_node = backbone_nodes[second_index]
            second_response = pair_response.v1
            response_value.v0 = response_value.v0 - backbone_upper[first_index] * second_response
            response_value.v1 = (
                original_response[second_node * closure_count + closure] - diagonal[second_node] * second_response
            )
            response_value.v1 = response_value.v1 - backbone_lower[second_index] * first_response
            if first_row + 2 < backbone_node_count:
                following_response = pair_response_solution[(index + 1) * closure_count + closure].v0
                response_value.v1 = response_value.v1 - backbone_upper[second_index] * following_response
        response_residual[response_index] = response_value


@wp.kernel
def scatter_refined_paired_tree_backbone_solution(
    pair_count: int,
    backbone_node_count: int,
    closure_count: int,
    backbone_nodes: wp.array[wp.int32],
    pair_solution: wp.array[Any],
    pair_response_solution: wp.array[Any],
    pair_correction: wp.array[Any],
    pair_response_correction: wp.array[Any],
    solution: wp.array[Any],
    response_solution: wp.array[Any],
):
    """Unpack the base solve and its defect correction in one write."""
    index = wp.tid()
    batch = index // pair_count
    local_pair = index - batch * pair_count
    backbone_base = batch * backbone_node_count
    first_row = 2 * local_pair
    first_node = backbone_nodes[backbone_base + first_row]
    value = pair_solution[index]
    correction = pair_correction[index]
    solution[first_node] = value.v0 + correction.v0
    if first_row + 1 < backbone_node_count:
        second_node = backbone_nodes[backbone_base + first_row + 1]
        solution[second_node] = value.v1 + correction.v1

    for closure in range(closure_count):
        response_index = index * closure_count + closure
        response = pair_response_solution[response_index]
        response_correction = pair_response_correction[response_index]
        first_response_index = first_node * closure_count + closure
        response_solution[first_response_index] = response.v0 + response_correction.v0
        if first_row + 1 < backbone_node_count:
            second_node = backbone_nodes[backbone_base + first_row + 1]
            second_response_index = second_node * closure_count + closure
            response_solution[second_response_index] = response.v1 + response_correction.v1


@wp.struct
class _SpatialPairMatrixD:
    """A 2x2 matrix whose entries are 6x6 spatial blocks."""

    m00: wp.spatial_matrixd
    m01: wp.spatial_matrixd
    m10: wp.spatial_matrixd
    m11: wp.spatial_matrixd


@wp.struct
class _SpatialPairVectorD:
    """A two-entry vector of spatial vectors."""

    v0: wp.spatial_vectord
    v1: wp.spatial_vectord


@wp.struct
class _SpatialPairResponseD:
    """A two-entry vector of 6-column spatial response blocks."""

    v0: wp.spatial_matrixd
    v1: wp.spatial_matrixd


@wp.struct
class _SpatialPairEliminationFactorD:
    """The two sparse multiplier rows adjacent to one eliminated pair."""

    to_left0: wp.spatial_matrixd
    to_left1: wp.spatial_matrixd
    to_right0: wp.spatial_matrixd
    to_right1: wp.spatial_matrixd


def _register_paired_precision(matrix, vector, pair_matrix, pair_vector, pair_response, pair_factor):
    """Register both numeric layouts before CUDA capture can specialize kernels."""
    wp.overload(
        initialize_tree_backbone_edges,
        [
            int,
            wp.array[wp.int32],
            wp.array[wp.int32],
            wp.array[matrix],
            wp.array[matrix],
            wp.array[matrix],
        ],
    )
    wp.overload(
        initialize_paired_tree_backbone,
        [
            int,
            int,
            int,
            wp.array[wp.int32],
            wp.array[matrix],
            wp.array[matrix],
            wp.array[matrix],
            wp.array[vector],
            wp.array[pair_matrix],
            wp.array[pair_matrix],
            wp.array[pair_matrix],
            wp.array[pair_vector],
        ],
    )
    wp.overload(
        initialize_paired_tree_backbone_response,
        [
            int,
            int,
            int,
            wp.array[wp.int32],
            wp.array[matrix],
            wp.array[pair_response],
        ],
    )
    wp.overload(
        invert_paired_tree_backbone_cr_eliminated,
        [
            int,
            int,
            int,
            wp.array[pair_matrix],
            wp.array[pair_matrix],
            wp.array[pair_matrix],
            wp.array[pair_factor],
        ],
    )
    wp.overload(
        reduce_paired_tree_backbone_cr_in_place,
        [
            int,
            int,
            int,
            int,
            int,
            wp.array[pair_matrix],
            wp.array[pair_matrix],
            wp.array[pair_matrix],
            wp.array[pair_factor],
            wp.array[pair_vector],
            wp.array[pair_response],
        ],
    )
    wp.overload(
        reduce_paired_tree_backbone_cr_lanes,
        [
            int,
            int,
            int,
            int,
            int,
            wp.array[pair_matrix],
            wp.array[pair_matrix],
            wp.array[pair_matrix],
            wp.array[pair_factor],
            wp.array[pair_vector],
            wp.array[pair_response],
        ],
    )
    wp.overload(
        solve_paired_tree_backbone_cr_coarse,
        [
            int,
            int,
            int,
            int,
            wp.array[pair_matrix],
            wp.array[pair_matrix],
            wp.array[pair_matrix],
            wp.array[pair_vector],
            wp.array[pair_response],
        ],
    )
    wp.overload(
        solve_paired_tree_backbone_cr_coarse_lanes,
        [
            int,
            int,
            int,
            int,
            wp.array[pair_matrix],
            wp.array[pair_matrix],
            wp.array[pair_matrix],
            wp.array[pair_vector],
            wp.array[pair_response],
        ],
    )
    wp.overload(
        back_substitute_paired_tree_backbone_cr_in_place,
        [
            int,
            int,
            int,
            int,
            wp.array[pair_matrix],
            wp.array[pair_matrix],
            wp.array[pair_matrix],
            wp.array[pair_vector],
            wp.array[pair_response],
        ],
    )
    wp.overload(
        back_substitute_paired_tree_backbone_cr_lanes,
        [
            int,
            int,
            int,
            int,
            wp.array[pair_matrix],
            wp.array[pair_matrix],
            wp.array[pair_matrix],
            wp.array[pair_vector],
            wp.array[pair_response],
        ],
    )
    wp.overload(
        scatter_paired_tree_backbone_solution,
        [
            int,
            int,
            int,
            wp.array[wp.int32],
            wp.array[pair_vector],
            wp.array[pair_response],
            wp.array[vector],
            wp.array[matrix],
        ],
    )
    wp.overload(
        compute_paired_tree_backbone_residual,
        [
            int,
            int,
            int,
            wp.array[wp.int32],
            wp.array[matrix],
            wp.array[matrix],
            wp.array[matrix],
            wp.array[vector],
            wp.array[matrix],
            wp.array[pair_vector],
            wp.array[pair_response],
            wp.array[pair_vector],
            wp.array[pair_response],
        ],
    )
    wp.overload(
        scatter_refined_paired_tree_backbone_solution,
        [
            int,
            int,
            int,
            wp.array[wp.int32],
            wp.array[pair_vector],
            wp.array[pair_response],
            wp.array[pair_vector],
            wp.array[pair_response],
            wp.array[vector],
            wp.array[matrix],
        ],
    )


_register_paired_precision(
    wp.spatial_matrix,
    wp.spatial_vector,
    _SpatialPairMatrix,
    _SpatialPairVector,
    _SpatialPairResponse,
    _SpatialPairEliminationFactor,
)
_register_paired_precision(
    wp.spatial_matrixd,
    wp.spatial_vectord,
    _SpatialPairMatrixD,
    _SpatialPairVectorD,
    _SpatialPairResponseD,
    _SpatialPairEliminationFactorD,
)


@wp.kernel
def invert_tree_backbone_cr_eliminated(
    stride: int,
    row_count: int,
    eliminated_count: int,
    backbone_nodes: wp.array[wp.int32],
    diagonal: wp.array[wp.spatial_matrix],
):
    """Invert one CR level directly in the persistent tree-node workspace."""
    index = wp.tid()
    batch = index // eliminated_count
    local = index - batch * eliminated_count
    row = stride + 2 * stride * local
    if local < eliminated_count and row < row_count:
        node = backbone_nodes[batch * row_count + row]
        diagonal[node] = _inverse_spatial_robust(diagonal[node])


@wp.kernel
def reduce_tree_backbone_cr_in_place(
    stride: int,
    row_count: int,
    survivor_count: int,
    closure_count: int,
    backbone_nodes: wp.array[wp.int32],
    lower: wp.array[wp.spatial_matrix],
    diagonal: wp.array[wp.spatial_matrix],
    upper: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
    response_rhs: wp.array[wp.spatial_matrix],
):
    """Reduce the primary and closure-response right-hand sides together."""
    index = wp.tid()
    batch = index // survivor_count
    local = index - batch * survivor_count
    base = batch * row_count
    local_row = 2 * stride * local
    if local >= survivor_count or local_row >= row_count:
        return
    row = base + local_row
    node = backbone_nodes[row]

    lower_value = wp.spatial_matrix(0.0)
    diagonal_value = diagonal[node]
    upper_value = wp.spatial_matrix(0.0)
    rhs_value = rhs[node]
    if local_row >= stride:
        eliminated_row = local_row - stride
        eliminated_index = base + eliminated_row
        eliminated = backbone_nodes[eliminated_index]
        factor = lower[row] * diagonal[eliminated]
        lower_value = -(factor * lower[eliminated_index])
        diagonal_value = diagonal_value - factor * upper[eliminated_index]
        rhs_value = rhs_value - factor * rhs[eliminated]
        for closure in range(closure_count):
            response_index = node * closure_count + closure
            eliminated_response_index = eliminated * closure_count + closure
            response_rhs[response_index] = (
                response_rhs[response_index] - factor * response_rhs[eliminated_response_index]
            )
    if local_row + stride < row_count:
        eliminated_row = local_row + stride
        eliminated_index = base + eliminated_row
        eliminated = backbone_nodes[eliminated_index]
        factor = upper[row] * diagonal[eliminated]
        diagonal_value = diagonal_value - factor * lower[eliminated_index]
        rhs_value = rhs_value - factor * rhs[eliminated]
        for closure in range(closure_count):
            response_index = node * closure_count + closure
            eliminated_response_index = eliminated * closure_count + closure
            response_rhs[response_index] = (
                response_rhs[response_index] - factor * response_rhs[eliminated_response_index]
            )
        if local_row + 2 * stride < row_count:
            upper_value = -(factor * upper[eliminated_index])

    lower[row] = lower_value
    diagonal[node] = diagonal_value
    upper[row] = upper_value
    rhs[node] = rhs_value


@wp.kernel
def solve_tree_backbone_cr_coarse(
    stride: int,
    row_count: int,
    closure_count: int,
    backbone_nodes: wp.array[wp.int32],
    lower: wp.array[wp.spatial_matrix],
    diagonal: wp.array[wp.spatial_matrix],
    upper: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
    response_rhs: wp.array[wp.spatial_matrix],
):
    """Finish the reduced tree backbone with one small block-Thomas solve."""
    base = wp.tid() * row_count
    active_count = (row_count + stride - 1) // stride
    for local in range(1, active_count):
        previous_row = (local - 1) * stride
        row = local * stride
        previous = backbone_nodes[base + previous_row]
        node = backbone_nodes[base + row]
        diagonal[previous] = _inverse_spatial_robust(diagonal[previous])
        multiplier = lower[base + row] * diagonal[previous]
        diagonal[node] = diagonal[node] - multiplier * upper[base + previous_row]
        rhs[node] = rhs[node] - multiplier * rhs[previous]
        for closure in range(closure_count):
            response_index = node * closure_count + closure
            previous_response_index = previous * closure_count + closure
            response_rhs[response_index] = (
                response_rhs[response_index] - multiplier * response_rhs[previous_response_index]
            )

    last_row = (active_count - 1) * stride
    last = backbone_nodes[base + last_row]
    diagonal[last] = _inverse_spatial_robust(diagonal[last])
    rhs[last] = diagonal[last] * rhs[last]
    for closure in range(closure_count):
        response_index = last * closure_count + closure
        response_rhs[response_index] = diagonal[last] * response_rhs[response_index]

    for reverse_local in range(1, active_count):
        row = (active_count - 1 - reverse_local) * stride
        following_row = row + stride
        node = backbone_nodes[base + row]
        following = backbone_nodes[base + following_row]
        rhs[node] = diagonal[node] * (rhs[node] - upper[base + row] * rhs[following])
        for closure in range(closure_count):
            response_index = node * closure_count + closure
            following_response_index = following * closure_count + closure
            response_rhs[response_index] = diagonal[node] * (
                response_rhs[response_index] - upper[base + row] * response_rhs[following_response_index]
            )


@wp.kernel
def back_substitute_tree_backbone_cr_in_place(
    stride: int,
    row_count: int,
    eliminated_count: int,
    closure_count: int,
    backbone_nodes: wp.array[wp.int32],
    lower: wp.array[wp.spatial_matrix],
    upper: wp.array[wp.spatial_matrix],
    diagonal: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
    response_rhs: wp.array[wp.spatial_matrix],
):
    """Recover one eliminated backbone level for every right-hand side."""
    index = wp.tid()
    batch = index // eliminated_count
    local = index - batch * eliminated_count
    base = batch * row_count
    local_row = stride + 2 * stride * local
    if local >= eliminated_count or local_row >= row_count:
        return
    row = base + local_row
    node = backbone_nodes[row]
    previous = backbone_nodes[base + local_row - stride]
    rhs_value = rhs[node] - lower[row] * rhs[previous]
    if local_row + stride < row_count:
        following = backbone_nodes[base + local_row + stride]
        rhs_value = rhs_value - upper[row] * rhs[following]
    rhs[node] = diagonal[node] * rhs_value

    for closure in range(closure_count):
        response_index = node * closure_count + closure
        previous_response_index = previous * closure_count + closure
        response_value = response_rhs[response_index] - lower[row] * response_rhs[previous_response_index]
        if local_row + stride < row_count:
            following = backbone_nodes[base + local_row + stride]
            following_response_index = following * closure_count + closure
            response_value = response_value - upper[row] * response_rhs[following_response_index]
        response_rhs[response_index] = diagonal[node] * response_value


@wp.kernel
def compute_path_correction(
    body_ids: wp.array[wp.int32],
    body_slot_by_id: wp.array[wp.int32],
    body_incident_rows: wp.array[wp.vec2i],
    row_active: wp.array[wp.int32],
    jacobian_parent: wp.array[wp.spatial_matrix],
    jacobian_child: wp.array[wp.spatial_matrix],
    multiplier: wp.array[wp.spatial_vector],
    body_inverse: wp.array[wp.spatial_matrix],
    body_free: wp.array[wp.spatial_vector],
    body_inv_mass: wp.array[float],
    body_correction: wp.array[wp.spatial_vector],
):
    """Map solved path multipliers back to one coherent body correction.

    Incident codes store ``(bucket_row << 1) | side`` for at most two rows.
    """
    index = wp.tid()
    body = body_ids[index]
    slot = body_slot_by_id[body]
    if body_inv_mass[body] <= 0.0:
        return

    reaction = wp.spatial_vector()
    active = int(0)
    incident_rows = body_incident_rows[index]
    for incident in range(2):
        code = incident_rows[incident]
        if code >= 0:
            row = code // 2
            jacobian = jacobian_parent[row] if code % 2 == 0 else jacobian_child[row]
            reaction = reaction + wp.transpose(jacobian) * multiplier[row]
            active = active + row_active[row]
    if active == 0:
        body_correction[slot] = wp.spatial_vector()
        return

    body_correction[slot] = body_free[slot] - body_inverse[slot] * reaction


@wp.func
def _corrected_pose(
    pose: wp.transform,
    correction: wp.spatial_vector,
    com_local: wp.vec3,
    alpha: float,
):
    # Preserve represented fixed/unchanged poses, including offset COMs, exactly.
    if alpha == 0.0 or (
        correction[0] == 0.0
        and correction[1] == 0.0
        and correction[2] == 0.0
        and correction[3] == 0.0
        and correction[4] == 0.0
        and correction[5] == 0.0
    ):
        return pose
    linear = alpha * wp.spatial_top(correction)
    angular = alpha * wp.spatial_bottom(correction)
    rotation = wp.transform_get_rotation(pose)

    angle = wp.length(angular)
    delta_rotation = wp.quat_identity()
    if angle > _SMALL_ANGLE_EPS:
        delta_rotation = wp.quat_from_axis_angle(angular / angle, angle)
    else:
        half = 0.5 * angular
        delta_rotation = wp.normalize(wp.quat(half[0], half[1], half[2], 1.0))
    rotation_new = wp.normalize(delta_rotation * rotation)
    dx_pose = linear + (wp.quat_rotate(rotation, com_local) - wp.quat_rotate(rotation_new, com_local))
    position_new = wp.transform_get_translation(pose) + dx_pose
    return wp.transform(position_new, rotation_new)


@wp.kernel
def limit_dynamic_contact_jacobi_step(
    body_ids: wp.array[wp.int32],
    graph_body_island: wp.array[wp.int32],
    island_contact_state: wp.array[wp.int32],
    correction: wp.array[wp.spatial_vector],
    dynamic_contact_hessian: wp.array[wp.spatial_matrix],
    dt: float,
    body_q: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    island_step_scale: wp.array[float],
):
    """Relax a structural correction by its dynamic-contact overlap.

    The global structural islands are solved concurrently.  At an active
    dynamic pair, each endpoint therefore responds as if the other endpoint
    were fixed.  This is block Jacobi on the omitted contact edge.  Weight its
    standard half relaxation by the fraction of the proposed correction's
    local quadratic curvature that comes from those dynamic contacts:

        omega = 1 / (1 + contact_curvature / total_curvature).

    The coefficient is one without dynamic-contact influence and approaches
    one half when contact dominates.  It contains no scene scale or authored
    tolerance, and local VBD relinearizes the nonlinear contact afterward.
    """
    slot = wp.tid()
    island = graph_body_island[slot]
    if island_contact_state[island] != -1:
        return

    body = body_ids[slot]
    value = correction[slot]
    linear = wp.spatial_top(value)
    angular = wp.spatial_bottom(value)
    contact_hessian_ll = _spatial_block(dynamic_contact_hessian[slot], 0, 0)
    contact_hessian_al = _spatial_block(dynamic_contact_hessian[slot], 1, 0)
    contact_hessian_aa = _spatial_block(dynamic_contact_hessian[slot], 1, 1)
    contact_curvature = 2.0 * (
        wp.dot(linear, contact_hessian_ll * linear)
        + 2.0 * wp.dot(angular, contact_hessian_al * linear)
        + wp.dot(angular, contact_hessian_aa * angular)
    )
    contact_curvature = wp.max(contact_curvature, 0.0)

    dt_inv_sq = 1.0 / (dt * dt)
    rotation = wp.quat_to_matrix(wp.transform_get_rotation(body_q[body]))
    inertia_world = rotation * body_inertia[body] * wp.transpose(rotation)
    inertial_curvature = dt_inv_sq * (
        body_mass[body] * wp.dot(linear, linear) + wp.dot(angular, inertia_world * angular)
    )
    total_curvature = inertial_curvature + contact_curvature
    if total_curvature > 0.0:
        overlap = wp.clamp(contact_curvature / total_curvature, 0.0, 1.0)
        wp.atomic_min(island_step_scale, island, 1.0 / (1.0 + overlap))


@wp.kernel
def refresh_translation_freedom(
    candidate: wp.array[int],
    anchor_offsets: wp.array[int],
    anchor_joints: wp.array[int],
    joint_enabled: wp.array[bool],
    translation_free: wp.array[int],
    active: wp.array[int],
):
    """Read live anchor enables without rebuilding the structural graph."""
    island = wp.tid()
    free = candidate[island]
    for index in range(anchor_offsets[island], anchor_offsets[island + 1]):
        if joint_enabled[anchor_joints[index]]:
            free = 0
    translation_free[island] = free
    if free != 0:
        wp.atomic_max(active, 0, 1)


@wp.kernel
def accumulate_free_translation_system(
    body_ids: wp.array[int],
    body_island: wp.array[int],
    body_slot: wp.array[int],
    translation_free: wp.array[int],
    island_state: wp.array[int],
    step_scale: wp.array[float],
    correction: wp.array[wp.spatial_vector],
    dt: float,
    body_q: wp.array[wp.transform],
    body_inertia_q: wp.array[wp.transform],
    body_disp: wp.array[wp.vec3],
    body_disp_ang: wp.array[wp.vec3],
    use_displacement: int,
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[float],
    body_inv_mass: wp.array[float],
    contact_hessian_ll: wp.array[wp.mat33],
    contact_hessian_al: wp.array[wp.mat33],
    contact_forces: wp.array[wp.vec3],
    dynamic_contact_hessian: wp.array[wp.spatial_matrix],
    has_contacts: int,
    contact_count: wp.array[int],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    shape_body: wp.array[int],
    contact_buffer_size: int,
    body_contact_counts: wp.array[int],
    body_contact_indices: wp.array[int],
    system: wp.array[wp.mat44d],
):
    """Restrict the physical quadratic to an island's rigid translation.

    Internal joints and contacts do no work in this subspace. Their diagonal
    solve majorizers must not damp it. External dynamic contacts disable this
    independent correction because their relative motion couples islands.
    """
    slot = wp.tid()
    island = body_island[slot]
    if translation_free[island] == 0 or island_state[island] < -1:
        return
    body = body_ids[slot]
    contribution = wp.mat44d(0.0)
    if has_contacts != 0:
        for i in range(wp.min(body_contact_counts[body], contact_buffer_size)):
            contact = body_contact_indices[body * contact_buffer_size + i]
            if contact >= contact_count[0]:
                continue
            shape0, shape1 = contact_shape0[contact], contact_shape1[contact]
            body0 = shape_body[shape0] if shape0 >= 0 else -1
            body1 = shape_body[shape1] if shape1 >= 0 else -1
            other = body1 if body == body0 else body0
            if other >= 0 and body_inv_mass[other] > 0.0:
                other_slot = body_slot[other]
                if other_slot < 0 or body_island[other_slot] != island:
                    contribution[3, 3] = wp.float64(1.0)
                    wp.atomic_add(system, island, contribution)
                    return

    com_world = wp.transform_point(body_q[body], body_com[body])
    inertial_delta = _pose_com_diff(body_inertia_q[body], body_q[body], body_com[body])
    if use_displacement != 0:
        inertial_delta = -body_disp[body]
    mass = wp.float64(body_mass[body]) / (wp.float64(dt) * wp.float64(dt))
    for col in range(3):
        contribution[3, col] = mass * wp.float64(wp.abs(com_world[col]))
    delta = correction[slot] * step_scale[island]
    h_ll, h_al = contact_hessian_ll[body], contact_hessian_al[body]
    dynamic = dynamic_contact_hessian[slot]
    for row in range(3):
        rhs = mass * wp.float64(inertial_delta[row]) + wp.float64(contact_forces[body][row])
        for column in range(3):
            # Subtract before adding inertia, in double precision: internal
            # contact curvature can be much larger than the island's mass.
            value = wp.float64(h_ll[row, column]) - wp.float64(dynamic[row, column])
            if row == column:
                value += mass
            contribution[row, column] = value
            rhs -= value * wp.float64(delta[column])
            angular = wp.float64(h_al[column, row]) - wp.float64(dynamic[column + 3, row])
            rhs -= angular * wp.float64(delta[column + 3])
        contribution[row, 3] = rhs
    wp.atomic_add(system, island, contribution)


@wp.kernel
def correct_free_translation(
    body_island: wp.array[int],
    translation_free: wp.array[int],
    island_state: wp.array[int],
    step_scale: wp.array[float],
    system: wp.array[wp.mat44d],
    correction: wp.array[wp.spatial_vector],
    apply_translation: bool,
    track_pending: bool,
    stop_at_roundoff: bool,
    carry_remainder: bool,
    support: wp.array[_TranslationSupport],
    body_ids: wp.array[int],
    use_displacement: int,
    body_com: wp.array[wp.vec3],
    body_target_q: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    pending: wp.array[int],
    remainder: wp.array[wp.vec4],
    trial: wp.array[wp.vec4d],
    body_disp: wp.array[wp.vec3],
):
    """Minimize common translation, optionally applying its isolated update."""
    slot = wp.tid()
    island = body_island[slot]
    scale = step_scale[island]
    packed = system[island]
    # Per body and pass: the unstored tangential shift the system was evaluated with, and whether friction
    # could not balance the island when the pass started.
    carried = wp.vec3d(0.0)
    overloaded = bool(False)
    if apply_translation:
        if carry_remainder:
            state = remainder[slot]
            carried = wp.vec3d(wp.float64(state[0]), wp.float64(state[1]), wp.float64(state[2]))
            overloaded = state[3] != 0.0
        else:
            remainder[slot] = wp.vec4(0.0)
    if translation_free[island] == 0 or island_state[island] < -1 or packed[3, 3] != wp.float64(0.0) or scale <= 0.0:
        return
    matrix = wp.mat33d(0.0)
    rhs = wp.vec3d(0.0)
    for row in range(3):
        rhs[row] = packed[row, 3]
        for column in range(3):
            matrix[row, column] = packed[row, column]
    shift = wp.inverse(matrix) * rhs
    if apply_translation and not carry_remainder:
        # With one support plane, friction cannot balance the island at the pose the pass starts from once
        # the tangential part of the rest of its right-hand side exceeds the Coulomb capacity of its contacts.
        # A tangential shift below the resolution of the positions must then not round away: the following
        # contact evaluations of this pass read the unstored part instead of recomputing the same lost step
        # against stick curvature. Decided once per pass; a trigger, not a convergence certificate.
        supported = support[island]
        capacity = supported[3]
        normal = wp.vec3d(supported[4], supported[5], supported[6])
        aligned = wp.dot(normal, normal)
        if capacity > wp.float64(0.0) and aligned >= wp.float64(0.999999) * capacity * capacity:
            demand = wp.vec3d(rhs[0] - supported[0], rhs[1] - supported[1], rhs[2] - supported[2])
            tangential = demand - normal * (wp.dot(normal, demand) / aligned)
            if wp.dot(tangential, tangential) > capacity * capacity:
                overloaded = bool(True)
                remainder[slot] = wp.vec4(0.0, 0.0, 0.0, 1.0)
    if apply_translation and stop_at_roundoff and not overloaded:
        # A coordinate is at roundoff when neither the island residual nor the solved step taken
        # through the metric's own diagonal exceeds a relative tolerance of 2^-22 (two float32
        # epsilons) on the island's inertia-weighted coordinates (packed in row 3, cols 0..2).
        # The step test keeps a correction that off-diagonal contact curvature moves onto
        # another coordinate.
        tolerance = wp.float64(2.384185791015625e-7)
        kept = shift
        moving = int(0)
        for axis in range(3):
            if wp.max(wp.abs(rhs[axis]), wp.abs(matrix[axis, axis] * shift[axis])) <= tolerance * packed[3, axis]:
                kept[axis] = wp.float64(0.0)
            else:
                moving += 1
        # Stop when every coordinate is at roundoff, and when the step without the coordinates
        # at roundoff, a model of what a coarse position keeps, does not lower the quadratic.
        if moving == 0 or wp.dot(rhs, kept) <= wp.float64(0.5) * wp.dot(kept, matrix * kept):
            trial[slot] = wp.vec4d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.dot(rhs, rhs))
            return
    if apply_translation:
        # Contacts open and close and friction saturates between evaluations, and undamped Newton steps
        # can cycle between such states. A step after which the island's residual grew is withdrawn by
        # half; the entry holds that solved step and the squared residual before it. A heuristic: the
        # last evaluation's step stays unverified, and a withdrawal below position resolution is lost.
        norm = wp.dot(rhs, rhs)
        previous = trial[slot]
        if carry_remainder and norm > previous[3]:
            half = wp.float64(0.5)
            shift = wp.vec3d(-half * previous[0], -half * previous[1], -half * previous[2])
            trial[slot] = wp.vec4d(half * previous[0], half * previous[1], half * previous[2], previous[3])
        else:
            trial[slot] = wp.vec4d(shift[0], shift[1], shift[2], norm)
    if wp.isfinite(shift[0]) and wp.isfinite(shift[1]) and wp.isfinite(shift[2]):
        if apply_translation:
            # This path solves only common translation, with unit step scale
            # and zero base correction. Keep the correction buffer zero for
            # the next restricted solve instead of writing then clearing it.
            body = body_ids[slot]
            pose = body_q[body]
            position = wp.transform_get_translation(pose)
            wanted = shift + carried
            translated = position + wp.vec3(float(wanted[0]), float(wanted[1]), float(wanted[2]))
            if use_displacement != 0:
                # The solved shift goes into the displacement and the position is rebuilt from the
                # inertial target; the unstored part below is what that one rounding leaves.
                body_disp[body] = body_disp[body] + wp.vec3(float(shift[0]), float(shift[1]), float(shift[2]))
                translated = wp.transform_get_translation(
                    pose_from_displacement(
                        body_target_q[body], body_com[body], body_disp[body], wp.transform_get_rotation(pose)
                    )
                )
            body_q[body] = wp.transform(translated, wp.transform_get_rotation(pose))
            if overloaded:
                plane = support[island]
                unstored = wanted - (wp.vec3d(translated) - wp.vec3d(position))
                across = wp.vec3d(plane[4], plane[5], plane[6])
                unstored -= across * (wp.dot(across, unstored) / wp.dot(across, across))
                remainder[slot] = wp.vec4(float(unstored[0]), float(unstored[1]), float(unstored[2]), 1.0)
            if track_pending and stop_at_roundoff:
                wp.atomic_max(pending, 0, 1)
        else:
            value = correction[slot]
            for axis in range(3):
                value[axis] += float(shift[axis]) / scale
            correction[slot] = value


@wp.kernel
def _finish_translation_iteration(remaining: wp.array[int], pending: wp.array[int]):
    """Bound a captured reconciliation loop without changing its numerical stop."""
    remaining[0] -= 1
    if remaining[0] == 0:
        pending[0] = 0


@wp.func
def _free_rigid_basis(offset: wp.vec3d):
    basis = wp.spatial_matrixd(0.0)
    for axis in range(6):
        basis[axis, axis] = wp.float64(1.0)
    cross = -wp.skew(offset)
    for row in range(3):
        for column in range(3):
            basis[row, column + 3] = cross[row, column]
    return basis


@wp.kernel
def accumulate_free_rigid_center(
    body_ids: wp.array[int],
    body_island: wp.array[int],
    free: wp.array[int],
    candidate: wp.array[int],
    state: wp.array[int],
    translation_system: wp.array[wp.mat44d],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[float],
    center: wp.array[wp.vec4d],
):
    slot = wp.tid()
    island = body_island[slot]
    # An active contact pair inside the island does no work under collective rigid motion
    # and must not disqualify it; external dynamic contact blocks translation_system[3, 3].
    if (
        candidate[island] == 0
        or free[island] == 0
        or state[island] < -1
        or translation_system[island][3, 3] != wp.float64(0.0)
    ):
        return
    body = body_ids[slot]
    mass = wp.float64(body_mass[body])
    position = wp.vec3d(wp.transform_get_translation(body_q[body])) + wp.vec3d(
        wp.quat_rotate(wp.transform_get_rotation(body_q[body]), body_com[body])
    )
    wp.atomic_add(center, island, wp.vec4d(mass * position[0], mass * position[1], mass * position[2], mass))


@wp.kernel
def accumulate_free_rigid_system(
    body_ids: wp.array[int],
    body_island: wp.array[int],
    island_state: wp.array[int],
    center: wp.array[wp.vec4d],
    scale: wp.array[float],
    correction: wp.array[wp.spatial_vector],
    dt: float,
    body_q: wp.array[wp.transform],
    body_inertia_q: wp.array[wp.transform],
    body_disp: wp.array[wp.vec3],
    body_disp_ang: wp.array[wp.vec3],
    use_displacement: int,
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[float],
    body_inv_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    contact_hessian_ll: wp.array[wp.mat33],
    contact_hessian_al: wp.array[wp.mat33],
    contact_hessian_aa: wp.array[wp.mat33],
    contact_forces: wp.array[wp.vec3],
    contact_torques: wp.array[wp.vec3],
    dynamic_hessian: wp.array[wp.spatial_matrix],
    has_contacts: int,
    contact_damping_bidirectional: bool,
    contact_normal_rho: wp.array[float],
    contact_material_ke: wp.array[float],
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[wp.vec3],
    contact_tangent_rho: wp.array[wp.mat22],
    contact_torsional_rho: wp.array[float],
    contact_rolling_rho: wp.array[wp.mat22],
    contact_lambda: wp.array[wp.vec3],
    contact_lambda_angular: wp.array[wp.vec3],
    contact_C0: wp.array[wp.vec3],
    stab_alpha: float,
    contact_count: wp.array[int],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_surface_velocity: wp.array[wp.vec3],
    contact_offset0: wp.array[wp.vec3],
    contact_offset1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_margin0: wp.array[float],
    contact_margin1: wp.array[float],
    shape_body: wp.array[int],
    contact_buffer_size: int,
    body_contact_counts: wp.array[int],
    body_contact_indices: wp.array[int],
    matrix: wp.array[wp.spatial_matrixd],
    rhs: wp.array[wp.spatial_vectord],
):
    slot = wp.tid()
    island = body_island[slot]
    packed = center[island]
    if packed[3] <= wp.float64(0.0) or scale[island] <= 0.0:
        return
    body = body_ids[slot]
    pose = body_q[body]
    position = wp.vec3d(wp.transform_get_translation(pose)) + wp.vec3d(
        wp.quat_rotate(wp.transform_get_rotation(pose), body_com[body])
    )
    origin = wp.vec3d(packed[0], packed[1], packed[2]) / packed[3]
    basis = _free_rigid_basis(position - origin)
    inv_dt_sq = 1.0 / (dt * dt)
    if island_state[island] == 1 or has_contacts == 0:
        # Fast bitwise path when the island has no active internal dynamic pairs.
        metric = _body_surrogate_matrix(
            wp.transform_get_rotation(pose),
            body_mass[body],
            body_inertia[body],
            contact_hessian_ll[body],
            contact_hessian_al[body],
            contact_hessian_aa[body],
            dynamic_hessian[slot],
            inv_dt_sq,
        )
        force = _body_objective_rhs(
            pose,
            body_inertia_q[body],
            body_mass[body],
            body_inertia[body],
            body_com[body],
            contact_forces[body],
            contact_torques[body],
            inv_dt_sq,
        )
        if use_displacement != 0:
            force = _body_objective_rhs_displacement(
                pose,
                body_mass[body],
                body_inertia[body],
                contact_forces[body],
                contact_torques[body],
                inv_dt_sq,
                body_disp[body],
                body_disp_ang[body],
            )
        metric_d = wp.spatial_matrixd(metric)
    else:
        # Eligible island with active internal self-contact: evaluate ONLY external (static/kinematic)
        # contacts in registers in FP64. Zero extra launches, zero scratch mutation, zero inertia
        # cancellation, and exact linear & angular momentum conservation.
        ext_f = wp.vec3(0.0)
        ext_tau = wp.vec3(0.0)
        ext_ll = wp.mat33d(0.0)
        ext_al = wp.mat33d(0.0)
        ext_aa = wp.mat33d(0.0)
        n_contacts = wp.min(body_contact_counts[body], contact_buffer_size)
        c_total = contact_count[0]
        for idx in range(n_contacts):
            c = body_contact_indices[body * contact_buffer_size + idx]
            if c >= c_total:
                continue
            s0, s1 = contact_shape0[c], contact_shape1[c]
            b0 = shape_body[s0] if s0 >= 0 else -1
            b1 = shape_body[s1] if s1 >= 0 else -1
            if b0 != body and b1 != body:
                continue
            other = b1 if body == b0 else b0
            if other >= 0 and body_inv_mass[other] > 0.0:
                continue
            p0_loc = contact_point0[c]
            p1_loc = contact_point1[c]
            nrm = contact_normal[c]
            p0 = wp.transform_point(body_q[b0], p0_loc) if b0 >= 0 else p0_loc
            p1 = wp.transform_point(body_q[b1], p1_loc) if b1 >= 0 else p1_loc
            n_err = -contact_surface_separation(p0, p1, nrm, contact_margin0[c], contact_margin1[c])
            n_weight = contact_normal_rho[c]
            mat_k = contact_material_ke[c]
            mult = contact_lambda[c]
            mult_n = wp.dot(mult, nrm)
            c0_val = contact_C0[c]
            c0_n = wp.dot(nrm, c0_val)
            stab_err = n_err - c0_n
            tan_c0 = (1.0 - stab_alpha) * (c0_val - nrm * c0_n)
            n_pk, mult_n_eff = _material_force_terms(n_weight, mat_k, mult_n)
            if mult_n <= 0.0 and (n_err <= _SMALL_LENGTH_EPS or n_pk * stab_err + mult_n_eff <= 0.0):
                continue
            surf_vel = wp.vec3(0.0)
            if contact_surface_velocity:
                surf_vel = contact_surface_velocity[c]
            friction_mu = contact_material_mu[c]
            f0, t0, hll0, hal0, haa0, f1, t1, hll1, hal1, haa1 = evaluate_rigid_contact_normal_sliding(
                b0,
                b1,
                body_q,
                body_q_prev,
                body_com,
                p0_loc,
                p1_loc,
                surf_vel,
                contact_offset0[c],
                contact_offset1[c],
                nrm,
                stab_err,
                n_weight,
                mat_k,
                contact_tangent_rho[c],
                contact_material_kd[c],
                mult,
                friction_mu[0],
                dt,
                tan_c0,
                contact_damping_bidirectional,
            )
            mu_torsional = friction_mu[1]
            mu_rolling = friction_mu[2]
            normal_load = wp.max(n_pk * stab_err + mult_n_eff, 0.0)
            if normal_load > 0.0 and (mu_torsional > 0.0 or mu_rolling > 0.0):
                pure_torque_b, k_angular = evaluate_rigid_contact_angular_friction(
                    b0,
                    b1,
                    body_q,
                    body_q_prev,
                    nrm,
                    normal_load,
                    mu_torsional,
                    mu_rolling,
                    contact_torsional_rho[c],
                    contact_rolling_rho[c],
                    contact_lambda_angular[c],
                )
                t0 -= pure_torque_b
                t1 += pure_torque_b
                haa0 += k_angular
                haa1 += k_angular
            if body == b0:
                ext_f += f0
                ext_tau += t0
                ext_ll += wp.mat33d(hll0)
                ext_al += wp.mat33d(hal0)
                ext_aa += wp.mat33d(haa0)
            else:
                ext_f += f1
                ext_tau += t1
                ext_ll += wp.mat33d(hll1)
                ext_al += wp.mat33d(hal1)
                ext_aa += wp.mat33d(haa1)
        inv_dt2_d = wp.float64(1.0) / (wp.float64(dt) * wp.float64(dt))
        rot_d = wp.mat33d(wp.quat_to_matrix(wp.transform_get_rotation(pose)))
        iner_d = inv_dt2_d * rot_d * wp.mat33d(body_inertia[body]) * wp.transpose(rot_d)
        mass_d = wp.float64(body_mass[body]) * inv_dt2_d
        h_ll_d = mass_d * wp.identity(3, wp.float64) + ext_ll
        h_aa_d = iner_d + ext_aa
        eps_l_d = wp.float64(1.0e-9) * (
            (h_ll_d[0, 0] + h_ll_d[1, 1] + h_ll_d[2, 2]) / wp.float64(3.0) + wp.float64(1.0)
        )
        eps_a_d = wp.float64(1.0e-9) * (
            (h_aa_d[0, 0] + h_aa_d[1, 1] + h_aa_d[2, 2]) / wp.float64(3.0) + wp.float64(1.0)
        )
        metric_d = wp.spatial_matrixd(0.0)
        for r_i in range(3):
            for c_i in range(3):
                metric_d[r_i, c_i] = h_ll_d[r_i, c_i] + (eps_l_d if r_i == c_i else wp.float64(0.0))
                metric_d[r_i, c_i + 3] = ext_al[c_i, r_i]
                metric_d[r_i + 3, c_i] = ext_al[r_i, c_i]
                metric_d[r_i + 3, c_i + 3] = h_aa_d[r_i, c_i] + (eps_a_d if r_i == c_i else wp.float64(0.0))
        force = _body_objective_rhs(
            pose,
            body_inertia_q[body],
            body_mass[body],
            body_inertia[body],
            body_com[body],
            ext_f,
            ext_tau,
            inv_dt_sq,
        )
        if use_displacement != 0:
            force = _body_objective_rhs_displacement(
                pose,
                body_mass[body],
                body_inertia[body],
                ext_f,
                ext_tau,
                inv_dt_sq,
                body_disp[body],
                body_disp_ang[body],
            )
    current = wp.spatial_vectord(correction[slot]) * wp.float64(scale[island])
    defect = wp.spatial_vectord(force) - metric_d * current
    wp.atomic_add(matrix, island, wp.transpose(basis) * metric_d * basis)
    wp.atomic_add(rhs, island, wp.transpose(basis) * defect)


@wp.kernel
def solve_free_rigid_system(
    center: wp.array[wp.vec4d],
    matrix: wp.array[wp.spatial_matrixd],
    rhs: wp.array[wp.spatial_vectord],
    result: wp.array[wp.spatial_vectord],
):
    island = wp.tid()
    result[island] = wp.spatial_vectord(0.0)
    if center[island][3] <= wp.float64(0.0):
        return
    lower = wp.spatial_matrixd(0.0)
    for row in range(6):
        for column in range(row + 1):
            value = matrix[island][row, column]
            for k in range(column):
                value -= lower[row, k] * lower[column, k]
            if row == column:
                if not wp.isfinite(value) or value <= wp.float64(0.0):
                    return
                lower[row, column] = wp.sqrt(value)
            else:
                lower[row, column] = value / lower[column, column]
    solution = wp.spatial_vectord(0.0)
    for row in range(6):
        value = rhs[island][row]
        for k in range(row):
            value -= lower[row, k] * solution[k]
        solution[row] = value / lower[row, row]
    for reverse in range(6):
        row = 5 - reverse
        value = solution[row]
        for k in range(row + 1, 6):
            value -= lower[k, row] * solution[k]
        solution[row] = value / lower[row, row]
        if not wp.isfinite(solution[row]):
            return
    result[island] = solution


@wp.kernel
def enrich_free_rigid_correction(
    body_ids: wp.array[int],
    body_island: wp.array[int],
    center: wp.array[wp.vec4d],
    shift: wp.array[wp.spatial_vectord],
    scale: wp.array[float],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    correction: wp.array[wp.spatial_vector],
):
    slot = wp.tid()
    island = body_island[slot]
    packed = center[island]
    if packed[3] <= wp.float64(0.0) or scale[island] <= 0.0:
        return
    body = body_ids[slot]
    position = wp.vec3d(wp.transform_get_translation(body_q[body])) + wp.vec3d(
        wp.quat_rotate(wp.transform_get_rotation(body_q[body]), body_com[body])
    )
    origin = wp.vec3d(packed[0], packed[1], packed[2]) / packed[3]
    update = _free_rigid_basis(position - origin) * shift[island]
    value = correction[slot]
    for axis in range(6):
        value[axis] = float(wp.float64(value[axis]) + update[axis] / wp.float64(scale[island]))
    correction[slot] = value


@wp.kernel
def exclude_supported_rigid_centers(
    body_ids: wp.array[int],
    body_island: wp.array[int],
    body_q: wp.array[wp.transform],
    body_inv_mass: wp.array[float],
    contact_lambda: wp.array[wp.vec3],
    contact_count: wp.array[int],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_margin0: wp.array[float],
    contact_margin1: wp.array[float],
    shape_body: wp.array[int],
    contact_buffer_size: int,
    body_contact_counts: wp.array[int],
    body_contact_indices: wp.array[int],
    center: wp.array[wp.vec4d],
):
    """Drop islands that touch a static or kinematic body from the post-sweep rigid reconciliation.

    Such an island keeps its translation pass, with its friction capacity and stop rules. The
    activity test is the one of the island classification.
    """
    slot = wp.tid()
    body = body_ids[slot]
    total = contact_count[0]
    for i in range(wp.min(body_contact_counts[body], contact_buffer_size)):
        contact = body_contact_indices[body * contact_buffer_size + i]
        if contact >= total:
            continue
        shape0 = contact_shape0[contact]
        shape1 = contact_shape1[contact]
        body0 = shape_body[shape0] if shape0 >= 0 else -1
        body1 = shape_body[shape1] if shape1 >= 0 else -1
        other = body1 if body == body0 else body0
        if other >= 0 and body_inv_mass[other] > 0.0:
            continue
        normal = contact_normal[contact]
        point0 = contact_point0[contact]
        if body0 >= 0:
            point0 = wp.transform_point(body_q[body0], point0)
        point1 = contact_point1[contact]
        if body1 >= 0:
            point1 = wp.transform_point(body_q[body1], point1)
        separation = wp.dot(normal, point1 - point0) - contact_margin0[contact] - contact_margin1[contact]
        if separation > 0.0 and wp.dot(contact_lambda[contact], normal) <= 0.0:
            continue
        center[body_island[slot]] = wp.vec4d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
        return


@wp.kernel
def apply_free_rigid_reconciliation(
    body_ids: wp.array[int],
    body_island: wp.array[int],
    center: wp.array[wp.vec4d],
    correction: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    use_displacement: int,
    body_target_q: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_disp: wp.array[wp.vec3],
    body_disp_ang: wp.array[wp.vec3],
):
    """Apply the collective rigid motion of a free island after the local sweeps."""
    slot = wp.tid()
    if center[body_island[slot]][3] <= wp.float64(0.0):
        return
    body = body_ids[slot]
    body_q[body] = _corrected_pose(body_q[body], correction[slot], body_com[body], 1.0)
    if use_displacement != 0:
        delta = correction[slot]
        if _spatial_vector_nonzero(delta):
            body_disp[body] = body_disp[body] + wp.spatial_top(delta)
            body_disp_ang[body] = rotation_vector_compose(wp.spatial_bottom(delta), body_disp_ang[body])
            body_q[body] = pose_from_displacement(
                body_target_q[body], body_com[body], body_disp[body], wp.transform_get_rotation(body_q[body])
            )


@wp.kernel
def apply_global_correction(
    body_ids: wp.array[wp.int32],
    graph_body_island: wp.array[wp.int32],
    island_contact_state: wp.array[wp.int32],
    correction: wp.array[wp.spatial_vector],
    island_step_scale: wp.array[float],
    body_com: wp.array[wp.vec3],
    use_displacement: int,
    body_target_q: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_disp: wp.array[wp.vec3],
    body_disp_ang: wp.array[wp.vec3],
):
    slot = wp.tid()
    body = body_ids[slot]
    island = graph_body_island[slot]
    if island_contact_state[island] < -1:
        return
    scale = island_step_scale[island]
    if scale == 0.0 or not _spatial_vector_nonzero(correction[slot]):
        return
    body_q[body] = _corrected_pose(body_q[body], correction[slot], body_com[body], scale)
    if use_displacement != 0:
        delta = correction[slot]
        if scale != 0.0 and _spatial_vector_nonzero(delta):
            body_disp[body] = body_disp[body] + scale * wp.spatial_top(delta)
            body_disp_ang[body] = rotation_vector_compose(scale * wp.spatial_bottom(delta), body_disp_ang[body])
            body_q[body] = pose_from_displacement(
                body_target_q[body], body_com[body], body_disp[body], wp.transform_get_rotation(body_q[body])
            )


@wp.kernel
def accumulate_body_directional_derivative(
    body_ids: wp.array[int],
    body_island: wp.array[int],
    enabled: wp.array[bool],
    body_q: wp.array[wp.transform],
    body_inertia_q: wp.array[wp.transform],
    body_disp: wp.array[wp.vec3],
    body_disp_ang: wp.array[wp.vec3],
    use_displacement: int,
    body_mass: wp.array[float],
    body_inv_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    body_com: wp.array[wp.vec3],
    contact_forces: wp.array[wp.vec3],
    contact_torques: wp.array[wp.vec3],
    dt: float,
    correction: wp.array[wp.spatial_vector],
    merit: wp.array[wp.float64],
):
    """Evaluate the slope without rebuilding unused body Hessians."""
    slot = wp.tid()
    island = body_island[slot]
    body = body_ids[slot]
    if not enabled[island] or body_inv_mass[body] <= 0.0:
        return
    rhs = _body_objective_rhs(
        body_q[body],
        body_inertia_q[body],
        body_mass[body],
        body_inertia[body],
        body_com[body],
        contact_forces[body],
        contact_torques[body],
        1.0 / (dt * dt),
    )
    if use_displacement != 0:
        rhs = _body_objective_rhs_displacement(
            body_q[body],
            body_mass[body],
            body_inertia[body],
            contact_forces[body],
            contact_torques[body],
            1.0 / (dt * dt),
            body_disp[body],
            body_disp_ang[body],
        )
    delta = correction[slot]
    linear = wp.float64(0.0)
    for row in range(6):
        linear -= wp.float64(rhs[row]) * wp.float64(delta[row])
    wp.atomic_add(merit, island, linear)


@wp.kernel
def suppress_nonfinite_correction(
    graph_body_island: wp.array[wp.int32],
    correction: wp.array[wp.spatial_vector],
    island_contact_state: wp.array[wp.int32],
):
    """Hand an island back to local VBD when its correction is not finite.

    The structural correction is a float32 solve of a linearized system whose
    conditioning is not bounded a priori. A closed island with many closures at
    near-hard stiffness can produce a non-finite multiplier and correction.
    The Schur buffer is factorized in place; inspecting it after a solve does
    not identify whether the original assembly or the factorization failed.
    Owned pre-factor captures are needed to distinguish these failure stages.

    A partial correction is not an option: applying it to some bodies and not
    others would violate the very joint rows the global solve just satisfied.
    So the whole island is suppressed, reusing the ``-2`` sentinel that already
    means "this island keeps its local-VBD pose" for truncated contact
    incidence. Healthy islands are untouched and remain bit-identical.

    This is containment, not an accuracy fix. It does not detect inaccurate
    finite corrections or certify convergence of the remaining local sweeps.
    """
    slot = wp.tid()
    value = correction[slot]
    finite = int(1)
    for component in range(6):
        if not wp.isfinite(value[component]):
            finite = 0
    if finite == 0:
        wp.atomic_min(island_contact_state, graph_body_island[slot], -2)


@dataclass
class _Path:
    joints: list[int]
    bodies: list[int]
    body_rows: list[list[tuple[int, int]]]


@dataclass
class _Tree:
    joints: list[int]
    bodies: list[int]
    node_body: list[int]
    node_row: list[int]
    parent_node: list[int]
    coupling_side: list[int]
    elimination_levels: list[list[int]]
    root: int


@dataclass
class _ClosedTree:
    """A complete structural island represented by a tree plus closure rows."""

    tree: _Tree
    closure_joints: list[int]
    joints: list[int]
    bodies: list[int]


@dataclass(frozen=True)
class _BucketDiagnostics:
    """Host-only construction decision for one normalized topology bucket."""

    kind: str
    batch_count: int
    body_count_per_island: int
    body_count_total: int
    row_count_per_island: int
    row_count_total: int
    cycle_rank_per_island: int
    closure_count_per_island: int
    closure_count_total: int
    planned_route: str
    selected_route: str
    fill_edge_count_total: int
    estimated_bytes: int
    closure_response_bytes: int = 0
    closure_schur_bytes: int = 0
    fallback_reason: str | None = None
    cooperative_paired: bool = False


@dataclass
class _BucketPlan:
    """A device-allocation-free bucket plan used by the global preflight."""

    kind: str
    components: list[_Path] | list[_Tree] | list[_ClosedTree]
    diagnostics: _BucketDiagnostics
    # One schedule, or one per tree (keyed by id) for a merged contraction bucket.
    tree_symbolics: _TreeBucketSymbolics | dict[int, _TreeBucketSymbolics] | None = None


@dataclass(frozen=True)
class _TreeBucketSymbolics:
    """Host symbolic schedules shared by accounting and bucket construction."""

    backbone: list[int]
    branch_levels: list[list[int]]
    contraction: _TreeContraction | None
    use_tree_contraction: bool
    use_backbone_cr: bool
    selected_route: str
    use_paired_open_backbone: bool = False


def _tree_backbone_partition(tree: _Tree) -> tuple[list[int], list[list[int]]]:
    """Split a rooted tree into one long backbone and shallow side branches.

    The backbone joins the deepest two root branches, so it contains the root
    and preserves the existing parent orientation. Side branches can therefore
    be eliminated toward the backbone before a logarithmic CR solve.
    """
    if len(tree.parent_node) <= 1:
        return [tree.root], []

    children: list[list[int]] = [[] for _ in tree.parent_node]
    for node, parent in enumerate(tree.parent_node):
        if parent >= 0:
            children[parent].append(node)

    depth = [-1] * len(tree.parent_node)
    root_branch = [-1] * len(tree.parent_node)
    depth[tree.root] = 0
    traversal = [tree.root]
    for parent in traversal:
        for node in sorted(children[parent]):
            depth[node] = depth[parent] + 1
            root_branch[node] = node if parent == tree.root else root_branch[parent]
            traversal.append(node)

    root_branches: dict[int, tuple[int, int]] = {}
    for node in traversal:
        if node == tree.root:
            continue
        root_child = root_branch[node]
        previous = root_branches.get(root_child)
        if previous is None or depth[node] > previous[0] or (depth[node] == previous[0] and node < previous[1]):
            root_branches[root_child] = (depth[node], node)

    endpoints = sorted(root_branches.values(), key=lambda item: (-item[0], item[1]))[:2]

    def path_to_root(endpoint: int) -> list[int]:
        path = [endpoint]
        while path[-1] != tree.root:
            path.append(tree.parent_node[path[-1]])
        return path

    first = path_to_root(endpoints[0][1]) if endpoints else [tree.root]
    second = path_to_root(endpoints[1][1]) if len(endpoints) > 1 else [tree.root]
    backbone = [*first, *reversed(second[:-1])]
    backbone_set = set(backbone)

    branch_depth = [0] * len(tree.parent_node)
    for node in traversal:
        if node not in backbone_set:
            branch_depth[node] = branch_depth[tree.parent_node[node]] + 1

    levels = [
        sorted(node for node, node_depth in enumerate(branch_depth) if node_depth == level)
        for level in range(max(branch_depth, default=0), 0, -1)
    ]
    return backbone, levels


def _make_device_tree_levels(levels: list[list[int]], parent_node: list[int], device):
    """Encode a host leaf schedule for graph-capturable tree elimination."""
    result = []
    for level in levels:
        leaves = wp.array(level, dtype=wp.int32, device=device)
        if len({parent_node[node] for node in level}) == len(level):
            result.append((leaves, None, None, None, True))
            continue
        grouped: dict[int, list[int]] = defaultdict(list)
        for node in level:
            grouped[parent_node[node]].append(node)
        recipients = sorted(grouped)
        message_nodes: list[int] = []
        offsets = [0]
        for recipient in recipients:
            message_nodes.extend(sorted(grouped[recipient]))
            offsets.append(len(message_nodes))
        result.append(
            (
                leaves,
                wp.array(recipients, dtype=wp.int32, device=device),
                wp.array(offsets, dtype=wp.int32, device=device),
                wp.array(message_nodes, dtype=wp.int32, device=device),
                False,
            )
        )
    return result


def _fused_tree_level_schedule(levels: list[list[int]], parent_node: list[int]):
    """Flatten a per-tree leaf schedule for the single-block fused solve.

    Node identifiers stay per tree so one device schedule serves every batch
    member. Recipient and message ordering reproduces
    ``_make_device_tree_levels`` exactly, which is what keeps the fused solve
    bit-identical to the per-level launches.
    """
    kinds: list[int] = []
    node_start: list[int] = [0]
    nodes: list[int] = []
    recipient_start: list[int] = [0]
    recipients: list[int] = []
    message_offsets: list[int] = [0]
    message_nodes: list[int] = []
    for level in levels:
        nodes.extend(level)
        node_start.append(len(nodes))
        if len({parent_node[node] for node in level}) == len(level):
            kinds.append(1)
            recipient_start.append(len(recipients))
            continue
        kinds.append(0)
        grouped: dict[int, list[int]] = defaultdict(list)
        for node in level:
            grouped[parent_node[node]].append(node)
        for recipient in sorted(grouped):
            recipients.append(recipient)
            message_nodes.extend(sorted(grouped[recipient]))
            message_offsets.append(len(message_nodes))
        recipient_start.append(len(recipients))
    # An all-unique schedule has no shared recipients at all. Warp reports a
    # null pointer for a zero-length array, so keep one dummy element, exactly
    # as the closure-response workspace does. The kernel never reads it
    # because every level then has a zero recipient count.
    if not recipients:
        recipients.append(0)
    if not message_nodes:
        message_nodes.append(0)
    return (
        kinds,
        node_start,
        nodes,
        recipient_start,
        recipients,
        message_offsets,
        message_nodes,
    )


def _batched_fused_tree_level_schedule(
    trees: list[_Tree],
    node_offsets: list[int],
) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int], list[int], list[int]]:
    """Encode heterogeneous leaf-rake schedules back to back with per-tree level offsets."""
    tree_level_offsets: list[int] = [0]
    kinds: list[int] = []
    node_start: list[int] = [0]
    nodes: list[int] = []
    recipient_start: list[int] = [0]
    recipients: list[int] = []
    message_offsets: list[int] = [0]
    message_nodes: list[int] = []
    for candidate, offset in zip(trees, node_offsets, strict=True):
        parent = candidate.parent_node
        for level in candidate.elimination_levels:
            nodes.extend(offset + node for node in level)
            node_start.append(len(nodes))
            if len({parent[node] for node in level}) == len(level):
                kinds.append(1)
                recipient_start.append(len(recipients))
                continue
            kinds.append(0)
            grouped: dict[int, list[int]] = defaultdict(list)
            for node in level:
                grouped[parent[node]].append(node)
            for recipient in sorted(grouped):
                recipients.append(offset + recipient)
                message_nodes.extend(offset + m for m in sorted(grouped[recipient]))
                message_offsets.append(len(message_nodes))
            recipient_start.append(len(recipients))
        tree_level_offsets.append(len(kinds))
    if not recipients:
        recipients.append(0)
    if not message_nodes:
        message_nodes.append(0)
    return (
        tree_level_offsets,
        kinds,
        node_start,
        nodes,
        recipient_start,
        recipients,
        message_offsets,
        message_nodes,
    )


def _make_fused_tree_levels(levels: list[list[int]], parent_node: list[int], device):
    """Encode the fused leaf-rake schedule as contiguous device arrays."""
    (
        kinds,
        node_start,
        nodes,
        recipient_start,
        recipients,
        message_offsets,
        message_nodes,
    ) = _fused_tree_level_schedule(levels, parent_node)
    return (
        wp.array(kinds, dtype=wp.int32, device=device),
        wp.array(node_start, dtype=wp.int32, device=device),
        wp.array(nodes, dtype=wp.int32, device=device),
        wp.array(recipient_start, dtype=wp.int32, device=device),
        wp.array(recipients, dtype=wp.int32, device=device),
        wp.array(message_offsets, dtype=wp.int32, device=device),
        wp.array(message_nodes, dtype=wp.int32, device=device),
    )


def _make_batched_fused_tree_levels(trees: list[_Tree], node_offsets: list[int], device):
    """Encode heterogeneous leaf-rake schedules as contiguous device arrays."""
    (
        tree_level_offsets,
        kinds,
        node_start,
        nodes,
        recipient_start,
        recipients,
        message_offsets,
        message_nodes,
    ) = _batched_fused_tree_level_schedule(trees, node_offsets)
    return (
        wp.array(tree_level_offsets, dtype=wp.int32, device=device),
        wp.array(kinds, dtype=wp.int32, device=device),
        wp.array(node_start, dtype=wp.int32, device=device),
        wp.array(nodes, dtype=wp.int32, device=device),
        wp.array(recipient_start, dtype=wp.int32, device=device),
        wp.array(recipients, dtype=wp.int32, device=device),
        wp.array(message_offsets, dtype=wp.int32, device=device),
        wp.array(message_nodes, dtype=wp.int32, device=device),
    )


def _fused_tree_levels_payload_bytes(levels: list[list[int]], parent_node: list[int]) -> int:
    """Count the arrays emitted by ``_make_fused_tree_levels`` exactly."""
    (
        kinds,
        node_start,
        nodes,
        recipient_start,
        recipients,
        message_offsets,
        message_nodes,
    ) = _fused_tree_level_schedule(levels, parent_node)
    total = 0
    for payload in (
        kinds,
        node_start,
        nodes,
        recipient_start,
        recipients,
        message_offsets,
        message_nodes,
    ):
        total += _array_payload_bytes(len(payload), wp.int32)
    return total


def _batched_fused_tree_levels_payload_bytes(trees: list[_Tree], node_offsets: list[int]) -> int:
    """Count the arrays emitted by ``_make_batched_fused_tree_levels`` exactly."""
    payloads = _batched_fused_tree_level_schedule(trees, node_offsets)
    return sum(_array_payload_bytes(len(payload), wp.int32) for payload in payloads)


def _fused_tree_levels_supported(levels: list[list[int]], device) -> bool:
    """Avoid per-level launches when one block can replay the leaf schedule."""
    if len(levels) < _FUSED_TREE_MIN_LEVELS:
        return False
    # CPU uses one lane per tree, so the same kernel needs neither a barrier
    # nor a width limit. Node and shared-recipient ordering remain identical.
    return not device.is_cuda or all(len(level) <= _FUSED_TREE_MAX_LEVEL_WIDTH for level in levels)


def _batched_fused_tree_levels_supported(
    trees: list[_Tree],
    tree_symbolics: list[_TreeBucketSymbolics],
    device,
) -> bool:
    """Return whether every tree in a bucket supports single-block fused leaf rake."""
    return bool(trees) and all(
        not sym.use_tree_contraction
        and not sym.use_backbone_cr
        and _fused_tree_levels_supported(candidate.elimination_levels, device)
        for candidate, sym in zip(trees, tree_symbolics, strict=True)
    )


def _tree_level_launch_count(levels: list[list[int]], parent_node: list[int]) -> int:
    """Count dependent factor/back-substitution launches for one tree schedule."""
    forward = sum(1 if len({parent_node[node] for node in level}) == len(level) else 2 for level in levels)
    return forward + len(levels)


@dataclass
class _TreeContractionLevel:
    """One immutable exact rake/compress level before device encoding."""

    nodes: list[int]
    neighbors: list[tuple[int, int]]
    edges: list[tuple[int, int]]
    transpose_edges: list[tuple[int, int]]
    generated_edges: list[int]
    recipients: list[int]
    recipient_offsets: list[int]
    message_slots: list[int]


@dataclass
class _TreeContraction:
    """Host schedule for exact independent-node tree contraction."""

    levels: list[_TreeContractionLevel]
    root: int
    edge_count: int


def _tree_contraction_schedule(tree: _Tree) -> _TreeContraction:
    """Build an exact logarithmic-depth rake/compress schedule.

    Every selected node has current degree one or two, and selected nodes are
    independent. Eliminating a degree-two node creates one fill edge between
    its surviving neighbors. The symbolic edge and message layout is fixed at
    construction, so the numeric pass remains graph-capturable.
    """
    node_count = len(tree.parent_node)
    adjacency: list[dict[int, int]] = [{} for _ in range(node_count)]
    edge_u = [-1] * node_count
    edge_v = [-1] * node_count
    for node, parent in enumerate(tree.parent_node):
        if parent < 0:
            continue
        # Original edge ``node`` stores A(node, parent), matching coupling.
        edge_u[node] = node
        edge_v[node] = parent
        adjacency[node][parent] = node
        adjacency[parent][node] = node

    backed = [tree.node_body[node] >= 0 for node in range(node_count)]
    next_edge = node_count
    active = set(range(node_count))
    levels: list[_TreeContractionLevel] = []
    round_index = 0
    while len(active) > 1:
        # Never eliminate a constraint row node before at least one incident
        # body has Schur-complemented its inertia into the row (0 bare -C_row
        # pivots across all trees, including world-anchor rows up to K=1e12).
        candidates = [node for node in active if 0 < len(adjacency[node]) <= 2 and backed[node]]
        if not candidates:
            candidates = [node for node in active if 0 < len(adjacency[node]) <= 2]
        # Rake leaves first, then body nodes before backed rows. Rotate the
        # deterministic degree-two ordering so a path stays balanced.
        candidates.sort(
            key=lambda node: (
                len(adjacency[node]) != 1,
                tree.node_body[node] < 0,
                (node - round_index) % node_count,
            )
        )
        selected: list[int] = []
        blocked: set[int] = set()
        for node in candidates:
            if node in blocked:
                continue
            selected.append(node)
            blocked.add(node)
            blocked.update(adjacency[node])
        if len(selected) >= len(active):
            selected.pop()
        if not selected:
            raise RuntimeError("Unable to construct an independent tree-contraction level")

        nodes: list[int] = []
        neighbors: list[tuple[int, int]] = []
        edges: list[tuple[int, int]] = []
        transpose_edges: list[tuple[int, int]] = []
        generated_edges: list[int] = []
        grouped_messages: dict[int, list[int]] = defaultdict(list)
        generated: list[tuple[int, int, int]] = []
        for level_index, node in enumerate(sorted(selected)):
            incident = sorted(adjacency[node].items())
            local_neighbors = [neighbor for neighbor, _ in incident]
            local_edges = [edge for _, edge in incident]
            local_transpose = [int(edge_u[edge] != node) for edge in local_edges]
            while len(local_neighbors) < 2:
                local_neighbors.append(-1)
                local_edges.append(-1)
                local_transpose.append(0)

            generated_edge = -1
            if len(incident) == 2:
                generated_edge = next_edge
                next_edge += 1
                first, second = local_neighbors
                edge_u.append(first)
                edge_v.append(second)
                generated.append((first, second, generated_edge))

            nodes.append(node)
            neighbors.append((local_neighbors[0], local_neighbors[1]))
            edges.append((local_edges[0], local_edges[1]))
            transpose_edges.append((local_transpose[0], local_transpose[1]))
            generated_edges.append(generated_edge)
            for local, neighbor in enumerate(local_neighbors):
                if neighbor >= 0:
                    grouped_messages[neighbor].append(2 * level_index + local)
                    if tree.node_body[node] >= 0:
                        backed[neighbor] = True

        # Remove the independent nodes, then insert their degree-two fill
        # edges. Every neighbor is a survivor because selected nodes are not
        # adjacent, so no generated edge can reference a dead node.
        for node in selected:
            for neighbor in list(adjacency[node]):
                adjacency[neighbor].pop(node)
            adjacency[node].clear()
        active.difference_update(selected)
        for first, second, edge in generated:
            adjacency[first][second] = edge
            adjacency[second][first] = edge

        recipients = sorted(grouped_messages)
        recipient_offsets = [0]
        message_slots: list[int] = []
        for recipient in recipients:
            message_slots.extend(sorted(grouped_messages[recipient]))
            recipient_offsets.append(len(message_slots))
        levels.append(
            _TreeContractionLevel(
                nodes,
                neighbors,
                edges,
                transpose_edges,
                generated_edges,
                recipients,
                recipient_offsets,
                message_slots,
            )
        )
        round_index += 1

    return _TreeContraction(levels, next(iter(active)), next_edge)


def _make_device_batched_tree_contraction(
    schedules: list[_TreeContraction],
    node_counts: list[int],
    device,
):
    """Merge per-tree contraction schedules level by level with compact offsets.

    Trees may differ in topology. Each tree keeps its own node, edge and message
    order; tree edges are offset by the tree's first node and fill edges by the
    tree's first fill slot after all tree nodes.
    """
    total_node_count = sum(node_counts)
    node_offsets = np.cumsum([0, *node_counts[:-1]]).tolist()
    fill_counts = [schedule.edge_count - count for schedule, count in zip(schedules, node_counts, strict=True)]
    fill_offsets = np.cumsum([0, *fill_counts[:-1]]).tolist()
    result = []
    for level_index in range(max(len(schedule.levels) for schedule in schedules)):
        nodes: list[int] = []
        neighbors: list[tuple[int, int]] = []
        edges: list[tuple[int, int]] = []
        transpose_edges: list[tuple[int, int]] = []
        generated_edges: list[int] = []
        recipients: list[int] = []
        recipient_offsets = [0]
        message_slots: list[int] = []
        for schedule, node_count, node_offset, fill_offset in zip(
            schedules, node_counts, node_offsets, fill_offsets, strict=True
        ):
            if level_index >= len(schedule.levels):
                continue
            level = schedule.levels[level_index]
            level_start = len(nodes)

            def global_edge(edge: int, node_count=node_count, node_offset=node_offset, fill_offset=fill_offset) -> int:
                if edge < node_count:
                    return node_offset + edge
                return total_node_count + fill_offset + edge - node_count

            nodes.extend(node_offset + node for node in level.nodes)
            neighbors.extend(
                tuple(node_offset + neighbor if neighbor >= 0 else -1 for neighbor in pair) for pair in level.neighbors
            )
            edges.extend(tuple(global_edge(edge) if edge >= 0 else -1 for edge in pair) for pair in level.edges)
            transpose_edges.extend(level.transpose_edges)
            generated_edges.extend(global_edge(edge) if edge >= 0 else -1 for edge in level.generated_edges)
            for recipient_index, recipient in enumerate(level.recipients):
                recipients.append(node_offset + recipient)
                begin = level.recipient_offsets[recipient_index]
                end = level.recipient_offsets[recipient_index + 1]
                for slot in level.message_slots[begin:end]:
                    local_index = slot // 2
                    local = slot - 2 * local_index
                    message_slots.append(2 * (level_start + local_index) + local)
                recipient_offsets.append(len(message_slots))
        result.append(
            (
                wp.array(nodes, dtype=wp.int32, device=device),
                wp.array(neighbors, dtype=wp.vec2i, device=device),
                wp.array(edges, dtype=wp.vec2i, device=device),
                wp.array(transpose_edges, dtype=wp.vec2i, device=device),
                wp.array(generated_edges, dtype=wp.int32, device=device),
                wp.array(recipients, dtype=wp.int32, device=device),
                wp.array(recipient_offsets, dtype=wp.int32, device=device),
                wp.array(message_slots, dtype=wp.int32, device=device),
            )
        )
    return result


def _tree_topology_signature(tree: _Tree) -> tuple:
    """Return the normalized symbolic layout used for safe numeric batching."""
    return (
        len(tree.bodies),
        len(tree.joints),
        tuple(tree.parent_node),
        tuple(tree.coupling_side),
    )


def _closed_tree_topology_signature(
    component: _ClosedTree,
    parent: np.ndarray,
    child: np.ndarray,
) -> tuple:
    """Return the normalized spanning-tree and closure endpoint layout."""
    body_node = {body: node for node, body in enumerate(component.tree.bodies)}
    closure_endpoints = tuple(
        (body_node.get(int(parent[joint]), -1), body_node.get(int(child[joint]), -1))
        for joint in component.closure_joints
    )
    return _tree_topology_signature(component.tree), closure_endpoints


def _array_payload_bytes(count: int, dtype) -> int:
    """Return the exact logical payload of one contiguous Warp array."""
    if count < 0:
        raise ValueError(f"Array element count must be nonnegative, got {count}")
    return int(count) * int(wp.types.type_size_in_bytes(dtype))


def _row_pass_payload_bytes(row_count: int) -> int:
    """Count one bucket's joint-type row order and stored line-search slope inputs."""
    return (
        2 * _array_payload_bytes(row_count, wp.int32)
        + 2 * _array_payload_bytes(row_count, wp.spatial_matrix)
        + 3 * _array_payload_bytes(row_count, wp.spatial_vector)
    )


def _structural_payload_budget(device) -> int:
    """Bound planned payload deterministically without claiming an allocation reserve."""
    if not device.is_cuda:
        # Warp cannot report host capacity without an optional psutil
        # dependency. Keep CPU construction dependency-free and deterministic.
        return _STRUCTURAL_MEMORY_CAP_BYTES
    total_memory = int(device.total_memory)
    if total_memory > 0:
        return total_memory // _STRUCTURAL_MEMORY_CAPACITY_DIVISOR
    return _STRUCTURAL_MEMORY_CAP_BYTES


def _within_structural_payload_budget(estimated_bytes: int, budget_bytes: int) -> bool:
    """Use an inclusive boundary with Python integers, which cannot overflow."""
    return 0 <= estimated_bytes <= budget_bytes


def _tree_anchor_row_count(tree: _Tree) -> int:
    """Count rows with a single dynamic body (joints to the world or to static bodies)."""
    degree = [int(parent >= 0) for parent in tree.parent_node]
    for parent in tree.parent_node:
        if parent >= 0:
            degree[parent] += 1
    return sum(1 for node, row in enumerate(tree.node_row) if row >= 0 and degree[node] == 1)


def _tree_reoriented(tree: _Tree, root: int) -> _Tree:
    """Return the same tree with every edge oriented toward ``root`` (host-only planning aid)."""
    adjacency: list[list[int]] = [[] for _ in tree.parent_node]
    for node, parent in enumerate(tree.parent_node):
        if parent >= 0:
            adjacency[node].append(parent)
            adjacency[parent].append(node)
    parent_node = [-1] * len(tree.parent_node)
    order = [root]
    for node in order:
        for neighbor in adjacency[node]:
            if neighbor != parent_node[node]:
                parent_node[neighbor] = node
                order.append(neighbor)
    height = [0] * len(tree.parent_node)
    for node in reversed(order[1:]):
        height[parent_node[node]] = max(height[parent_node[node]], height[node] + 1)
    levels: list[list[int]] = [[] for _ in range(height[root])]
    for node, level in enumerate(height):
        if node != root:
            levels[level].append(node)
    return _Tree(tree.joints, tree.bodies, tree.node_body, tree.node_row, parent_node, tree.coupling_side, levels, root)


def _tree_bucket_symbolics(
    tree: _Tree,
    *,
    include_contraction: bool = True,
    enable_paired_open_chain: bool = False,
) -> _TreeBucketSymbolics:
    """Select one exact tree route using host-only topology."""
    backbone, branch_levels = _tree_backbone_partition(tree)
    # A tree rooted at its row to the world (see _make_tree) is planned as it was oriented before,
    # from that row's body, so the new root does not change the route.
    anchor_root = tree.node_row[tree.root] >= 0
    plan_tree = _tree_reoriented(tree, tree.parent_node.index(tree.root)) if anchor_root else tree
    plan_backbone, plan_branch_levels = (
        _tree_backbone_partition(plan_tree) if anchor_root else (backbone, branch_levels)
    )
    if enable_paired_open_chain and not branch_levels:
        alternating = all(
            tree.node_body[node] >= 0 if index % 2 == 0 else tree.node_row[node] >= 0
            for index, node in enumerate(backbone)
        )
        if not alternating:
            reversed_backbone = list(reversed(backbone))
            alternating = all(
                tree.node_body[node] >= 0 if index % 2 == 0 else tree.node_row[node] >= 0
                for index, node in enumerate(reversed_backbone)
            )
            if alternating:
                backbone = reversed_backbone
        if alternating and len(tree.bodies) >= _PAIRED_OPEN_CHAIN_MIN_BODIES:
            return _TreeBucketSymbolics(
                backbone,
                branch_levels,
                None,
                False,
                False,
                "tree_paired_open_backbone_cr",
                True,
            )
    if not include_contraction:
        return _TreeBucketSymbolics(
            backbone,
            branch_levels,
            None,
            False,
            True,
            "tree_backbone_cr",
            False,
        )

    backbone_cr_levels = 0
    stride = 1
    active_count = len(plan_backbone)
    while active_count > _CR_SERIAL_TERMINAL_SIZE:
        backbone_cr_levels += 1
        stride *= 2
        active_count = len(range(0, len(plan_backbone), stride))

    level_launches = _tree_level_launch_count(plan_tree.elimination_levels, plan_tree.parent_node) + 1
    backbone_launches = (
        _tree_level_launch_count(plan_branch_levels, plan_tree.parent_node) + 1 + 3 * backbone_cr_levels + 1
    )
    contraction = _tree_contraction_schedule(tree)
    contraction_launches = 3 * len(contraction.levels) + 1
    use_tree_contraction = contraction_launches < min(level_launches, backbone_launches)
    anchor_rows = _tree_anchor_row_count(tree)
    if anchor_rows >= 2 or (anchor_rows == 1 and not anchor_root):
        # A row to the world that is a leaf of the leaf rake would be eliminated before its body.
        # The contraction schedule eliminates every row after one of its bodies.
        use_tree_contraction = True
    # The unpaired 6x6 CR route can expose near-hard constraint Schur blocks
    # as individual float32 pivots. Closed trees can instead use the guarded
    # body-row paired factorization; open trees retain leaf rake unless the
    # already-oracle-covered contraction schedule wins outright.
    use_backbone_cr = False
    if use_tree_contraction:
        selected_route = "tree_contraction"
    else:
        selected_route = "tree_leaf_rake"
    return _TreeBucketSymbolics(
        backbone,
        branch_levels,
        contraction,
        use_tree_contraction,
        use_backbone_cr,
        selected_route,
        False,
    )


def _tree_levels_payload_bytes(levels: list[list[int]], parent_node: list[int], batch_count: int) -> int:
    """Count the arrays emitted by ``_make_device_tree_levels`` exactly."""
    total = 0
    for level in levels:
        level_size = len(level)
        total += _array_payload_bytes(batch_count * level_size, wp.int32)
        parent_count = len({parent_node[node] for node in level})
        if parent_count == level_size:
            continue
        total += _array_payload_bytes(batch_count * parent_count, wp.int32)
        total += _array_payload_bytes(batch_count * parent_count + 1, wp.int32)
        total += _array_payload_bytes(batch_count * level_size, wp.int32)
    return total


def _tree_contraction_payload_bytes(schedules: list[_TreeContraction], node_counts: list[int]) -> int:
    """Count the merged rake-and-compress device schedule of one tree bucket."""
    total = 0
    for level_index in range(max(len(schedule.levels) for schedule in schedules)):
        levels = [schedule.levels[level_index] for schedule in schedules if level_index < len(schedule.levels)]
        level_size = sum(len(level.nodes) for level in levels)
        recipient_count = sum(len(level.recipients) for level in levels)
        message_count = sum(len(level.message_slots) for level in levels)
        total += _array_payload_bytes(level_size, wp.int32)
        total += 3 * _array_payload_bytes(level_size, wp.vec2i)
        total += _array_payload_bytes(level_size, wp.int32)
        total += _array_payload_bytes(recipient_count, wp.int32)
        total += _array_payload_bytes(recipient_count + 1, wp.int32)
        total += _array_payload_bytes(message_count, wp.int32)
    total += _array_payload_bytes(len(schedules), wp.int32)
    total += _array_payload_bytes(
        sum(schedule.edge_count - count for schedule, count in zip(schedules, node_counts, strict=True)),
        wp.spatial_matrix,
    )
    return total


def _tree_body_row_count(trees: list[_Tree]) -> int:
    """Count the compact body-to-row incidence array built by ``_TreeBucket``."""
    count = 0
    for tree in trees:
        for body_node in range(len(tree.bodies)):
            for node, row in enumerate(tree.node_row):
                if row >= 0 and (tree.parent_node[node] == body_node or tree.parent_node[body_node] == node):
                    count += 1
    return count


def _path_bucket_diagnostics(paths: list[_Path], device, *, factor_dtype=wp.float32) -> _BucketDiagnostics:
    """Plan the exact device payload of one path bucket.

    Paths may differ in length; per-island counts report the largest path.
    """
    row_counts = [len(path.joints) for path in paths]
    batch_count = len(paths)
    size = sum(row_counts)
    body_size = sum(len(path.bodies) for path in paths)
    schedule = _path_cr_schedule(row_counts, device.is_cuda)
    estimated_bytes = (
        2 * _array_payload_bytes(size, wp.int32)
        + _array_payload_bytes(body_size, wp.int32)
        + _array_payload_bytes(body_size, wp.vec2i)
        + 5 * _array_payload_bytes(size, wp.spatial_matrix)
        + _array_payload_bytes(size, wp.spatial_vector)
        + 2 * _array_payload_bytes(size, wp.vec2i)
        + _path_cr_schedule_bytes(schedule)
        + _row_pass_payload_bytes(size)
        + (
            3 * _array_payload_bytes(size, wp.spatial_matrixd) + _array_payload_bytes(size, wp.spatial_vectord)
            if factor_dtype == wp.float64
            else 0
        )
    )
    suffix = "_fp64" if factor_dtype == wp.float64 else ""
    if schedule.tails:
        route = "path_fused_tail_cr" + suffix
    elif schedule.persistent:
        route = "path_persistent_cr" + suffix
    else:
        route = "path_level_cr" + suffix
    return _BucketDiagnostics(
        kind="path",
        batch_count=batch_count,
        body_count_per_island=max(len(path.bodies) for path in paths),
        body_count_total=body_size,
        row_count_per_island=max(row_counts),
        row_count_total=size,
        cycle_rank_per_island=0,
        closure_count_per_island=0,
        closure_count_total=0,
        planned_route=route,
        selected_route=route,
        fill_edge_count_total=0,
        estimated_bytes=estimated_bytes,
    )


def _tree_bucket_layout(
    trees: list[_Tree],
    symbolics: _TreeBucketSymbolics | dict[int, _TreeBucketSymbolics] | None,
):
    """Return per-tree symbolics and the merged host schedules shared by accounting and construction.

    A bucket is either one normalized topology or, when every tree takes the
    contraction route, trees of different topologies merged level by level.
    """
    uniform = all(_tree_topology_signature(tree) == _tree_topology_signature(trees[0]) for tree in trees[1:])
    if not uniform and not isinstance(symbolics, dict):
        raise ValueError("Nonuniform tree buckets require per-tree symbolics")
    if isinstance(symbolics, dict):
        tree_symbolics = [symbolics[id(tree)] for tree in trees]
    else:
        shared = symbolics if symbolics is not None else _tree_bucket_symbolics(trees[0])
        tree_symbolics = [shared] * len(trees)
    if not uniform and not all(
        (candidate.use_tree_contraction or not candidate.use_backbone_cr) and not candidate.use_paired_open_backbone
        for candidate in tree_symbolics
    ):
        raise ValueError(
            "A tree bucket requires one normalized symbolic topology unless every tree uses contraction or leaf rake"
        )
    node_counts = [len(tree.node_body) for tree in trees]
    node_offsets = np.cumsum([0, *node_counts[:-1]]).tolist()
    parent_node: list[int] = []
    for tree, offset in zip(trees, node_offsets, strict=True):
        parent_node.extend(offset + parent if parent >= 0 else -1 for parent in tree.parent_node)

    def merge(per_tree: list[list[list[int]]]) -> list[list[int]]:
        depth = max((len(levels) for levels in per_tree), default=0)
        return [
            [
                offset + node
                for levels, offset in zip(per_tree, node_offsets, strict=True)
                if index < len(levels)
                for node in levels[index]
            ]
            for index in range(depth)
        ]

    return (
        uniform,
        tree_symbolics,
        node_counts,
        node_offsets,
        parent_node,
        merge([tree.elimination_levels for tree in trees]),
        merge([candidate.branch_levels for candidate in tree_symbolics]),
        [
            offset + node
            for candidate, offset in zip(tree_symbolics, node_offsets, strict=True)
            for node in candidate.backbone
        ],
    )


def _tree_bucket_diagnostics(
    trees: list[_Tree],
    device,
    *,
    symbolics: _TreeBucketSymbolics | dict[int, _TreeBucketSymbolics] | None = None,
    allow_fused_tree_levels: bool = True,
    factor_dtype=wp.float32,
) -> _BucketDiagnostics:
    """Plan the exact device payload of one tree bucket.

    Contracted trees of different topologies report their largest island in the
    per-island counts.
    """
    (
        uniform,
        tree_symbolics,
        node_counts,
        node_offsets,
        parent_node,
        levels,
        branch_levels,
        backbone,
    ) = _tree_bucket_layout(trees, symbolics)
    tree = trees[0]
    symbolics = tree_symbolics[0]
    batch_count = len(trees)
    row_count = sum(len(candidate.joints) for candidate in trees)
    body_count = sum(len(candidate.bodies) for candidate in trees)
    node_count = sum(node_counts)
    backbone_size = len(backbone)

    matrix_dtype = wp.spatial_matrixd if factor_dtype == wp.float64 else wp.spatial_matrix
    vector_dtype = wp.spatial_vectord if factor_dtype == wp.float64 else wp.spatial_vector
    estimated_bytes = 0
    estimated_bytes += _array_payload_bytes(batch_count, wp.int32)
    estimated_bytes += 2 * _array_payload_bytes(row_count, wp.int32)
    estimated_bytes += _array_payload_bytes(node_count, wp.int32)
    estimated_bytes += _array_payload_bytes(body_count, wp.int32)
    if batch_count > 1:
        estimated_bytes += _array_payload_bytes(body_count, wp.int32)
    estimated_bytes += _array_payload_bytes(body_count + 1, wp.int32)
    estimated_bytes += _array_payload_bytes(_tree_body_row_count(trees), wp.int32)
    estimated_bytes += 3 * _array_payload_bytes(node_count, wp.int32)
    estimated_bytes += _array_payload_bytes(max(2 * row_count, node_count), wp.spatial_matrix)
    estimated_bytes += _array_payload_bytes(row_count, wp.spatial_matrix)
    estimated_bytes += 2 * _array_payload_bytes(row_count, wp.spatial_vector)
    estimated_bytes += _array_payload_bytes(row_count, wp.int32)
    estimated_bytes += _array_payload_bytes(node_count, vector_dtype)
    estimated_bytes += _array_payload_bytes(node_count, matrix_dtype)
    if factor_dtype == wp.float64:
        estimated_bytes += _array_payload_bytes(node_count, matrix_dtype)
    estimated_bytes += _tree_levels_payload_bytes(levels, parent_node, 1)
    if allow_fused_tree_levels and _batched_fused_tree_levels_supported(trees, tree_symbolics, device):
        if uniform:
            estimated_bytes += _fused_tree_levels_payload_bytes(tree.elimination_levels, tree.parent_node)
        else:
            estimated_bytes += _batched_fused_tree_levels_payload_bytes(trees, node_offsets)
    estimated_bytes += _array_payload_bytes(backbone_size, wp.int32)
    estimated_bytes += _tree_levels_payload_bytes(branch_levels, parent_node, 1)
    estimated_bytes += 2 * _array_payload_bytes(backbone_size, matrix_dtype)
    if symbolics.use_paired_open_backbone:
        pair_matrix_dtype = _SpatialPairMatrixD if factor_dtype == wp.float64 else _SpatialPairMatrix
        pair_factor_dtype = (
            _SpatialPairEliminationFactorD if factor_dtype == wp.float64 else _SpatialPairEliminationFactor
        )
        pair_vector_dtype = _SpatialPairVectorD if factor_dtype == wp.float64 else _SpatialPairVector
        pair_response_dtype = _SpatialPairResponseD if factor_dtype == wp.float64 else _SpatialPairResponse
        paired_backbone_count = (len(symbolics.backbone) + 1) // 2
        paired_backbone_size = batch_count * paired_backbone_count
        estimated_bytes += 3 * _array_payload_bytes(paired_backbone_size, pair_matrix_dtype)
        estimated_bytes += _array_payload_bytes(paired_backbone_size, pair_factor_dtype)
        estimated_bytes += _array_payload_bytes(paired_backbone_size, pair_vector_dtype)
        estimated_bytes += _array_payload_bytes(1, pair_response_dtype)
        if len(tree.bodies) > _PAIRED_OPEN_CHAIN_MIN_BODIES:
            estimated_bytes += _array_payload_bytes(paired_backbone_size, pair_vector_dtype)
            estimated_bytes += _array_payload_bytes(1, pair_response_dtype)
    fill_count = 0
    if symbolics.use_tree_contraction:
        if any(candidate.contraction is None for candidate in tree_symbolics):
            raise RuntimeError("Tree contraction was selected without a symbolic schedule")
        schedules = [candidate.contraction for candidate in tree_symbolics]
        estimated_bytes += _tree_contraction_payload_bytes(schedules, node_counts)
        fill_count = sum(schedule.edge_count - count for schedule, count in zip(schedules, node_counts, strict=True))
    estimated_bytes += _array_payload_bytes(1, matrix_dtype)
    estimated_bytes += _row_pass_payload_bytes(row_count)

    route_suffix = "_fp64" if factor_dtype == wp.float64 else ""
    route_name = f"{symbolics.selected_route}{route_suffix}"
    return _BucketDiagnostics(
        kind="tree",
        batch_count=batch_count,
        body_count_per_island=max(len(candidate.bodies) for candidate in trees),
        body_count_total=body_count,
        row_count_per_island=max(len(candidate.joints) for candidate in trees),
        row_count_total=row_count,
        cycle_rank_per_island=0,
        closure_count_per_island=0,
        closure_count_total=0,
        planned_route=route_name,
        selected_route=route_name,
        fill_edge_count_total=fill_count,
        estimated_bytes=estimated_bytes,
    )


def _closed_tree_body_closure_row_count(
    components: list[_ClosedTree],
    parent: np.ndarray,
    child: np.ndarray,
) -> int:
    """Count the compact body-to-closure incidence array."""
    count = 0
    for component in components:
        for body in component.tree.bodies:
            count += sum(body == int(parent[joint]) or body == int(child[joint]) for joint in component.closure_joints)
    return count


def _closed_tree_bucket_diagnostics(
    components: list[_ClosedTree],
    parent: np.ndarray,
    child: np.ndarray,
    device,
    *,
    tree_symbolics: _TreeBucketSymbolics | None = None,
    cooperative_paired: bool = True,
) -> _BucketDiagnostics:
    """Plan every tree, response, dense-Schur, and paired payload before allocation."""
    component = components[0]
    signature = _closed_tree_topology_signature(component, parent, child)
    if any(_closed_tree_topology_signature(candidate, parent, child) != signature for candidate in components[1:]):
        raise ValueError("A closed-tree bucket requires one normalized symbolic topology")

    trees = [candidate.tree for candidate in components]
    if tree_symbolics is None:
        tree_symbolics = _tree_bucket_symbolics(component.tree, include_contraction=False)
    tree_diagnostics = _tree_bucket_diagnostics(
        trees,
        device,
        symbolics=tree_symbolics,
        allow_fused_tree_levels=False,
        factor_dtype=wp.float64,
    )
    batch_count = len(components)
    closure_count = len(component.closure_joints)
    closure_size = batch_count * closure_count
    tree_node_count = len(component.tree.node_body)
    node_count = batch_count * tree_node_count
    body_count = tree_diagnostics.body_count_total

    matrix_dtype = wp.spatial_matrixd
    vector_dtype = wp.spatial_vectord
    estimated_bytes = tree_diagnostics.estimated_bytes + _row_pass_payload_bytes(closure_size)
    estimated_bytes += _array_payload_bytes(sum(map(len, component.tree.elimination_levels)), wp.int32)
    estimated_bytes += 3 * _array_payload_bytes(closure_size, wp.int32)
    estimated_bytes += _array_payload_bytes(body_count + 1, wp.int32)
    estimated_bytes += _array_payload_bytes(
        _closed_tree_body_closure_row_count(components, parent, child),
        wp.int32,
    )
    estimated_bytes += 3 * _array_payload_bytes(closure_size, wp.spatial_matrix)
    estimated_bytes += 2 * _array_payload_bytes(closure_size, wp.spatial_vector)
    estimated_bytes += _array_payload_bytes(closure_size, wp.int32)

    closure_response_bytes = _array_payload_bytes(node_count * closure_count, matrix_dtype)
    closure_schur_bytes = _array_payload_bytes(
        batch_count * closure_count * closure_count,
        matrix_dtype,
    )
    estimated_bytes += closure_response_bytes
    estimated_bytes += closure_schur_bytes
    if _closure_back_substitute_lanes(closure_count, device):
        estimated_bytes += _array_payload_bytes(closure_size, vector_dtype)
    estimated_bytes += _array_payload_bytes(closure_size, vector_dtype)

    local_backbone = tree_symbolics.backbone
    use_paired_backbone = not device.is_cpu and all(
        component.tree.node_body[node] >= 0 if index % 2 == 0 else component.tree.node_row[node] >= 0
        for index, node in enumerate(local_backbone)
    )
    cooperative_paired = (
        cooperative_paired
        and use_paired_backbone
        and device.is_cuda
        and closure_count >= _PAIRED_COOPERATIVE_MIN_CLOSURES
    )
    if use_paired_backbone:
        paired_backbone_count = (len(local_backbone) + 1) // 2
        paired_backbone_size = batch_count * paired_backbone_count
        estimated_bytes += 3 * _array_payload_bytes(paired_backbone_size, _SpatialPairMatrixD)
        estimated_bytes += _array_payload_bytes(paired_backbone_size, _SpatialPairEliminationFactorD)
        estimated_bytes += _array_payload_bytes(paired_backbone_size, _SpatialPairVectorD)
        paired_response_bytes = _array_payload_bytes(
            paired_backbone_size * closure_count,
            _SpatialPairResponseD,
        )
        estimated_bytes += paired_response_bytes
        closure_response_bytes += paired_response_bytes
        route = "closed_tree_paired_backbone_cr"
    else:
        route = "closed_tree_cpu_float64" if device.is_cpu else "closed_tree_serial_float64"

    return _BucketDiagnostics(
        kind="closed_tree",
        batch_count=batch_count,
        body_count_per_island=len(component.bodies),
        body_count_total=body_count,
        row_count_per_island=len(component.joints),
        row_count_total=sum(len(candidate.joints) for candidate in components),
        cycle_rank_per_island=closure_count,
        closure_count_per_island=closure_count,
        closure_count_total=closure_size,
        planned_route=route,
        selected_route=route,
        fill_edge_count_total=tree_diagnostics.fill_edge_count_total,
        estimated_bytes=estimated_bytes,
        closure_response_bytes=closure_response_bytes,
        closure_schur_bytes=closure_schur_bytes,
        cooperative_paired=cooperative_paired,
    )


def _shared_structural_payload_bytes(
    model_body_count: int,
    model_joint_count: int,
    graph_body_count: int,
    island_count: int,
    joint_count: int,
    *,
    is_cuda: bool,
) -> int:
    """Count arrays owned once by an active ``StructuralGraphKKT``."""
    return (
        2 * _array_payload_bytes(graph_body_count, wp.int32)
        + _array_payload_bytes(island_count, wp.int32)
        + _array_payload_bytes(joint_count, wp.int32)
        + _array_payload_bytes(model_body_count, wp.int32)
        + _array_payload_bytes(model_body_count, wp.transform)
        + 2 * _array_payload_bytes(model_body_count, wp.vec3)
        + _array_payload_bytes(island_count, wp.bool)
        + _array_payload_bytes(island_count, wp.int32)
        + 2 * _array_payload_bytes(1, wp.int32)
        + 2 * _array_payload_bytes(island_count, wp.float64)
        + _array_payload_bytes(1 + 2 * int(is_cuda), wp.int32)
        + _array_payload_bytes(graph_body_count, wp.spatial_vector)
        + _array_payload_bytes(island_count, wp.int32)
        + _array_payload_bytes(island_count, wp.mat44d)
        + _array_payload_bytes(island_count, _TranslationSupport)
        + _array_payload_bytes(graph_body_count, wp.vec4)
        + _array_payload_bytes(graph_body_count, wp.vec4d)
        + _array_payload_bytes(island_count, wp.int32)
        + _array_payload_bytes(island_count, wp.vec4d)
        + _array_payload_bytes(island_count, wp.spatial_matrixd)
        + 2 * _array_payload_bytes(island_count, wp.spatial_vectord)
        + _array_payload_bytes(joint_count + 2 * island_count + 2, wp.int32)
        + _array_payload_bytes(model_joint_count, wp.vec2i)
        + _array_payload_bytes(graph_body_count, wp.int32)
        + _array_payload_bytes(2 * joint_count, wp.mat33)
        + _array_payload_bytes(graph_body_count + 1, wp.int32)
        + _array_payload_bytes(island_count, wp.float32)
        + 2 * _array_payload_bytes(graph_body_count, wp.spatial_matrix)
        + _array_payload_bytes(graph_body_count, wp.spatial_vector)
        + _array_payload_bytes(island_count, wp.int32)
        + _array_payload_bytes(island_count, wp.int32)
        + _array_payload_bytes(1, wp.int32)
    )


def _component_order_key(component: _Path | _Tree | _ClosedTree) -> tuple[int, int]:
    """Order islands by model identifiers without preferring a topology."""
    return min(component.joints), min(component.bodies)


def _bucket_plan_diagnostics(
    plan: _BucketPlan,
    components: list[_Path] | list[_Tree] | list[_ClosedTree],
    parent: np.ndarray,
    child: np.ndarray,
    device,
) -> _BucketDiagnostics:
    """Resize one normalized bucket while reusing its symbolic schedule."""
    if plan.kind == "path":
        return _path_bucket_diagnostics(
            components,
            device,
            factor_dtype=wp.float64 if plan.diagnostics.planned_route.endswith("_fp64") else wp.float32,
        )
    if plan.kind == "tree":
        tree_dtype = wp.float64 if plan.diagnostics.planned_route.endswith("_fp64") else wp.float32
        return _tree_bucket_diagnostics(
            components,
            device,
            symbolics=plan.tree_symbolics,
            allow_fused_tree_levels=True,
            factor_dtype=tree_dtype,
        )
    return _closed_tree_bucket_diagnostics(
        components,
        parent,
        child,
        device,
        tree_symbolics=plan.tree_symbolics,
        cooperative_paired=plan.diagnostics.cooperative_paired,
    )


def _cached_bucket_prefix_diagnostics(
    cache: dict[int, _BucketDiagnostics],
    plan: _BucketPlan,
    count: int,
    parent: np.ndarray,
    child: np.ndarray,
    device,
) -> _BucketDiagnostics:
    """Return exact diagnostics for a resized normalized bucket."""
    if count not in cache:
        cache[count] = _bucket_plan_diagnostics(
            plan,
            plan.components[:count],
            parent,
            child,
            device,
        )
    return cache[count]


def _eligible_structural_joints(model, supported_joint_types: set[int] | None = None) -> list[int]:
    """Return supported enabled joints whose complete objective is represented."""
    joint_type = np.asarray(model.joint_type.numpy(), dtype=np.int64)
    joint_enabled = np.asarray(model.joint_enabled.numpy(), dtype=bool)
    child = np.asarray(model.joint_child.numpy(), dtype=np.int64)
    supported = supported_joint_types
    if supported is None:
        supported = {
            int(JointType.ROD),
            int(JointType.BALL),
            int(JointType.FIXED),
            int(JointType.REVOLUTE),
            int(JointType.PRISMATIC),
            int(JointType.D6),
        }
    result: list[int] = []
    for joint, kind in enumerate(joint_type):
        if kind not in supported or not joint_enabled[joint] or child[joint] < 0:
            continue
        result.append(joint)
    return result


def _build_paths(
    model,
    body_inv_mass: np.ndarray,
    *,
    allow_general_joints=False,
    supported_joint_types: set[int] | None = None,
    ignore_free_completion_joints=False,
) -> list[_Path]:
    """Return supported structural components whose line graph is a path."""
    joint_type = np.asarray(model.joint_type.numpy(), dtype=np.int64)
    joint_enabled = np.asarray(model.joint_enabled.numpy(), dtype=bool)
    parent = np.asarray(model.joint_parent.numpy(), dtype=np.int64)
    child = np.asarray(model.joint_child.numpy(), dtype=np.int64)
    structural_joints = _eligible_structural_joints(model, supported_joint_types)
    if not structural_joints:
        return []

    body_to_joints: dict[int, list[int]] = defaultdict(list)
    for joint in structural_joints:
        if parent[joint] >= 0:
            body_to_joints[int(parent[joint])].append(joint)
        body_to_joints[int(child[joint])].append(joint)

    all_body_to_joints: dict[int, set[int]] = defaultdict(set)
    for joint, enabled in enumerate(joint_enabled):
        if (
            not enabled
            or child[joint] < 0
            or (ignore_free_completion_joints and joint_type[joint] == int(JointType.FREE))
        ):
            continue
        if parent[joint] >= 0:
            all_body_to_joints[int(parent[joint])].add(joint)
        all_body_to_joints[int(child[joint])].add(joint)

    neighbors = {joint: set() for joint in structural_joints}
    for incident in body_to_joints.values():
        for joint in incident:
            neighbors[joint].update(other for other in incident if other != joint)

    paths: list[_Path] = []
    unvisited = set(structural_joints)
    while unvisited:
        seed = min(unvisited)
        component: set[int] = set()
        stack = [seed]
        while stack:
            joint = stack.pop()
            if joint in component:
                continue
            component.add(joint)
            unvisited.discard(joint)
            stack.extend(neighbors[joint] - component)

        # The compliance Schur path is fastest for rod chains, including a
        # world-attachment row at a boundary. General rigid and dynamically
        # mixed chains use the equilibrated primal-dual tree factorization,
        # which remains stable near the hard-joint limit.
        rod_joints = [joint for joint in component if joint_type[joint] == int(JointType.ROD)]
        non_rod_joints = component - set(rod_joints)
        if (
            not allow_general_joints
            and non_rod_joints
            and (not rod_joints or any(parent[joint] >= 0 for joint in non_rod_joints))
        ):
            continue

        component_bodies = {
            body for joint in component for body in (int(parent[joint]), int(child[joint])) if body >= 0
        }
        if not any(body_inv_mass[body] > 0.0 for body in component_bodies):
            continue
        if any(all_body_to_joints[body] - component for body in component_bodies):
            # A path correction must contain the body's complete persistent
            # structural neighborhood. Otherwise it can undo an attachment,
            # drive, or another joint that the KKT system did not assemble.
            continue

        path = _make_path(component, parent, child)
        if path is not None:
            paths.append(path)
    return paths


def _make_path(
    joints: set[int],
    parent: np.ndarray,
    child: np.ndarray,
    *,
    dynamic_body_mask: np.ndarray | None = None,
) -> _Path | None:
    """Order a Schur path, optionally omitting prescribed body unknowns."""
    incident: dict[int, set[int]] = defaultdict(set)
    for joint in joints:
        for body in (int(parent[joint]), int(child[joint])):
            if body >= 0 and (dynamic_body_mask is None or dynamic_body_mask[body]):
                incident[body].add(joint)
    neighbors = {joint: set() for joint in joints}
    for rows in incident.values():
        for joint in rows:
            neighbors[joint].update(rows - {joint})
    if any(len(rows) > 2 for rows in neighbors.values()):
        return None
    if sum(map(len, neighbors.values())) // 2 != len(joints) - 1:
        return None
    starts = [joint for joint in joints if len(neighbors[joint]) <= 1]
    if not starts:
        return None
    ordered: list[int] = []
    previous = -1
    current = min(starts, key=lambda joint: (parent[joint] >= 0, joint))
    while current >= 0:
        ordered.append(current)
        following = sorted(neighbors[current] - {previous})
        previous, current = current, (following[0] if following else -1)
    if len(ordered) != len(joints):
        return None
    for left, right in pairwise(ordered):
        left_bodies = {int(parent[left]), int(child[left])} & incident.keys()
        right_bodies = {int(parent[right]), int(child[right])} & incident.keys()
        if len(left_bodies & right_bodies) != 1:
            return None
    # Keep first-encounter body order and ascending incident row/side order.
    rows_by_body: dict[int, list[tuple[int, int]]] = {}
    for row, joint in enumerate(ordered):
        for side, body in enumerate((int(parent[joint]), int(child[joint]))):
            if body in incident:
                rows_by_body.setdefault(body, []).append((row, side))
    if any(len(rows) > 2 for rows in rows_by_body.values()):
        return None
    return _Path(ordered, list(rows_by_body), list(rows_by_body.values()))


def _has_rigid_angular_rows(joints, joint_type: np.ndarray) -> bool:
    """Whether a structure holds a rigid joint with angular rows (not a ball, rod or free joint).

    The contraction route loses such joints near rigidity in float32: a clamped chain of 24 fixed or 64
    revolute joints at K = 1e9 diverges on it within 120 steps of 1/600 s, while the leaf rake holds it.
    """
    kept = (int(JointType.BALL), int(JointType.ROD), int(JointType.FREE))
    return any(int(joint_type[joint]) not in kept for joint in joints)


def _make_tree(
    joints: list[int],
    parent: np.ndarray,
    child: np.ndarray,
    body_inv_mass: np.ndarray,
    *,
    root_at_anchor: bool = False,
) -> _Tree | None:
    """Build the bipartite body-joint tree used by the exact KKT solve."""
    bodies = sorted(
        {
            body
            for joint in joints
            for body in (int(parent[joint]), int(child[joint]))
            if body >= 0 and body_inv_mass[body] > 0.0
        }
    )
    if not bodies:
        return None

    body_node = {body: index for index, body in enumerate(bodies)}
    joint_node = {joint: len(bodies) + row for row, joint in enumerate(joints)}
    node_count = len(bodies) + len(joints)
    adjacency: list[set[int]] = [set() for _ in range(node_count)]
    edge_count = 0
    for joint in joints:
        node = joint_node[joint]
        for body in (int(parent[joint]), int(child[joint])):
            endpoint = body_node.get(body)
            if endpoint is None:
                continue
            adjacency[node].add(endpoint)
            adjacency[endpoint].add(node)
            edge_count += 1
        if not adjacency[node]:
            return None
    if edge_count != node_count - 1:
        return None

    reached = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node] - reached:
            reached.add(neighbor)
            stack.append(neighbor)
    if len(reached) != node_count:
        return None

    # In a tree, a node's eccentricity is the larger distance to the two
    # diameter endpoints. Three linear traversals therefore find the best
    # dynamic-body root; evaluating one BFS per body would be quadratic.
    def distances(start: int) -> list[int]:
        result = [-1] * node_count
        result[start] = 0
        queue = [start]
        for node in queue:
            for neighbor in adjacency[node]:
                if result[neighbor] < 0:
                    result[neighbor] = result[node] + 1
                    queue.append(neighbor)
        return result

    first_distance = distances(0)
    first = max(range(node_count), key=lambda node: (first_distance[node], -node))
    distance_from_first = distances(first)
    second = max(range(node_count), key=lambda node: (distance_from_first[node], -node))
    distance_from_second = distances(second)
    root = min(
        range(len(bodies)),
        key=lambda node: (
            max(distance_from_first[node], distance_from_second[node]),
            bodies[node],
        ),
    )
    # Every route eliminates a node before its parent. A row with a single dynamic body is a leaf,
    # so under a body root it would be eliminated before that body, as the bare pivot -C, and a stiff
    # row (K ~ 1e12) then erases the body's inertia in float32. Root the tree at such a row when it is
    # the only one and the leaf rake gets at most two levels deeper, or always when the tree keeps the
    # leaf rake (root_at_anchor: it has rigid joints with angular rows, which the contraction route
    # loses near rigidity in float32). The planner sends the other anchored trees to the contraction route.
    anchor_rows = [node for node in range(len(bodies), node_count) if len(adjacency[node]) == 1]
    if len(anchor_rows) == 1:
        anchor = anchor_rows[0]
        anchor_height = max(distance_from_first[anchor], distance_from_second[anchor])
        if root_at_anchor or anchor_height <= max(distance_from_first[root], distance_from_second[root]) + 2:
            root = anchor
    # The graph is a connected tree. Orient each edge toward the root,
    # then a node's leaf-removal round is its maximum child round plus one.
    # This avoids rescanning all surviving nodes at every level of a long tree.
    parent_node = [-1] * node_count
    order = [root]
    for node in order:
        for neighbor in adjacency[node]:
            if neighbor != parent_node[node]:
                parent_node[neighbor] = node
                order.append(neighbor)
    height = [0] * node_count
    for node in reversed(order[1:]):
        neighbor = parent_node[node]
        height[neighbor] = max(height[neighbor], height[node] + 1)
    elimination_levels: list[list[int]] = [[] for _ in range(height[root])]
    # Ascending node order preserves the previous sorted leaves exactly.
    for node, level in enumerate(height):
        if node != root:
            elimination_levels[level].append(node)

    node_body = [*bodies, *([-1] * len(joints))]
    node_row = [*([-1] * len(bodies)), *range(len(joints))]
    coupling_side = [-1] * node_count
    for node in range(node_count):
        if node == root:
            continue
        neighbor = parent_node[node]
        row = node_row[node] if node_row[node] >= 0 else node_row[neighbor]
        body = node_body[node] if node_body[node] >= 0 else node_body[neighbor]
        joint = joints[row]
        coupling_side[node] = 0 if int(parent[joint]) == body else 1

    return _Tree(
        joints,
        bodies,
        node_body,
        node_row,
        parent_node,
        coupling_side,
        elimination_levels,
        root,
    )


def _union_find_root(representative: dict[int, int], body: int) -> int:
    """Return and path-compress one disjoint-set representative."""
    while representative[body] != body:
        representative[body] = representative[representative[body]]
        body = representative[body]
    return body


def _build_trees(
    model,
    path_joint_sets: set[frozenset[int]],
    body_inv_mass: np.ndarray,
    supported_joint_types: set[int] | None = None,
    ignore_free_completion_joints=False,
) -> list[_Tree]:
    """Return eligible branched structural trees not handled by path CR."""
    joint_enabled = np.asarray(model.joint_enabled.numpy(), dtype=bool)
    joint_type = np.asarray(model.joint_type.numpy(), dtype=np.int64)
    parent = np.asarray(model.joint_parent.numpy(), dtype=np.int64)
    child = np.asarray(model.joint_child.numpy(), dtype=np.int64)
    structural_joints = _eligible_structural_joints(model, supported_joint_types)
    if not structural_joints:
        return []

    body_to_structural: dict[int, list[int]] = defaultdict(list)
    for joint in structural_joints:
        if parent[joint] >= 0:
            body_to_structural[int(parent[joint])].append(joint)
        body_to_structural[int(child[joint])].append(joint)

    all_body_to_joints: dict[int, set[int]] = defaultdict(set)
    for joint, enabled in enumerate(joint_enabled):
        if (
            not enabled
            or child[joint] < 0
            or (ignore_free_completion_joints and joint_type[joint] == int(JointType.FREE))
        ):
            continue
        if parent[joint] >= 0:
            all_body_to_joints[int(parent[joint])].add(joint)
        all_body_to_joints[int(child[joint])].add(joint)

    joint_neighbors = {joint: set() for joint in structural_joints}
    for incident in body_to_structural.values():
        for joint in incident:
            joint_neighbors[joint].update(other for other in incident if other != joint)

    trees: list[_Tree] = []
    unvisited = set(structural_joints)
    while unvisited:
        seed = min(unvisited)
        component: set[int] = set()
        stack = [seed]
        while stack:
            joint = stack.pop()
            if joint in component:
                continue
            component.add(joint)
            unvisited.discard(joint)
            stack.extend(joint_neighbors[joint] - component)

        if frozenset(component) in path_joint_sets:
            continue
        component_bodies = {
            body for joint in component for body in (int(parent[joint]), int(child[joint])) if body >= 0
        }
        if any(all_body_to_joints[body] - component for body in component_bodies):
            continue

        for dynamic_component in _dynamic_joint_components(component, parent, child, body_inv_mass):
            tree = _make_tree(
                dynamic_component,
                parent,
                child,
                body_inv_mass,
                root_at_anchor=_has_rigid_angular_rows(dynamic_component, joint_type),
            )
            if tree is not None:
                trees.append(tree)
    return trees


def _dynamic_joint_components(
    joints: set[int],
    parent: np.ndarray,
    child: np.ndarray,
    body_inv_mass: np.ndarray,
) -> list[list[int]]:
    """Split at prescribed bodies, retaining every dynamic boundary row.

    Static/kinematic poses are known data, not shared KKT unknowns. Their
    incident rows therefore do not couple otherwise disconnected dynamic
    systems. Rows with two prescribed endpoints have no body correction.
    """
    representative = {
        body: body
        for joint in joints
        for body in (int(parent[joint]), int(child[joint]))
        if body >= 0 and body_inv_mass[body] > 0.0
    }
    for joint in sorted(joints):
        p, c = int(parent[joint]), int(child[joint])
        if p in representative and c in representative:
            p_root = _union_find_root(representative, p)
            c_root = _union_find_root(representative, c)
            representative[max(p_root, c_root)] = min(p_root, c_root)
    components: dict[int, list[int]] = defaultdict(list)
    for joint in sorted(joints):
        p, c = int(parent[joint]), int(child[joint])
        endpoint = p if p in representative else c
        if endpoint in representative:
            components[_union_find_root(representative, endpoint)].append(joint)
    return sorted(components.values(), key=lambda component: component[0])


def _build_closed_trees(
    model,
    handled_joint_sets: set[frozenset[int]],
    body_inv_mass: np.ndarray,
    supported_joint_types: set[int] | None = None,
    ignore_free_completion_joints=False,
) -> list[_ClosedTree]:
    """Represent complete cyclic islands as a spanning tree plus closure rows."""
    joint_type = np.asarray(model.joint_type.numpy(), dtype=np.int64)
    joint_enabled = np.asarray(model.joint_enabled.numpy(), dtype=bool)
    parent = np.asarray(model.joint_parent.numpy(), dtype=np.int64)
    child = np.asarray(model.joint_child.numpy(), dtype=np.int64)
    structural_joints = _eligible_structural_joints(model, supported_joint_types)
    if not structural_joints:
        return []

    body_to_structural: dict[int, list[int]] = defaultdict(list)
    for joint in structural_joints:
        if parent[joint] >= 0:
            body_to_structural[int(parent[joint])].append(joint)
        body_to_structural[int(child[joint])].append(joint)

    all_body_to_joints: dict[int, set[int]] = defaultdict(set)
    for joint, enabled in enumerate(joint_enabled):
        if (
            not enabled
            or child[joint] < 0
            or (ignore_free_completion_joints and joint_type[joint] == int(JointType.FREE))
        ):
            continue
        if parent[joint] >= 0:
            all_body_to_joints[int(parent[joint])].add(joint)
        all_body_to_joints[int(child[joint])].add(joint)

    neighbors = {joint: set() for joint in structural_joints}
    for incident in body_to_structural.values():
        for joint in incident:
            neighbors[joint].update(other for other in incident if other != joint)

    closed_trees: list[_ClosedTree] = []
    unvisited = set(structural_joints)
    while unvisited:
        seed = min(unvisited)
        component: set[int] = set()
        stack = [seed]
        while stack:
            joint = stack.pop()
            if joint in component:
                continue
            component.add(joint)
            unvisited.discard(joint)
            stack.extend(neighbors[joint] - component)

        if frozenset(component) in handled_joint_sets:
            continue
        component_bodies = {
            body for joint in component for body in (int(parent[joint]), int(child[joint])) if body >= 0
        }
        if any(all_body_to_joints[body] - component for body in component_bodies):
            continue
        for dynamic_component in _dynamic_joint_components(component, parent, child, body_inv_mass):
            if frozenset(dynamic_component) in handled_joint_sets:
                continue
            dynamic_bodies = {
                body
                for joint in dynamic_component
                for body in (int(parent[joint]), int(child[joint]))
                if body >= 0 and body_inv_mass[body] > 0.0
            }
            representative = {body: body for body in dynamic_bodies}
            base_joints: list[int] = []
            closure_joints: list[int] = []
            for joint in dynamic_component:
                parent_body = int(parent[joint])
                child_body = int(child[joint])
                if parent_body not in dynamic_bodies or child_body not in dynamic_bodies:
                    # Static/world attachments remain rows in their dynamic component.
                    base_joints.append(joint)
                    continue
                parent_root = _union_find_root(representative, parent_body)
                child_root = _union_find_root(representative, child_body)
                if parent_root != child_root:
                    representative[child_root] = parent_root
                    base_joints.append(joint)
                else:
                    closure_joints.append(joint)

            if not closure_joints:
                continue
            base = _make_tree(
                base_joints,
                parent,
                child,
                body_inv_mass,
                root_at_anchor=_has_rigid_angular_rows(dynamic_component, joint_type),
            )
            if base is not None and set(base.bodies) == dynamic_bodies:
                closed_trees.append(_ClosedTree(base, closure_joints, dynamic_component, base.bodies))
    return closed_trees


@wp.kernel
def equilibrate_tree_rows(
    compliance: wp.array[wp.spatial_matrix],
    row_scale: wp.array[wp.spatial_vector],
):
    row = wp.tid()
    scale = wp.spatial_vector()
    for axis in range(6):
        scale[axis] = 1.0 / wp.sqrt(wp.max(wp.abs(compliance[row][axis, axis]), 1.0e-30))
    row_scale[row] = scale


@wp.kernel
def equilibrate_tree_bodies(
    body_slots: wp.array[wp.int32],
    body_matrix: wp.array[wp.spatial_matrix],
    body_scale: wp.array[wp.spatial_vector],
):
    body = body_slots[wp.tid()]
    scale = wp.spatial_vector()
    for axis in range(6):
        scale[axis] = 1.0 / wp.sqrt(wp.max(wp.abs(body_matrix[body][axis, axis]), 1.0e-30))
    body_scale[body] = scale


# Closed islands retain double precision through equilibration, elimination,
# closure assembly, and recovery; only the final body correction is rounded.
@wp.func
def _scale_spatial_matrix_fp64(
    left_scale: wp.spatial_vector,
    matrix: wp.spatial_matrix,
    right_scale: wp.spatial_vector,
):
    result = wp.spatial_matrixd(0.0)
    for row in range(6):
        for column in range(6):
            result[row, column] = (
                wp.float64(left_scale[row]) * wp.float64(matrix[row, column]) * wp.float64(right_scale[column])
            )
    return result


@wp.func
def _scale_spatial_vector_fp64(scale: wp.spatial_vector, value: wp.spatial_vector):
    result = wp.spatial_vectord()
    for row in range(6):
        result[row] = wp.float64(scale[row]) * wp.float64(value[row])
    return result


@wp.kernel
def _initialize_tree_nodes_fp64(
    node_body: wp.array[wp.int32],
    node_row: wp.array[wp.int32],
    body_matrix: wp.array[wp.spatial_matrix],
    body_rhs: wp.array[wp.spatial_vector],
    body_scale: wp.array[wp.spatial_vector],
    compliance: wp.array[wp.spatial_matrix],
    residual: wp.array[wp.spatial_vector],
    row_scale: wp.array[wp.spatial_vector],
    diagonal: wp.array[wp.spatial_matrixd],
    rhs: wp.array[wp.spatial_vectord],
):
    node = wp.tid()
    body = node_body[node]
    if body >= 0:
        scale = body_scale[body]
        diagonal[node] = _scale_spatial_matrix_fp64(scale, body_matrix[body], scale)
        rhs[node] = _scale_spatial_vector_fp64(scale, body_rhs[body])
    else:
        row = node_row[node]
        scale = row_scale[row]
        diagonal[node] = -_scale_spatial_matrix_fp64(scale, compliance[row], scale)
        rhs[node] = -_scale_spatial_vector_fp64(scale, residual[row])


@wp.kernel
def _initialize_tree_couplings_fp64(
    node_body: wp.array[wp.int32],
    node_row: wp.array[wp.int32],
    parent_node: wp.array[wp.int32],
    coupling_side: wp.array[wp.int32],
    body_scale: wp.array[wp.spatial_vector],
    row_scale: wp.array[wp.spatial_vector],
    jacobian_parent: wp.array[wp.spatial_matrix],
    jacobian_child: wp.array[wp.spatial_matrix],
    coupling: wp.array[wp.spatial_matrixd],
):
    node = wp.tid()
    parent = parent_node[node]
    if parent < 0:
        coupling[node] = wp.spatial_matrixd(0.0)
        return
    row = node_row[node]
    node_is_joint = row >= 0
    if not node_is_joint:
        row = node_row[parent]
    body = node_body[parent] if node_is_joint else node_body[node]
    jacobian = jacobian_parent[row] if coupling_side[node] == 0 else jacobian_child[row]
    value = _scale_spatial_matrix_fp64(row_scale[row], jacobian, body_scale[body])
    coupling[node] = value if node_is_joint else wp.transpose(value)


@wp.kernel
def _initialize_closure_response_fp64(
    node_body: wp.array[wp.int32],
    tree_node_count: int,
    closure_count: int,
    closure_parent_node: wp.array[wp.int32],
    closure_child_node: wp.array[wp.int32],
    body_scale: wp.array[wp.spatial_vector],
    row_scale: wp.array[wp.spatial_vector],
    jacobian_parent: wp.array[wp.spatial_matrix],
    jacobian_child: wp.array[wp.spatial_matrix],
    response_rhs: wp.array[wp.spatial_matrixd],
):
    """Build the six tree right-hand sides induced by every closure row."""
    index = wp.tid()
    node = index // closure_count
    batch = node // tree_node_count
    closure = batch * closure_count + index % closure_count
    value = wp.spatial_matrixd(0.0)
    if node == closure_parent_node[closure]:
        coupling = _scale_spatial_matrix_fp64(row_scale[closure], jacobian_parent[closure], body_scale[node_body[node]])
        value = value + wp.transpose(coupling)
    if node == closure_child_node[closure]:
        coupling = _scale_spatial_matrix_fp64(row_scale[closure], jacobian_child[closure], body_scale[node_body[node]])
        value = value + wp.transpose(coupling)
    response_rhs[index] = value


@wp.kernel
def _eliminate_tree_nodes_fp64(
    tree_node_count: int,
    elimination_nodes: wp.array[int],
    closure_count: int,
    parent_node: wp.array[int],
    coupling: wp.array[wp.spatial_matrixd],
    diagonal: wp.array[wp.spatial_matrixd],
    rhs: wp.array[wp.spatial_vectord],
    response: wp.array[wp.spatial_matrixd],
):
    """Eliminate one CPU tree in the existing message order."""
    base = wp.tid() * tree_node_count
    for index in range(elimination_nodes.shape[0]):
        node = base + elimination_nodes[index]
        parent = parent_node[node]
        inverse = _inverse_spatial_pivoted(diagonal[node])
        a = coupling[node]
        diagonal[node] = inverse
        diagonal[parent] = diagonal[parent] - wp.transpose(a) * inverse * a
        rhs[parent] = rhs[parent] - wp.transpose(a) * inverse * rhs[node]
        for closure in range(closure_count):
            target = parent * closure_count + closure
            response[target] = response[target] - (wp.transpose(a) * inverse * response[node * closure_count + closure])


@wp.kernel
def _solve_tree_roots_fp64(
    roots: wp.array[int],
    closure_count: int,
    diagonal: wp.array[wp.spatial_matrixd],
    rhs: wp.array[wp.spatial_vectord],
    response: wp.array[wp.spatial_matrixd],
):
    """Solve the remaining root block and its closure responses."""
    root = roots[wp.tid()]
    inverse = _inverse_spatial_pivoted(diagonal[root])
    rhs[root] = inverse * rhs[root]
    for closure in range(closure_count):
        index = root * closure_count + closure
        response[index] = inverse * response[index]


@wp.kernel
def _back_substitute_tree_nodes_fp64(
    tree_node_count: int,
    elimination_nodes: wp.array[int],
    closure_count: int,
    parent_node: wp.array[int],
    coupling: wp.array[wp.spatial_matrixd],
    diagonal: wp.array[wp.spatial_matrixd],
    rhs: wp.array[wp.spatial_vectord],
    response: wp.array[wp.spatial_matrixd],
):
    """Recover CPU tree motion and closure responses without level launches."""
    base = wp.tid() * tree_node_count
    for reverse in range(elimination_nodes.shape[0]):
        node = base + elimination_nodes[elimination_nodes.shape[0] - 1 - reverse]
        parent = parent_node[node]
        rhs[node] = diagonal[node] * (rhs[node] - coupling[node] * rhs[parent])
        for closure in range(closure_count):
            target = node * closure_count + closure
            response[target] = diagonal[node] * (
                response[target] - coupling[node] * response[parent * closure_count + closure]
            )


@wp.kernel
def _solve_block_dense_serial_fp64(
    block_count: int,
    matrix: wp.array[wp.spatial_matrixd],
    rhs_solution: wp.array[wp.spatial_vectord],
):
    """Solve one dense SPD block system per batch with Gaussian elimination."""
    batch = wp.tid()
    matrix_base = batch * block_count * block_count
    rhs_base = batch * block_count
    for pivot in range(block_count):
        pivot_index = matrix_base + pivot * block_count + pivot
        pivot_inverse = _inverse_spatial_pivoted(matrix[pivot_index])
        matrix[pivot_index] = pivot_inverse
        for row in range(pivot + 1, block_count):
            factor = matrix[matrix_base + row * block_count + pivot] * pivot_inverse
            for column in range(pivot + 1, block_count):
                target = matrix_base + row * block_count + column
                matrix[target] = matrix[target] - factor * matrix[matrix_base + pivot * block_count + column]
            rhs_solution[rhs_base + row] = rhs_solution[rhs_base + row] - factor * rhs_solution[rhs_base + pivot]
    for reverse_index in range(block_count):
        row = block_count - 1 - reverse_index
        value = rhs_solution[rhs_base + row]
        for column in range(row + 1, block_count):
            value = value - matrix[matrix_base + row * block_count + column] * rhs_solution[rhs_base + column]
        rhs_solution[rhs_base + row] = matrix[matrix_base + row * block_count + row] * value


@wp.kernel
def _assemble_closure_rhs_fp64(
    node_body: wp.array[wp.int32],
    closure_parent_node: wp.array[wp.int32],
    closure_child_node: wp.array[wp.int32],
    body_scale: wp.array[wp.spatial_vector],
    row_scale: wp.array[wp.spatial_vector],
    jacobian_parent: wp.array[wp.spatial_matrix],
    jacobian_child: wp.array[wp.spatial_matrix],
    residual: wp.array[wp.spatial_vector],
    tree_solution: wp.array[wp.spatial_vectord],
    closure_rhs: wp.array[wp.spatial_vectord],
):
    closure = wp.tid()
    rhs = _scale_spatial_vector_fp64(row_scale[closure], residual[closure])
    parent_node = closure_parent_node[closure]
    if parent_node >= 0:
        coupling = _scale_spatial_matrix_fp64(
            row_scale[closure],
            jacobian_parent[closure],
            body_scale[node_body[parent_node]],
        )
        rhs = rhs + coupling * tree_solution[parent_node]
    child_node = closure_child_node[closure]
    if child_node >= 0:
        coupling = _scale_spatial_matrix_fp64(
            row_scale[closure],
            jacobian_child[closure],
            body_scale[node_body[child_node]],
        )
        rhs = rhs + coupling * tree_solution[child_node]
    closure_rhs[closure] = rhs


@wp.kernel
def _assemble_closure_schur_fp64(
    closure_count: int,
    node_body: wp.array[wp.int32],
    closure_parent_node: wp.array[wp.int32],
    closure_child_node: wp.array[wp.int32],
    body_scale: wp.array[wp.spatial_vector],
    row_scale: wp.array[wp.spatial_vector],
    jacobian_parent: wp.array[wp.spatial_matrix],
    jacobian_child: wp.array[wp.spatial_matrix],
    compliance: wp.array[wp.spatial_matrix],
    response_solution: wp.array[wp.spatial_matrixd],
    schur: wp.array[wp.spatial_matrixd],
):
    index = wp.tid()
    row = index // closure_count
    column = index % closure_count
    local_row = row % closure_count
    value = wp.spatial_matrixd(0.0)
    if local_row == column:
        value = _scale_spatial_matrix_fp64(row_scale[row], compliance[row], row_scale[row])
    parent_node = closure_parent_node[row]
    if parent_node >= 0:
        coupling = _scale_spatial_matrix_fp64(row_scale[row], jacobian_parent[row], body_scale[node_body[parent_node]])
        value = value + coupling * response_solution[parent_node * closure_count + column]
    child_node = closure_child_node[row]
    if child_node >= 0:
        coupling = _scale_spatial_matrix_fp64(row_scale[row], jacobian_child[row], body_scale[node_body[child_node]])
        value = value + coupling * response_solution[child_node * closure_count + column]
    schur[index] = value


@wp.kernel
def _scatter_closed_tree_body_correction_fp64(
    body_nodes: wp.array[wp.int32],
    body_slots: wp.array[wp.int32],
    body_row_offsets: wp.array[wp.int32],
    body_rows: wp.array[wp.int32],
    tree_row_active: wp.array[wp.int32],
    body_closure_offsets: wp.array[wp.int32],
    body_closure_rows: wp.array[wp.int32],
    closure_row_active: wp.array[wp.int32],
    tree_body_count: int,
    closure_count: int,
    row_node: wp.array[wp.int32],
    parent_node: wp.array[wp.int32],
    coupling_side: wp.array[wp.int32],
    body_matrix: wp.array[wp.spatial_matrix],
    body_rhs: wp.array[wp.spatial_vector],
    body_scale: wp.array[wp.spatial_vector],
    tree_row_scale: wp.array[wp.spatial_vector],
    tree_jp: wp.array[wp.spatial_matrix],
    tree_jc: wp.array[wp.spatial_matrix],
    tree_compliance: wp.array[wp.spatial_matrix],
    tree_residual: wp.array[wp.spatial_vector],
    closure_parent_node: wp.array[wp.int32],
    closure_child_node: wp.array[wp.int32],
    closure_row_scale: wp.array[wp.spatial_vector],
    closure_jp: wp.array[wp.spatial_matrix],
    closure_jc: wp.array[wp.spatial_matrix],
    tree_solution: wp.array[wp.spatial_vectord],
    response_solution: wp.array[wp.spatial_matrixd],
    closure_multiplier: wp.array[wp.spatial_vectord],
    body_correction: wp.array[wp.spatial_vector],
):
    index = wp.tid()
    batch = index // tree_body_count
    node = body_nodes[index]
    slot = body_slots[index]
    active = int(0)
    for cursor in range(body_row_offsets[index], body_row_offsets[index + 1]):
        active = active + tree_row_active[body_rows[cursor]]
    for cursor in range(body_closure_offsets[index], body_closure_offsets[index + 1]):
        active = active + closure_row_active[body_closure_rows[cursor]]
    if active == 0:
        body_correction[slot] = wp.spatial_vector()
        return

    bscale = wp.spatial_vector()
    for axis in range(6):
        bscale[axis] = 1.0 / wp.sqrt(wp.max(wp.abs(body_matrix[slot][axis, axis]), 1.0e-30))
    # Detect whether this body owns a single degree-1 world-anchor leaf row in the spanning tree.
    # Such a leaf row u has parent_node[u] == node and only one dynamic endpoint (the other is < 0).
    # Eliminating u before node adds K_u ~ 1e12 onto D_node in float64; recovering mu_u from the
    # body-first 12x12 Schur system S_u = C_u + A_u * D_b^-1 * A_u^T avoids K_u * (J*dx - c) cancellation.
    leaf_count = int(0)
    leaf_row = int(-1)
    leaf_node = int(-1)
    for cursor in range(body_row_offsets[index], body_row_offsets[index + 1]):
        r_idx = body_rows[cursor]
        raw_node = row_node[r_idx]
        is_leaf_row = raw_node < -1
        r_node = (-2 - raw_node) if is_leaf_row else raw_node
        if is_leaf_row and r_node >= 0 and parent_node[r_node] == node:
            leaf_count += 1
            leaf_row = r_idx
            leaf_node = r_node

    correction = tree_solution[node]
    for closure in range(closure_count):
        correction = (
            correction
            - response_solution[node * closure_count + closure] * closure_multiplier[batch * closure_count + closure]
        )

    if leaf_count == 1 and leaf_node >= 0:
        f_eff = _scale_spatial_vector_fp64(bscale, body_rhs[slot])
        for cursor in range(body_row_offsets[index], body_row_offsets[index + 1]):
            r_idx = body_rows[cursor]
            if r_idx == leaf_row:
                continue
            raw_node = row_node[r_idx]
            r_node = (-2 - raw_node) if raw_node < -1 else raw_node
            mu_r = tree_solution[r_node]
            for closure in range(closure_count):
                mu_r = (
                    mu_r
                    - response_solution[r_node * closure_count + closure]
                    * closure_multiplier[batch * closure_count + closure]
                )
            side = coupling_side[r_node] if parent_node[r_node] == node else coupling_side[node]
            jac = tree_jp[r_idx] if side == 0 else tree_jc[r_idx]
            a_rb = _scale_spatial_matrix_fp64(tree_row_scale[r_idx], jac, bscale)
            f_eff = f_eff - wp.transpose(a_rb) * mu_r
        for cursor in range(body_closure_offsets[index], body_closure_offsets[index + 1]):
            c_idx = body_closure_rows[cursor]
            mu_c = closure_multiplier[c_idx]
            if closure_parent_node[c_idx] == node:
                a_cb = _scale_spatial_matrix_fp64(closure_row_scale[c_idx], closure_jp[c_idx], bscale)
                f_eff = f_eff - wp.transpose(a_cb) * mu_c
            if closure_child_node[c_idx] == node:
                a_cb = _scale_spatial_matrix_fp64(closure_row_scale[c_idx], closure_jc[c_idx], bscale)
                f_eff = f_eff - wp.transpose(a_cb) * mu_c
        d_b = _scale_spatial_matrix_fp64(bscale, body_matrix[slot], bscale)
        inv_db = _inverse_spatial_pivoted(d_b)
        leaf_jac = tree_jp[leaf_row] if coupling_side[leaf_node] == 0 else tree_jc[leaf_row]
        rscale = tree_row_scale[leaf_row]
        a_ub = _scale_spatial_matrix_fp64(rscale, leaf_jac, bscale)
        c_u = _scale_spatial_matrix_fp64(rscale, tree_compliance[leaf_row], rscale)
        rhs_u = _scale_spatial_vector_fp64(rscale, tree_residual[leaf_row])
        schur_u = c_u + a_ub * inv_db * wp.transpose(a_ub)
        mu_u = _inverse_spatial_pivoted(schur_u) * (rhs_u + a_ub * (inv_db * f_eff))
        correction = inv_db * (f_eff - wp.transpose(a_ub) * mu_u)
        # Store mu_u_total + resp_u * mu_closure into tree_solution[leaf_node] so
        # tree_solution[leaf_node] - resp_u * mu_closure equals exact mu_u!
        stored_leaf = mu_u
        for closure in range(closure_count):
            stored_leaf = (
                stored_leaf
                + response_solution[leaf_node * closure_count + closure]
                * closure_multiplier[batch * closure_count + closure]
            )
        tree_solution[leaf_node] = stored_leaf

    rounded = wp.spatial_vector()
    for axis in range(6):
        rounded[axis] = float(wp.float64(bscale[axis]) * correction[axis])
    body_correction[slot] = rounded


@wp.kernel
def initialize_tree_nodes(
    node_body: wp.array[wp.int32],
    node_row: wp.array[wp.int32],
    body_matrix: wp.array[wp.spatial_matrix],
    body_rhs: wp.array[wp.spatial_vector],
    body_scale: wp.array[wp.spatial_vector],
    compliance: wp.array[wp.spatial_matrix],
    residual: wp.array[wp.spatial_vector],
    row_scale: wp.array[wp.spatial_vector],
    diagonal: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
):
    node = wp.tid()
    body = node_body[node]
    if body >= 0:
        scale = body_scale[body]
        diagonal[node] = _scale_spatial_matrix(scale, body_matrix[body], scale)
        rhs[node] = _scale_spatial_vector(scale, body_rhs[body])
    else:
        row = node_row[node]
        scale = row_scale[row]
        diagonal[node] = -_scale_spatial_matrix(scale, compliance[row], scale)
        rhs[node] = -_scale_spatial_vector(scale, residual[row])


@wp.func
def _diag_equilibration_scale(matrix: wp.spatial_matrix):
    scale = wp.spatial_vector()
    for axis in range(6):
        scale[axis] = 1.0 / wp.sqrt(wp.max(wp.abs(matrix[axis, axis]), 1.0e-30))
    return scale


@wp.kernel
def prepare_tree_system(
    node_body: wp.array[wp.int32],
    node_row: wp.array[wp.int32],
    parent_node: wp.array[wp.int32],
    coupling_side: wp.array[wp.int32],
    body_matrix: wp.array[wp.spatial_matrix],
    compliance: wp.array[wp.spatial_matrix],
    jacobian_parent: wp.array[wp.spatial_matrix],
    jacobian_child: wp.array[wp.spatial_matrix],
    body_scale: wp.array[wp.spatial_vector],
    row_scale: wp.array[wp.spatial_vector],
    coupling: wp.array[wp.spatial_matrix],
):
    """Fused bit-identical equilibrate_tree_bodies + equilibrate_tree_rows + initialize_tree_couplings."""
    node = wp.tid()
    own_body = node_body[node]
    own_bscale = wp.spatial_vector()
    own_rscale = wp.spatial_vector()
    if own_body >= 0:
        own_bscale = _diag_equilibration_scale(body_matrix[own_body])
        body_scale[own_body] = own_bscale
    else:
        own_row = node_row[node]
        own_rscale = _diag_equilibration_scale(compliance[own_row])
        row_scale[own_row] = own_rscale
    parent = parent_node[node]
    if parent < 0:
        coupling[node] = wp.spatial_matrix(0.0)
        return
    row = node_row[node]
    node_is_joint = row >= 0
    if not node_is_joint:
        row = node_row[parent]
    body = node_body[parent] if node_is_joint else own_body
    bscale = _diag_equilibration_scale(body_matrix[body]) if node_is_joint else own_bscale
    rscale = own_rscale if node_is_joint else _diag_equilibration_scale(compliance[row])
    jacobian = jacobian_parent[row] if coupling_side[node] == 0 else jacobian_child[row]
    value = _scale_spatial_matrix(rscale, jacobian, bscale)
    coupling[node] = value if node_is_joint else wp.transpose(value)


@wp.kernel
def initialize_tree_couplings(
    node_body: wp.array[wp.int32],
    node_row: wp.array[wp.int32],
    parent_node: wp.array[wp.int32],
    coupling_side: wp.array[wp.int32],
    body_scale: wp.array[wp.spatial_vector],
    row_scale: wp.array[wp.spatial_vector],
    jacobian_parent: wp.array[wp.spatial_matrix],
    jacobian_child: wp.array[wp.spatial_matrix],
    coupling: wp.array[wp.spatial_matrix],
):
    node = wp.tid()
    parent = parent_node[node]
    if parent < 0:
        coupling[node] = wp.spatial_matrix(0.0)
        return
    row = node_row[node]
    node_is_joint = row >= 0
    if not node_is_joint:
        row = node_row[parent]
    body = node_body[parent] if node_is_joint else node_body[node]
    jacobian = jacobian_parent[row] if coupling_side[node] == 0 else jacobian_child[row]
    value = _scale_spatial_matrix(row_scale[row], jacobian, body_scale[body])
    coupling[node] = value if node_is_joint else wp.transpose(value)


@wp.kernel
def invert_tree_leaves(
    leaves: wp.array[wp.int32],
    diagonal: wp.array[Any],
):
    """Factor leaves whose messages share a recipient."""
    node = leaves[wp.tid()]
    diagonal[node] = _inverse_spatial_robust(diagonal[node])


@wp.kernel
def eliminate_tree_unique_leaves(
    leaves: wp.array[wp.int32],
    parent_node: wp.array[wp.int32],
    diagonal: wp.array[Any],
    rhs: wp.array[Any],
    coupling: wp.array[Any],
    saved_inverse: wp.array[Any],
    saved_rhs: wp.array[Any],
):
    """Eliminate leaves whose recipients are unique within this level."""
    node = leaves[wp.tid()]
    parent = parent_node[node]
    inverse = _inverse_spatial_robust(diagonal[node])
    a = coupling[node]
    saved_inverse[node] = inverse
    saved_rhs[node] = rhs[node]
    diagonal[parent] = diagonal[parent] - wp.transpose(a) * inverse * a
    rhs[parent] = rhs[parent] - wp.transpose(a) * inverse * rhs[node]


@wp.kernel
def accumulate_tree_messages(
    recipients: wp.array[wp.int32],
    recipient_offsets: wp.array[wp.int32],
    message_nodes: wp.array[wp.int32],
    coupling: wp.array[Any],
    saved_inverse: wp.array[Any],
    saved_rhs: wp.array[Any],
    diagonal: wp.array[Any],
    rhs: wp.array[Any],
):
    index = wp.tid()
    recipient = recipients[index]
    diagonal_value = diagonal[recipient]
    rhs_value = rhs[recipient]
    for cursor in range(recipient_offsets[index], recipient_offsets[index + 1]):
        node = message_nodes[cursor]
        a = coupling[node]
        inverse = saved_inverse[node]
        diagonal_value = diagonal_value - wp.transpose(a) * inverse * a
        rhs_value = rhs_value - wp.transpose(a) * inverse * saved_rhs[node]
    diagonal[recipient] = diagonal_value
    rhs[recipient] = rhs_value


@wp.kernel
def solve_tree_roots(
    roots: wp.array[wp.int32],
    diagonal: wp.array[Any],
    rhs: wp.array[Any],
    solution: wp.array[Any],
):
    """Solve one independent root block per identically scheduled tree."""
    root = roots[wp.tid()]
    solution[root] = _inverse_spatial_robust(diagonal[root]) * rhs[root]


@wp.kernel
def back_substitute_tree_level(
    leaves: wp.array[wp.int32],
    parent_node: wp.array[wp.int32],
    coupling: wp.array[Any],
    saved_inverse: wp.array[Any],
    saved_rhs: wp.array[Any],
    solution: wp.array[Any],
):
    node = leaves[wp.tid()]
    parent = parent_node[node]
    solution[node] = saved_inverse[node] * (saved_rhs[node] - coupling[node] * solution[parent])


@wp.kernel(module="unique")
def solve_tree_levels_fused(
    tree_node_count: int,
    level_count: int,
    tree_level_offsets: wp.array[wp.int32],
    level_kind: wp.array[wp.int32],
    level_node_start: wp.array[wp.int32],
    level_nodes: wp.array[wp.int32],
    level_recipient_start: wp.array[wp.int32],
    level_recipients: wp.array[wp.int32],
    message_offsets: wp.array[wp.int32],
    message_nodes: wp.array[wp.int32],
    roots: wp.array[wp.int32],
    parent_node: wp.array[wp.int32],
    coupling: wp.array[Any],
    diagonal: wp.array[Any],
    rhs: wp.array[Any],
):
    """Run one tree's whole leaf-rake schedule inside a single block.

    Supports both uniform batches (``tree_node_count > 0``, sharing one schedule)
    and heterogeneous merged trees (``tree_node_count == 0``, with global node
    indices and ``tree_level_offsets`` per block), in either ``float32`` or
    ``float64``. Walking the exact level schedule inside one block per tree is
    bit-identical to per-level launches while running all trees simultaneously
    across SMs in one kernel launch.
    """
    thread = wp.tid()
    block = wp.block_dim()
    lane = thread % block
    batch = thread // block
    base = batch * tree_node_count if tree_node_count > 0 else 0
    level_begin = 0 if tree_node_count > 0 else tree_level_offsets[batch]
    level_end = level_count if tree_node_count > 0 else tree_level_offsets[batch + 1]
    span = level_end - level_begin

    for step in range(span):
        level = level_begin + step
        start = level_node_start[level]
        count = level_node_start[level + 1] - start
        if level_kind[level] == 1:
            # Unique recipients: mirror eliminate_tree_unique_leaves.
            index = int(lane)
            while index < count:
                node = base + level_nodes[start + index]
                parent = parent_node[node]
                inverse = _inverse_spatial_robust(diagonal[node])
                a = coupling[node]
                eliminated_rhs = rhs[node]
                diagonal[node] = inverse
                diagonal[parent] = diagonal[parent] - wp.transpose(a) * inverse * a
                rhs[parent] = rhs[parent] - wp.transpose(a) * inverse * eliminated_rhs
                index += block
        else:
            # Shared recipients: mirror invert_tree_leaves, then
            # accumulate_tree_messages, separated by the same barrier the
            # two launches would have provided.
            index = int(lane)
            while index < count:
                node = base + level_nodes[start + index]
                diagonal[node] = _inverse_spatial_robust(diagonal[node])
                index += block
            _synchronize_cr_block()
            recipient_start = level_recipient_start[level]
            recipient_count = level_recipient_start[level + 1] - recipient_start
            index = int(lane)
            while index < recipient_count:
                slot = recipient_start + index
                recipient = base + level_recipients[slot]
                diagonal_value = diagonal[recipient]
                rhs_value = rhs[recipient]
                for cursor in range(message_offsets[slot], message_offsets[slot + 1]):
                    node = base + message_nodes[cursor]
                    a = coupling[node]
                    inverse = diagonal[node]
                    diagonal_value = diagonal_value - wp.transpose(a) * inverse * a
                    rhs_value = rhs_value - wp.transpose(a) * inverse * rhs[node]
                diagonal[recipient] = diagonal_value
                rhs[recipient] = rhs_value
                index += block
        _synchronize_cr_block()

    if lane == 0:
        # Mirror solve_tree_roots for this block's own tree.
        root = roots[batch]
        rhs[root] = _inverse_spatial_robust(diagonal[root]) * rhs[root]
    _synchronize_cr_block()

    for reverse in range(span):
        level = level_end - 1 - reverse
        start = level_node_start[level]
        count = level_node_start[level + 1] - start
        index = int(lane)
        while index < count:
            node = base + level_nodes[start + index]
            parent = parent_node[node]
            rhs[node] = diagonal[node] * (rhs[node] - coupling[node] * rhs[parent])
            index += block
        _synchronize_cr_block()


@wp.func
def _load_tree_contraction_edge(
    edge: int,
    transpose_edge: int,
    original_edge_count: int,
    coupling: wp.array[wp.spatial_matrix],
    fill_edges: wp.array[wp.spatial_matrix],
):
    """Load one original or generated edge in a requested orientation."""
    value = coupling[edge] if edge < original_edge_count else fill_edges[edge - original_edge_count]
    return wp.transpose(value) if transpose_edge != 0 else value


@wp.kernel
def eliminate_tree_contraction_level(
    original_edge_count: int,
    nodes: wp.array[wp.int32],
    neighbors: wp.array[wp.vec2i],
    edges: wp.array[wp.vec2i],
    transpose_edges: wp.array[wp.vec2i],
    generated_edges: wp.array[wp.int32],
    diagonal: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
    coupling: wp.array[wp.spatial_matrix],
    fill_edges: wp.array[wp.spatial_matrix],
    saved_inverse: wp.array[wp.spatial_matrix],
    saved_rhs: wp.array[wp.spatial_vector],
):
    """Factor independent degree-one/two nodes and form fill edges."""
    index = wp.tid()
    node = nodes[index]
    inverse = _inverse_spatial_robust(diagonal[node])
    node_rhs = rhs[node]
    saved_inverse[node] = inverse
    saved_rhs[node] = node_rhs

    second_neighbor = neighbors[index][1]
    if second_neighbor >= 0:
        first_edge = edges[index][0]
        first_a = _load_tree_contraction_edge(
            first_edge,
            transpose_edges[index][0],
            original_edge_count,
            coupling,
            fill_edges,
        )
        second_edge = edges[index][1]
        second_a = _load_tree_contraction_edge(
            second_edge,
            transpose_edges[index][1],
            original_edge_count,
            coupling,
            fill_edges,
        )
        # The generated edge is oriented neighbor[0] -> neighbor[1].
        fill_edges[generated_edges[index] - original_edge_count] = -(wp.transpose(first_a) * inverse * second_a)


@wp.kernel
def accumulate_tree_contraction_messages(
    original_edge_count: int,
    nodes: wp.array[wp.int32],
    edges: wp.array[wp.vec2i],
    transpose_edges: wp.array[wp.vec2i],
    recipients: wp.array[wp.int32],
    recipient_offsets: wp.array[wp.int32],
    message_slots: wp.array[wp.int32],
    coupling: wp.array[wp.spatial_matrix],
    fill_edges: wp.array[wp.spatial_matrix],
    saved_inverse: wp.array[wp.spatial_matrix],
    saved_rhs: wp.array[wp.spatial_vector],
    diagonal: wp.array[wp.spatial_matrix],
    rhs: wp.array[wp.spatial_vector],
):
    """Accumulate exact Schur messages without matrix atomics."""
    index = wp.tid()
    recipient = recipients[index]
    diagonal_value = diagonal[recipient]
    rhs_value = rhs[recipient]
    for cursor in range(recipient_offsets[index], recipient_offsets[index + 1]):
        slot = message_slots[cursor]
        eliminated_index = slot // 2
        local = slot - 2 * eliminated_index
        node = nodes[eliminated_index]
        a = _load_tree_contraction_edge(
            edges[eliminated_index][local],
            transpose_edges[eliminated_index][local],
            original_edge_count,
            coupling,
            fill_edges,
        )
        inverse = saved_inverse[node]
        diagonal_value = diagonal_value - wp.transpose(a) * inverse * a
        rhs_value = rhs_value - wp.transpose(a) * inverse * saved_rhs[node]
    diagonal[recipient] = diagonal_value
    rhs[recipient] = rhs_value


@wp.kernel
def back_substitute_tree_contraction_level(
    original_edge_count: int,
    nodes: wp.array[wp.int32],
    neighbors: wp.array[wp.vec2i],
    edges: wp.array[wp.vec2i],
    transpose_edges: wp.array[wp.vec2i],
    coupling: wp.array[wp.spatial_matrix],
    fill_edges: wp.array[wp.spatial_matrix],
    saved_inverse: wp.array[wp.spatial_matrix],
    saved_rhs: wp.array[wp.spatial_vector],
    solution: wp.array[wp.spatial_vector],
):
    """Recover one exact rake/compress level in parallel."""
    index = wp.tid()
    node = nodes[index]
    value = saved_rhs[node]

    first_a = _load_tree_contraction_edge(
        edges[index][0],
        transpose_edges[index][0],
        original_edge_count,
        coupling,
        fill_edges,
    )
    value = value - first_a * solution[neighbors[index][0]]

    second_neighbor = neighbors[index][1]
    if second_neighbor >= 0:
        second_a = _load_tree_contraction_edge(
            edges[index][1],
            transpose_edges[index][1],
            original_edge_count,
            coupling,
            fill_edges,
        )
        value = value - second_a * solution[second_neighbor]
    solution[node] = saved_inverse[node] * value


@wp.func
def _actual_row_node(encoded: int):
    return encoded if encoded >= 0 else (-2 - encoded)


@wp.kernel
def build_tree_body_sqp_rhs(
    body_nodes: wp.array[wp.int32],
    body_slots: wp.array[wp.int32],
    body_row_offsets: wp.array[wp.int32],
    body_rows: wp.array[wp.int32],
    graph_body_ids: wp.array[wp.int32],
    joint_ids: wp.array[wp.int32],
    joint_parent: wp.array[int],
    row_node: wp.array[wp.int32],
    parent_node: wp.array[wp.int32],
    dt: float,
    body_q: wp.array[wp.transform],
    body_inertia_q: wp.array[wp.transform],
    body_disp: wp.array[wp.vec3],
    body_disp_ang: wp.array[wp.vec3],
    use_displacement: int,
    body_mass: wp.array[float],
    body_inv_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    body_com: wp.array[wp.vec3],
    body_matrix: wp.array[wp.spatial_matrix],
    slope_offset: int,
    slope_jacobian_parent: wp.array[wp.spatial_matrix],
    slope_jacobian_child: wp.array[wp.spatial_matrix],
    slope_mode: wp.array[wp.int32],
    row_scale: wp.array[wp.spatial_vector],
    coupling: wp.array[wp.spatial_matrix],
    prev_solution: wp.array[wp.spatial_vector],
    include_sqp_torque: int,
    body_rhs_out: wp.array[wp.spatial_vector],
    body_scale_out: wp.array[wp.spatial_vector],
):
    """Fused body inertial RHS, equilibration scale, and division-free SQP torque -(J1 - J0)^T mu^(1)."""
    index = wp.tid()
    b_node = body_nodes[index]
    slot = body_slots[index]
    body = graph_body_ids[slot]
    if body_inv_mass[body] <= 0.0:
        body_rhs_out[slot] = wp.spatial_vector()
        body_scale_out[slot] = wp.spatial_vector()
        return
    dt_inv_sq = 1.0 / (dt * dt)
    pose = body_q[body]
    zero_v = wp.vec3(0.0)
    if use_displacement != 0:
        rhs = _body_objective_rhs_displacement(
            pose,
            body_mass[body],
            body_inertia[body],
            zero_v,
            zero_v,
            dt_inv_sq,
            body_disp[body],
            body_disp_ang[body],
        )
    else:
        rhs = _body_objective_rhs(
            pose,
            body_inertia_q[body],
            body_mass[body],
            body_inertia[body],
            body_com[body],
            zero_v,
            zero_v,
            dt_inv_sq,
        )
    bscale = _diag_equilibration_scale(body_matrix[slot])
    if include_sqp_torque != 0:
        inv_bscale = wp.spatial_vector()
        for axis in range(6):
            inv_bscale[axis] = 1.0 / bscale[axis] if bscale[axis] > 0.0 else 0.0
        delta_f = wp.spatial_vector()
        for cursor in range(body_row_offsets[index], body_row_offsets[index + 1]):
            row = body_rows[cursor]
            slope_row = slope_offset + row
            if slope_mode[slope_row] == 0:
                continue
            r_node = _actual_row_node(row_node[row])
            u_r = prev_solution[r_node]
            mu_r = _scale_spatial_vector(row_scale[row], u_r)
            joint = joint_ids[row]
            j1 = slope_jacobian_parent[slope_row] if joint_parent[joint] == body else slope_jacobian_child[slope_row]
            j0_scaled_u = (
                wp.transpose(coupling[r_node]) * u_r if parent_node[r_node] == b_node else coupling[b_node] * u_r
            )
            j0_t_mu = _scale_spatial_vector(inv_bscale, j0_scaled_u)
            df = wp.transpose(j1) * mu_r - j0_t_mu
            f0_sq = wp.dot(j0_t_mu, j0_t_mu)
            df_sq = wp.dot(df, df)
            w = f0_sq / (f0_sq + 25.0 * df_sq + 1.0e-12)
            delta_f = delta_f - w * df
        rhs = rhs + delta_f
    body_rhs_out[slot] = rhs
    body_scale_out[slot] = bscale


@wp.kernel
def build_tree_body_sqp_rhs_fp64(
    body_nodes: wp.array[wp.int32],
    body_slots: wp.array[wp.int32],
    body_row_offsets: wp.array[wp.int32],
    body_rows: wp.array[wp.int32],
    graph_body_ids: wp.array[wp.int32],
    joint_ids: wp.array[wp.int32],
    joint_parent: wp.array[int],
    row_node: wp.array[wp.int32],
    dt: float,
    body_q: wp.array[wp.transform],
    body_inertia_q: wp.array[wp.transform],
    body_disp: wp.array[wp.vec3],
    body_disp_ang: wp.array[wp.vec3],
    use_displacement: int,
    body_mass: wp.array[float],
    body_inv_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    body_com: wp.array[wp.vec3],
    body_matrix: wp.array[wp.spatial_matrix],
    jacobian_parent_0: wp.array[wp.spatial_matrix],
    jacobian_child_0: wp.array[wp.spatial_matrix],
    slope_offset: int,
    slope_jacobian_parent: wp.array[wp.spatial_matrix],
    slope_jacobian_child: wp.array[wp.spatial_matrix],
    slope_mode: wp.array[wp.int32],
    row_scale: wp.array[wp.spatial_vector],
    prev_solution: wp.array[wp.spatial_vectord],
    include_sqp_torque: int,
    body_rhs_out: wp.array[wp.spatial_vector],
    body_scale_out: wp.array[wp.spatial_vector],
):
    """FP64-factor variant of build_tree_body_sqp_rhs (Stage-0 Jacobians remain intact)."""
    index = wp.tid()
    _ = body_nodes[index]
    slot = body_slots[index]
    body = graph_body_ids[slot]
    if body_inv_mass[body] <= 0.0:
        body_rhs_out[slot] = wp.spatial_vector()
        body_scale_out[slot] = wp.spatial_vector()
        return
    dt_inv_sq = 1.0 / (dt * dt)
    pose = body_q[body]
    zero_v = wp.vec3(0.0)
    if use_displacement != 0:
        rhs = _body_objective_rhs_displacement(
            pose,
            body_mass[body],
            body_inertia[body],
            zero_v,
            zero_v,
            dt_inv_sq,
            body_disp[body],
            body_disp_ang[body],
        )
    else:
        rhs = _body_objective_rhs(
            pose,
            body_inertia_q[body],
            body_mass[body],
            body_inertia[body],
            body_com[body],
            zero_v,
            zero_v,
            dt_inv_sq,
        )
    bscale = _diag_equilibration_scale(body_matrix[slot])
    if include_sqp_torque != 0:
        delta_f = wp.spatial_vector()
        for cursor in range(body_row_offsets[index], body_row_offsets[index + 1]):
            row = body_rows[cursor]
            slope_row = slope_offset + row
            if slope_mode[slope_row] == 0:
                continue
            r_node = _actual_row_node(row_node[row])
            u_r64 = prev_solution[r_node]
            u_r = wp.spatial_vector()
            for axis in range(6):
                u_r[axis] = float(u_r64[axis])
            mu_r = _scale_spatial_vector(row_scale[row], u_r)
            joint = joint_ids[row]
            is_parent = joint_parent[joint] == body
            j1 = slope_jacobian_parent[slope_row] if is_parent else slope_jacobian_child[slope_row]
            j0 = jacobian_parent_0[row] if is_parent else jacobian_child_0[row]
            j0_t_mu = wp.transpose(j0) * mu_r
            df = wp.transpose(j1) * mu_r - j0_t_mu
            f0_sq = wp.dot(j0_t_mu, j0_t_mu)
            df_sq = wp.dot(df, df)
            w = f0_sq / (f0_sq + 25.0 * df_sq + 1.0e-12)
            delta_f = delta_f - w * df
        rhs = rhs + delta_f
    body_rhs_out[slot] = rhs
    body_scale_out[slot] = bscale


@wp.kernel
def initialize_tree_saddle_resolve_rhs(
    node_body: wp.array[wp.int32],
    node_row: wp.array[wp.int32],
    slope_offset: int,
    body_rhs: wp.array[wp.spatial_vector],
    body_scale: wp.array[wp.spatial_vector],
    compliance_0: wp.array[wp.spatial_matrix],
    slope_compliance: wp.array[wp.spatial_vector],
    slope_residual: wp.array[wp.spatial_vector],
    row_scale: wp.array[wp.spatial_vector],
    rhs_out: wp.array[wp.spatial_vector],
):
    """Division-free saddle-RHS resolve: exact nonlinear c(x) and SQP g(x), uniform as C -> 0."""
    node = wp.tid()
    body = node_body[node]
    if body >= 0:
        rhs_out[node] = _scale_spatial_vector(body_scale[body], body_rhs[body])
    else:
        row = node_row[node]
        s_row = slope_offset + row
        c_res = slope_residual[s_row]
        c_cur = slope_compliance[s_row]
        c_ref = compliance_0[row]
        masked = wp.spatial_vector()
        for axis in range(6):
            c0 = c_ref[axis, axis]
            c1 = c_cur[axis]
            masked[axis] = 0.0 if wp.abs(c1 - c0) > 1.0e-5 * wp.max(wp.abs(c0), 1.0e-12) else c_res[axis]
        rhs_out[node] = -_scale_spatial_vector(row_scale[row], masked)


@wp.kernel
def initialize_tree_saddle_resolve_rhs_fp64(
    node_body: wp.array[wp.int32],
    node_row: wp.array[wp.int32],
    slope_offset: int,
    body_rhs: wp.array[wp.spatial_vector],
    body_scale: wp.array[wp.spatial_vector],
    compliance_0: wp.array[wp.spatial_matrix],
    slope_compliance: wp.array[wp.spatial_vector],
    slope_residual: wp.array[wp.spatial_vector],
    row_scale: wp.array[wp.spatial_vector],
    rhs_out: wp.array[wp.spatial_vectord],
):
    """FP64 division-free saddle-RHS resolve."""
    node = wp.tid()
    body = node_body[node]
    if body >= 0:
        rhs_out[node] = _scale_spatial_vector_fp64(body_scale[body], body_rhs[body])
    else:
        row = node_row[node]
        s_row = slope_offset + row
        c_res = slope_residual[s_row]
        c_cur = slope_compliance[s_row]
        c_ref = compliance_0[row]
        masked = wp.spatial_vector()
        for axis in range(6):
            c0 = c_ref[axis, axis]
            c1 = c_cur[axis]
            masked[axis] = 0.0 if wp.abs(c1 - c0) > 1.0e-5 * wp.max(wp.abs(c0), 1.0e-12) else c_res[axis]
        rhs_out[node] = -_scale_spatial_vector_fp64(row_scale[row], masked)


@wp.kernel
def pack_paired_tree_backbone_rhs(
    pair_count: int,
    backbone_node_count: int,
    backbone_nodes: wp.array[wp.int32],
    rhs: wp.array[wp.spatial_vector],
    pair_rhs: wp.array[_SpatialPairVector],
):
    """Pack RHS-only into paired backbone vectors without overwriting stored factors."""
    index = wp.tid()
    batch = index // pair_count
    local_pair = index - batch * pair_count
    backbone_base = batch * backbone_node_count
    first_row = 2 * local_pair
    first_node = backbone_nodes[backbone_base + first_row]
    rhs_value = _SpatialPairVector()
    rhs_value.v0 = rhs[first_node]
    rhs_value.v1 = wp.spatial_vector()
    if first_row + 1 < backbone_node_count:
        second_node = backbone_nodes[backbone_base + first_row + 1]
        rhs_value.v1 = rhs[second_node]
    pair_rhs[index] = rhs_value


@wp.kernel
def resolve_tree_unique_leaves(
    leaves: wp.array[wp.int32],
    parent_node: wp.array[wp.int32],
    coupling: wp.array[Any],
    saved_inverse: wp.array[Any],
    rhs: wp.array[Any],
):
    """Forward RHS-only leaf elimination reusing stored pivot inverses."""
    node = leaves[wp.tid()]
    parent = parent_node[node]
    rhs[parent] = rhs[parent] - wp.transpose(coupling[node]) * (saved_inverse[node] * rhs[node])


@wp.kernel
def resolve_tree_messages(
    recipients: wp.array[wp.int32],
    recipient_offsets: wp.array[wp.int32],
    message_nodes: wp.array[wp.int32],
    coupling: wp.array[Any],
    saved_inverse: wp.array[Any],
    saved_rhs: wp.array[Any],
    rhs: wp.array[Any],
):
    """Forward RHS-only shared-recipient elimination reusing stored pivot inverses."""
    index = wp.tid()
    recipient = recipients[index]
    rhs_value = rhs[recipient]
    for cursor in range(recipient_offsets[index], recipient_offsets[index + 1]):
        node = message_nodes[cursor]
        rhs_value = rhs_value - wp.transpose(coupling[node]) * (saved_inverse[node] * saved_rhs[node])
    rhs[recipient] = rhs_value


@wp.kernel(module="unique")
def resolve_tree_levels_fused(
    tree_node_count: int,
    level_count: int,
    tree_level_offsets: wp.array[wp.int32],
    level_kind: wp.array[wp.int32],
    level_node_start: wp.array[wp.int32],
    level_nodes: wp.array[wp.int32],
    level_recipient_start: wp.array[wp.int32],
    level_recipients: wp.array[wp.int32],
    message_offsets: wp.array[wp.int32],
    message_nodes: wp.array[wp.int32],
    roots: wp.array[wp.int32],
    parent_node: wp.array[wp.int32],
    coupling: wp.array[Any],
    diagonal: wp.array[Any],
    rhs: wp.array[Any],
):
    """Single-block RHS-only forward/back substitution reusing Stage-0 tree factors."""
    thread = wp.tid()
    block = wp.block_dim()
    lane = thread % block
    batch = thread // block
    base = batch * tree_node_count if tree_node_count > 0 else 0
    level_begin = 0 if tree_node_count > 0 else tree_level_offsets[batch]
    level_end = level_count if tree_node_count > 0 else tree_level_offsets[batch + 1]
    span = level_end - level_begin

    for step in range(span):
        level = level_begin + step
        start = level_node_start[level]
        count = level_node_start[level + 1] - start
        if level_kind[level] == 1:
            index = int(lane)
            while index < count:
                node = base + level_nodes[start + index]
                parent = parent_node[node]
                rhs[parent] = rhs[parent] - wp.transpose(coupling[node]) * (diagonal[node] * rhs[node])
                index += block
        else:
            recipient_start = level_recipient_start[level]
            recipient_count = level_recipient_start[level + 1] - recipient_start
            index = int(lane)
            while index < recipient_count:
                slot = recipient_start + index
                recipient = base + level_recipients[slot]
                rhs_value = rhs[recipient]
                for cursor in range(message_offsets[slot], message_offsets[slot + 1]):
                    node = base + message_nodes[cursor]
                    rhs_value = rhs_value - wp.transpose(coupling[node]) * (diagonal[node] * rhs[node])
                rhs[recipient] = rhs_value
                index += block
        _synchronize_cr_block()

    if lane == 0:
        root = roots[batch]
        rhs[root] = _inverse_spatial_robust(diagonal[root]) * rhs[root]
    _synchronize_cr_block()

    for reverse in range(span):
        level = level_end - 1 - reverse
        start = level_node_start[level]
        count = level_node_start[level + 1] - start
        index = int(lane)
        while index < count:
            node = base + level_nodes[start + index]
            parent = parent_node[node]
            rhs[node] = diagonal[node] * (rhs[node] - coupling[node] * rhs[parent])
            index += block
        _synchronize_cr_block()


@wp.kernel
def resolve_tree_contraction_messages(
    original_edge_count: int,
    nodes: wp.array[wp.int32],
    edges: wp.array[wp.vec2i],
    transpose_edges: wp.array[wp.vec2i],
    recipients: wp.array[wp.int32],
    recipient_offsets: wp.array[wp.int32],
    message_slots: wp.array[wp.int32],
    coupling: wp.array[wp.spatial_matrix],
    fill_edges: wp.array[wp.spatial_matrix],
    saved_inverse: wp.array[wp.spatial_matrix],
    saved_rhs: wp.array[wp.spatial_vector],
    rhs: wp.array[wp.spatial_vector],
):
    """Forward RHS-only rake/compress message accumulation reusing Stage-0 factors."""
    index = wp.tid()
    recipient = recipients[index]
    rhs_value = rhs[recipient]
    for cursor in range(recipient_offsets[index], recipient_offsets[index + 1]):
        slot = message_slots[cursor]
        eliminated_index = slot // 2
        local = slot - 2 * eliminated_index
        node = nodes[eliminated_index]
        a = _load_tree_contraction_edge(
            edges[eliminated_index][local],
            transpose_edges[eliminated_index][local],
            original_edge_count,
            coupling,
            fill_edges,
        )
        rhs_value = rhs_value - wp.transpose(a) * (saved_inverse[node] * saved_rhs[node])
    rhs[recipient] = rhs_value


@wp.kernel
def scatter_tree_body_correction_indexed(
    body_nodes: wp.array[wp.int32],
    body_slots: wp.array[wp.int32],
    body_row_offsets: wp.array[wp.int32],
    body_rows: wp.array[wp.int32],
    row_active: wp.array[wp.int32],
    body_scale: wp.array[wp.spatial_vector],
    solution: wp.array[Any],
    body_correction: wp.array[wp.spatial_vector],
):
    """Scatter body nodes from multiple flattened tree instances."""
    index = wp.tid()
    node = body_nodes[index]
    slot = body_slots[index]
    active = int(0)
    for cursor in range(body_row_offsets[index], body_row_offsets[index + 1]):
        active = active + row_active[body_rows[cursor]]
    if active == 0:
        body_correction[slot] = wp.spatial_vector()
        return
    correction = _scale_spatial_vector(body_scale[slot], solution[node])
    body_correction[slot] = correction


@wp.kernel
def initialize_closure_response(
    node_body: wp.array[wp.int32],
    tree_node_count: int,
    closure_count: int,
    closure_parent_node: wp.array[wp.int32],
    closure_child_node: wp.array[wp.int32],
    body_scale: wp.array[wp.spatial_vector],
    row_scale: wp.array[wp.spatial_vector],
    jacobian_parent: wp.array[wp.spatial_matrix],
    jacobian_child: wp.array[wp.spatial_matrix],
    response_rhs: wp.array[wp.spatial_matrix],
):
    """Build the six tree right-hand sides induced by every closure row."""
    index = wp.tid()
    node = index // closure_count
    batch = node // tree_node_count
    closure = batch * closure_count + index % closure_count
    value = wp.spatial_matrix(0.0)
    if node == closure_parent_node[closure]:
        coupling = _scale_spatial_matrix(row_scale[closure], jacobian_parent[closure], body_scale[node_body[node]])
        value = value + wp.transpose(coupling)
    if node == closure_child_node[closure]:
        coupling = _scale_spatial_matrix(row_scale[closure], jacobian_child[closure], body_scale[node_body[node]])
        value = value + wp.transpose(coupling)
    response_rhs[index] = value


@wp.kernel
def eliminate_tree_response_unique_leaves(
    leaves: wp.array[wp.int32],
    closure_count: int,
    parent_node: wp.array[wp.int32],
    coupling: wp.array[Any],
    saved_inverse: wp.array[Any],
    response_rhs: wp.array[Any],
):
    index = wp.tid()
    node = leaves[index // closure_count]
    closure = index % closure_count
    parent = parent_node[node]
    response_rhs[parent * closure_count + closure] = response_rhs[parent * closure_count + closure] - (
        wp.transpose(coupling[node]) * saved_inverse[node] * response_rhs[node * closure_count + closure]
    )


@wp.kernel
def accumulate_tree_response_messages(
    recipients: wp.array[wp.int32],
    recipient_offsets: wp.array[wp.int32],
    message_nodes: wp.array[wp.int32],
    closure_count: int,
    coupling: wp.array[Any],
    saved_inverse: wp.array[Any],
    response_rhs: wp.array[Any],
):
    index = wp.tid()
    recipient_index = index // closure_count
    closure = index % closure_count
    recipient = recipients[recipient_index]
    value = response_rhs[recipient * closure_count + closure]
    for cursor in range(recipient_offsets[recipient_index], recipient_offsets[recipient_index + 1]):
        node = message_nodes[cursor]
        value = value - (
            wp.transpose(coupling[node]) * saved_inverse[node] * response_rhs[node * closure_count + closure]
        )
    response_rhs[recipient * closure_count + closure] = value


@wp.kernel
def back_substitute_tree_response_level(
    leaves: wp.array[wp.int32],
    closure_count: int,
    parent_node: wp.array[wp.int32],
    coupling: wp.array[Any],
    saved_inverse: wp.array[Any],
    response_rhs: wp.array[Any],
    response_solution: wp.array[Any],
):
    index = wp.tid()
    node = leaves[index // closure_count]
    closure = index % closure_count
    parent = parent_node[node]
    response_solution[node * closure_count + closure] = saved_inverse[node] * (
        response_rhs[node * closure_count + closure]
        - coupling[node] * response_solution[parent * closure_count + closure]
    )


@wp.kernel
def assemble_closure_rhs(
    node_body: wp.array[wp.int32],
    closure_parent_node: wp.array[wp.int32],
    closure_child_node: wp.array[wp.int32],
    body_scale: wp.array[wp.spatial_vector],
    row_scale: wp.array[wp.spatial_vector],
    jacobian_parent: wp.array[wp.spatial_matrix],
    jacobian_child: wp.array[wp.spatial_matrix],
    residual: wp.array[wp.spatial_vector],
    tree_solution: wp.array[wp.spatial_vector],
    closure_rhs: wp.array[wp.spatial_vector],
):
    closure = wp.tid()
    rhs = _scale_spatial_vector(row_scale[closure], residual[closure])
    parent_node = closure_parent_node[closure]
    if parent_node >= 0:
        coupling = _scale_spatial_matrix(
            row_scale[closure],
            jacobian_parent[closure],
            body_scale[node_body[parent_node]],
        )
        rhs = rhs + coupling * tree_solution[parent_node]
    child_node = closure_child_node[closure]
    if child_node >= 0:
        coupling = _scale_spatial_matrix(
            row_scale[closure],
            jacobian_child[closure],
            body_scale[node_body[child_node]],
        )
        rhs = rhs + coupling * tree_solution[child_node]
    closure_rhs[closure] = rhs


@wp.kernel
def assemble_closure_schur(
    closure_count: int,
    node_body: wp.array[wp.int32],
    closure_parent_node: wp.array[wp.int32],
    closure_child_node: wp.array[wp.int32],
    body_scale: wp.array[wp.spatial_vector],
    row_scale: wp.array[wp.spatial_vector],
    jacobian_parent: wp.array[wp.spatial_matrix],
    jacobian_child: wp.array[wp.spatial_matrix],
    compliance: wp.array[wp.spatial_matrix],
    response_solution: wp.array[wp.spatial_matrix],
    schur: wp.array[wp.spatial_matrix],
):
    index = wp.tid()
    row = index // closure_count
    column = index % closure_count
    local_row = row % closure_count
    value = wp.spatial_matrix(0.0)
    if local_row == column:
        value = _scale_spatial_matrix(row_scale[row], compliance[row], row_scale[row])
    parent_node = closure_parent_node[row]
    if parent_node >= 0:
        coupling = _scale_spatial_matrix(row_scale[row], jacobian_parent[row], body_scale[node_body[parent_node]])
        value = value + coupling * response_solution[parent_node * closure_count + column]
    child_node = closure_child_node[row]
    if child_node >= 0:
        coupling = _scale_spatial_matrix(row_scale[row], jacobian_child[row], body_scale[node_body[child_node]])
        value = value + coupling * response_solution[child_node * closure_count + column]
    schur[index] = value


@wp.kernel
def solve_block_dense_serial(
    block_count: int,
    matrix: wp.array[wp.spatial_matrix],
    rhs_solution: wp.array[wp.spatial_vector],
):
    """Solve one dense SPD block system per batch with Gaussian elimination."""
    batch = wp.tid()
    matrix_base = batch * block_count * block_count
    rhs_base = batch * block_count
    for pivot in range(block_count):
        pivot_index = matrix_base + pivot * block_count + pivot
        pivot_inverse = _inverse_spatial_robust(matrix[pivot_index])
        matrix[pivot_index] = pivot_inverse
        for row in range(pivot + 1, block_count):
            factor = matrix[matrix_base + row * block_count + pivot] * pivot_inverse
            for column in range(pivot + 1, block_count):
                target = matrix_base + row * block_count + column
                matrix[target] = matrix[target] - factor * matrix[matrix_base + pivot * block_count + column]
            rhs_solution[rhs_base + row] = rhs_solution[rhs_base + row] - factor * rhs_solution[rhs_base + pivot]
    for reverse_index in range(block_count):
        row = block_count - 1 - reverse_index
        value = rhs_solution[rhs_base + row]
        for column in range(row + 1, block_count):
            value = value - matrix[matrix_base + row * block_count + column] * rhs_solution[rhs_base + column]
        rhs_solution[rhs_base + row] = matrix[matrix_base + row * block_count + row] * value


@wp.kernel
def invert_block_dense_pivot(
    pivot: int,
    block_count: int,
    matrix: wp.array[Any],
):
    """Invert one block pivot in every independent dense system."""
    batch = wp.tid()
    index = batch * block_count * block_count + pivot * block_count + pivot
    matrix[index] = _inverse_spatial_robust(matrix[index])


@wp.kernel
def factor_block_ldlt_pivot_rows(
    pivot: int,
    block_count: int,
    matrix: wp.array[Any],
    rhs: wp.array[Any],
):
    """Form one block-LDLT column and forward-eliminate its rhs rows."""
    index = wp.tid()
    remaining = block_count - pivot - 1
    batch = index // remaining
    row = pivot + 1 + index - batch * remaining
    if row >= block_count:
        return
    matrix_base = batch * block_count * block_count
    rhs_base = batch * block_count
    multiplier = matrix[matrix_base + row * block_count + pivot] * matrix[matrix_base + pivot * block_count + pivot]
    # The strict upper triangle is scratch once symmetry is used. Store
    # L(row,pivot) transposed by block index, without another factor buffer.
    matrix[matrix_base + pivot * block_count + row] = multiplier
    rhs[rhs_base + row] = rhs[rhs_base + row] - multiplier * rhs[rhs_base + pivot]


@wp.kernel
def update_block_ldlt_lower_triangle(
    pivot: int,
    block_count: int,
    trailing_count: int,
    matrix: wp.array[Any],
):
    """Update the symmetric trailing block matrix in parallel."""
    index = wp.tid()
    trailing_size = trailing_count * trailing_count
    batch = index // trailing_size
    local_index = index - batch * trailing_size
    local_row = local_index // trailing_count
    local_column = local_index - local_row * trailing_count
    if local_column > local_row:
        return
    row = pivot + 1 + local_row
    column = pivot + 1 + local_column
    matrix_base = batch * block_count * block_count
    multiplier = matrix[matrix_base + pivot * block_count + row]
    original_column = matrix[matrix_base + column * block_count + pivot]
    target = matrix_base + row * block_count + column
    matrix[target] = matrix[target] - multiplier * wp.transpose(original_column)


@wp.kernel
def solve_block_ldlt_diagonal(
    block_count: int,
    matrix: wp.array[Any],
    rhs_solution: wp.array[Any],
):
    """Apply independent inverse diagonal blocks after forward elimination."""
    index = wp.tid()
    batch = index // block_count
    row = index - batch * block_count
    if row < block_count:
        rhs_solution[index] = matrix[batch * block_count * block_count + row * block_count + row] * rhs_solution[index]


@wp.kernel
def back_substitute_block_ldlt_serial(
    block_count: int,
    matrix: wp.array[Any],
    rhs_solution: wp.array[Any],
):
    """Back-substitute one block-LDLT system per batch."""
    batch = wp.tid()
    matrix_base = batch * block_count * block_count
    rhs_base = batch * block_count
    for reverse_index in range(block_count):
        row = block_count - 1 - reverse_index
        value = rhs_solution[rhs_base + row]
        for column in range(row + 1, block_count):
            multiplier = matrix[matrix_base + row * block_count + column]
            value = value - wp.transpose(multiplier) * rhs_solution[rhs_base + column]
        rhs_solution[rhs_base + row] = value


@wp.kernel
def back_substitute_block_ldlt_lanes(
    block_count: int,
    matrix: wp.array[Any],
    partial: wp.array[Any],
    rhs_solution: wp.array[Any],
):
    """Back-substitute one dense block-LDLT system with lane-parallel products.

    The row loop is irreducibly sequential, but within a row the products over
    higher columns are independent. Measured on a 127-block closure system, the
    serial form spends about 360 ns per block operation -- roughly 870 cycles --
    because one thread waits on its own dependent 144-byte matrix load with
    nothing else in flight. Simply unrolling that loop is counterproductive:
    holding several 6x6 temporaries at once exceeds the register budget and
    spills, measured 33% *slower*.

    So the products are spread across lanes, one block per system, and written
    to a scratch column. Lane zero then subtracts them in **ascending column
    order**, exactly as the serial loop did. Bit-identity depends on that: the
    partial products are unchanged values and their accumulation order is
    unchanged, so every intermediate rounding matches. A cross-lane tree
    reduction would be faster still and is deliberately not used, because
    regrouping a floating-point sum would change the result.
    """
    thread = wp.tid()
    width = wp.block_dim()
    lane = thread % width
    batch = thread // width
    matrix_base = batch * block_count * block_count
    rhs_base = batch * block_count
    for reverse_index in range(block_count):
        row = block_count - 1 - reverse_index
        row_base = matrix_base + row * block_count
        column = row + 1 + int(lane)
        while column < block_count:
            partial[rhs_base + column] = wp.transpose(matrix[row_base + column]) * rhs_solution[rhs_base + column]
            column += width
        _synchronize_cr_block()
        if lane == 0:
            value = rhs_solution[rhs_base + row]
            for ordered in range(row + 1, block_count):
                value = value - partial[rhs_base + ordered]
            rhs_solution[rhs_base + row] = value
        _synchronize_cr_block()


@wp.kernel
def solve_block_ldlt_persistent(
    block_count: int,
    matrix: wp.array[Any],
    rhs: wp.array[Any],
):
    """Replay dense block LDLT within one CUDA block per independent system.

    The lower triangle holds Schur updates; the strict upper triangle stores
    L by transposed block index, exactly as in the multi-launch schedule.
    Launch complete blocks: every lane must reach every block-wide barrier.
    """
    tid = wp.tid()
    width = wp.block_dim()
    lane = tid % width
    batch = tid // width
    matrix_base = batch * block_count * block_count
    rhs_base = batch * block_count
    for pivot in range(block_count):
        if lane == 0:
            index = matrix_base + pivot * block_count + pivot
            matrix[index] = _inverse_spatial_robust(matrix[index])
        _synchronize_cr_block()
        row = pivot + 1 + lane
        while row < block_count:
            multiplier = (
                matrix[matrix_base + row * block_count + pivot] * matrix[matrix_base + pivot * block_count + pivot]
            )
            matrix[matrix_base + pivot * block_count + row] = multiplier
            rhs[rhs_base + row] = rhs[rhs_base + row] - multiplier * rhs[rhs_base + pivot]
            row += width
        _synchronize_cr_block()
        remaining = block_count - pivot - 1
        index = int(lane)
        while index < remaining * remaining:
            row = pivot + 1 + index // remaining
            column = pivot + 1 + index % remaining
            if column <= row:
                target = matrix_base + row * block_count + column
                matrix[target] = matrix[target] - matrix[matrix_base + pivot * block_count + row] * wp.transpose(
                    matrix[matrix_base + column * block_count + pivot]
                )
            index += width
        _synchronize_cr_block()
    row = int(lane)
    while row < block_count:
        rhs[rhs_base + row] = matrix[matrix_base + row * block_count + row] * rhs[rhs_base + row]
        row += width
    _synchronize_cr_block()
    if lane == 0:
        for reverse_index in range(block_count):
            row = block_count - 1 - reverse_index
            value = rhs[rhs_base + row]
            for column in range(row + 1, block_count):
                value = value - wp.transpose(matrix[matrix_base + row * block_count + column]) * rhs[rhs_base + column]
            rhs[rhs_base + row] = value


@wp.kernel
def scatter_closed_tree_body_correction(
    body_nodes: wp.array[wp.int32],
    body_slots: wp.array[wp.int32],
    body_row_offsets: wp.array[wp.int32],
    body_rows: wp.array[wp.int32],
    tree_row_active: wp.array[wp.int32],
    body_closure_offsets: wp.array[wp.int32],
    body_closure_rows: wp.array[wp.int32],
    closure_row_active: wp.array[wp.int32],
    tree_body_count: int,
    closure_count: int,
    body_scale: wp.array[wp.spatial_vector],
    tree_solution: wp.array[wp.spatial_vector],
    response_solution: wp.array[wp.spatial_matrix],
    closure_multiplier: wp.array[wp.spatial_vector],
    body_correction: wp.array[wp.spatial_vector],
):
    index = wp.tid()
    batch = index // tree_body_count
    node = body_nodes[index]
    slot = body_slots[index]
    active = int(0)
    for cursor in range(body_row_offsets[index], body_row_offsets[index + 1]):
        active = active + tree_row_active[body_rows[cursor]]
    for cursor in range(body_closure_offsets[index], body_closure_offsets[index + 1]):
        active = active + closure_row_active[body_closure_rows[cursor]]
    if active == 0:
        body_correction[slot] = wp.spatial_vector()
        return
    correction = tree_solution[node]
    for closure in range(closure_count):
        correction = (
            correction
            - response_solution[node * closure_count + closure] * closure_multiplier[batch * closure_count + closure]
        )
    correction = _scale_spatial_vector(body_scale[slot], correction)
    body_correction[slot] = correction


@dataclass
class _PathCRSchedule:
    """Host routes and merged level rows for the paths of one bucket."""

    persistent: list[tuple[int, int]]
    tails: list[tuple[int, int, int]]
    coarse: list[tuple[int, int, int]]
    levels: list[tuple[int, list[tuple[int, int, int]], list[tuple[int, int, int]]]]


def _path_cr_schedule(row_counts: list[int], fused: bool) -> _PathCRSchedule:
    """Route back-to-back path rows and merge equal-stride CR levels.

    With ``fused`` (CUDA), a path of at most ``_CR_PERSISTENT_MAX_ROWS`` rows
    is solved by one block. Longer paths share one launch per level until at
    most ``_CR_TAIL_MAX_ROWS`` rows remain, then finish in one block each.
    Otherwise every path reduces to ``_CR_SERIAL_TERMINAL_SIZE`` rows and
    finishes with block Thomas. Every path keeps the elimination order of a
    standalone solve, so batching paths of different lengths is exact. Level
    entries store the path's first row, the local row, and the path row count.
    """
    persistent: list[tuple[int, int]] = []
    tails: list[tuple[int, int, int]] = []
    coarse: list[tuple[int, int, int]] = []
    eliminated: list[list[tuple[int, int, int]]] = []
    survivors: list[list[tuple[int, int, int]]] = []
    limit = _CR_TAIL_MAX_ROWS if fused else _CR_SERIAL_TERMINAL_SIZE
    start = 0
    for count in row_counts:
        if fused and count <= _CR_PERSISTENT_MAX_ROWS:
            persistent.append((start, count))
        else:
            stride = 1
            level = 0
            while (count + stride - 1) // stride > limit:
                if level == len(eliminated):
                    eliminated.append([])
                    survivors.append([])
                eliminated[level].extend((start, row, count) for row in range(stride, count, 2 * stride))
                survivors[level].extend((start, row, count) for row in range(0, count, 2 * stride))
                stride *= 2
                level += 1
            (tails if fused else coarse).append((start, count, stride))
        start += count
    levels = [(1 << level, eliminated[level], survivors[level]) for level in range(len(eliminated))]
    return _PathCRSchedule(persistent, tails, coarse, levels)


def _path_cr_schedule_bytes(schedule: _PathCRSchedule) -> int:
    """Count the device arrays allocated for one path CR schedule."""
    return (
        _array_payload_bytes(len(schedule.persistent), wp.vec2i)
        + _array_payload_bytes(len(schedule.tails), wp.vec3i)
        + _array_payload_bytes(len(schedule.coarse), wp.vec3i)
        + sum(
            _array_payload_bytes(len(eliminated), wp.vec3i) + _array_payload_bytes(len(survivors), wp.vec3i)
            for _, eliminated, survivors in schedule.levels
        )
    )


class _PathBucket:
    """Batch of joint paths of any lengths, stored back to back."""

    def __init__(self, paths: list[_Path], device, *, factor_dtype=wp.float32):
        self.factor_dtype = factor_dtype
        self.device = device
        self.spatial_block_dim = _SPATIAL_GPU_BLOCK_DIM if device.is_cuda else 1
        self.batch_count = len(paths)
        self.row_counts = [len(path.joints) for path in paths]
        self.size = sum(self.row_counts)
        self.body_size = sum(len(path.bodies) for path in paths)

        joint_ids: list[int] = []
        body_ids: list[int] = []
        body_incident_rows: list[tuple[int, int]] = []
        row_bounds: list[tuple[int, int]] = []
        start = 0
        for path in paths:
            end = start + len(path.joints)
            joint_ids.extend(path.joints)
            body_ids.extend(path.bodies)
            row_bounds.extend([(start, end)] * (end - start))
            for incident in path.body_rows:
                encoded = [((start + row) << 1) | side for row, side in incident]
                body_incident_rows.append(
                    (
                        encoded[0] if encoded else -1,
                        encoded[1] if len(encoded) > 1 else -1,
                    )
                )
            start = end

        self.joint_ids_host = np.asarray(joint_ids, dtype=np.int64)
        self.joint_ids = wp.array(self.joint_ids_host, dtype=wp.int32, device=device)
        self.row_body = None
        self.row_bounds = wp.array(np.asarray(row_bounds, dtype=np.int32), dtype=wp.vec2i, device=device)
        self.body_ids_host = np.asarray(body_ids, dtype=np.int32)
        self.body_ids = wp.array(self.body_ids_host, dtype=wp.int32, device=device)
        self.body_incident_rows = wp.array(body_incident_rows, dtype=wp.vec2i, device=device)
        self.jacobian_parent = wp.empty(self.size, dtype=wp.spatial_matrix, device=device)
        self.jacobian_child = wp.empty_like(self.jacobian_parent)
        self.lower = [wp.empty_like(self.jacobian_parent)]
        self.diagonal = [wp.empty_like(self.jacobian_parent)]
        self.upper = [wp.empty_like(self.jacobian_parent)]
        self.rhs = [wp.empty(self.size, dtype=wp.spatial_vector, device=device)]
        self.row_active = wp.zeros(self.size, dtype=wp.int32, device=device)

        # Linearization storage is dead as soon as the path system is
        # assembled. Reuse the diagonal and rhs buffers for compliance and
        # residual, then overwrite them with CR factors and the solution.
        self.compliance = self.diagonal[0]
        self.residual = self.rhs[0]
        self.solution = self.rhs
        if factor_dtype == wp.float64:
            self.factor_lower = wp.empty(self.size, dtype=wp.spatial_matrixd, device=device)
            self.factor_diagonal = wp.empty_like(self.factor_lower)
            self.factor_upper = wp.empty_like(self.factor_lower)
            self.factor_rhs = wp.empty(self.size, dtype=wp.spatial_vectord, device=device)
        else:
            self.factor_lower = None
            self.factor_diagonal = None
            self.factor_upper = None
            self.factor_rhs = None

        self.plan_cr(fused=device.is_cuda)

    def bind_body_endpoints(self, parent: np.ndarray, child: np.ndarray, body_slot: dict[int, int]):
        self.row_body = wp.array(
            [
                (
                    body_slot.get(int(parent[joint]), -1),
                    body_slot.get(int(child[joint]), -1),
                )
                for joint in self.joint_ids_host
            ],
            dtype=wp.vec2i,
            device=self.device,
        )

    def plan_cr(self, *, fused: bool):
        """Allocate the fused CUDA schedule, or the plain level schedule used on CPU."""

        def device_array(values, dtype):
            if not values:
                return None
            return wp.array(np.asarray(values, dtype=np.int32), dtype=dtype, device=self.device)

        self.cr_fused = bool(fused)
        self.cr_schedule = _path_cr_schedule(self.row_counts, fused)
        self.persistent_paths = device_array(self.cr_schedule.persistent, wp.vec2i)
        self.tail_paths = device_array(self.cr_schedule.tails, wp.vec3i)
        self.coarse_paths = device_array(self.cr_schedule.coarse, wp.vec3i)
        self.cr_levels = [
            (stride, device_array(eliminated, wp.vec3i), device_array(survivors, wp.vec3i))
            for stride, eliminated, survivors in self.cr_schedule.levels
        ]
        # Each path takes exactly one route and owns its rows, so CUDA may run
        # the persistent route beside the level route without changing arithmetic.
        long_route = bool(self.cr_levels) or self.tail_paths is not None
        self.cr_side_stream = (
            wp.Stream(self.device) if fused and self.persistent_paths is not None and long_route else None
        )
        # Scalar aliases share the four owned row buffers; no extra device storage.
        self.cr_float_buffers = (
            [
                array.view(wp.float64)
                for array in (self.factor_lower, self.factor_diagonal, self.factor_upper, self.factor_rhs)
            ]
            if self.device.is_cuda and fused and long_route and self.factor_dtype == wp.float64
            else [array.view(float) for array in (self.lower[0], self.diagonal[0], self.upper[0], self.rhs[0])]
            if self.device.is_cuda and fused and long_route
            else None
        )

    def solve_rows(self):
        """Solve every block-tridiagonal path in four in-place row buffers."""
        use_fp64 = self.factor_dtype == wp.float64
        if (
            use_fp64
            and self.persistent_paths is not None
            and not self.cr_levels
            and self.tail_paths is None
            and self.coarse_paths is None
        ):
            wp.launch(
                solve_cr_persistent_fp64_io,
                self.persistent_paths.shape[0] * self.spatial_block_dim,
                inputs=[
                    self.persistent_paths,
                    _CR_SERIAL_TERMINAL_SIZE,
                    self.lower[0],
                    self.diagonal[0],
                    self.upper[0],
                    self.rhs[0],
                ],
                outputs=[self.factor_lower, self.factor_diagonal, self.factor_upper, self.factor_rhs],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
            return
        if use_fp64:
            wp.launch(
                _cast_path_rows_to_fp64,
                self.size,
                inputs=[self.lower[0], self.diagonal[0], self.upper[0], self.rhs[0]],
                outputs=[self.factor_lower, self.factor_diagonal, self.factor_upper, self.factor_rhs],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
            buffers = [self.factor_lower, self.factor_diagonal, self.factor_upper, self.factor_rhs]
            buffers_f = self.cr_float_buffers
        else:
            buffers = [self.lower[0], self.diagonal[0], self.upper[0], self.rhs[0]]
            buffers_f = self.cr_float_buffers
        invert_limit = _CR_TAIL_MAX_ROWS if self.cr_fused else _CR_SERIAL_TERMINAL_SIZE
        side = self.cr_side_stream
        if side is not None:
            # The level route is the longer dependent chain: start it first, join before recovery.
            main = wp.get_stream(self.device)
            side.wait_stream(main)
            with wp.ScopedStream(side, sync_enter=False):
                self._solve_long_rows(buffers, buffers_f, invert_limit)
        if self.persistent_paths is not None:
            wp.launch(
                solve_cr_persistent,
                self.persistent_paths.shape[0] * self.spatial_block_dim,
                inputs=[self.persistent_paths, _CR_SERIAL_TERMINAL_SIZE],
                outputs=buffers,
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
        if side is not None:
            main.wait_stream(side)
        else:
            self._solve_long_rows(buffers, buffers_f, invert_limit)
        if use_fp64:
            wp.launch(
                _cast_path_solution_to_fp32,
                self.size,
                inputs=[self.factor_rhs],
                outputs=[self.rhs[0]],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )

    def _solve_long_rows(self, buffers, buffers_f, invert_limit):
        """Reduce paths too long for one block level by level, then finish their tails."""
        for level_idx, (stride, eliminated, survivors) in enumerate(self.cr_levels):
            if buffers_f is not None or level_idx == 0:
                wp.launch(
                    invert_cr_eliminated_in_place,
                    eliminated.shape[0],
                    inputs=[eliminated],
                    outputs=[buffers[1]],
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )
            if buffers_f is not None:
                wp.launch(
                    reduce_cr_in_place_rows_fp64 if self.factor_dtype == wp.float64 else reduce_cr_in_place_rows,
                    (survivors.shape[0], 6),
                    inputs=[stride, survivors],
                    outputs=buffers_f,
                    device=self.device,
                    block_dim=128,
                )
            else:
                wp.launch(
                    reduce_cr_in_place,
                    survivors.shape[0],
                    inputs=[stride, invert_limit, survivors],
                    outputs=buffers,
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )
        if self.tail_paths is not None:
            tail_kernel = (
                (
                    solve_cr_tail_persistent_rows_fp64
                    if self.factor_dtype == wp.float64
                    else solve_cr_tail_persistent_rows
                )
                if buffers_f is not None
                else solve_cr_tail_persistent
            )
            wp.launch(
                tail_kernel,
                self.tail_paths.shape[0] * self.spatial_block_dim,
                inputs=[self.tail_paths],
                outputs=buffers + buffers_f if buffers_f is not None else buffers,
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
        if self.coarse_paths is not None:
            wp.launch(
                solve_strided_cr_coarse,
                self.coarse_paths.shape[0],
                inputs=[self.coarse_paths],
                outputs=buffers,
                device=self.device,
            )
        for stride, eliminated, _ in reversed(self.cr_levels):
            wp.launch(
                back_substitute_cr_in_place,
                eliminated.shape[0],
                inputs=[stride, eliminated],
                outputs=buffers,
                device=self.device,
                block_dim=self.spatial_block_dim,
            )


class _TreeBucket:
    """A batch of graph-capturable trees with exact symbolic schedules.

    Trees share one normalized topology, or every tree takes the contraction
    route and trees of different topologies are stored back to back.
    """

    def __init__(
        self,
        trees: _Tree | list[_Tree],
        body_slot: dict[int, int],
        device,
        *,
        symbolics: _TreeBucketSymbolics | dict[int, _TreeBucketSymbolics] | None = None,
        allow_fused_tree_levels: bool = True,
        factor_dtype=wp.float32,
    ):
        if isinstance(trees, _Tree):
            trees = [trees]
        tree = trees[0]
        (
            uniform,
            tree_symbolics,
            node_counts,
            node_offsets,
            parent_node,
            levels,
            branch_levels,
            backbone_nodes,
        ) = _tree_bucket_layout(trees, symbolics)
        symbolics = tree_symbolics[0]

        matrix_dtype = wp.spatial_matrixd if factor_dtype == wp.float64 else wp.spatial_matrix
        vector_dtype = wp.spatial_vectord if factor_dtype == wp.float64 else wp.spatial_vector
        self.factor_dtype = factor_dtype
        self.device = device
        self.spatial_block_dim = _SPATIAL_GPU_BLOCK_DIM if device.is_cuda else 1
        self.batch_count = len(trees)
        # Per-tree sizes exist only for one normalized topology; contracted
        # trees of different topologies are stored back to back.
        self.tree_row_count = len(tree.joints) if uniform else None
        self.tree_body_count = len(tree.bodies) if uniform else None
        self.tree_node_count = len(tree.node_body) if uniform else None
        self.row_count = sum(len(candidate.joints) for candidate in trees)
        self.body_count = sum(len(candidate.bodies) for candidate in trees)
        self.node_count = sum(node_counts)
        self.size = self.row_count
        self.body_size = self.body_count

        joint_ids: list[int] = []
        body_ids: list[int] = []
        node_body: list[int] = []
        body_nodes: list[int] = []
        body_slots: list[int] = []
        body_row_offsets: list[int] = [0]
        body_rows: list[int] = []
        node_row: list[int] = []
        coupling_side: list[int] = []
        roots: list[int] = []
        row_offset = 0
        for candidate, node_offset in zip(trees, node_offsets, strict=True):
            joint_ids.extend(candidate.joints)
            body_ids.extend(candidate.bodies)
            node_body.extend(body_slot[body] if body >= 0 else -1 for body in candidate.node_body)
            body_nodes.extend(node_offset + node for node in range(len(candidate.bodies)))
            body_slots.extend(body_slot[body] for body in candidate.bodies)
            for body_node in range(len(candidate.bodies)):
                incident_rows = []
                for node, row in enumerate(candidate.node_row):
                    if row < 0:
                        continue
                    if candidate.parent_node[node] == body_node or candidate.parent_node[body_node] == node:
                        incident_rows.append(row_offset + row)
                body_rows.extend(incident_rows)
                body_row_offsets.append(len(body_rows))
            node_row.extend(row_offset + row if row >= 0 else -1 for row in candidate.node_row)
            coupling_side.extend(candidate.coupling_side)
            roots.append(node_offset + candidate.root)
            row_offset += len(candidate.joints)

        self.root = roots[0]
        self.roots = wp.array(roots, dtype=wp.int32, device=device)
        self.joint_ids_host = np.asarray(joint_ids, dtype=np.int64)
        self.joint_ids = wp.array(self.joint_ids_host, dtype=wp.int32, device=device)
        self.body_ids_host = np.asarray(body_ids, dtype=np.int32)
        self.node_body = wp.array(node_body, dtype=wp.int32, device=device)
        self.body_nodes = wp.array(body_nodes, dtype=wp.int32, device=device)
        # In the common single-tree case body nodes occupy the leading slots,
        # so retain the original zero-copy alias.  Batched trees interleave
        # body and row nodes per instance and therefore need an explicit map.
        self.body_slots = (
            self.node_body if self.batch_count == 1 else wp.array(body_slots, dtype=wp.int32, device=device)
        )
        self.body_row_offsets = wp.array(body_row_offsets, dtype=wp.int32, device=device)
        self.body_rows = wp.array(body_rows, dtype=wp.int32, device=device)
        self.node_row = wp.array(node_row, dtype=wp.int32, device=device)
        parent_set = set(parent_node)
        row_node_host = np.full(self.row_count, -1, dtype=np.int32)
        for n_i, r_i in enumerate(node_row):
            if r_i >= 0:
                row_node_host[r_i] = (-2 - n_i) if n_i not in parent_set else n_i
        self.row_node = wp.array(row_node_host, dtype=wp.int32, device=device)
        self.parent_node = wp.array(parent_node, dtype=wp.int32, device=device)
        self.coupling_side = wp.array(coupling_side, dtype=wp.int32, device=device)
        # Float32 open trees reuse dead row Jacobians for numeric factors.
        # Precise closed trees retain these source rows in separate storage.
        self.matrix_workspace = wp.zeros(
            max(2 * self.row_count, self.node_count),
            dtype=wp.spatial_matrix,
            device=device,
        )
        self.jacobian_parent = self.matrix_workspace[: self.row_count]
        self.jacobian_child = self.matrix_workspace[self.row_count : 2 * self.row_count]
        self.diagonal = (
            wp.zeros(self.node_count, dtype=matrix_dtype, device=device)
            if factor_dtype == wp.float64
            else self.matrix_workspace[: self.node_count]
        )
        self.compliance = wp.zeros(self.row_count, dtype=wp.spatial_matrix, device=device)
        self.row_scale = wp.zeros(self.row_count, dtype=wp.spatial_vector, device=device)
        self.residual = wp.zeros(self.row_count, dtype=wp.spatial_vector, device=device)
        self.row_active = wp.zeros(self.row_count, dtype=wp.int32, device=device)
        self.row_slices = None
        self.rhs = wp.zeros(self.node_count, dtype=vector_dtype, device=device)
        self.coupling = wp.zeros(self.node_count, dtype=matrix_dtype, device=device)
        # Eliminated node blocks and rhs values are dead in the forward pass;
        # keep their inverse and later solution in those same node slots.
        self.saved_inverse = self.diagonal
        self.saved_rhs = self.rhs
        self.solution = self.rhs

        self.levels = _make_device_tree_levels(levels, parent_node, device)
        # The leaf-rake schedule is a chain of dependent launches, one per
        # level, so a long chain pays O(bodies) launch depth for two-thread
        # grids. Keep the exact per-level path and add a bit-identical
        # single-block replay of the same levels for schedules that fit.
        self.tree_levels_host = tree.elimination_levels if uniform else None
        self.use_fused_tree_levels = bool(
            allow_fused_tree_levels and _batched_fused_tree_levels_supported(trees, tree_symbolics, device)
        )
        if self.use_fused_tree_levels:
            if uniform:
                self.fused_tree_node_count = int(self.tree_node_count)
                self.fused_level_count = len(tree.elimination_levels)
                # Zero-copy placeholder: uniform batches never read tree_level_offsets.
                self.fused_tree_level_offsets = self.roots
                (
                    self.fused_level_kind,
                    self.fused_level_node_start,
                    self.fused_level_nodes,
                    self.fused_level_recipient_start,
                    self.fused_level_recipients,
                    self.fused_message_offsets,
                    self.fused_message_nodes,
                ) = _make_fused_tree_levels(tree.elimination_levels, tree.parent_node, device)
            else:
                self.fused_tree_node_count = 0
                self.fused_level_count = 0
                (
                    self.fused_tree_level_offsets,
                    self.fused_level_kind,
                    self.fused_level_node_start,
                    self.fused_level_nodes,
                    self.fused_level_recipient_start,
                    self.fused_level_recipients,
                    self.fused_message_offsets,
                    self.fused_message_nodes,
                ) = _make_batched_fused_tree_levels(trees, node_offsets, device)
        self.backbone_node_count = len(symbolics.backbone) if uniform else None
        self.backbone_nodes = wp.array(backbone_nodes, dtype=wp.int32, device=device)
        self.branch_levels = _make_device_tree_levels(branch_levels, parent_node, device)
        self.backbone_lower = wp.zeros(len(backbone_nodes), dtype=matrix_dtype, device=device)
        self.backbone_upper = wp.zeros_like(self.backbone_lower)
        self.backbone_cr_levels: list[tuple[int, int, int]] = []
        stride = 1
        active_count = self.backbone_node_count if uniform else 0
        while active_count > _CR_SERIAL_TERMINAL_SIZE:
            eliminated_count = len(range(stride, self.backbone_node_count, 2 * stride))
            survivor_count = len(range(0, self.backbone_node_count, 2 * stride))
            self.backbone_cr_levels.append((stride, eliminated_count, survivor_count))
            stride *= 2
            active_count = survivor_count
        self.backbone_terminal_stride = stride
        # Select once from immutable topology. Open trees use oracle-covered
        # leaf rake or contraction; closed-tree owners reuse the backbone
        # workspace below for paired or closure-response solves.
        self.use_tree_contraction = symbolics.use_tree_contraction
        self.use_backbone_cr = symbolics.use_backbone_cr
        if self.use_tree_contraction:
            if any(candidate.contraction is None for candidate in tree_symbolics):
                raise RuntimeError("Tree contraction was selected without a symbolic schedule")
            schedules = [candidate.contraction for candidate in tree_symbolics]
            self.contraction_levels = _make_device_batched_tree_contraction(schedules, node_counts, device)
            contraction_roots = [
                offset + candidate.contraction.root
                for candidate, offset in zip(tree_symbolics, node_offsets, strict=True)
            ]
            self.contraction_root = contraction_roots[0]
            self.contraction_roots = wp.array(contraction_roots, dtype=wp.int32, device=device)
            self.contraction_fill_edges = wp.zeros(
                sum(schedule.edge_count - count for schedule, count in zip(schedules, node_counts, strict=True)),
                dtype=wp.spatial_matrix,
                device=device,
            )
        else:
            self.contraction_levels = []
            self.contraction_root = -1
            self.contraction_roots = None
            self.contraction_fill_edges = None
        # The shared backbone kernels also propagate optional closure-response
        # right-hand sides.  A tree has none; one dummy element keeps the same
        # kernels and graph shape without allocating per-node response storage.
        self.no_closure_response = wp.zeros(1, dtype=matrix_dtype, device=device)
        self.use_paired_open_backbone = symbolics.use_paired_open_backbone
        # An open backbone borrows _ClosedTreeBucket._solve_paired_backbone_system
        # unbound, so it must carry that method's schedule flags. It has no
        # closure columns, so the lane split has nothing to widen.
        self.use_lane_split_coarse = False
        self.supports_sqp_chord_resolve = False
        self.chain_route = None
        if self.use_paired_open_backbone:
            pair_matrix_dtype = _SpatialPairMatrixD if factor_dtype == wp.float64 else _SpatialPairMatrix
            pair_factor_dtype = (
                _SpatialPairEliminationFactorD if factor_dtype == wp.float64 else _SpatialPairEliminationFactor
            )
            pair_vector_dtype = _SpatialPairVectorD if factor_dtype == wp.float64 else _SpatialPairVector
            pair_response_dtype = _SpatialPairResponseD if factor_dtype == wp.float64 else _SpatialPairResponse
            self.closure_count = 0
            self.paired_backbone_count = (self.backbone_node_count + 1) // 2
            self.paired_backbone_size = self.batch_count * self.paired_backbone_count
            self.paired_lower = wp.zeros(self.paired_backbone_size, dtype=pair_matrix_dtype, device=self.device)
            self.paired_diagonal = wp.zeros_like(self.paired_lower)
            self.paired_upper = wp.zeros_like(self.paired_lower)
            self.paired_elimination_factor = wp.zeros(
                self.paired_backbone_size,
                dtype=pair_factor_dtype,
                device=self.device,
            )
            self.paired_rhs = wp.zeros(self.paired_backbone_size, dtype=pair_vector_dtype, device=self.device)
            self.paired_response = wp.zeros(1, dtype=pair_response_dtype, device=self.device)
            # A retained-factor defect correction pays for itself on longer
            # chains by suppressing float32 elimination drift. Short chains
            # already have a substantially smaller residual than the stock
            # route and retain the minimum-launch path.
            self.paired_correction_rhs = (
                wp.zeros_like(self.paired_rhs) if self.tree_body_count > _PAIRED_OPEN_CHAIN_MIN_BODIES else None
            )
            self.paired_correction_response = (
                wp.zeros(1, dtype=pair_response_dtype, device=self.device)
                if self.paired_correction_rhs is not None
                else None
            )
            self.paired_cr_levels = []
            stride = 1
            active_count = self.paired_backbone_count
            while active_count > _CR_SERIAL_TERMINAL_SIZE:
                eliminated_count = len(range(stride, self.paired_backbone_count, 2 * stride))
                survivor_count = len(range(0, self.paired_backbone_count, 2 * stride))
                self.paired_cr_levels.append((stride, eliminated_count, survivor_count))
                stride *= 2
                active_count = survivor_count
            self.paired_terminal_stride = stride
        else:
            self.closure_count = 0
            self.paired_backbone_count = 0
            self.paired_backbone_size = 0
            self.paired_lower = None
            self.paired_diagonal = None
            self.paired_upper = None
            self.paired_elimination_factor = None
            self.paired_rhs = None
            self.paired_response = None
            self.paired_correction_rhs = None
            self.paired_correction_response = None
            self.paired_cr_levels = []
            self.paired_terminal_stride = 1

    def initialize_system(self, body_matrix, body_rhs, body_scale):
        """Initialize the equilibrated bipartite KKT nodes and edge blocks."""
        if self.factor_dtype == wp.float64:
            wp.launch(
                equilibrate_tree_bodies,
                self.body_count,
                inputs=[self.body_slots, body_matrix],
                outputs=[body_scale],
                device=self.device,
            )
            wp.launch(
                equilibrate_tree_rows,
                self.row_count,
                inputs=[self.compliance],
                outputs=[self.row_scale],
                device=self.device,
            )
            wp.launch(
                _initialize_tree_couplings_fp64,
                self.node_count,
                inputs=[
                    self.node_body,
                    self.node_row,
                    self.parent_node,
                    self.coupling_side,
                    body_scale,
                    self.row_scale,
                    self.jacobian_parent,
                    self.jacobian_child,
                ],
                outputs=[self.coupling],
                device=self.device,
            )
            wp.launch(
                _initialize_tree_nodes_fp64,
                self.node_count,
                inputs=[
                    self.node_body,
                    self.node_row,
                    body_matrix,
                    body_rhs,
                    body_scale,
                    self.compliance,
                    self.residual,
                    self.row_scale,
                ],
                outputs=[self.diagonal, self.rhs],
                device=self.device,
            )
            return
        wp.launch(
            prepare_tree_system,
            self.node_count,
            inputs=[
                self.node_body,
                self.node_row,
                self.parent_node,
                self.coupling_side,
                body_matrix,
                self.compliance,
                self.jacobian_parent,
                self.jacobian_child,
            ],
            outputs=[body_scale, self.row_scale, self.coupling],
            device=self.device,
            block_dim=self.spatial_block_dim,
        )
        wp.launch(
            initialize_tree_nodes,
            self.node_count,
            inputs=[
                self.node_body,
                self.node_row,
                body_matrix,
                body_rhs,
                body_scale,
                self.compliance,
                self.residual,
                self.row_scale,
            ],
            outputs=[self.diagonal, self.rhs],
            device=self.device,
            block_dim=self.spatial_block_dim,
        )

    def eliminate_levels(self, levels):
        """Eliminate a precomputed set of tree leaves toward its survivor."""
        for (
            leaves,
            recipients,
            recipient_offsets,
            message_nodes,
            recipients_are_unique,
        ) in levels:
            if recipients_are_unique:
                wp.launch(
                    eliminate_tree_unique_leaves,
                    leaves.shape[0],
                    inputs=[
                        leaves,
                        self.parent_node,
                        self.diagonal,
                        self.rhs,
                        self.coupling,
                    ],
                    outputs=[self.saved_inverse, self.saved_rhs],
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )
            else:
                wp.launch(
                    invert_tree_leaves,
                    leaves.shape[0],
                    inputs=[leaves],
                    outputs=[self.saved_inverse],
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )
                wp.launch(
                    accumulate_tree_messages,
                    recipients.shape[0],
                    inputs=[
                        recipients,
                        recipient_offsets,
                        message_nodes,
                        self.coupling,
                        self.saved_inverse,
                        self.saved_rhs,
                    ],
                    outputs=[self.diagonal, self.rhs],
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )

    def solve_levels_fused(self):
        """Replay the whole leaf-rake schedule in one block per tree."""
        wp.launch(
            solve_tree_levels_fused,
            self.batch_count * self.spatial_block_dim,
            inputs=[
                self.fused_tree_node_count,
                self.fused_level_count,
                self.fused_tree_level_offsets,
                self.fused_level_kind,
                self.fused_level_node_start,
                self.fused_level_nodes,
                self.fused_level_recipient_start,
                self.fused_level_recipients,
                self.fused_message_offsets,
                self.fused_message_nodes,
                self.roots,
                self.parent_node,
                self.coupling,
            ],
            outputs=[self.diagonal, self.rhs],
            device=self.device,
            block_dim=self.spatial_block_dim,
        )

    def back_substitute_levels(self, levels):
        """Recover a precomputed set of eliminated tree leaves."""
        for leaves, _, _, _, _ in reversed(levels):
            wp.launch(
                back_substitute_tree_level,
                leaves.shape[0],
                inputs=[
                    leaves,
                    self.parent_node,
                    self.coupling,
                    self.saved_inverse,
                    self.saved_rhs,
                ],
                outputs=[self.solution],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )

    def solve_backbone(self):
        """Solve the surviving exact tree backbone with block CR."""
        wp.launch(
            initialize_tree_backbone_edges,
            self.batch_count * self.backbone_node_count,
            inputs=[
                self.backbone_node_count,
                self.backbone_nodes,
                self.parent_node,
                self.coupling,
            ],
            outputs=[self.backbone_lower, self.backbone_upper],
            device=self.device,
            block_dim=self.spatial_block_dim,
        )
        for stride, eliminated_count, survivor_count in self.backbone_cr_levels:
            wp.launch(
                invert_tree_backbone_cr_eliminated,
                self.batch_count * eliminated_count,
                inputs=[
                    stride,
                    self.backbone_node_count,
                    eliminated_count,
                    self.backbone_nodes,
                ],
                outputs=[self.diagonal],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
            wp.launch(
                reduce_tree_backbone_cr_in_place,
                self.batch_count * survivor_count,
                inputs=[
                    stride,
                    self.backbone_node_count,
                    survivor_count,
                    0,
                    self.backbone_nodes,
                ],
                outputs=[
                    self.backbone_lower,
                    self.diagonal,
                    self.backbone_upper,
                    self.rhs,
                    self.no_closure_response,
                ],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
        wp.launch(
            solve_tree_backbone_cr_coarse,
            self.batch_count,
            inputs=[
                self.backbone_terminal_stride,
                self.backbone_node_count,
                0,
                self.backbone_nodes,
            ],
            outputs=[
                self.backbone_lower,
                self.diagonal,
                self.backbone_upper,
                self.rhs,
                self.no_closure_response,
            ],
            device=self.device,
        )
        for stride, eliminated_count, _ in reversed(self.backbone_cr_levels):
            wp.launch(
                back_substitute_tree_backbone_cr_in_place,
                self.batch_count * eliminated_count,
                inputs=[
                    stride,
                    self.backbone_node_count,
                    eliminated_count,
                    0,
                    self.backbone_nodes,
                    self.backbone_lower,
                    self.backbone_upper,
                ],
                outputs=[self.diagonal, self.rhs, self.no_closure_response],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )

    def group_rows_by_joint_type(self, joint_type: np.ndarray):
        """Order rows by joint type so each type can linearize in its own launch.

        Only precise closed trees call this: their row Jacobians do not alias
        numeric factors. Rows are reached through ``joint_ids``, ``node_row`` and
        ``body_rows`` alone, so per-row arithmetic and per-body sum orders hold.
        """
        types = joint_type[self.joint_ids_host]
        order = np.argsort(types, kind="stable")
        bounds = (np.flatnonzero(np.diff(types[order])) + 1).tolist()
        if self.batch_count != 1 or not bounds:
            return
        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.shape[0])
        self.joint_ids_host = self.joint_ids_host[order]
        self.joint_ids.assign(self.joint_ids_host)
        node_row = self.node_row.numpy()
        node_row[node_row >= 0] = inverse[node_row[node_row >= 0]]
        self.node_row.assign(node_row)
        row_node = self.row_node.numpy()
        self.row_node.assign(row_node[order])
        self.body_rows.assign(inverse[self.body_rows.numpy()])
        self.row_slices = [
            (
                self.joint_ids[start:end],
                end - start,
                self.jacobian_parent[start:end],
                self.jacobian_child[start:end],
                self.compliance[start:end],
                self.residual[start:end],
                self.row_active[start:end],
            )
            for start, end in zip([0, *bounds], [*bounds, self.row_count], strict=True)
        ]

    def solve_tree_contraction(self):
        """Solve an arbitrary acyclic island by exact rake-and-compress."""
        for (
            nodes,
            neighbors,
            edges,
            transpose_edges,
            generated_edges,
            recipients,
            offsets,
            slots,
        ) in self.contraction_levels:
            wp.launch(
                eliminate_tree_contraction_level,
                nodes.shape[0],
                inputs=[
                    self.node_count,
                    nodes,
                    neighbors,
                    edges,
                    transpose_edges,
                    generated_edges,
                ],
                outputs=[
                    self.diagonal,
                    self.rhs,
                    self.coupling,
                    self.contraction_fill_edges,
                    self.saved_inverse,
                    self.saved_rhs,
                ],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
            wp.launch(
                accumulate_tree_contraction_messages,
                recipients.shape[0],
                inputs=[
                    self.node_count,
                    nodes,
                    edges,
                    transpose_edges,
                    recipients,
                    offsets,
                    slots,
                    self.coupling,
                    self.contraction_fill_edges,
                ],
                outputs=[self.saved_inverse, self.saved_rhs, self.diagonal, self.rhs],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
        wp.launch(
            solve_tree_roots,
            self.batch_count,
            inputs=[self.contraction_roots, self.diagonal, self.rhs],
            outputs=[self.solution],
            device=self.device,
        )
        for nodes, neighbors, edges, transpose_edges, _, _, _, _ in reversed(self.contraction_levels):
            wp.launch(
                back_substitute_tree_contraction_level,
                nodes.shape[0],
                inputs=[
                    self.node_count,
                    nodes,
                    neighbors,
                    edges,
                    transpose_edges,
                    self.coupling,
                    self.contraction_fill_edges,
                ],
                outputs=[self.saved_inverse, self.saved_rhs, self.solution],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )

    def solve_paired_open_backbone(self):
        """Solve batched branch-free open backbones with stable body-row pivots."""
        wp.launch(
            initialize_tree_backbone_edges,
            self.batch_count * self.backbone_node_count,
            inputs=[
                self.backbone_node_count,
                self.backbone_nodes,
                self.parent_node,
                self.coupling,
            ],
            outputs=[self.backbone_lower, self.backbone_upper],
            device=self.device,
            block_dim=self.spatial_block_dim,
        )
        wp.launch(
            initialize_paired_tree_backbone,
            self.paired_backbone_size,
            inputs=[
                1,
                self.paired_backbone_count,
                self.backbone_node_count,
                self.backbone_nodes,
                self.backbone_lower,
                self.backbone_upper,
                self.diagonal,
                self.rhs,
            ],
            outputs=[
                self.paired_lower,
                self.paired_diagonal,
                self.paired_upper,
                self.paired_rhs,
            ],
            device=self.device,
            block_dim=self.spatial_block_dim,
        )
        _ClosedTreeBucket._solve_paired_backbone_system(
            self,
            True,
            self.paired_rhs,
            self.paired_response,
        )
        if self.paired_correction_rhs is not None:
            # Form the defect before scattering while diagonal/rhs still hold
            # the original equilibrated tree system. Reuse the factors for one
            # classical iterative-refinement step.
            wp.launch(
                compute_paired_tree_backbone_residual,
                self.paired_backbone_size,
                inputs=[
                    self.paired_backbone_count,
                    self.backbone_node_count,
                    0,
                    self.backbone_nodes,
                    self.backbone_lower,
                    self.backbone_upper,
                    self.diagonal,
                    self.rhs,
                    self.no_closure_response,
                    self.paired_rhs,
                    self.paired_response,
                ],
                outputs=[self.paired_correction_rhs, self.paired_correction_response],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
        if self.paired_correction_rhs is not None:
            _ClosedTreeBucket._solve_paired_backbone_system(
                self,
                False,
                self.paired_correction_rhs,
                self.paired_correction_response,
            )
            wp.launch(
                scatter_refined_paired_tree_backbone_solution,
                self.paired_backbone_size,
                inputs=[
                    self.paired_backbone_count,
                    self.backbone_node_count,
                    0,
                    self.backbone_nodes,
                    self.paired_rhs,
                    self.paired_response,
                    self.paired_correction_rhs,
                    self.paired_correction_response,
                ],
                outputs=[self.rhs, self.no_closure_response],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
        else:
            wp.launch(
                scatter_paired_tree_backbone_solution,
                self.paired_backbone_size,
                inputs=[
                    self.paired_backbone_count,
                    self.backbone_node_count,
                    0,
                    self.backbone_nodes,
                    self.paired_rhs,
                    self.paired_response,
                ],
                outputs=[self.rhs, self.no_closure_response],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )

    def solve_tree(
        self,
        body_matrix,
        body_rhs,
        body_scale,
        body_correction,
        *,
        scatter=True,
        allow_paired_open_backbone=False,
    ):
        self.initialize_system(body_matrix, body_rhs, body_scale)
        if self.use_paired_open_backbone and allow_paired_open_backbone:
            self.solve_paired_open_backbone()
        elif self.use_tree_contraction:
            self.solve_tree_contraction()
        elif self.use_backbone_cr:
            # Rake only side branches. Solving a long serial part by leaf
            # levels would expose O(n) dependent launches; block CR solves the
            # exact same KKT backbone in O(log n) launch depth.
            self.eliminate_levels(self.branch_levels)
            self.solve_backbone()
            self.back_substitute_levels(self.branch_levels)
        elif self.use_fused_tree_levels:
            self.solve_levels_fused()
        else:
            self.eliminate_levels(self.levels)
            wp.launch(
                solve_tree_roots,
                self.batch_count,
                inputs=[self.roots, self.diagonal, self.rhs],
                outputs=[self.solution],
                device=self.device,
            )
            self.back_substitute_levels(self.levels)

        if scatter:
            wp.launch(
                scatter_tree_body_correction_indexed,
                self.body_count,
                inputs=[
                    self.body_nodes,
                    self.body_slots,
                    self.body_row_offsets,
                    self.body_rows,
                    self.row_active,
                    body_scale,
                    self.solution,
                ],
                outputs=[body_correction],
                device=self.device,
            )

    def resolve_tree(
        self,
        graph_body_ids,
        joint_parent,
        dt: float,
        body_q,
        body_inertia_q,
        body_disp,
        body_disp_ang,
        use_displacement: int,
        body_mass,
        body_inv_mass,
        body_inertia,
        body_com,
        body_matrix,
        slope_offset: int,
        slope_jacobian_parent,
        slope_jacobian_child,
        slope_compliance,
        slope_residual,
        slope_mode,
        body_rhs,
        body_scale,
        body_correction,
        *,
        allow_paired_open_backbone=False,
        include_sqp_torque=True,
    ):
        if not self.supports_sqp_chord_resolve:
            return False
        if self.use_backbone_cr and not (self.use_paired_open_backbone and allow_paired_open_backbone):
            return False
        if self.factor_dtype == wp.float64:
            wp.launch(
                build_tree_body_sqp_rhs_fp64,
                self.body_count,
                inputs=[
                    self.body_nodes,
                    self.body_slots,
                    self.body_row_offsets,
                    self.body_rows,
                    graph_body_ids,
                    self.joint_ids,
                    joint_parent,
                    self.row_node,
                    dt,
                    body_q,
                    body_inertia_q,
                    body_disp,
                    body_disp_ang,
                    use_displacement,
                    body_mass,
                    body_inv_mass,
                    body_inertia,
                    body_com,
                    body_matrix,
                    self.jacobian_parent,
                    self.jacobian_child,
                    slope_offset,
                    slope_jacobian_parent,
                    slope_jacobian_child,
                    slope_mode,
                    self.row_scale,
                    self.rhs,
                    int(include_sqp_torque),
                ],
                outputs=[body_rhs, body_scale],
                device=self.device,
            )
        else:
            wp.launch(
                build_tree_body_sqp_rhs,
                self.body_count,
                inputs=[
                    self.body_nodes,
                    self.body_slots,
                    self.body_row_offsets,
                    self.body_rows,
                    graph_body_ids,
                    self.joint_ids,
                    joint_parent,
                    self.row_node,
                    self.parent_node,
                    dt,
                    body_q,
                    body_inertia_q,
                    body_disp,
                    body_disp_ang,
                    use_displacement,
                    body_mass,
                    body_inv_mass,
                    body_inertia,
                    body_com,
                    body_matrix,
                    slope_offset,
                    slope_jacobian_parent,
                    slope_jacobian_child,
                    slope_mode,
                    self.row_scale,
                    self.coupling,
                    self.rhs,
                    int(include_sqp_torque),
                ],
                outputs=[body_rhs, body_scale],
                device=self.device,
            )
        wp.launch(
            initialize_tree_saddle_resolve_rhs_fp64
            if self.factor_dtype == wp.float64
            else initialize_tree_saddle_resolve_rhs,
            self.node_count,
            inputs=[
                self.node_body,
                self.node_row,
                slope_offset,
                body_rhs,
                body_scale,
                self.compliance,
                slope_compliance,
                slope_residual,
                self.row_scale,
            ],
            outputs=[self.rhs],
            device=self.device,
            block_dim=self.spatial_block_dim,
        )
        if self.use_paired_open_backbone and allow_paired_open_backbone:
            wp.launch(
                pack_paired_tree_backbone_rhs,
                self.paired_backbone_size,
                inputs=[
                    self.paired_backbone_count,
                    self.backbone_node_count,
                    self.backbone_nodes,
                    self.rhs,
                ],
                outputs=[self.paired_rhs],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
            _ClosedTreeBucket._solve_paired_backbone_system(
                self,
                False,
                self.paired_rhs,
                self.paired_response,
            )
            wp.launch(
                scatter_paired_tree_backbone_solution,
                self.paired_backbone_size,
                inputs=[
                    self.paired_backbone_count,
                    self.backbone_node_count,
                    0,
                    self.backbone_nodes,
                    self.paired_rhs,
                    self.paired_response,
                ],
                outputs=[self.rhs, self.no_closure_response],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
        elif self.use_tree_contraction:
            for (
                nodes,
                _neighbors,
                edges,
                transpose_edges,
                _generated_edges,
                recipients,
                offsets,
                slots,
            ) in self.contraction_levels:
                wp.launch(
                    resolve_tree_contraction_messages,
                    recipients.shape[0],
                    inputs=[
                        self.node_count,
                        nodes,
                        edges,
                        transpose_edges,
                        recipients,
                        offsets,
                        slots,
                        self.coupling,
                        self.contraction_fill_edges,
                        self.saved_inverse,
                        self.saved_rhs,
                    ],
                    outputs=[self.rhs],
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )
            wp.launch(
                solve_tree_roots,
                self.batch_count,
                inputs=[self.contraction_roots, self.diagonal, self.rhs],
                outputs=[self.solution],
                device=self.device,
            )
            for nodes, neighbors, edges, transpose_edges, _, _, _, _ in reversed(self.contraction_levels):
                wp.launch(
                    back_substitute_tree_contraction_level,
                    nodes.shape[0],
                    inputs=[
                        self.node_count,
                        nodes,
                        neighbors,
                        edges,
                        transpose_edges,
                        self.coupling,
                        self.contraction_fill_edges,
                    ],
                    outputs=[self.saved_inverse, self.saved_rhs, self.solution],
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )
        elif self.use_fused_tree_levels:
            wp.launch(
                resolve_tree_levels_fused,
                self.batch_count * self.spatial_block_dim,
                inputs=[
                    self.fused_tree_node_count,
                    self.fused_level_count,
                    self.fused_tree_level_offsets,
                    self.fused_level_kind,
                    self.fused_level_node_start,
                    self.fused_level_nodes,
                    self.fused_level_recipient_start,
                    self.fused_level_recipients,
                    self.fused_message_offsets,
                    self.fused_message_nodes,
                    self.roots,
                    self.parent_node,
                    self.coupling,
                ],
                outputs=[self.diagonal, self.rhs],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
        else:
            for leaves, recipients, recipient_offsets, message_nodes, recipients_are_unique in self.levels:
                if recipients_are_unique:
                    wp.launch(
                        resolve_tree_unique_leaves,
                        leaves.shape[0],
                        inputs=[leaves, self.parent_node, self.coupling, self.saved_inverse],
                        outputs=[self.rhs],
                        device=self.device,
                        block_dim=self.spatial_block_dim,
                    )
                else:
                    wp.launch(
                        resolve_tree_messages,
                        recipients.shape[0],
                        inputs=[
                            recipients,
                            recipient_offsets,
                            message_nodes,
                            self.coupling,
                            self.saved_inverse,
                            self.saved_rhs,
                        ],
                        outputs=[self.rhs],
                        device=self.device,
                        block_dim=self.spatial_block_dim,
                    )
            wp.launch(
                solve_tree_roots,
                self.batch_count,
                inputs=[self.roots, self.diagonal, self.rhs],
                outputs=[self.solution],
                device=self.device,
            )
            self.back_substitute_levels(self.levels)

        wp.launch(
            scatter_tree_body_correction_indexed,
            self.body_count,
            inputs=[
                self.body_nodes,
                self.body_slots,
                self.body_row_offsets,
                self.body_rows,
                self.row_active,
                body_scale,
                self.solution,
            ],
            outputs=[body_correction],
            device=self.device,
        )
        return True


@wp.func
def _inverse_spatial_cooperative(
    work: wp.array[wp.float64], value: int, result: int, pivots: wp.array[wp.int32], pair: int, lane: int
):
    """Use the serial FP64 partial-pivot algebra with block-uniform barriers."""
    row = lane // 6
    col = lane % 6
    if lane < 36:
        work[result + lane] = wp.float64(1.0) if row == col else wp.float64(0.0)
    _synchronize_cr_block()
    for k in range(6):
        if lane == 0:
            chosen = k
            mag = wp.abs(work[value + k * 6 + k])
            for r in range(k + 1, 6):
                candidate = wp.abs(work[value + r * 6 + k])
                if candidate > mag:
                    chosen = r
                    mag = candidate
            pivots[pair] = chosen
        _synchronize_cr_block()
        chosen = pivots[pair]
        if lane < 6 and chosen != k:
            a = work[value + k * 6 + lane]
            b = work[value + chosen * 6 + lane]
            ia = work[result + k * 6 + lane]
            ib = work[result + chosen * 6 + lane]
            work[value + k * 6 + lane] = b
            work[value + chosen * 6 + lane] = a
            work[result + k * 6 + lane] = ib
            work[result + chosen * 6 + lane] = ia
        _synchronize_cr_block()
        reciprocal = wp.float64(1.0) / work[value + k * 6 + k]
        _synchronize_cr_block()
        if lane < 6:
            work[value + k * 6 + lane] = reciprocal * work[value + k * 6 + lane]
            work[result + k * 6 + lane] = reciprocal * work[result + k * 6 + lane]
        _synchronize_cr_block()
        factor = wp.float64(0.0)
        pv = wp.float64(0.0)
        pi = wp.float64(0.0)
        if lane < 36:
            factor = work[value + row * 6 + k]
            pv = work[value + k * 6 + col]
            pi = work[result + k * 6 + col]
        _synchronize_cr_block()
        if lane < 36 and row != k:
            work[value + lane] = work[value + lane] - factor * pv
            work[result + lane] = work[result + lane] - factor * pi
        _synchronize_cr_block()


@wp.func
def _cooperative_matrix_product(a: wp.array[wp.float64], ao: int, b: wp.array[wp.float64], bo: int, row: int, col: int):
    value = wp.float64(0.0)
    for k in range(6):
        value = value + a[ao + row * 6 + k] * b[bo + k * 6 + col]
    return value


@wp.func
def _cooperative_matrix_product_fma(
    a: wp.array[wp.float64], ao: int, b: wp.array[wp.float64], bo: int, row: int, col: int
):
    """Left-associated fused multiply-add, matching a spatial-matrix product."""
    value = wp.float64(0.0)
    for k in range(6):
        value = _fma_f64(a[ao + row * 6 + k], b[bo + k * 6 + col], value)
    return value


@wp.kernel(module="unique", enable_backward=False)
def eliminate_tree_unique_leaves_cooperative(
    leaves: wp.array[wp.int32],
    parent_node: wp.array[wp.int32],
    diagonal: wp.array[wp.float64],
    rhs: wp.array[wp.float64],
    coupling: wp.array[wp.float64],
):
    """Eliminate unique-recipient FP64 leaves with one 64-thread block each.

    The arithmetic of :func:`eliminate_tree_unique_leaves`: the serial partial-pivot
    inverse, the left-associated fused products of ``transpose(a) * inverse * a`` and
    the column-ordered ``* rhs`` product. The inverse overwrites the eliminated
    diagonal, which the tree's saved-inverse buffer aliases. Requires ``block_dim`` 64.
    """
    tid = wp.tid()
    width = wp.block_dim()
    lane = tid % width
    index = tid // width
    node = leaves[index]
    parent = parent_node[node]
    row = lane // 6
    col = lane % 6
    s = 0
    scratch = _block_shared_words()
    pivot_slot = _block_shared_pivot()
    d = node * 36
    if lane < 36:
        scratch[s + lane] = diagonal[d + lane]
    _synchronize_cr_block()
    _inverse_spatial_cooperative(scratch, s, s + 36, pivot_slot, 0, lane)
    if lane < 36:
        value = wp.float64(0.0)
        for k in range(6):
            value = _fma_f64(coupling[d + k * 6 + row], scratch[s + 36 + k * 6 + col], value)
        scratch[s + 72 + lane] = value
        diagonal[d + lane] = scratch[s + 36 + lane]
    _synchronize_cr_block()
    if lane < 36:
        value = wp.float64(0.0)
        for k in range(6):
            value = _fma_f64(scratch[s + 72 + row * 6 + k], coupling[d + k * 6 + col], value)
        diagonal[parent * 36 + lane] = diagonal[parent * 36 + lane] - value
    if lane < 6:
        value = scratch[s + 72 + lane * 6] * rhs[node * 6]
        for k in range(1, 6):
            value = value + scratch[s + 72 + lane * 6 + k] * rhs[node * 6 + k]
        rhs[parent * 6 + lane] = rhs[parent * 6 + lane] - value


@wp.kernel(module="unique", enable_backward=False)
def invert_paired_tree_backbone_cooperative(
    stride: int,
    pair_count: int,
    selected_count: int,
    row_offset: int,
    write_factors: int,
    lower: wp.array[wp.float64],
    diagonal: wp.array[wp.float64],
    upper: wp.array[wp.float64],
    factors: wp.array[wp.float64],
):
    """Invert body-first paired pivots using block-shared work and pivot storage."""
    tid = wp.tid()
    width = wp.block_dim()
    lane = tid % width
    index = tid // width
    batch = index // selected_count
    local = index % selected_count
    local_row = row_offset + 2 * stride * local
    pair = batch * pair_count + local_row
    if local_row >= pair_count:
        return
    row = lane // 6
    col = lane % 6
    d = pair * 144
    s = 0
    scratch = _block_shared_words()
    pivot_slot = _block_shared_pivot()
    # B^-1, X=B^-1*Jt, S=C-J*X; all original blocks remain read-only.
    if lane < 36:
        scratch[s + lane] = diagonal[d + lane]
    _synchronize_cr_block()
    _inverse_spatial_cooperative(scratch, s, s + 36, pivot_slot, 0, lane)
    if lane < 36:
        scratch[s + 72 + lane] = _cooperative_matrix_product(scratch, s + 36, diagonal, d + 36, row, col)
    _synchronize_cr_block()
    if lane < 36:
        scratch[s + 108 + lane] = diagonal[d + 108 + lane] - _cooperative_matrix_product(
            diagonal, d + 72, scratch, s + 72, row, col
        )
    _synchronize_cr_block()
    _inverse_spatial_cooperative(scratch, s + 108, s + 144, pivot_slot, 0, lane)
    # Preserve the native left-associated matrix-product expressions.
    if lane < 36:
        scratch[s + 180 + lane] = _cooperative_matrix_product(scratch, s + 72, scratch, s + 144, row, col)
        scratch[s + 216 + lane] = _cooperative_matrix_product(scratch, s + 144, diagonal, d + 72, row, col)
    _synchronize_cr_block()
    if lane < 36:
        scratch[s + 252 + lane] = _cooperative_matrix_product(scratch, s + 216, scratch, s + 36, row, col)
        scratch[s + 288 + lane] = _cooperative_matrix_product(scratch, s + 180, diagonal, d + 72, row, col)
    _synchronize_cr_block()
    if lane < 36:
        scratch[s + 324 + lane] = _cooperative_matrix_product(scratch, s + 288, scratch, s + 36, row, col)
    _synchronize_cr_block()
    if lane < 36:
        diagonal[d + lane] = scratch[s + 36 + lane] + scratch[s + 324 + lane]
        diagonal[d + 36 + lane] = -scratch[s + 180 + lane]
        diagonal[d + 72 + lane] = -scratch[s + 252 + lane]
        diagonal[d + 108 + lane] = scratch[s + 144 + lane]
    _synchronize_cr_block()
    if write_factors != 0 and lane < 36:
        left0 = wp.float64(0.0)
        left1 = wp.float64(0.0)
        right0 = wp.float64(0.0)
        right1 = wp.float64(0.0)
        if local_row >= stride:
            coupling = (pair - stride) * 144 + 72
            left0 = _cooperative_matrix_product(upper, coupling, diagonal, d, row, col)
            left1 = _cooperative_matrix_product(upper, coupling, diagonal, d + 36, row, col)
        if local_row + stride < pair_count:
            coupling = (pair + stride) * 144 + 36
            right0 = _cooperative_matrix_product(lower, coupling, diagonal, d + 72, row, col)
            right1 = _cooperative_matrix_product(lower, coupling, diagonal, d + 108, row, col)
        factors[d + lane] = left0
        factors[d + 36 + lane] = left1
        factors[d + 72 + lane] = right0
        factors[d + 108 + lane] = right1


@wp.func
def _load_paired_response_column(values: wp.array[wp.float64], entry: int, part: int, column: int):
    result = wp.spatial_vectord()
    for r in range(6):
        result[r] = values[entry * 72 + part * 36 + r * 6 + column]
    return result


@wp.func
def _store_paired_response_column(
    values: wp.array[wp.float64], entry: int, part: int, column: int, value: wp.spatial_vectord
):
    for r in range(6):
        values[entry * 72 + part * 36 + r * 6 + column] = value[r]


@wp.kernel(module="unique", enable_backward=False)
def reduce_paired_tree_backbone_columns(
    stride: int,
    pair_count: int,
    survivor_count: int,
    closure_count: int,
    factorize: int,
    lower: wp.array[wp.float64],
    diagonal: wp.array[wp.float64],
    upper: wp.array[wp.float64],
    factors: wp.array[wp.float64],
    factors_struct: wp.array[_SpatialPairEliminationFactorD],
    rhs: wp.array[_SpatialPairVectorD],
    responses: wp.array[wp.float64],
):
    """Reduce one survivor with a 64-thread block.

    Lanes 0-35 write the four 6x6 block updates with the same left-associated
    fma as a spatial-matrix product. The physical right-hand side and the
    closure responses stay on the struct product so the packed solution matches
    the serial schedule. Requires ``block_dim`` of 64.
    """
    tid = wp.tid()
    width = wp.block_dim()
    lane = tid % width
    index = tid // width
    batch = index // survivor_count
    local = index % survivor_count
    local_row = 2 * stride * local
    base = batch * pair_count
    row = base + local_row
    if local_row >= pair_count:
        return
    block = lane // 6
    entry = lane % 6
    packed = row * 144

    if factorize != 0 and lane < _PAIRED_COOPERATIVE_MATRIX_LANES:
        low_01 = wp.float64(0.0)
        diag_00 = diagonal[packed + lane]
        diag_11 = diagonal[packed + 108 + lane]
        up_10 = wp.float64(0.0)
        if local_row >= stride:
            eliminated = (row - stride) * 144
            low_01 = -_cooperative_matrix_product_fma(factors, eliminated + 72, lower, eliminated + 36, block, entry)
            diag_00 = diag_00 - _cooperative_matrix_product_fma(
                factors, eliminated + 108, upper, eliminated + 72, block, entry
            )
        if local_row + stride < pair_count:
            eliminated = (row + stride) * 144
            diag_11 = diag_11 - _cooperative_matrix_product_fma(
                factors, eliminated, lower, eliminated + 36, block, entry
            )
            if local_row + 2 * stride < pair_count:
                up_10 = -_cooperative_matrix_product_fma(factors, eliminated + 36, upper, eliminated + 72, block, entry)
        lower[packed + lane] = wp.float64(0.0)
        lower[packed + 36 + lane] = low_01
        lower[packed + 72 + lane] = wp.float64(0.0)
        lower[packed + 108 + lane] = wp.float64(0.0)
        diagonal[packed + lane] = diag_00
        diagonal[packed + 108 + lane] = diag_11
        upper[packed + lane] = wp.float64(0.0)
        upper[packed + 36 + lane] = wp.float64(0.0)
        upper[packed + 72 + lane] = up_10
        upper[packed + 108 + lane] = wp.float64(0.0)

    if lane == _PAIRED_COOPERATIVE_RHS_LANE:
        value = rhs[row]
        if local_row >= stride:
            eliminated = row - stride
            factor = factors_struct[eliminated]
            e = rhs[eliminated]
            value.v0 = value.v0 - (factor.to_right0 * e.v0 + factor.to_right1 * e.v1)
        if local_row + stride < pair_count:
            eliminated = row + stride
            factor = factors_struct[eliminated]
            e = rhs[eliminated]
            value.v1 = value.v1 - (factor.to_left0 * e.v0 + factor.to_left1 * e.v1)
        rhs[row] = value

    work = int(lane)
    while work < closure_count * 6:
        closure = work // 6
        column = work % 6
        response_entry = row * closure_count + closure
        v0 = _load_paired_response_column(responses, response_entry, 0, column)
        v1 = _load_paired_response_column(responses, response_entry, 1, column)
        if local_row >= stride:
            eliminated = row - stride
            factor = factors_struct[eliminated]
            other = eliminated * closure_count + closure
            v0 = v0 - (
                factor.to_right0 * _load_paired_response_column(responses, other, 0, column)
                + factor.to_right1 * _load_paired_response_column(responses, other, 1, column)
            )
        if local_row + stride < pair_count:
            eliminated = row + stride
            factor = factors_struct[eliminated]
            other = eliminated * closure_count + closure
            v1 = v1 - (
                factor.to_left0 * _load_paired_response_column(responses, other, 0, column)
                + factor.to_left1 * _load_paired_response_column(responses, other, 1, column)
            )
        _store_paired_response_column(responses, response_entry, 0, column, v0)
        _store_paired_response_column(responses, response_entry, 1, column, v1)
        work += width


@wp.kernel(module="unique", enable_backward=False)
def back_substitute_paired_tree_backbone_columns(
    stride: int,
    pair_count: int,
    eliminated_count: int,
    closure_count: int,
    lower: wp.array[Any],
    diagonal: wp.array[Any],
    upper: wp.array[Any],
    rhs: wp.array[Any],
    responses: wp.array[wp.float64],
):
    tid = wp.tid()
    width = wp.block_dim()
    lane = tid % width
    index = tid // width
    batch = index // eliminated_count
    local = index % eliminated_count
    local_row = stride + 2 * stride * local
    row = batch * pair_count + local_row
    if local_row >= pair_count:
        return
    if lane == _PAIRED_COOPERATIVE_PHYSICAL_LANE:
        value = rhs[row]
        value.v0 = value.v0 - lower[row].m01 * rhs[row - stride].v1
        if local_row + stride < pair_count:
            value.v1 = value.v1 - upper[row].m10 * rhs[row + stride].v0
        rhs[row] = _spatial_pair_matrix_vector_multiply(diagonal[row], value)
    work = int(lane)
    while work < closure_count * 6:
        closure = work // 6
        column = work % 6
        entry = row * closure_count + closure
        v0 = _load_paired_response_column(responses, entry, 0, column)
        v1 = _load_paired_response_column(responses, entry, 1, column)
        v0 = v0 - lower[row].m01 * _load_paired_response_column(
            responses, (row - stride) * closure_count + closure, 1, column
        )
        if local_row + stride < pair_count:
            v1 = v1 - upper[row].m10 * _load_paired_response_column(
                responses, (row + stride) * closure_count + closure, 0, column
            )
        diag = diagonal[row]
        result0 = diag.m00 * v0 + diag.m01 * v1
        result1 = diag.m10 * v0 + diag.m11 * v1
        _store_paired_response_column(responses, entry, 0, column, result0)
        _store_paired_response_column(responses, entry, 1, column, result1)
        work += width


@wp.kernel(module="unique", enable_backward=False)
def solve_paired_tree_backbone_single(
    pair_count: int,
    closure_count: int,
    diagonal: wp.array[_SpatialPairMatrixD],
    rhs: wp.array[_SpatialPairVectorD],
    responses: wp.array[wp.float64],
):
    """Apply the final paired inverse to independent physical/closure RHS.

    Requires ``block_dim`` of 64. The physical product runs on the second warp
    so it overlaps the first 32 response columns.
    """
    tid = wp.tid()
    width = wp.block_dim()
    lane = tid % width
    row = (tid // width) * pair_count
    if lane == _PAIRED_COOPERATIVE_PHYSICAL_LANE:
        rhs[row] = _spatial_pair_matrix_vector_multiply(diagonal[row], rhs[row])
    column_index = int(lane)
    while column_index < closure_count * 6:
        closure = column_index // 6
        column = column_index % 6
        entry = row * closure_count + closure
        value0 = _load_paired_response_column(responses, entry, 0, column)
        value1 = _load_paired_response_column(responses, entry, 1, column)
        matrix = diagonal[row]
        result0 = matrix.m00 * value0 + matrix.m01 * value1
        result1 = matrix.m10 * value0 + matrix.m11 * value1
        _store_paired_response_column(responses, entry, 0, column, result0)
        _store_paired_response_column(responses, entry, 1, column, result1)
        column_index += width


def _float64_words_view(array, words: int):
    """Alias an FP64 matrix or vector array as flat words without allocating."""
    if wp.types.type_size_in_bytes(array.dtype) != words * 8:
        raise ValueError("Unexpected FP64 block layout")
    return wp.array(ptr=array.ptr, dtype=wp.float64, shape=array.shape[0] * words, device=array.device, copy=False)


def _paired_float64_view(array, words: int):
    """Alias the private packed FP64 layout without allocating device memory."""
    if np.dtype(array.dtype.numpy_dtype()).itemsize != words * 8:
        raise ValueError("Unexpected paired KKT buffer layout")
    return wp.array(
        ptr=array.ptr,
        dtype=wp.float64,
        shape=array.size * words,
        device=array.device,
        copy=False,
    )


class _ClosedTreeBucket:
    """Batched exact backbone-CR solves completed by closure Schur systems."""

    def __init__(
        self,
        components: _ClosedTree | list[_ClosedTree],
        parent: np.ndarray,
        child: np.ndarray,
        body_slot: dict[int, int],
        device,
        *,
        tree_symbolics: _TreeBucketSymbolics | None = None,
        cooperative_paired: bool = True,
    ):
        if isinstance(components, _ClosedTree):
            components = [components]
        component = components[0]
        signature = _closed_tree_topology_signature(component, parent, child)
        if any(_closed_tree_topology_signature(candidate, parent, child) != signature for candidate in components[1:]):
            raise ValueError("A closed-tree bucket requires one normalized symbolic topology")

        if tree_symbolics is None:
            tree_symbolics = _tree_bucket_symbolics(component.tree, include_contraction=False)
        self.device = device
        self.side_streams = None
        self.chain_route = None
        self.tree = _TreeBucket(
            [candidate.tree for candidate in components],
            body_slot,
            device,
            symbolics=tree_symbolics,
            allow_fused_tree_levels=False,
            factor_dtype=wp.float64,
        )
        self.spatial_block_dim = self.tree.spatial_block_dim
        # Cyclic islands use the same exact tree backbone schedule before the
        # closure Schur solve; share its immutable topology and work buffers.
        self.backbone_node_count = self.tree.backbone_node_count
        self.backbone_nodes = self.tree.backbone_nodes
        self.branch_levels = self.tree.branch_levels
        self.backbone_lower = self.tree.backbone_lower
        self.backbone_upper = self.tree.backbone_upper
        self.backbone_cr_levels = self.tree.backbone_cr_levels
        self.backbone_terminal_stride = self.tree.backbone_terminal_stride
        self.batch_count = len(components)
        self.size = sum(len(candidate.joints) for candidate in components)
        self.body_size = self.tree.body_count
        self.closure_count = len(component.closure_joints)
        self.closure_size = self.batch_count * self.closure_count
        matrix_dtype = wp.spatial_matrixd
        vector_dtype = wp.spatial_vectord
        # Keep the original row Jacobians separate from numeric factors.
        self.elimination_nodes = wp.array(
            [node for level in component.tree.elimination_levels for node in sorted(level)],
            dtype=wp.int32,
            device=device,
        )
        self.factor_diagonal = self.tree.diagonal
        self.factor_coupling = self.tree.coupling
        self.use_fused_closure = (
            device.is_cuda and _CR_SERIAL_TERMINAL_SIZE < self.closure_count <= _FUSED_CLOSURE_MAX_BLOCKS
        )
        closure_joint_ids: list[int] = []
        closure_parent_node: list[int] = []
        closure_child_node: list[int] = []
        body_closure_offsets: list[int] = [0]
        body_closure_rows: list[int] = []
        for batch, candidate in enumerate(components):
            node_offset = batch * self.tree.tree_node_count
            closure_offset = batch * self.closure_count
            body_node = {body: node for node, body in enumerate(candidate.tree.bodies)}
            for joint in candidate.closure_joints:
                closure_joint_ids.append(joint)
                parent_node = body_node.get(int(parent[joint]), -1)
                child_node = body_node.get(int(child[joint]), -1)
                closure_parent_node.append(node_offset + parent_node if parent_node >= 0 else -1)
                closure_child_node.append(node_offset + child_node if child_node >= 0 else -1)
            for body in candidate.tree.bodies:
                for local_closure, joint in enumerate(candidate.closure_joints):
                    if body == int(parent[joint]) or body == int(child[joint]):
                        body_closure_rows.append(closure_offset + local_closure)
                body_closure_offsets.append(len(body_closure_rows))
        self.closure_joint_ids = wp.array(closure_joint_ids, dtype=wp.int32, device=device)
        self.closure_parent_node = wp.array(closure_parent_node, dtype=wp.int32, device=device)
        self.closure_child_node = wp.array(closure_child_node, dtype=wp.int32, device=device)
        self.body_closure_offsets = wp.array(body_closure_offsets, dtype=wp.int32, device=device)
        self.body_closure_rows = wp.array(body_closure_rows, dtype=wp.int32, device=device)

        self.closure_jacobian_parent = wp.zeros(self.closure_size, dtype=wp.spatial_matrix, device=device)
        self.closure_jacobian_child = wp.zeros_like(self.closure_jacobian_parent)
        self.closure_compliance = wp.zeros_like(self.closure_jacobian_parent)
        self.closure_residual = wp.zeros(self.closure_size, dtype=wp.spatial_vector, device=device)
        self.closure_row_active = wp.zeros(self.closure_size, dtype=wp.int32, device=device)
        self.closure_row_scale = wp.zeros(self.closure_size, dtype=wp.spatial_vector, device=device)
        response_size = self.tree.node_count * self.closure_count
        self.response_rhs = wp.zeros(response_size, dtype=matrix_dtype, device=device)
        self.response_solution = self.response_rhs
        self.closure_schur = wp.zeros(
            self.batch_count * self.closure_count * self.closure_count,
            dtype=matrix_dtype,
            device=device,
        )
        self.closure_rhs = wp.zeros(self.closure_size, dtype=vector_dtype, device=device)
        self.closure_multiplier = self.closure_rhs
        self.use_lane_back_substitute = _closure_back_substitute_lanes(self.closure_count, device)
        self.closure_back_partial = (
            wp.zeros(self.closure_size, dtype=vector_dtype, device=device) if self.use_lane_back_substitute else None
        )

        # Closed-tree backbones alternate body and row nodes. Group each body
        # with its following row so near-hard compliance is never selected as
        # an isolated CR pivot. An odd body-ended backbone receives one padded
        # identity row in the packing kernel.
        local_backbone = tree_symbolics.backbone
        self.use_paired_backbone = not device.is_cpu and all(
            component.tree.node_body[node] >= 0 if index % 2 == 0 else component.tree.node_row[node] >= 0
            for index, node in enumerate(local_backbone)
        )
        self.paired_backbone_count = (self.backbone_node_count + 1) // 2
        self.paired_backbone_size = self.batch_count * self.paired_backbone_count
        # The coarse tail's pair recurrence is serial, but its closure columns
        # are independent. Give the block that width only when there are enough
        # columns to repay the barriers; one column measurably regresses.
        self.use_lane_split_coarse = self.device.is_cuda and self.closure_count >= _PAIRED_COARSE_MIN_CLOSURES
        self.use_cooperative_paired = (
            cooperative_paired
            and self.use_paired_backbone
            and device.is_cuda
            and self.closure_count >= _PAIRED_COOPERATIVE_MIN_CLOSURES
        )
        self.cooperative_leaf_levels = [False] * len(self.branch_levels)
        if self.use_paired_backbone:
            self.paired_lower = wp.zeros(self.paired_backbone_size, dtype=_SpatialPairMatrixD, device=self.device)
            self.paired_diagonal = wp.zeros_like(self.paired_lower)
            self.paired_upper = wp.zeros_like(self.paired_lower)
            self.paired_elimination_factor = wp.zeros(
                self.paired_backbone_size,
                dtype=_SpatialPairEliminationFactorD,
                device=self.device,
            )
            self.paired_rhs = wp.zeros(self.paired_backbone_size, dtype=_SpatialPairVectorD, device=self.device)
            self.paired_response = wp.zeros(
                self.paired_backbone_size * self.closure_count,
                dtype=_SpatialPairResponseD,
                device=self.device,
            )
            if self.use_cooperative_paired:
                self.cooperative_lower = _paired_float64_view(self.paired_lower, 144)
                self.cooperative_diagonal = _paired_float64_view(self.paired_diagonal, 144)
                self.cooperative_upper = _paired_float64_view(self.paired_upper, 144)
                self.cooperative_factors = _paired_float64_view(self.paired_elimination_factor, 144)
                self.cooperative_response = _paired_float64_view(self.paired_response, 72)
                # Retain the established leaf scheduling envelope; scratch is block-local.
                leaf_capacity = self.paired_backbone_size
                self.cooperative_leaf_levels = [
                    unique and leaves.shape[0] <= leaf_capacity for leaves, _, _, _, unique in self.branch_levels
                ]
                if self.tree.saved_inverse is self.tree.diagonal and self.tree.saved_rhs is self.tree.rhs:
                    self.cooperative_tree_diagonal = _float64_words_view(self.tree.diagonal, 36)
                    self.cooperative_tree_rhs = _float64_words_view(self.tree.rhs, 6)
                    self.cooperative_tree_coupling = _float64_words_view(self.tree.coupling, 36)
                else:
                    self.cooperative_leaf_levels = [False] * len(self.branch_levels)
            self.paired_cr_levels: list[tuple[int, int, int]] = []
            stride = 1
            active_count = self.paired_backbone_count
            terminal_size = 1 if self.use_cooperative_paired else _CR_SERIAL_TERMINAL_SIZE
            while active_count > terminal_size:
                eliminated_count = len(range(stride, self.paired_backbone_count, 2 * stride))
                survivor_count = len(range(0, self.paired_backbone_count, 2 * stride))
                self.paired_cr_levels.append((stride, eliminated_count, survivor_count))
                stride *= 2
                active_count = survivor_count
            self.paired_terminal_stride = stride
        else:
            self.paired_lower = None
            self.paired_diagonal = None
            self.paired_upper = None
            self.paired_elimination_factor = None
            self.paired_rhs = None
            self.paired_response = None
            self.paired_cr_levels = []
            self.paired_terminal_stride = 1

    def _solve_paired_backbone_system(self, factorize: bool, rhs, response_rhs):
        """Solve packed right-hand sides, building or reusing the pair factor."""
        # Open FP32 tree buckets also borrow this method; they keep their route.
        if getattr(self, "use_cooperative_paired", False):
            self._solve_cooperative_paired_backbone(factorize, rhs, response_rhs)
            return
        for stride, eliminated_count, survivor_count in self.paired_cr_levels:
            if factorize:
                wp.launch(
                    invert_paired_tree_backbone_cr_eliminated,
                    self.batch_count * eliminated_count,
                    inputs=[stride, self.paired_backbone_count, eliminated_count],
                    outputs=[
                        self.paired_lower,
                        self.paired_diagonal,
                        self.paired_upper,
                        self.paired_elimination_factor,
                    ],
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )
            wp.launch(
                reduce_paired_tree_backbone_cr_lanes
                if self.use_lane_split_coarse
                else reduce_paired_tree_backbone_cr_in_place,
                self.batch_count * survivor_count * (self.spatial_block_dim if self.use_lane_split_coarse else 1),
                inputs=[
                    stride,
                    self.paired_backbone_count,
                    survivor_count,
                    self.closure_count,
                    int(factorize),
                ],
                outputs=[
                    self.paired_lower,
                    self.paired_diagonal,
                    self.paired_upper,
                    self.paired_elimination_factor,
                    rhs,
                    response_rhs,
                ],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
        coarse_inputs = [
            self.paired_terminal_stride,
            self.paired_backbone_count,
            self.closure_count,
            int(factorize),
        ]
        coarse_outputs = [
            self.paired_lower,
            self.paired_diagonal,
            self.paired_upper,
            rhs,
            response_rhs,
        ]
        if self.use_lane_split_coarse:
            wp.launch(
                solve_paired_tree_backbone_cr_coarse_lanes,
                self.batch_count * self.spatial_block_dim,
                inputs=coarse_inputs,
                outputs=coarse_outputs,
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
        else:
            wp.launch(
                solve_paired_tree_backbone_cr_coarse,
                self.batch_count,
                inputs=coarse_inputs,
                outputs=coarse_outputs,
                device=self.device,
            )
        for stride, eliminated_count, _ in reversed(self.paired_cr_levels):
            wp.launch(
                back_substitute_paired_tree_backbone_cr_lanes
                if self.use_lane_split_coarse
                else back_substitute_paired_tree_backbone_cr_in_place,
                self.batch_count * eliminated_count * (self.spatial_block_dim if self.use_lane_split_coarse else 1),
                inputs=[
                    stride,
                    self.paired_backbone_count,
                    eliminated_count,
                    self.closure_count,
                ],
                outputs=[
                    self.paired_lower,
                    self.paired_diagonal,
                    self.paired_upper,
                    rhs,
                    response_rhs,
                ],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )

    def _solve_cooperative_paired_backbone(self, factorize: bool, rhs, response_rhs):
        """Same FP64 body-first KKT, with one CUDA block per pair/response row.

        Partial pivoting does not require SPD blocks. Each block owns its
        scratch and all barriers are block-uniform. Factor-reuse calls retain
        the identical level schedule and never refactor or allocate.
        """
        if response_rhs is not self.paired_response:
            raise ValueError("Unexpected paired closure-response buffer")
        width = _PAIRED_COOPERATIVE_BLOCK_DIM
        for stride, eliminated_count, survivor_count in self.paired_cr_levels:
            if factorize:
                self._invert_cooperative_pairs(stride, eliminated_count, stride, True)
            wp.launch(
                reduce_paired_tree_backbone_columns,
                self.batch_count * survivor_count * width,
                inputs=[stride, self.paired_backbone_count, survivor_count, self.closure_count, int(factorize)],
                outputs=[
                    self.cooperative_lower,
                    self.cooperative_diagonal,
                    self.cooperative_upper,
                    self.cooperative_factors,
                    self.paired_elimination_factor,
                    rhs,
                    self.cooperative_response,
                ],
                device=self.device,
                block_dim=width,
            )
        if factorize:
            self._invert_cooperative_pairs(self.paired_terminal_stride, 1, 0, False)
        wp.launch(
            solve_paired_tree_backbone_single,
            self.batch_count * width,
            inputs=[self.paired_backbone_count, self.closure_count, self.paired_diagonal],
            outputs=[rhs, self.cooperative_response],
            device=self.device,
            block_dim=width,
        )
        for stride, eliminated_count, _ in reversed(self.paired_cr_levels):
            wp.launch(
                back_substitute_paired_tree_backbone_columns,
                self.batch_count * eliminated_count * width,
                inputs=[
                    stride,
                    self.paired_backbone_count,
                    eliminated_count,
                    self.closure_count,
                    self.paired_lower,
                    self.paired_diagonal,
                    self.paired_upper,
                ],
                outputs=[rhs, self.cooperative_response],
                device=self.device,
                block_dim=width,
            )

    def _invert_cooperative_pairs(self, stride: int, count: int, offset: int, write_factors: bool):
        """Launch whole 64-thread blocks: 36 matrix entries plus idle lanes."""
        wp.launch(
            invert_paired_tree_backbone_cooperative,
            self.batch_count * count * 64,
            inputs=[stride, self.paired_backbone_count, count, offset, int(write_factors)],
            outputs=[
                self.cooperative_lower,
                self.cooperative_diagonal,
                self.cooperative_upper,
                self.cooperative_factors,
            ],
            device=self.device,
            block_dim=64,
        )

    def solve_paired_backbone(self, response_stream=None):
        """Solve the precise cyclic base with body-first pair pivots."""
        wp.launch(
            initialize_paired_tree_backbone,
            self.paired_backbone_size,
            inputs=[
                1,
                self.paired_backbone_count,
                self.backbone_node_count,
                self.backbone_nodes,
                self.backbone_lower,
                self.backbone_upper,
                self.tree.diagonal,
                self.tree.rhs,
            ],
            outputs=[
                self.paired_lower,
                self.paired_diagonal,
                self.paired_upper,
                self.paired_rhs,
            ],
            device=self.device,
            block_dim=self.spatial_block_dim,
        )
        with _stream_scope(response_stream):
            wp.launch(
                initialize_paired_tree_backbone_response,
                self.paired_backbone_size * self.closure_count,
                inputs=[
                    self.paired_backbone_count,
                    self.backbone_node_count,
                    self.closure_count,
                    self.backbone_nodes,
                    self.response_rhs,
                ],
                outputs=[self.paired_response],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
        if response_stream is not None:
            wp.get_stream(self.device).wait_stream(response_stream)
        self._solve_paired_backbone_system(True, self.paired_rhs, self.paired_response)
        wp.launch(
            scatter_paired_tree_backbone_solution,
            self.paired_backbone_size * (self.closure_count + 1),
            inputs=[
                self.paired_backbone_count,
                self.backbone_node_count,
                self.closure_count,
                self.backbone_nodes,
                self.paired_rhs,
                self.paired_response,
            ],
            outputs=[self.tree.solution, self.response_rhs],
            device=self.device,
            block_dim=self.spatial_block_dim,
        )

    def solve_closure_schur(self):
        """Solve the dense closure system, parallelizing independent rows."""
        if self.device.is_cpu and self.closure_schur.dtype == wp.spatial_matrixd:
            wp.launch(
                _solve_block_dense_serial_fp64,
                self.batch_count,
                inputs=[self.closure_count],
                outputs=[self.closure_schur, self.closure_multiplier],
                device=self.device,
            )
            return
        if self.use_fused_closure:
            wp.launch(
                solve_block_ldlt_persistent,
                self.batch_count * _SPATIAL_GPU_BLOCK_DIM,
                inputs=[self.closure_count],
                outputs=[self.closure_schur, self.closure_multiplier],
                device=self.device,
                block_dim=_SPATIAL_GPU_BLOCK_DIM,
            )
            return
        if self.closure_count <= _CR_SERIAL_TERMINAL_SIZE or not self.device.is_cuda:
            wp.launch(
                _solve_block_dense_serial_fp64
                if self.closure_schur.dtype == wp.spatial_matrixd
                else solve_block_dense_serial,
                self.batch_count,
                inputs=[self.closure_count],
                outputs=[self.closure_schur, self.closure_multiplier],
                device=self.device,
            )
            return

        for pivot in range(self.closure_count):
            wp.launch(
                invert_block_dense_pivot,
                self.batch_count,
                inputs=[pivot, self.closure_count],
                outputs=[self.closure_schur],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
            remaining = self.closure_count - pivot - 1
            if remaining > 0:
                wp.launch(
                    factor_block_ldlt_pivot_rows,
                    self.batch_count * remaining,
                    inputs=[pivot, self.closure_count, self.closure_schur],
                    outputs=[self.closure_multiplier],
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )
                wp.launch(
                    update_block_ldlt_lower_triangle,
                    self.batch_count * remaining * remaining,
                    inputs=[pivot, self.closure_count, remaining],
                    outputs=[self.closure_schur],
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )
        wp.launch(
            solve_block_ldlt_diagonal,
            self.closure_size,
            inputs=[self.closure_count, self.closure_schur],
            outputs=[self.closure_multiplier],
            device=self.device,
            block_dim=self.spatial_block_dim,
        )
        if self.use_lane_back_substitute:
            wp.launch(
                back_substitute_block_ldlt_lanes,
                self.batch_count * self.spatial_block_dim,
                inputs=[
                    self.closure_count,
                    self.closure_schur,
                    self.closure_back_partial,
                ],
                outputs=[self.closure_multiplier],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
        else:
            wp.launch(
                back_substitute_block_ldlt_serial,
                self.batch_count,
                inputs=[self.closure_count, self.closure_schur],
                outputs=[self.closure_multiplier],
                device=self.device,
            )

    def solve_tree(self, body_matrix, body_rhs, body_scale, body_correction):
        """Solve a closed island without rounding its closure response or recovery.

        With side streams, the closure-response chain and the coupling setup run beside the
        factor chain; every kernel keeps its inputs and arithmetic.
        """
        tree = self.tree
        device = self.device
        streams = self.side_streams if device.is_cuda and self.use_paired_backbone else None
        response_stream, coupling_stream = streams[:2] if streams is not None else (None, None)
        main = wp.get_stream(device) if streams is not None else None
        wp.launch(
            equilibrate_tree_bodies,
            tree.body_count,
            inputs=[tree.body_slots, body_matrix],
            outputs=[body_scale],
            device=device,
        )
        wp.launch(
            equilibrate_tree_rows,
            tree.row_count,
            inputs=[tree.compliance],
            outputs=[tree.row_scale],
            device=device,
        )
        wp.launch(
            equilibrate_tree_rows,
            self.closure_size,
            inputs=[self.closure_compliance],
            outputs=[self.closure_row_scale],
            device=device,
        )
        if streams is not None:
            response_stream.wait_stream(main)
            coupling_stream.wait_stream(main)
        with _stream_scope(coupling_stream):
            wp.launch(
                _initialize_tree_couplings_fp64,
                tree.node_count,
                inputs=[
                    tree.node_body,
                    tree.node_row,
                    tree.parent_node,
                    tree.coupling_side,
                    body_scale,
                    tree.row_scale,
                    tree.jacobian_parent,
                    tree.jacobian_child,
                ],
                outputs=[self.factor_coupling],
                device=device,
            )
        with _stream_scope(response_stream):
            wp.launch(
                _initialize_closure_response_fp64,
                tree.node_count * self.closure_count,
                inputs=[
                    tree.node_body,
                    tree.tree_node_count,
                    self.closure_count,
                    self.closure_parent_node,
                    self.closure_child_node,
                    body_scale,
                    self.closure_row_scale,
                    self.closure_jacobian_parent,
                    self.closure_jacobian_child,
                ],
                outputs=[self.response_rhs],
                device=device,
            )
        wp.launch(
            _initialize_tree_nodes_fp64,
            tree.node_count,
            inputs=[
                tree.node_body,
                tree.node_row,
                body_matrix,
                body_rhs,
                body_scale,
                tree.compliance,
                tree.residual,
                tree.row_scale,
            ],
            outputs=[self.factor_diagonal, tree.solution],
            device=device,
        )
        if streams is not None:
            main.wait_stream(coupling_stream)
        if self.device.is_cuda and self.use_paired_backbone:
            self._solve_tree_gpu(response_stream)
        else:
            wp.launch(
                _eliminate_tree_nodes_fp64,
                self.batch_count,
                inputs=[
                    tree.tree_node_count,
                    self.elimination_nodes,
                    self.closure_count,
                    tree.parent_node,
                    self.factor_coupling,
                ],
                outputs=[self.factor_diagonal, tree.solution, self.response_rhs],
                device=device,
            )
            wp.launch(
                _solve_tree_roots_fp64,
                self.batch_count,
                inputs=[tree.roots, self.closure_count],
                outputs=[self.factor_diagonal, tree.solution, self.response_rhs],
                device=device,
            )
            wp.launch(
                _back_substitute_tree_nodes_fp64,
                self.batch_count,
                inputs=[
                    tree.tree_node_count,
                    self.elimination_nodes,
                    self.closure_count,
                    tree.parent_node,
                    self.factor_coupling,
                ],
                outputs=[self.factor_diagonal, tree.solution, self.response_rhs],
                device=device,
            )
        with _stream_scope(response_stream):
            wp.launch(
                _assemble_closure_schur_fp64,
                self.batch_count * self.closure_count * self.closure_count,
                inputs=[
                    self.closure_count,
                    tree.node_body,
                    self.closure_parent_node,
                    self.closure_child_node,
                    body_scale,
                    self.closure_row_scale,
                    self.closure_jacobian_parent,
                    self.closure_jacobian_child,
                    self.closure_compliance,
                    self.response_solution,
                ],
                outputs=[self.closure_schur],
                device=device,
            )
        wp.launch(
            _assemble_closure_rhs_fp64,
            self.closure_size,
            inputs=[
                tree.node_body,
                self.closure_parent_node,
                self.closure_child_node,
                body_scale,
                self.closure_row_scale,
                self.closure_jacobian_parent,
                self.closure_jacobian_child,
                self.closure_residual,
                tree.solution,
            ],
            outputs=[self.closure_rhs],
            device=device,
        )
        if streams is not None:
            main.wait_stream(response_stream)
        self.solve_closure_schur()
        wp.launch(
            _scatter_closed_tree_body_correction_fp64,
            tree.body_count,
            inputs=[
                tree.body_nodes,
                tree.body_slots,
                tree.body_row_offsets,
                tree.body_rows,
                tree.row_active,
                self.body_closure_offsets,
                self.body_closure_rows,
                self.closure_row_active,
                tree.tree_body_count,
                self.closure_count,
                tree.row_node,
                tree.parent_node,
                tree.coupling_side,
                body_matrix,
                body_rhs,
                body_scale,
                tree.row_scale,
                tree.jacobian_parent,
                tree.jacobian_child,
                tree.compliance,
                tree.residual,
                self.closure_parent_node,
                self.closure_child_node,
                self.closure_row_scale,
                self.closure_jacobian_parent,
                self.closure_jacobian_child,
                tree.solution,
                self.response_solution,
                self.closure_multiplier,
            ],
            outputs=[body_correction],
            device=device,
        )

    def _solve_tree_gpu(self, response_stream=None):
        # The response chain needs only each level's saved inverses: it trails
        # the factor chain by one level on its own stream.
        main = wp.get_stream(self.device) if response_stream is not None else None
        for level, (
            leaves,
            recipients,
            recipient_offsets,
            message_nodes,
            recipients_are_unique,
        ) in enumerate(self.branch_levels):
            if self.cooperative_leaf_levels[level]:
                wp.launch(
                    eliminate_tree_unique_leaves_cooperative,
                    leaves.shape[0] * _PAIRED_COOPERATIVE_BLOCK_DIM,
                    inputs=[leaves, self.tree.parent_node],
                    outputs=[
                        self.cooperative_tree_diagonal,
                        self.cooperative_tree_rhs,
                        self.cooperative_tree_coupling,
                    ],
                    device=self.device,
                    block_dim=_PAIRED_COOPERATIVE_BLOCK_DIM,
                )
            elif recipients_are_unique:
                wp.launch(
                    eliminate_tree_unique_leaves,
                    leaves.shape[0],
                    inputs=[
                        leaves,
                        self.tree.parent_node,
                        self.tree.diagonal,
                        self.tree.rhs,
                        self.tree.coupling,
                    ],
                    outputs=[self.tree.saved_inverse, self.tree.saved_rhs],
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )
            else:
                wp.launch(
                    invert_tree_leaves,
                    leaves.shape[0],
                    inputs=[leaves],
                    outputs=[self.tree.saved_inverse],
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )
                wp.launch(
                    accumulate_tree_messages,
                    recipients.shape[0],
                    inputs=[
                        recipients,
                        recipient_offsets,
                        message_nodes,
                        self.tree.coupling,
                        self.tree.saved_inverse,
                        self.tree.saved_rhs,
                    ],
                    outputs=[self.tree.diagonal, self.tree.rhs],
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )
            if response_stream is not None:
                response_stream.wait_stream(main)
            with _stream_scope(response_stream):
                if recipients_are_unique:
                    wp.launch(
                        eliminate_tree_response_unique_leaves,
                        leaves.shape[0] * self.closure_count,
                        inputs=[
                            leaves,
                            self.closure_count,
                            self.tree.parent_node,
                            self.tree.coupling,
                            self.tree.saved_inverse,
                        ],
                        outputs=[self.response_rhs],
                        device=self.device,
                        block_dim=self.spatial_block_dim,
                    )
                else:
                    wp.launch(
                        accumulate_tree_response_messages,
                        recipients.shape[0] * self.closure_count,
                        inputs=[
                            recipients,
                            recipient_offsets,
                            message_nodes,
                            self.closure_count,
                            self.tree.coupling,
                            self.tree.saved_inverse,
                        ],
                        outputs=[self.response_rhs],
                        device=self.device,
                        block_dim=self.spatial_block_dim,
                    )

        wp.launch(
            initialize_tree_backbone_edges,
            self.batch_count * self.backbone_node_count,
            inputs=[
                self.backbone_node_count,
                self.backbone_nodes,
                self.tree.parent_node,
                self.tree.coupling,
            ],
            outputs=[self.backbone_lower, self.backbone_upper],
            device=self.device,
            block_dim=self.spatial_block_dim,
        )
        self.solve_paired_backbone(response_stream)
        if response_stream is not None:
            response_stream.wait_stream(main)

        for leaves, _, _, _, _ in reversed(self.branch_levels):
            wp.launch(
                back_substitute_tree_level,
                leaves.shape[0],
                inputs=[
                    leaves,
                    self.tree.parent_node,
                    self.tree.coupling,
                    self.tree.saved_inverse,
                    self.tree.saved_rhs,
                ],
                outputs=[self.tree.solution],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )
            with _stream_scope(response_stream):
                wp.launch(
                    back_substitute_tree_response_level,
                    leaves.shape[0] * self.closure_count,
                    inputs=[
                        leaves,
                        self.closure_count,
                        self.tree.parent_node,
                        self.tree.coupling,
                        self.tree.saved_inverse,
                        self.response_rhs,
                    ],
                    outputs=[self.response_solution],
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )


_FP32_MANTISSA_RECIPROCAL = float(1 << 24)


def _rigid_at_step(rigidity: float, step: float | None) -> bool:
    """Whether a rigidity ``omega^2`` [s^-2] reaches ``omega^2 h^2 >= 2^24`` at a step ``h`` [s].

    With no step known (``None``) every positive rigidity counts as rigid: the safe side of each decision.
    """
    if rigidity <= 0.0:
        return False
    return step is None or rigidity * step * step >= _FP32_MANTISSA_RECIPROCAL


@wp.kernel
def flag_nonlinear_correction(
    body_flags: wp.array[wp.int32],
    body_island: wp.array[wp.int32],
    island_contact_state: wp.array[wp.int32],
    island_step_scale: wp.array[float],
    correction: wp.array[wp.spatial_vector],
    needed: wp.array[wp.int32],
):
    """Flag a correction whose rotation the linearized rows do not resolve in single precision.

    A body turned by ``theta`` moves a joint anchor by ``theta^2 / 2`` of its arm beyond the linearization. Once
    that reaches the binary32 resolution ``2^-24`` a second Newton stage changes the result.
    """
    slot = wp.tid()
    if (body_flags[slot] & 8) == 0:
        return
    island = body_island[slot]
    if island_contact_state[island] < -1:
        return
    turn = island_step_scale[island] * wp.spatial_bottom(correction[slot])
    if 0.5 * wp.dot(turn, turn) * 16777216.0 >= 1.0:
        needed[0] = 1


@wp.kernel
def flag_any_island(
    island_flags: wp.array[wp.int32],
    needed: wp.array[wp.int32],
):
    """Set ``needed[0]`` when a structural island is flagged."""
    if island_flags[wp.tid()] != 0:
        needed[0] = 1


def _compute_joint_delassus_mobilities(
    model,
    parent: np.ndarray,
    child: np.ndarray,
    body_inv_mass_host: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute single-body pivot translational [kg^-1] and rotational [(kg*m^2)^-1] mobilities.

    In paired cyclic reduction, each ``(body b, row j)`` saddle pivot evaluates the Woodbury
    complement ``B_b^-1 + B_b^-1 J_{j,b}^T (C_j - J_{j,b} B_b^-1 J_{j,b}^T)^-1 J_{j,b} B_b^-1`` on
    a single incident body ``b in {parent[j], child[j]}``. Taking ``max_b ||J_{j,b} M_b^-1 J_{j,b}^T||``
    gives ``k_lin * w_lin`` and ``k_ang * w_ang`` exact squared natural-frequency units ``[s^-2]``,
    invariant under arbitrary mass and length unit rescalings, and matches the exact IEEE-754
    ``float32`` Woodbury cancellation threshold ``chi_nom = omega^2 h_nom^2 = 2^24``.
    """
    n_joints = len(parent)
    w_lin = np.zeros(n_joints, dtype=np.float64)
    w_ang = np.zeros(n_joints, dtype=np.float64)
    if n_joints == 0 or model.body_count == 0:
        return w_lin, w_ang
    xp = np.asarray(model.joint_X_p.numpy(), dtype=np.float64)
    xc = np.asarray(model.joint_X_c.numpy(), dtype=np.float64)
    com = np.asarray(model.body_com.numpy(), dtype=np.float64)
    inv_inertia = np.asarray(model.body_inv_inertia.numpy(), dtype=np.float64)
    # The largest inverse inertia bounds the rigidity of every row direction of a body. The mean understates
    # the stiffest direction of a thin body (its twist), which is the one that cancels first.
    body_max_inv_i = np.linalg.eigvalsh(inv_inertia).max(axis=1) if len(inv_inertia) else np.zeros(0)
    for j in range(n_joints):
        c = int(child[j])
        if c < 0:
            continue
        p = int(parent[j])
        wl = 0.0
        wa = 0.0
        if body_inv_mass_host[c] > 0.0:
            rc = xc[j, :3] - com[c]
            ic = float(body_max_inv_i[c])
            wl = max(wl, float(body_inv_mass_host[c]) + float(np.dot(rc, rc)) * ic)
            wa = max(wa, ic)
        if p >= 0 and body_inv_mass_host[p] > 0.0:
            rp = xp[j, :3] - com[p]
            ip = float(body_max_inv_i[p])
            wl = max(wl, float(body_inv_mass_host[p]) + float(np.dot(rp, rp)) * ip)
            wa = max(wa, ip)
        w_lin[j] = wl
        w_ang[j] = wa
    return w_lin, w_ang


def _tree_is_branched_and_depth(
    tree: _Tree, parent: np.ndarray, child: np.ndarray, inv_mass: np.ndarray
) -> tuple[bool, int]:
    """Return whether ``tree`` contains a branching dynamic junction (degree >= 3) and its joint depth."""
    deg: dict[int, int] = defaultdict(int)
    for j in tree.joints:
        p, c = int(parent[j]), int(child[j])
        if p >= 0 and inv_mass[p] > 0.0:
            deg[p] += 1
        if c >= 0 and inv_mass[c] > 0.0:
            deg[c] += 1
    is_branched = any(count >= 3 for count in deg.values())
    joint_depth = max(1, (len(tree.elimination_levels) + 1) // 2)
    return is_branched, joint_depth


class StructuralGraphKKT:
    """Graph-capturable compliance-KKT backend for paths, trees, and cyclic graphs."""

    def __init__(
        self,
        model,
        body_inv_mass,
        *,
        enable_paired_open_chains=False,
        allow_general_joint_paths=False,
        supported_joint_types: set[int] | None = None,
        ignore_free_completion_joints=False,
        enable_stream_concurrency=False,
        joint_max_stiffness: np.ndarray | None = None,
        joint_linear_stiffness: np.ndarray | None = None,
        joint_angular_stiffness: np.ndarray | None = None,
        plan_step: float | None = None,
    ):
        self.device = model.device
        # Routes and factor precisions depend on the step through chi = omega^2 h^2; this plan is made for
        # plan_step (None: no step known, every decision takes its safe side), and plan_change(dt) tells whether
        # another step needs another plan.
        self.plan_step = None if plan_step is None else float(plan_step)
        self.plan_rigidities: tuple[float, ...] = ()
        # Independent row, contact and closed-tree work may overlap on CUDA.
        # Reproducible modes keep one stream and therefore one reduction order.
        self.side_streams = (
            tuple(wp.Stream(self.device) for _ in range(5))
            if enable_stream_concurrency and self.device.is_cuda
            else None
        )
        # Row launches use the current stream and every side stream except the
        # contact stream, which may refresh the contact objective meanwhile.
        self.row_lanes = (None, self.side_streams[0], *self.side_streams[2:]) if self.side_streams is not None else None
        self.spatial_block_dim = _SPATIAL_GPU_BLOCK_DIM if self.device.is_cuda else 1
        self.path_buckets: list[_PathBucket] = []
        self.tree_buckets: list[_TreeBucket] = []
        self.closed_tree_buckets: list[_ClosedTreeBucket] = []
        self.buckets = []
        self._bucket_diagnostics: tuple[_BucketDiagnostics, ...] = ()
        self._payload_budget_bytes = _structural_payload_budget(self.device)
        self._estimated_bytes = 0
        self.graph_body_count = 0
        self._fuse_rod_stress = False
        self.use_fused_solve_reset = True
        self.use_fused_contact_classification = True
        self.chain_route = None

        body_inv_mass_host = np.asarray(body_inv_mass.numpy(), dtype=float)
        self.dynamic_body_mask_host = body_inv_mass_host > 0.0
        paths = _build_paths(
            model,
            body_inv_mass_host,
            allow_general_joints=allow_general_joint_paths,
            supported_joint_types=supported_joint_types,
            ignore_free_completion_joints=ignore_free_completion_joints,
        )
        trees = _build_trees(
            model,
            {frozenset(path.joints) for path in paths},
            body_inv_mass_host,
            supported_joint_types=supported_joint_types,
            ignore_free_completion_joints=ignore_free_completion_joints,
        )
        joint_type = np.asarray(model.joint_type.numpy(), dtype=np.int64)
        parent = np.asarray(model.joint_parent.numpy(), dtype=np.int32)
        child = np.asarray(model.joint_child.numpy(), dtype=np.int32)
        remaining_trees = []
        for tree in trees:
            path = None
            if all(joint_type[joint] == int(JointType.ROD) for joint in tree.joints):
                # Interior prescribed supports can cut a branched cable into
                # serial systems. Batch those with the existing rod paths;
                # their boundary poses stay in the rows, not the unknowns.
                path = _make_path(set(tree.joints), parent, child, dynamic_body_mask=self.dynamic_body_mask_host)
            if path is not None:
                paths.append(path)
            else:
                remaining_trees.append(tree)
        trees = remaining_trees
        handled_joint_sets = {frozenset(component.joints) for component in [*paths, *trees]}
        closed_trees = _build_closed_trees(
            model,
            handled_joint_sets,
            body_inv_mass_host,
            supported_joint_types=supported_joint_types,
            ignore_free_completion_joints=ignore_free_completion_joints,
        )
        all_components = [*paths, *trees, *closed_trees]
        grouped_trees: dict[tuple, list[_Tree]] = defaultdict(list)
        for tree in trees:
            grouped_trees[_tree_topology_signature(tree)].append(tree)
        grouped_closed_trees: dict[tuple, list[_ClosedTree]] = defaultdict(list)
        for component in closed_trees:
            grouped_closed_trees[_closed_tree_topology_signature(component, parent, child)].append(component)

        # Build every host symbolic schedule and exact device payload before
        # the first bucket allocation. Order normalized groups by original
        # model identifiers, never by topology kind or an unmeasured quality
        # heuristic. If a repeated-topology batch is too large, retain the
        # largest deterministic prefix that fits and leave the rest to VBD.
        # Paths of every length share one bucket and one launch per CR level.
        w_lin_h, w_ang_h = _compute_joint_delassus_mobilities(model, parent, child, body_inv_mass_host)
        joint_omega2 = np.zeros(len(joint_type), dtype=np.float64)
        if joint_linear_stiffness is not None and joint_angular_stiffness is not None:
            joint_omega2 = np.maximum(
                np.asarray(joint_linear_stiffness, dtype=np.float64) * w_lin_h,
                np.asarray(joint_angular_stiffness, dtype=np.float64) * w_ang_h,
            )
        elif joint_max_stiffness is not None:
            k_max_h = np.asarray(joint_max_stiffness, dtype=np.float64)
            is_ball = joint_type == int(JointType.BALL)
            eff_w = np.where(is_ball, w_lin_h, np.maximum(w_lin_h, w_ang_h))
            joint_omega2 = k_max_h * eff_w
        # joint_omega2 [s^-2]: squared natural frequency of a joint's stiffest row against its mobility. A
        # decision below compares a rigidity r [s^-2] times plan_step^2 with 2^24 (_rigid_at_step);
        # plan_rigidities keeps every r whose decision changes with the step.
        plan_rigidities: list[float] = []

        plans: list[_BucketPlan] = []
        path_factor_dtype = wp.float32
        if paths and not allow_general_joint_paths:
            qd_start_h = np.asarray(model.joint_qd_start.numpy(), dtype=np.int64)
            dof_dim_h = np.asarray(model.joint_dof_dim.numpy(), dtype=np.int64)
            ke_h = np.asarray(model.joint_target_ke.numpy(), dtype=np.float64)
            path_rigidity = 0.0
            path_contrast = 0.0
            for cand_path in paths:
                max_omega2_drive = 0.0
                max_lin_w, min_lin_w = 0.0, np.inf
                max_ang_w, min_ang_w = 0.0, np.inf
                for j in cand_path.joints:
                    d0 = int(qd_start_h[j])
                    lc, ac = int(dof_dim_h[j, 0]), int(dof_dim_h[j, 1])
                    if lc > 0 and w_lin_h[j] > 0.0:
                        lpos = ke_h[d0 : d0 + lc]
                        lpos = lpos[lpos > 0.0]
                        if lpos.size > 0:
                            lw_hi = float(lpos.max()) * float(w_lin_h[j])
                            lw_lo = float(lpos.min()) * float(w_lin_h[j])
                            max_lin_w = max(max_lin_w, lw_hi)
                            min_lin_w = min(min_lin_w, lw_lo)
                            max_omega2_drive = max(max_omega2_drive, lw_hi)
                    if ac > 0 and w_ang_h[j] > 0.0:
                        apos = ke_h[d0 + lc : d0 + lc + ac]
                        apos = apos[apos > 0.0]
                        if apos.size > 0:
                            aw_hi = float(apos.max()) * float(w_ang_h[j])
                            aw_lo = float(apos.min()) * float(w_ang_h[j])
                            max_ang_w = max(max_ang_w, aw_hi)
                            min_ang_w = min(min_ang_w, aw_lo)
                            max_omega2_drive = max(max_omega2_drive, aw_hi)
                lin_c = (max_lin_w / min_lin_w) if (0.0 < min_lin_w < np.inf) else 1.0
                ang_c = (max_ang_w / min_ang_w) if (0.0 < min_ang_w < np.inf) else 1.0
                n_rows_sq = float(len(cand_path.joints)) ** 2
                path_rigidity = max(path_rigidity, n_rows_sq * max_omega2_drive)
                path_contrast = max(path_contrast, n_rows_sq * max(lin_c, ang_c))
            if path_contrast >= _FP32_MANTISSA_RECIPROCAL or _rigid_at_step(path_rigidity, self.plan_step):
                path_factor_dtype = wp.float64
            if path_contrast < _FP32_MANTISSA_RECIPROCAL:
                plan_rigidities.append(path_rigidity)
        if paths:
            ordered = sorted(paths, key=_component_order_key)
            plans.append(
                _BucketPlan(
                    "path",
                    ordered,
                    _path_bucket_diagnostics(ordered, self.device, factor_dtype=path_factor_dtype),
                )
            )
        for group in grouped_trees.values():
            ordered = sorted(group, key=_component_order_key)
            # Dimensionless conditioning: chi = omega^2 h^2 of the stiffest joint at the planned step. Paired
            # cyclic reduction cancels in every (body, row) pair; tip-to-root leaf rake does not.
            max_tree_omega2 = max(
                (float(joint_omega2[joint]) for candidate in ordered for joint in candidate.joints),
                default=0.0,
            )
            tree_is_rigid = _rigid_at_step(max_tree_omega2, self.plan_step)
            rigid_angular_tree = any(_has_rigid_angular_rows(candidate.joints, joint_type) for candidate in ordered)
            paired_allowed = bool(enable_paired_open_chains and self.device.is_cuda and rigid_angular_tree)
            paired_probe = _tree_bucket_symbolics(ordered[0], enable_paired_open_chain=True) if paired_allowed else None
            # Ultra-long near-rigid open chains (>= 512 joints) retain O(log N) paired CR in float64;
            # shorter near-rigid chains use single-block fused leaf rake in float32/float64.
            use_fp64_paired_cr = bool(
                paired_probe is not None
                and paired_probe.use_paired_open_backbone
                and tree_is_rigid
                and len(ordered[0].joints) >= _FUSED_TREE_MAX_LEVEL_WIDTH
            )
            symbolics = _tree_bucket_symbolics(
                ordered[0],
                enable_paired_open_chain=paired_allowed and (not tree_is_rigid or use_fp64_paired_cr),
            )
            if paired_probe is not None and paired_probe.use_paired_open_backbone:
                plan_rigidities.append(max_tree_omega2)
            # Generic rake/compress changes the pivot order of an indefinite
            # compliance KKT system.  Its homogeneous routes have independent
            # numerical coverage, but a mixed rod/rigid
            # tree can combine section-scale cable rows with attachment and
            # mechanism rows spanning many orders of magnitude.  Preserve the
            # rooted leaf order for that unsupported numerical envelope.  On
            # CUDA, eligible schedules still collapse to the bit-identical
            # fused leaf kernel, so this is a stability boundary rather than a
            # return to O(depth) launch count.
            mixed_rod_tree = any(
                any(joint_type[joint] == int(JointType.ROD) for joint in candidate.joints)
                and any(joint_type[joint] != int(JointType.ROD) for joint in candidate.joints)
                for candidate in ordered
            )
            # A tree with exactly one row to the world is rooted at that row, so every row follows a body and
            # float32 holds it up to K = 1e12. Free trees and trees with two or more rows to the world, whose
            # leaf rake eliminates the other rows to the world first, take float64 factors once the
            # single-pivot rigidity of the planned step reaches the IEEE-754 float32 mantissa reciprocal 2^24.
            unanchored_tree = _tree_anchor_row_count(ordered[0]) != 1
            near_rigid_tree = tree_is_rigid and unanchored_tree
            if unanchored_tree:
                plan_rigidities.append(max_tree_omega2)
            tree_factor_dtype = wp.float64 if (near_rigid_tree or use_fp64_paired_cr) else wp.float32
            if (mixed_rod_tree or rigid_angular_tree or near_rigid_tree) and symbolics.use_tree_contraction:
                symbolics = replace(
                    symbolics,
                    contraction=None,
                    use_tree_contraction=False,
                    selected_route="tree_leaf_rake",
                )
            plans.append(
                _BucketPlan(
                    "tree",
                    ordered,
                    _tree_bucket_diagnostics(
                        ordered,
                        self.device,
                        symbolics=symbolics,
                        allow_fused_tree_levels=True,
                        factor_dtype=tree_factor_dtype,
                    ),
                    symbolics,
                )
            )
        for group in grouped_closed_trees.values():
            ordered = sorted(group, key=_component_order_key)
            symbolics = _tree_bucket_symbolics(ordered[0].tree, include_contraction=False)
            plans.append(
                _BucketPlan(
                    "closed_tree",
                    ordered,
                    _closed_tree_bucket_diagnostics(
                        ordered,
                        parent,
                        child,
                        self.device,
                        tree_symbolics=symbolics,
                        # Block-shared cooperative scratch adds no global payload.
                        cooperative_paired=True,
                    ),
                    symbolics,
                )
            )
        # Contracted trees of every topology share one bucket and one launch per
        # contraction level; each tree keeps its own elimination schedule.
        contracted = [
            plan
            for plan in plans
            if plan.kind == "tree"
            and plan.tree_symbolics.use_tree_contraction
            and not plan.tree_symbolics.use_paired_open_backbone
        ]
        if len(contracted) > 1:
            symbolics_by_tree = {id(tree): plan.tree_symbolics for plan in contracted for tree in plan.components}
            ordered = sorted((tree for plan in contracted for tree in plan.components), key=_component_order_key)
            plans = [plan for plan in plans if all(plan is not merged for merged in contracted)]
            plans.append(
                _BucketPlan(
                    "tree",
                    ordered,
                    _tree_bucket_diagnostics(ordered, self.device, symbolics=symbolics_by_tree),
                    symbolics_by_tree,
                )
            )
        # Heterogeneous fused leaf-rake trees of the same precision and rod class
        # also share one multi-CTA bucket so distinct digits/limbs solve in parallel
        # across SMs in one kernel launch rather than serial per-topology buckets.
        if self.device.is_cuda:
            fused_rake_groups: dict[tuple[bool, bool], list[_BucketPlan]] = defaultdict(list)
            for plan in plans:
                if (
                    plan.kind == "tree"
                    and isinstance(plan.tree_symbolics, _TreeBucketSymbolics)
                    and not plan.tree_symbolics.use_tree_contraction
                    and not plan.tree_symbolics.use_backbone_cr
                    and _fused_tree_levels_supported(plan.components[0].elimination_levels, self.device)
                ):
                    is_fp64 = plan.diagnostics.planned_route.endswith("_fp64")
                    has_rod = any(
                        int(joint_type[j]) == int(JointType.ROD) for cand in plan.components for j in cand.joints
                    )
                    in_chain_scope = all(
                        _chain_island_in_scope(cand, joint_type, parent, child, body_inv_mass_host)
                        for cand in plan.components
                    )
                    fused_rake_groups[(is_fp64, has_rod, in_chain_scope)].append(plan)
            for (is_fp64, _has_rod, _in_chain_scope), rake_plans in fused_rake_groups.items():
                if len(rake_plans) > 1:
                    symbolics_by_tree = {
                        id(tree): plan.tree_symbolics for plan in rake_plans for tree in plan.components
                    }
                    ordered = sorted(
                        (tree for plan in rake_plans for tree in plan.components), key=_component_order_key
                    )
                    plans = [plan for plan in plans if all(plan is not merged for merged in rake_plans)]
                    merged_dtype = wp.float64 if is_fp64 else wp.float32
                    plans.append(
                        _BucketPlan(
                            "tree",
                            ordered,
                            _tree_bucket_diagnostics(
                                ordered,
                                self.device,
                                symbolics=symbolics_by_tree,
                                allow_fused_tree_levels=True,
                                factor_dtype=merged_dtype,
                            ),
                            symbolics_by_tree,
                        )
                    )
        plans.sort(key=lambda plan: _component_order_key(plan.components[0]))

        selected_component_ids: set[int] = set()
        selected_body_ids: set[int] = set()
        selected_joint_ids: set[int] = set()
        selected_bucket_bytes = 0
        selected_island_count = 0
        selected_plans: list[_BucketPlan] = []
        diagnostics: list[_BucketDiagnostics] = []
        for plan in plans:
            diagnostic_cache = {len(plan.components): plan.diagnostics}

            lower = 0
            upper = len(plan.components)
            while lower < upper:
                middle = (lower + upper + 1) // 2
                prefix = plan.components[:middle]
                prefix_diagnostics = _cached_bucket_prefix_diagnostics(
                    diagnostic_cache,
                    plan,
                    middle,
                    parent,
                    child,
                    self.device,
                )
                prospective_body_ids = selected_body_ids | {body for component in prefix for body in component.bodies}
                prospective_joint_ids = selected_joint_ids | {
                    joint for component in prefix for joint in component.joints
                }
                prospective_total = selected_bucket_bytes + prefix_diagnostics.estimated_bytes
                prospective_total += _shared_structural_payload_bytes(
                    model.body_count,
                    model.joint_count,
                    len(prospective_body_ids),
                    selected_island_count + middle,
                    len(prospective_joint_ids),
                    is_cuda=self.device.is_cuda,
                )
                if _within_structural_payload_budget(prospective_total, self._payload_budget_bytes):
                    lower = middle
                else:
                    upper = middle - 1
            accepted_count = lower

            if accepted_count > 0:
                accepted_components = plan.components[:accepted_count]
                accepted_diagnostics = _cached_bucket_prefix_diagnostics(
                    diagnostic_cache,
                    plan,
                    accepted_count,
                    parent,
                    child,
                    self.device,
                )
                selected_plans.append(
                    _BucketPlan(
                        plan.kind,
                        accepted_components,
                        accepted_diagnostics,
                        plan.tree_symbolics,
                    )
                )
                selected_component_ids.update(id(component) for component in accepted_components)
                selected_body_ids.update(body for component in accepted_components for body in component.bodies)
                selected_joint_ids.update(joint for component in accepted_components for joint in component.joints)
                selected_island_count += accepted_count
                selected_bucket_bytes += accepted_diagnostics.estimated_bytes
                diagnostics.append(accepted_diagnostics)

            fallback_count = len(plan.components) - accepted_count
            if fallback_count > 0:
                # Path lengths and contracted tree topologies can differ, so
                # describe the remaining components rather than the whole plan.
                fallback_diagnostics = _bucket_plan_diagnostics(
                    plan,
                    plan.components[accepted_count:],
                    parent,
                    child,
                    self.device,
                )
                diagnostics.append(
                    replace(
                        fallback_diagnostics,
                        selected_route="local_vbd",
                        fallback_reason="payload_budget",
                    )
                )
        self._bucket_diagnostics = tuple(diagnostics)

        components = [component for component in all_components if id(component) in selected_component_ids]
        if not components:
            return

        body_structural_island = np.full(model.body_count, -1, dtype=np.int32)
        for island, component in enumerate(components):
            body_structural_island[component.bodies] = island
        graph_body_ids = sorted(selected_body_ids)
        graph_body_island = body_structural_island[graph_body_ids]
        graph_joint_ids = sorted(selected_joint_ids)
        # Fuse only when every selected row can own its full stress update.
        # Mixed/world-anchored graphs retain the separate row-metric pass.
        self._fuse_rod_stress = all(
            joint_type[joint] == int(JointType.ROD) and parent[joint] >= 0 for joint in graph_joint_ids
        )
        body_slot = {body: slot for slot, body in enumerate(graph_body_ids)}

        for plan in selected_plans:
            if plan.kind == "path":
                bucket = _PathBucket(plan.components, self.device, factor_dtype=path_factor_dtype)
                self.path_buckets.append(bucket)
            elif plan.kind == "tree":
                tree_dtype = wp.float64 if plan.diagnostics.planned_route.endswith("_fp64") else wp.float32
                bucket = _TreeBucket(
                    plan.components,
                    body_slot,
                    self.device,
                    symbolics=plan.tree_symbolics,
                    allow_fused_tree_levels=True,
                    factor_dtype=tree_dtype,
                )
                self.tree_buckets.append(bucket)
            else:
                bucket = _ClosedTreeBucket(
                    plan.components,
                    parent,
                    child,
                    body_slot,
                    self.device,
                    tree_symbolics=plan.tree_symbolics,
                    cooperative_paired=plan.diagnostics.cooperative_paired,
                )
                self.closed_tree_buckets.append(bucket)
            bucket._diagnostics = plan.diagnostics
            bucket._estimated_bytes = plan.diagnostics.estimated_bytes
            bucket._components = plan.components

        self.buckets = [
            *self.path_buckets,
            *self.tree_buckets,
            *self.closed_tree_buckets,
        ]
        for bucket in self.path_buckets:
            bucket.bind_body_endpoints(parent, child, body_slot)
        for bucket in self.closed_tree_buckets:
            bucket.side_streams = self.side_streams
            if self.side_streams is not None:
                bucket.tree.group_rows_by_joint_type(model.joint_type.numpy())
        candidates = [*self.tree_buckets, *self.closed_tree_buckets]
        self.chain_route, row_offsets = ChainCondensation.plan(
            [
                (
                    bucket._components,
                    np.concatenate((bucket.tree.joint_ids_host, bucket.closure_joint_ids.numpy()))
                    if isinstance(bucket, _ClosedTreeBucket)
                    else bucket.joint_ids_host,
                )
                for bucket in candidates
            ],
            joint_type,
            parent,
            child,
            body_inv_mass_host,
            model.body_count,
            self.device,
        )
        chain_diag_ids: set[int] = set()
        for bucket, offset in zip(candidates, row_offsets, strict=True):
            if offset is None:
                continue
            chain_diag_ids.add(id(bucket._diagnostics))
            bucket.chain_route = self.chain_route
            bucket._diagnostics = replace(bucket._diagnostics, selected_route="chain_condensation")
            tree = bucket.tree if isinstance(bucket, _ClosedTreeBucket) else bucket
            (
                tree.jacobian_parent,
                tree.jacobian_child,
                tree.compliance,
                tree.residual,
                tree.row_active,
            ) = self.chain_route.rows(offset, tree.row_count)
            if tree.row_slices is not None:
                starts = np.cumsum([0] + [group[1] for group in tree.row_slices])
                tree.row_slices = [
                    (
                        tree.joint_ids[start:end],
                        end - start,
                        tree.jacobian_parent[start:end],
                        tree.jacobian_child[start:end],
                        tree.compliance[start:end],
                        tree.residual[start:end],
                        tree.row_active[start:end],
                    )
                    for start, end in zip(starts[:-1].tolist(), starts[1:].tolist(), strict=True)
                ]
            if isinstance(bucket, _ClosedTreeBucket):
                (
                    bucket.closure_jacobian_parent,
                    bucket.closure_jacobian_child,
                    bucket.closure_compliance,
                    bucket.closure_residual,
                    bucket.closure_row_active,
                ) = self.chain_route.rows(offset + tree.row_count, bucket.closure_size)
        if chain_diag_ids:
            self._bucket_diagnostics = tuple(
                replace(diag, selected_route="chain_condensation") if id(diag) in chain_diag_ids else diag
                for diag in self._bucket_diagnostics
            )
        # Row passes visit each launch's rows grouped by joint type; line-search slopes replay the
        # stored rows in identity order.
        self.row_block_dim = _ROW_PASS_GPU_BLOCK_DIM if self.device.is_cuda else 1
        self.row_orders = {}
        self.row_joint_type = np.asarray(model.joint_type.numpy())
        for bucket in self.buckets:
            if isinstance(bucket, _ClosedTreeBucket):
                launches = (
                    [bucket.tree.joint_ids]
                    if bucket.tree.row_slices is None
                    else [s[0] for s in bucket.tree.row_slices]
                )
                launches.append(bucket.closure_joint_ids)
            else:
                launches = [bucket.joint_ids]
            for ids in launches:
                self._row_order(ids)

        self.graph_body_count = len(graph_body_ids)
        self.graph_body_ids = wp.array(graph_body_ids, dtype=wp.int32, device=self.device)
        self.graph_body_island = wp.array(graph_body_island, dtype=wp.int32, device=self.device)
        # Per-island contact topology: 1 = no active dynamic-dynamic contact,
        # -1 = active dynamic-dynamic contact.
        self.island_contact_state = wp.ones(len(components), dtype=wp.int32, device=self.device)
        # Disabling a represented world/static joint can release translation
        # inside an existing CUDA graph. FREE rows carry no anchoring energy.
        translation_candidate = [
            int(all(body_inv_mass_host[body] > 0.0 for body in component.bodies)) for component in components
        ]
        anchor_offsets = [0]
        anchor_joints = []
        joint_enabled_host = np.asarray(model.joint_enabled.numpy(), dtype=bool)
        translation_free = []
        for candidate, component in zip(translation_candidate, components, strict=True):
            anchors = [
                joint
                for joint in component.joints
                if joint_type[joint] != JointType.FREE
                and (
                    parent[joint] < 0
                    or child[joint] < 0
                    or body_inv_mass_host[parent[joint]] <= 0.0
                    or body_inv_mass_host[child[joint]] <= 0.0
                )
            ]
            anchor_joints.extend(anchors)
            anchor_offsets.append(len(anchor_joints))
            translation_free.append(int(candidate and not any(joint_enabled_host[joint] for joint in anchors)))
        self.has_free_translation = any(translation_candidate)
        self.has_switchable_translation = self.has_free_translation and bool(anchor_joints)
        # Bound storage by the selected joint count, also used by the planner.
        anchor_storage = np.full(len(graph_joint_ids), -1, dtype=np.int32)
        anchor_storage[: len(anchor_joints)] = anchor_joints
        self.translation_candidate = wp.array(translation_candidate, dtype=wp.int32, device=self.device)
        self.translation_anchor_offsets = wp.array(anchor_offsets, dtype=wp.int32, device=self.device)
        self.translation_anchor_joints = wp.array(anchor_storage, dtype=wp.int32, device=self.device)
        self.translation_active = wp.array([int(any(translation_free))], dtype=wp.int32, device=self.device)
        self.island_translation_free = wp.array(translation_free, dtype=wp.int32, device=self.device)
        self.translation_system = wp.zeros(len(components), dtype=wp.mat44d, device=self.device)
        self.translation_support = wp.zeros(len(components), dtype=_TranslationSupport, device=self.device)
        self.translation_remainder = wp.zeros(len(graph_body_ids), dtype=wp.vec4, device=self.device)
        self.translation_trial = wp.zeros(len(graph_body_ids), dtype=wp.vec4d, device=self.device)
        # Material-frame rod residuals are invariant under collective rigid motion.
        # Ordinary world-frame joint duals need their projected rows included.
        rigid_mode_candidates = [
            int(all(joint_type[joint] in (JointType.ROD, JointType.FREE) for joint in component.joints))
            for component in components
        ]
        self.has_free_rigid_modes = any(
            rod and dynamic for rod, dynamic in zip(rigid_mode_candidates, translation_candidate, strict=True)
        )
        self.rigid_mode_candidate = wp.array(
            rigid_mode_candidates,
            dtype=wp.int32,
            device=self.device,
        )
        self.rigid_mode_center = wp.zeros(len(components), dtype=wp.vec4d, device=self.device)
        # State handed to the rigid-mode system after the local sweeps: every island evaluates
        # its external contacts only (the translation pass does not classify internal pairs).
        self.rigid_reconcile_state = wp.full(len(components), -1, dtype=wp.int32, device=self.device)
        self.rigid_mode_matrix = wp.zeros(len(components), dtype=wp.spatial_matrixd, device=self.device)
        self.rigid_mode_rhs = wp.zeros(len(components), dtype=wp.spatial_vectord, device=self.device)
        self.rigid_mode_shift = wp.zeros(len(components), dtype=wp.spatial_vectord, device=self.device)
        body_slot_by_id = np.full(model.body_count, -1, dtype=np.int32)
        body_slot_by_id[graph_body_ids] = np.arange(len(graph_body_ids), dtype=np.int32)
        self.graph_joint_ids = wp.array(graph_joint_ids, dtype=wp.int32, device=self.device)
        self.body_slot_by_id = wp.array(body_slot_by_id, dtype=wp.int32, device=self.device)
        self._plan_stress_incidence(parent, child, body_slot_by_id, body_inv_mass_host, graph_joint_ids)
        lim_ke_h = model.joint_limit_ke.numpy()
        lim_lo_h = model.joint_limit_lower.numpy()
        lim_hi_h = model.joint_limit_upper.numpy()
        joint_has_limit = (
            (lim_ke_h > 0.0) & ((lim_lo_h > -MAXVAL) | (lim_hi_h < MAXVAL))
            if len(lim_ke_h) > 0
            else np.zeros(0, dtype=bool)
        )
        drv_ke_h = model.joint_target_ke.numpy()
        drv_kd_h = model.joint_target_kd.numpy()
        lim_kd_h = model.joint_limit_kd.numpy()
        self.has_any_joint_drives_or_limits = bool(
            np.any(joint_has_limit)
            or (len(drv_ke_h) > 0 and np.any((drv_ke_h > 0.0) | (drv_kd_h > 0.0) | (lim_kd_h > 0.0)))
        )

        def _bucket_joint_ids_host(b):
            return (
                [*b.tree.joint_ids_host, *b.closure_joint_ids.numpy()]
                if isinstance(b, _ClosedTreeBucket)
                else b.joint_ids_host
            )

        self.plan_rigidities = tuple(plan_rigidities)
        # Largest joint omega^2 [s^-2] of every non-rod bucket (0 for buckets with rod rows); a step of size h
        # skips the predicted-pose stress term of a bucket whose omega^2 h^2 reaches 2^24 (bucket_skips_stress).
        self.bucket_omega2 = {
            id(bucket): (
                0.0
                if any(joint_type[j] == int(JointType.ROD) for j in _bucket_joint_ids_host(bucket))
                else max((float(joint_omega2[j]) for j in _bucket_joint_ids_host(bucket)), default=0.0)
            )
            for bucket in self.buckets
        }
        # Fallback of the second-stage decision for a captured graph without conditional nodes
        # (second_stage_without_flag): the largest depth^2 * omega^2 [s^-2] over branched non-rod trees
        # (0 without float32 tree buckets).
        branched_tree_rigidity = 0.0
        for plan in selected_plans:
            if plan.kind != "tree":
                continue
            for tree_cand in plan.components:
                if any(joint_type[j] == int(JointType.ROD) for j in tree_cand.joints):
                    continue
                is_branched, joint_depth = _tree_is_branched_and_depth(tree_cand, parent, child, body_inv_mass_host)
                if not is_branched:
                    continue
                tree_omega2 = max((float(joint_omega2[j]) for j in tree_cand.joints), default=0.0)
                branched_tree_rigidity = max(branched_tree_rigidity, float(joint_depth * joint_depth) * tree_omega2)
        has_float32_trees = any(b.solution.dtype == wp.spatial_vector for b in self.tree_buckets)
        self.branched_tree_rigidity = branched_tree_rigidity if has_float32_trees else 0.0
        # A non-rod float32 tree builds the stress term of every stage from the multipliers its
        # last solve stored (add_clean_saddle_stress_majorizer), at every joint stiffness.
        self.stored_stress_buckets = {
            id(bucket)
            for bucket in self.tree_buckets
            if bucket.solution.dtype == wp.spatial_vector
            and not any(joint_type[j] == int(JointType.ROD) for j in bucket.joint_ids_host)
        }
        self.has_any_joint_limits = bool(np.any(joint_has_limit))
        for bucket in self.tree_buckets:
            # The sweep owns the stress rows of rod joints, so rod trees keep it.
            has_rod = any(int(joint_type[j]) == int(JointType.ROD) for j in bucket.joint_ids_host)
            bucket.supports_sqp_chord_resolve = bool(
                bucket.chain_route is None
                and not has_rod
                and not (bucket.use_backbone_cr and not bucket.use_paired_open_backbone)
            )
        self.has_resolvable_trees = any(b.supports_sqp_chord_resolve for b in self.tree_buckets)
        # Path assembly consumes bit 0; resolved-tree sweep skipping consumes bit 2
        # (zero extra device bytes).
        body_path_inverse = np.zeros(self.graph_body_count, dtype=np.int32)
        for bucket in self.path_buckets:
            path_slots = body_slot_by_id[bucket.body_ids_host]
            body_path_inverse[path_slots[path_slots >= 0]] |= 1
        resolved_dynamic_bodies: set[int] = set()
        for bucket in self.tree_buckets:
            if bucket.supports_sqp_chord_resolve:
                res_slots = body_slot_by_id[bucket.body_ids_host]
                body_path_inverse[res_slots[res_slots >= 0]] |= 4
                for body_id in bucket.body_ids_host.tolist():
                    if body_inv_mass_host[int(body_id)] > 0.0:
                        resolved_dynamic_bodies.add(int(body_id))
        self.all_dynamic_rigid_resolved = bool(
            self.has_resolvable_trees and len(resolved_dynamic_bodies) == int(np.sum(body_inv_mass_host > 0.0))
        )
        # Bit 3: bodies of the float32 non-rod trees. The rotation of their first correction decides
        # whether a second stage follows (flag_second_stage; zero extra device bytes).
        for bucket in self.tree_buckets:
            if id(bucket) in self.stored_stress_buckets:
                stage_slots = body_slot_by_id[bucket.body_ids_host]
                body_path_inverse[stage_slots[stage_slots >= 0]] |= 8
        self.has_second_stage_bodies = bool(self.stored_stress_buckets)
        self.second_stage_needed = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.body_path_inverse = wp.array(body_path_inverse, dtype=wp.int32, device=self.device)
        slope_rows = sum(
            bucket.tree.size + bucket.closure_size if isinstance(bucket, _ClosedTreeBucket) else bucket.size
            for bucket in self.buckets
        )
        self.slope_jacobian_parent = wp.zeros(slope_rows, dtype=wp.spatial_matrix, device=self.device)
        self.slope_jacobian_child = wp.zeros(slope_rows, dtype=wp.spatial_matrix, device=self.device)
        self.slope_compliance = wp.zeros(slope_rows, dtype=wp.spatial_vector, device=self.device)
        self.slope_residual = wp.zeros(slope_rows, dtype=wp.spatial_vector, device=self.device)
        self.slope_objective = wp.zeros(slope_rows, dtype=wp.spatial_vector, device=self.device)
        self.slope_mode = wp.zeros(slope_rows, dtype=wp.int32, device=self.device)
        self.line_search_pose = wp.empty_like(model.body_q)
        self.line_search_disp = wp.zeros(model.body_count, dtype=wp.vec3, device=self.device)
        self.line_search_disp_ang = wp.zeros(model.body_count, dtype=wp.vec3, device=self.device)
        self.line_search_enabled = wp.ones(len(components), dtype=bool, device=self.device)
        self.line_search_slope = wp.zeros(len(components), dtype=wp.float64, device=self.device)
        self.line_search_initial_slope = wp.empty_like(self.line_search_slope)
        self.line_search_pending = wp.ones(1, dtype=int, device=self.device)
        self.limit_search_needed = wp.zeros(1, dtype=int, device=self.device)
        self.limit_search_islands = wp.zeros(len(components), dtype=wp.int32, device=self.device)
        # Islands with an active rigid contact, and whether there is one: the predicate of the solve under contact.
        self.contact_solve_islands = wp.zeros(len(components), dtype=wp.int32, device=self.device)
        self.contact_solve_needed = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.has_limit_rows = bool(np.any(joint_has_limit))
        self.line_search_trial = wp.zeros(1, dtype=int, device=self.device) if self.device.is_cuda else None
        self.translation_iterations_remaining = (
            wp.empty(1, dtype=int, device=self.device) if self.device.is_cuda else None
        )
        self.body_correction = wp.zeros(self.graph_body_count, dtype=wp.spatial_vector, device=self.device)
        self.island_step_scale = wp.ones(len(components), dtype=float, device=self.device)
        self.body_matrix = wp.zeros(self.graph_body_count, dtype=wp.spatial_matrix, device=self.device)
        # The relaxation pass reads contact curvature after body factorization.
        # Its lifetime overlaps the body metric and its in-place path inverse.
        self.body_dynamic_contact_hessian = wp.zeros(self.graph_body_count, dtype=wp.spatial_matrix, device=self.device)
        self.body_rhs = wp.zeros(self.graph_body_count, dtype=wp.spatial_vector, device=self.device)
        # Tree equilibration is consumed before its scatter writes the final
        # correction, so both lifetimes share the same compact body buffer.
        self.body_scale = self.body_correction
        # Tree bodies keep their metric/rhs for tree factorization; path bodies
        # hold their inverse/free motion in the same storage (disjoint slots).
        self.body_inverse = self.body_matrix
        self.body_free = self.body_rhs
        self._estimated_bytes = selected_bucket_bytes + _shared_structural_payload_bytes(
            model.body_count,
            model.joint_count,
            self.graph_body_count,
            len(components),
            len(graph_joint_ids),
            is_cuda=self.device.is_cuda,
        )

    def _row_order(self, ids: wp.array) -> wp.array:
        """Return the stable joint-type order of one launch's rows, computing it on first use."""
        cached = self.row_orders.get(id(ids))
        if cached is None:
            grouped = np.argsort(self.row_joint_type[ids.numpy()], kind="stable").astype(np.int32)
            cached = (ids, wp.array(grouped, dtype=wp.int32, device=self.device))
            self.row_orders[id(ids)] = cached
        return cached[1]

    def _plan_stress_incidence(
        self,
        parent: np.ndarray,
        child: np.ndarray,
        body_slot_by_id: np.ndarray,
        body_inv_mass_host: np.ndarray,
        graph_joint_ids: list[int],
    ):
        """Give every dynamic graph body a contiguous range of stress slots.

        Each range lists the body's incident joint sides in ascending joint id
        order (a joint's parent side before its child side), a sum order that is
        the same on CPU and CUDA and independent of the bucket layout.
        ``joint_stress_slot[joint]`` holds the parent and child destinations,
        ``-1`` for a static, kinematic, or out-of-graph endpoint.
        """
        joint_count = len(graph_joint_ids)
        incident: list[list[tuple[int, int]]] = [[] for _ in range(self.graph_body_count)]
        for joint in graph_joint_ids:
            for side, body in enumerate((int(parent[joint]), int(child[joint]))):
                if body >= 0 and body_slot_by_id[body] >= 0 and body_inv_mass_host[body] > 0.0:
                    incident[int(body_slot_by_id[body])].append((int(joint), side))
        offsets = np.zeros(self.graph_body_count + 1, dtype=np.int32)
        offsets[1:] = np.cumsum([len(entries) for entries in incident])
        joint_stress_slot = np.full((len(parent), 2), -1, dtype=np.int32)
        for slot, entries in enumerate(incident):
            for rank, (joint, side) in enumerate(entries):
                joint_stress_slot[joint, side] = offsets[slot] + rank
        self.joint_stress_slot = wp.array(joint_stress_slot, dtype=wp.vec2i, device=self.device)
        self.stress_offsets = wp.array(offsets, dtype=wp.int32, device=self.device)
        self.body_stress = wp.zeros(2 * joint_count, dtype=wp.mat33, device=self.device)

    @property
    def active(self) -> bool:
        return bool(self.buckets)

    @property
    def island_count(self) -> int:
        return sum(bucket.batch_count for bucket in self.buckets)

    @property
    def joint_count(self) -> int:
        return sum(bucket.size for bucket in self.buckets)

    def refresh_translation_freedom(self, joint_enabled):
        """Keep released-anchor momentum reconciliation graph resident."""
        if not self.has_switchable_translation:
            return
        self.translation_active.zero_()
        wp.launch(
            refresh_translation_freedom,
            self.island_count,
            inputs=[
                self.translation_candidate,
                self.translation_anchor_offsets,
                self.translation_anchor_joints,
                joint_enabled,
            ],
            outputs=[self.island_translation_free, self.translation_active],
            device=self.device,
        )

    def clear_stored_stress(self, world_mask, world_count, joint_world):
        """Zero the multipliers from which the next stage of the selected worlds would build its stress term."""
        for bucket in self.tree_buckets:
            if id(bucket) in self.stored_stress_buckets:
                wp.launch(
                    clear_stored_row_multipliers,
                    bucket.size,
                    inputs=[
                        world_mask,
                        world_mask is None,
                        world_count,
                        joint_world,
                        bucket.joint_ids,
                        bucket.row_node,
                    ],
                    outputs=[bucket.solution],
                    device=self.device,
                )

    def plan_change(self, dt: float) -> int:
        """Compare the decisions of this plan with those a step of size ``dt`` asks for.

        Returns 1 when ``dt`` needs a more robust route or factor precision than was planned, -1 when it would
        allow a faster one, and 0 when every decision is the same.
        """
        change = 0
        for rigidity in self.plan_rigidities:
            was = _rigid_at_step(rigidity, self.plan_step)
            now = _rigid_at_step(rigidity, float(dt))
            if now and not was:
                return 1
            if was and not now:
                change = -1
        return change

    def flag_second_stage(self) -> wp.array:
        """Set ``second_stage_needed`` from the correction the last solve applied and return it."""
        self.second_stage_needed.zero_()
        wp.launch(
            flag_nonlinear_correction,
            self.graph_body_count,
            inputs=[
                self.body_path_inverse,
                self.graph_body_island,
                self.island_contact_state,
                self.island_step_scale,
                self.body_correction,
            ],
            outputs=[self.second_stage_needed],
            device=self.device,
            block_dim=self.spatial_block_dim,
        )
        return self.second_stage_needed

    def flag_contact_solve(self) -> wp.array:
        """Set ``contact_solve_needed`` when ``contact_solve_islands`` marks an island and return it."""
        self.contact_solve_needed.zero_()
        wp.launch(
            flag_any_island,
            self.contact_solve_islands.shape[0],
            inputs=[self.contact_solve_islands],
            outputs=[self.contact_solve_needed],
            device=self.device,
        )
        return self.contact_solve_needed

    def second_stage_without_flag(self, dt: float) -> bool:
        """Whether a step of size ``dt`` that cannot read ``second_stage_needed`` takes the second stage.

        A captured graph without conditional nodes has no device predicate. The decision then comes from the
        structure and the step: branched float32 trees take the stage once ``depth^2 * omega^2 * dt^2``
        reaches 2^24. A mechanism that does not turn gets the same result as with the flag.
        """
        return self.branched_tree_rigidity * float(dt) * float(dt) >= _FP32_MANTISSA_RECIPROCAL

    def bucket_skips_stress(self, bucket, dt: float) -> bool:
        """Whether a step of size ``dt`` skips the predicted-pose stress term of ``bucket`` (near rigidity)."""
        return self.bucket_omega2.get(id(bucket), 0.0) * float(dt) * float(dt) >= _FP32_MANTISSA_RECIPROCAL

    def solve(
        self,
        *,
        dt,
        contacts,
        body_q,
        body_inertia_q,
        body_disp,
        body_disp_ang,
        use_displacement,
        body_q_prev,
        body_q_rest,
        body_mass,
        body_inv_mass,
        body_inertia,
        body_com,
        contact_hessian_ll,
        contact_hessian_al,
        contact_hessian_aa,
        contact_forces,
        contact_torques,
        dynamic_contact_hessian,
        joint_type,
        joint_enabled,
        joint_parent,
        joint_child,
        joint_X_p,
        joint_X_c,
        joint_axis,
        joint_rod_rest_kb_local,
        joint_rod_rest_twist,
        joint_qd_start,
        joint_target_q_start,
        joint_dof_dim,
        joint_constraint_start,
        joint_material_k,
        joint_rho,
        joint_penalty_kd,
        joint_target_ke,
        joint_target_kd,
        joint_target_q,
        joint_target_qd,
        joint_limit_lower,
        joint_limit_upper,
        joint_limit_ke,
        joint_limit_kd,
        joint_drive_limit_support,
        joint_drive_lambda,
        joint_limit_lambda,
        joint_lambda_lin,
        joint_lambda_ang,
        joint_C0_lin,
        joint_C0_ang,
        joint_rest_angle,
        joint_angle_prev,
        joint_sigma_start,
        joint_C_fric,
        stab_alpha,
        shape_body,
        body_contact_buffer_size,
        body_contact_counts,
        body_contact_indices,
        refresh_contacts=None,
        translation_only=False,
        resolve_only=False,
        resolve_sqp_torque=True,
        contact_stream=None,
        contact_damping_bidirectional=False,
        contact_normal_rho=None,
        contact_material_ke=None,
        contact_material_kd=None,
        contact_material_mu=None,
        contact_tangent_rho=None,
        contact_torsional_rho=None,
        contact_rolling_rho=None,
        contact_lambda=None,
        contact_lambda_angular=None,
        contact_C0=None,
        contact_stab_alpha=0.0,
        sync_alm_duals=False,
    ):
        def solve_chain_route():
            self.chain_route.solve(
                body_slot_by_id=self.body_slot_by_id,
                graph_body_island=self.graph_body_island,
                island_contact_state=self.island_contact_state,
                include_stress=1,
                stress_offsets=self.stress_offsets,
                body_stress=self.body_stress,
                body_correction=self.body_correction,
                dt=dt,
                body_q=body_q,
                body_inertia_q=body_inertia_q,
                body_disp=body_disp,
                body_disp_ang=body_disp_ang,
                use_displacement=use_displacement,
                body_q_prev=body_q_prev,
                body_mass=body_mass,
                body_inv_mass=body_inv_mass,
                body_inertia=body_inertia,
                body_com=body_com,
                contacts=contacts,
                shape_body=shape_body,
                body_contact_buffer_size=body_contact_buffer_size,
                body_contact_counts=body_contact_counts,
                body_contact_indices=body_contact_indices,
                contact_damping_bidirectional=contact_damping_bidirectional,
                contact_normal_rho=contact_normal_rho,
                contact_material_ke=contact_material_ke,
                contact_material_kd=contact_material_kd,
                contact_material_mu=contact_material_mu,
                contact_tangent_rho=contact_tangent_rho,
                contact_torsional_rho=contact_torsional_rho,
                contact_rolling_rho=contact_rolling_rho,
                contact_lambda=contact_lambda,
                contact_lambda_angular=contact_lambda_angular,
                contact_C0=contact_C0,
                contact_stab_alpha=contact_stab_alpha,
            )

        def linearizations_closure(bucket):
            return (
                bucket.closure_joint_ids,
                bucket.closure_size,
                bucket.closure_jacobian_parent,
                bucket.closure_jacobian_child,
                bucket.closure_compliance,
                bucket.closure_residual,
                bucket.closure_row_active,
            )

        def linearize(
            *,
            build_majorizer=True,
            build_body=True,
            slope_only=False,
            contact_stream=None,
            stored_slope=False,
            store_slope_rows=False,
        ):
            # Rows are linearized first: the stress majorizer writes per-row
            # blocks that the body surrogate then sums in fixed order.
            # Independent row groups (joint types, closures) overlap on the row
            # lanes; each launch writes only its own rows.
            lanes = self.row_lanes
            main = wp.get_stream(self.device) if lanes is not None else None
            forked = []
            launch_index = 0
            slope_offset = 0
            for bucket in self.buckets:
                linearizations = (
                    (
                        (
                            bucket.tree.joint_ids,
                            bucket.tree.size,
                            bucket.tree.jacobian_parent,
                            bucket.tree.jacobian_child,
                            bucket.tree.compliance,
                            bucket.tree.residual,
                            bucket.tree.row_active,
                        ),
                        (
                            bucket.closure_joint_ids,
                            bucket.closure_size,
                            bucket.closure_jacobian_parent,
                            bucket.closure_jacobian_child,
                            bucket.closure_compliance,
                            bucket.closure_residual,
                            bucket.closure_row_active,
                        ),
                    )
                    if isinstance(bucket, _ClosedTreeBucket) and bucket.tree.row_slices is None
                    else (*bucket.tree.row_slices, linearizations_closure(bucket))
                    if isinstance(bucket, _ClosedTreeBucket)
                    else (
                        (
                            bucket.joint_ids,
                            bucket.size,
                            bucket.jacobian_parent,
                            bucket.jacobian_child,
                            bucket.compliance,
                            bucket.residual,
                            bucket.row_active,
                        ),
                    )
                )
                for (
                    linearization_joint_ids,
                    linearization_size,
                    jacobian_parent,
                    jacobian_child,
                    compliance,
                    residual,
                    row_active,
                ) in linearizations:
                    stream = lanes[launch_index % len(lanes)] if lanes is not None else None
                    launch_index += 1
                    if stream is not None and all(stream is not lane for lane in forked):
                        stream.wait_stream(main)
                        forked.append(stream)
                    row_offset = slope_offset
                    slope_offset += linearization_size
                    if stored_slope:
                        with _stream_scope(stream):
                            wp.launch(
                                accumulate_stored_joint_slope,
                                linearization_size,
                                inputs=[
                                    linearization_joint_ids,
                                    joint_parent,
                                    joint_child,
                                    row_offset,
                                    self.slope_jacobian_parent,
                                    self.slope_jacobian_child,
                                    self.slope_compliance,
                                    self.slope_residual,
                                    self.slope_objective,
                                    self.slope_mode,
                                    self.body_slot_by_id,
                                    self.graph_body_island,
                                    self.body_correction,
                                ],
                                outputs=[self.line_search_slope],
                                device=self.device,
                                block_dim=self.spatial_block_dim,
                            )
                        continue
                    grouped_rows = self._row_order(linearization_joint_ids)
                    with _stream_scope(stream):
                        flag_limits = not self._fuse_rod_stress and not slope_only and self.has_limit_rows
                        wp.launch(
                            _linearize_fused_rod_rows if self._fuse_rod_stress else linearize_joint_path_rows,
                            linearization_size,
                            inputs=[
                                linearization_joint_ids,
                                grouped_rows,
                                joint_type,
                                joint_enabled,
                                joint_parent,
                                joint_child,
                                joint_X_p,
                                joint_X_c,
                                joint_axis,
                                joint_rod_rest_kb_local,
                                joint_rod_rest_twist,
                                joint_qd_start,
                                joint_target_q_start,
                                joint_dof_dim,
                                joint_constraint_start,
                                joint_material_k,
                                joint_rho,
                                joint_penalty_kd,
                                joint_lambda_lin,
                                joint_lambda_ang,
                                joint_C0_lin,
                                joint_C0_ang,
                                joint_sigma_start,
                                joint_C_fric,
                                joint_target_ke,
                                joint_target_kd,
                                joint_target_q,
                                joint_target_qd,
                                joint_limit_lower,
                                joint_limit_upper,
                                joint_limit_ke,
                                joint_limit_kd,
                                joint_rest_angle,
                                joint_angle_prev,
                                joint_drive_limit_support,
                                joint_drive_lambda,
                                joint_limit_lambda,
                                stab_alpha,
                                body_q,
                                body_q_prev,
                                body_q_rest,
                                body_com,
                                dt,
                                slope_only,
                                self.body_slot_by_id,
                                self.graph_body_island,
                                self.body_correction,
                                self.line_search_slope,
                            ],
                            outputs=[
                                jacobian_parent,
                                jacobian_child,
                                compliance,
                                residual,
                                row_active,
                                build_majorizer and self._fuse_rod_stress,
                                self.joint_stress_slot,
                                body_inv_mass,
                                self.body_stress,
                                store_slope_rows,
                                row_offset,
                                self.slope_jacobian_parent,
                                self.slope_jacobian_child,
                                self.slope_compliance,
                                self.slope_residual,
                                self.slope_objective,
                                self.slope_mode,
                                self.limit_search_needed if (flag_limits and contacts is None) else None,
                                self.limit_search_islands if flag_limits else None,
                            ],
                            device=self.device,
                            block_dim=self.row_block_dim,
                        )
                        if slope_only:
                            wp.launch(
                                accumulate_stored_joint_slope,
                                linearization_size,
                                inputs=[
                                    linearization_joint_ids,
                                    joint_parent,
                                    joint_child,
                                    row_offset,
                                    self.slope_jacobian_parent,
                                    self.slope_jacobian_child,
                                    self.slope_compliance,
                                    self.slope_residual,
                                    self.slope_objective,
                                    self.slope_mode,
                                    self.body_slot_by_id,
                                    self.graph_body_island,
                                    self.body_correction,
                                ],
                                outputs=[self.line_search_slope],
                                device=self.device,
                                block_dim=self.spatial_block_dim,
                            )
                        if (
                            build_majorizer
                            and not self._fuse_rod_stress
                            and not self.bucket_skips_stress(bucket, dt)
                            and id(bucket) not in self.stored_stress_buckets
                        ):
                            wp.launch(
                                add_joint_stress_majorizer,
                                linearization_size,
                                inputs=[
                                    self.joint_stress_slot,
                                    linearization_joint_ids,
                                    grouped_rows,
                                    joint_type,
                                    joint_parent,
                                    joint_child,
                                    joint_X_p,
                                    joint_X_c,
                                    body_q,
                                    body_com,
                                    body_inv_mass,
                                    self.body_slot_by_id,
                                    jacobian_parent,
                                    jacobian_child,
                                    compliance,
                                    residual,
                                ],
                                outputs=[self.body_stress],
                                device=self.device,
                                block_dim=self.row_block_dim,
                            )
                        elif build_majorizer and not self._fuse_rod_stress and id(bucket) in self.stored_stress_buckets:
                            wp.launch(
                                add_clean_saddle_stress_majorizer,
                                linearization_size,
                                inputs=[
                                    self.joint_stress_slot,
                                    linearization_joint_ids,
                                    grouped_rows,
                                    joint_type,
                                    joint_parent,
                                    joint_child,
                                    joint_X_p,
                                    joint_X_c,
                                    body_q,
                                    body_com,
                                    body_inv_mass,
                                    self.body_slot_by_id,
                                    jacobian_parent,
                                    jacobian_child,
                                    bucket.row_node,
                                    bucket.row_scale,
                                    bucket.row_active,
                                    bucket.solution,
                                ],
                                outputs=[self.body_stress],
                                device=self.device,
                                block_dim=self.row_block_dim,
                            )
            for stream in forked:
                main.wait_stream(stream)
            if contact_stream is not None:
                # The surrogate is the first consumer of the contact objective.
                main.wait_stream(contact_stream)
            if build_body:
                wp.launch(
                    build_body_surrogate,
                    self.graph_body_ids.shape[0],
                    inputs=[
                        self.graph_body_ids,
                        dt,
                        body_q,
                        body_inertia_q,
                        body_disp,
                        body_disp_ang,
                        use_displacement,
                        body_mass,
                        body_inv_mass,
                        body_inertia,
                        body_com,
                        contact_hessian_ll,
                        contact_hessian_al,
                        contact_hessian_aa,
                        contact_forces,
                        contact_torques,
                        dynamic_contact_hessian,
                        int(contacts is not None and contacts.rigid_contact_max > 0),
                        int(build_majorizer),
                        self.stress_offsets,
                        self.body_stress,
                        self.body_path_inverse,
                    ],
                    outputs=[self.body_matrix, self.body_rhs],
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )

        def refresh_contact_objective(*, translation=False, forces_only=False):
            if translation:
                self.translation_system.zero_()
                refresh_contacts(True)
                return
            if forces_only:
                wp.launch(
                    clear_structural_contact_forces,
                    self.graph_body_count,
                    inputs=[self.graph_body_ids],
                    outputs=[contact_forces, contact_torques],
                    device=self.device,
                )
                refresh_contacts(False, True)
                return
            wp.launch(
                clear_structural_contact_objective,
                self.graph_body_count,
                inputs=[self.graph_body_ids],
                outputs=[
                    contact_forces,
                    contact_torques,
                    contact_hessian_ll,
                    contact_hessian_al,
                    contact_hessian_aa,
                    dynamic_contact_hessian,
                ],
                device=self.device,
            )
            refresh_contacts(translation)

        def directional_derivative(*, refresh=True, stored_joint_slope=None):
            contact_side = self.side_streams[1] if self.side_streams is not None else None
            refresh_objective = refresh and contacts is not None and refresh_contacts is not None
            if contact_side is None and refresh_objective:
                refresh_contact_objective(forces_only=True)
            self.line_search_slope.zero_()
            if contact_side is not None:
                # Body contact and joint rows add independent slope terms.
                main = wp.get_stream(self.device)
                contact_side.wait_stream(main)
            with _stream_scope(contact_side):
                if contact_side is not None and refresh_objective:
                    refresh_contact_objective(forces_only=True)
                wp.launch(
                    accumulate_body_directional_derivative,
                    self.graph_body_count,
                    inputs=[
                        self.graph_body_ids,
                        self.graph_body_island,
                        self.line_search_enabled,
                        body_q,
                        body_inertia_q,
                        body_disp,
                        body_disp_ang,
                        use_displacement,
                        body_mass,
                        body_inv_mass,
                        body_inertia,
                        body_com,
                        contact_forces,
                        contact_torques,
                        dt,
                        self.body_correction,
                    ],
                    outputs=[self.line_search_slope],
                    device=self.device,
                    block_dim=self.spatial_block_dim,
                )
            # Contact line search reuses stored step-zero rows; contact-free limit search
            # evaluates joint slope only when a limit island actually requests the search,
            # paying zero slope-buffer writes on ordinary standby steps.
            use_stored = (not refresh) if stored_joint_slope is None else stored_joint_slope
            linearize(build_majorizer=False, build_body=False, slope_only=True, stored_slope=use_stored)
            if contact_side is not None:
                main.wait_stream(contact_side)

        def balance_translation(
            *, apply_translation=False, track_pending=False, stop_at_roundoff=True, restricted_ready=False, carry=False
        ):
            if self.has_free_translation and not restricted_ready:
                self.translation_system.zero_()
                wp.launch(
                    accumulate_free_translation_system,
                    self.graph_body_count,
                    inputs=[
                        self.graph_body_ids,
                        self.graph_body_island,
                        self.body_slot_by_id,
                        self.island_translation_free,
                        self.island_contact_state,
                        self.island_step_scale,
                        self.body_correction,
                        dt,
                        body_q,
                        body_inertia_q,
                        body_disp,
                        body_disp_ang,
                        use_displacement,
                        body_com,
                        body_mass,
                        body_inv_mass,
                        contact_hessian_ll,
                        contact_hessian_al,
                        contact_forces,
                        dynamic_contact_hessian,
                        int(contacts is not None),
                        contacts.rigid_contact_count if contacts is not None else None,
                        contacts.rigid_contact_shape0 if contacts is not None else None,
                        contacts.rigid_contact_shape1 if contacts is not None else None,
                        shape_body,
                        body_contact_buffer_size,
                        body_contact_counts,
                        body_contact_indices,
                    ],
                    outputs=[self.translation_system],
                    device=self.device,
                )
            if self.has_free_translation:
                wp.launch(
                    correct_free_translation,
                    self.graph_body_count,
                    inputs=[
                        self.graph_body_island,
                        self.island_translation_free,
                        self.island_contact_state,
                        self.island_step_scale,
                        self.translation_system,
                        self.body_correction,
                        apply_translation,
                        track_pending,
                        stop_at_roundoff,
                        carry,
                        self.translation_support,
                        self.graph_body_ids,
                        use_displacement,
                        body_com,
                        body_inertia_q,
                    ],
                    outputs=[
                        body_q,
                        self.line_search_pending,
                        self.translation_remainder,
                        self.translation_trial,
                        body_disp,
                    ],
                    device=self.device,
                )

        if translation_only:
            # Local contact reconciliation can disturb common translation.
            # Resolve only this three-dimensional mode; internal joint errors
            # and internal contact separations are invariant under the shift.
            if not self.use_fused_solve_reset:
                self.island_step_scale.fill_(1.0)
            self.body_correction.zero_()
            if contacts is None:
                # Nothing to reconcile without contacts: one projection, always applied.
                balance_translation(apply_translation=True, stop_at_roundoff=False)
                return

            def reconcile_free_rigid_modes():
                # The local sweeps take angular momentum out of an island that has a contact
                # inside it, and the translation pass above restores three of the six rigid
                # modes. Rod rows and internal contacts do no work under a rigid motion of the
                # island, so the inertial residual restricted to those modes is the island's
                # momentum defect. Islands that touch a static or kinematic body keep the
                # translation pass only.
                self.rigid_mode_center.zero_()
                self.rigid_mode_matrix.zero_()
                self.rigid_mode_rhs.zero_()
                wp.launch(
                    accumulate_free_rigid_center,
                    self.graph_body_count,
                    inputs=[
                        self.graph_body_ids,
                        self.graph_body_island,
                        self.island_translation_free,
                        self.rigid_mode_candidate,
                        self.island_contact_state,
                        self.translation_system,
                        body_q,
                        body_com,
                        body_mass,
                    ],
                    outputs=[self.rigid_mode_center],
                    device=self.device,
                )
                wp.launch(
                    exclude_supported_rigid_centers,
                    self.graph_body_count,
                    inputs=[
                        self.graph_body_ids,
                        self.graph_body_island,
                        body_q,
                        body_inv_mass,
                        contact_lambda,
                        contacts.rigid_contact_count,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        contacts.rigid_contact_point0,
                        contacts.rigid_contact_point1,
                        contacts.rigid_contact_normal,
                        contacts.rigid_contact_margin0,
                        contacts.rigid_contact_margin1,
                        shape_body,
                        body_contact_buffer_size,
                        body_contact_counts,
                        body_contact_indices,
                    ],
                    outputs=[self.rigid_mode_center],
                    device=self.device,
                )
                wp.launch(
                    accumulate_free_rigid_system,
                    self.graph_body_count,
                    inputs=[
                        self.graph_body_ids,
                        self.graph_body_island,
                        self.rigid_reconcile_state,
                        self.rigid_mode_center,
                        self.island_step_scale,
                        self.body_correction,
                        dt,
                        body_q,
                        body_inertia_q,
                        body_disp,
                        body_disp_ang,
                        use_displacement,
                        body_q_prev,
                        body_com,
                        body_mass,
                        body_inv_mass,
                        body_inertia,
                        contact_hessian_ll,
                        contact_hessian_al,
                        contact_hessian_aa,
                        contact_forces,
                        contact_torques,
                        dynamic_contact_hessian,
                        1,
                        bool(contact_damping_bidirectional),
                        contact_normal_rho,
                        contact_material_ke,
                        contact_material_kd,
                        contact_material_mu,
                        contact_tangent_rho,
                        contact_torsional_rho,
                        contact_rolling_rho,
                        contact_lambda,
                        contact_lambda_angular,
                        contact_C0,
                        float(contact_stab_alpha),
                        contacts.rigid_contact_count,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        contacts.rigid_contact_point0,
                        contacts.rigid_contact_point1,
                        contacts.rigid_contact_surface_velocity,
                        contacts.rigid_contact_offset0,
                        contacts.rigid_contact_offset1,
                        contacts.rigid_contact_normal,
                        contacts.rigid_contact_margin0,
                        contacts.rigid_contact_margin1,
                        shape_body,
                        body_contact_buffer_size,
                        body_contact_counts,
                        body_contact_indices,
                    ],
                    outputs=[self.rigid_mode_matrix, self.rigid_mode_rhs],
                    device=self.device,
                )
                wp.launch(
                    solve_free_rigid_system,
                    self.island_count,
                    inputs=[self.rigid_mode_center, self.rigid_mode_matrix, self.rigid_mode_rhs],
                    outputs=[self.rigid_mode_shift],
                    device=self.device,
                )
                wp.launch(
                    enrich_free_rigid_correction,
                    self.graph_body_count,
                    inputs=[
                        self.graph_body_ids,
                        self.graph_body_island,
                        self.rigid_mode_center,
                        self.rigid_mode_shift,
                        self.island_step_scale,
                        body_q,
                        body_com,
                    ],
                    outputs=[self.body_correction],
                    device=self.device,
                )
                wp.launch(
                    apply_free_rigid_reconciliation,
                    self.graph_body_count,
                    inputs=[
                        self.graph_body_ids,
                        self.graph_body_island,
                        self.rigid_mode_center,
                        self.body_correction,
                        body_com,
                        use_displacement,
                        body_inertia_q,
                    ],
                    outputs=[body_q, body_disp, body_disp_ang],
                    device=self.device,
                )

            track_pending = self.device.is_cpu or (self.device.is_capturing and wp.is_conditional_graph_supported())
            if track_pending:
                self.line_search_pending.fill_(1)

            def translate(*, refresh=True):
                if refresh:
                    refresh_contact_objective(translation=True)
                # An island accepts a correction only under the stop's relative tolerance and guard;
                # every island that accepted one in this evaluation raises the flag again.
                if track_pending:
                    self.line_search_pending.zero_()
                balance_translation(
                    apply_translation=True,
                    track_pending=track_pending,
                    restricted_ready=contacts.rigid_contact_max > 0,
                    # Only the refreshed restricted objective reads the unstored shifts.
                    carry=refresh and contacts.rigid_contact_max > 0,
                )

            # One evaluation on the stored objective and at most fifteen refreshed ones, with
            # the same stopping rule in every determinism mode.
            evaluations = 16
            if self.device.is_cuda and track_pending:
                # One graph loop instead of fifteen replicated conditional bodies.
                translate(refresh=False)
                self.translation_iterations_remaining.fill_(evaluations - 1)

                def repeat_translation():
                    translate()
                    wp.launch(
                        _finish_translation_iteration,
                        1,
                        inputs=[self.translation_iterations_remaining, self.line_search_pending],
                        device=self.device,
                    )

                wp.capture_while(self.line_search_pending, repeat_translation)
            else:
                for iteration in range(evaluations):
                    if track_pending:
                        wp.capture_if(self.line_search_pending, translate, refresh=iteration > 0)
                    else:
                        translate(refresh=iteration > 0)
            if self.has_free_rigid_modes and contacts.rigid_contact_max > 0 and contact_normal_rho is not None:
                reconcile_free_rigid_modes()
            return

        if resolve_only:
            if not self.has_resolvable_trees:
                return
            if not self.use_fused_solve_reset:
                self.body_correction.zero_()
            any_resolved = False
            slope_offset = 0
            for bucket in self.tree_buckets:
                row_offset = slope_offset
                slope_offset += bucket.size
                if not bucket.supports_sqp_chord_resolve:
                    continue
                if bucket.use_backbone_cr and not (bucket.use_paired_open_backbone and contacts is None):
                    continue
                grouped_rows = self._row_order(bucket.joint_ids)
                wp.launch(
                    _linearize_fused_rod_rows if self._fuse_rod_stress else linearize_joint_path_rows,
                    bucket.size,
                    inputs=[
                        bucket.joint_ids,
                        grouped_rows,
                        joint_type,
                        joint_enabled,
                        joint_parent,
                        joint_child,
                        joint_X_p,
                        joint_X_c,
                        joint_axis,
                        joint_rod_rest_kb_local,
                        joint_rod_rest_twist,
                        joint_qd_start,
                        joint_target_q_start,
                        joint_dof_dim,
                        joint_constraint_start,
                        joint_material_k,
                        joint_rho,
                        joint_penalty_kd,
                        joint_lambda_lin,
                        joint_lambda_ang,
                        joint_C0_lin,
                        joint_C0_ang,
                        joint_sigma_start,
                        joint_C_fric,
                        joint_target_ke,
                        joint_target_kd,
                        joint_target_q,
                        joint_target_qd,
                        joint_limit_lower,
                        joint_limit_upper,
                        joint_limit_ke,
                        joint_limit_kd,
                        joint_rest_angle,
                        joint_angle_prev,
                        joint_drive_limit_support,
                        joint_drive_lambda,
                        joint_limit_lambda,
                        stab_alpha,
                        body_q,
                        body_q_prev,
                        body_q_rest,
                        body_com,
                        dt,
                        True,
                        self.body_slot_by_id,
                        self.graph_body_island,
                        self.body_correction,
                        self.line_search_slope,
                    ],
                    outputs=[
                        bucket.jacobian_parent,
                        bucket.jacobian_child,
                        bucket.compliance,
                        bucket.residual,
                        bucket.row_active,
                        False,
                        self.joint_stress_slot,
                        body_inv_mass,
                        self.body_stress,
                        True,
                        row_offset,
                        self.slope_jacobian_parent,
                        self.slope_jacobian_child,
                        self.slope_compliance,
                        self.slope_residual,
                        self.slope_objective,
                        self.slope_mode,
                        self.limit_search_needed,
                        self.limit_search_islands,
                    ],
                    device=self.device,
                    block_dim=self.row_block_dim,
                )
                ok = bucket.resolve_tree(
                    self.graph_body_ids,
                    joint_parent,
                    dt,
                    body_q,
                    body_inertia_q,
                    body_disp,
                    body_disp_ang,
                    use_displacement,
                    body_mass,
                    body_inv_mass,
                    body_inertia,
                    body_com,
                    self.body_matrix,
                    row_offset,
                    self.slope_jacobian_parent,
                    self.slope_jacobian_child,
                    self.slope_compliance,
                    self.slope_residual,
                    self.slope_mode,
                    self.body_rhs,
                    self.body_scale,
                    self.body_correction,
                    allow_paired_open_backbone=contacts is None,
                    include_sqp_torque=resolve_sqp_torque,
                )
                any_resolved = any_resolved or ok
            if not any_resolved:
                return
        else:
            # Only the contact line search below reads the stored step-zero slope rows.
            linearize(
                contact_stream=contact_stream,
                store_slope_rows=contacts is not None and refresh_contacts is not None,
            )

            # Trees consume the body metric directly. Path Schur systems consume
            # its inverse, which the surrogate build already wrote for path bodies.
            if not self.use_fused_solve_reset:
                self.body_correction.zero_()
            for bucket in self.tree_buckets:
                if bucket.chain_route is not None:
                    continue
                bucket.solve_tree(
                    self.body_matrix,
                    self.body_rhs,
                    self.body_scale,
                    self.body_correction,
                    allow_paired_open_backbone=contacts is None,
                )
        if not resolve_only:
            for bucket in self.closed_tree_buckets:
                if bucket.chain_route is not None:
                    continue
                bucket.solve_tree(self.body_matrix, self.body_rhs, self.body_scale, self.body_correction)
            if self.chain_route is not None:
                solve_chain_route()

        for bucket in [] if resolve_only else self.path_buckets:
            # Scalar-row CUDA assembly exposes the six independent output rows
            # without changing Schur equations or adding temporary buffers.
            row_parallel = self.device.is_cuda
            wp.launch(
                assemble_joint_path_system_rows if row_parallel else assemble_joint_path_system,
                (bucket.size, 6) if row_parallel else bucket.size,
                inputs=[
                    bucket.row_bounds,
                    bucket.row_body,
                    *[
                        array.view(float) if row_parallel else array
                        for array in (
                            self.body_inverse,
                            self.body_free,
                            bucket.jacobian_parent,
                            bucket.jacobian_child,
                            bucket.compliance,
                            bucket.residual,
                        )
                    ],
                ],
                outputs=[
                    array.view(float) if row_parallel else array
                    for array in (bucket.lower[0], bucket.diagonal[0], bucket.upper[0], bucket.rhs[0])
                ],
                device=self.device,
                block_dim=128 if row_parallel else self.spatial_block_dim,
            )
            bucket.solve_rows()
            wp.launch(
                compute_path_correction_fp64 if bucket.factor_rhs is not None else compute_path_correction,
                bucket.body_size,
                inputs=[
                    bucket.body_ids,
                    self.body_slot_by_id,
                    bucket.body_incident_rows,
                    bucket.row_active,
                    bucket.jacobian_parent,
                    bucket.jacobian_child,
                    bucket.factor_rhs if bucket.factor_rhs is not None else bucket.solution[0],
                    self.body_inverse,
                    self.body_free,
                    body_inv_mass,
                ],
                outputs=[self.body_correction],
                device=self.device,
                block_dim=self.spatial_block_dim,
            )

        # Factor first, then accept one coherent correction per structural
        # island. Dynamic-pair overlap determines a coefficient-free Jacobi
        # relaxation; static/kinematic contact remains in the local VBD solve.
        if not self.use_fused_solve_reset:
            self.island_step_scale.fill_(1.0)
        # The surrogate majorizes each represented dynamic contact edge with
        # H_static + 2 H_dynamic, which is block Jacobi's half relaxation.
        # The directional search below evaluates the step on the true merit;
        # without it, keep the additional island limit.
        if contacts is not None and contacts.rigid_contact_max > 0 and refresh_contacts is None:
            wp.launch(
                limit_dynamic_contact_jacobi_step,
                self.graph_body_count,
                inputs=[
                    self.graph_body_ids,
                    self.graph_body_island,
                    self.island_contact_state,
                    self.body_correction,
                    dynamic_contact_hessian,
                    dt,
                    body_q,
                    body_mass,
                    body_inertia,
                ],
                outputs=[self.island_step_scale],
                device=self.device,
            )
        # Finite limits remain compliant force rows. An additional hard
        # feasibility projection can suppress an entire island at equilibrium
        # (where load requires finite violation), or freeze its active set.
        # The following local sweeps own limit/contact dual reconciliation.
        wp.launch(
            suppress_nonfinite_correction,
            self.graph_body_count,
            inputs=[self.graph_body_island, self.body_correction],
            outputs=[self.island_contact_state],
            device=self.device,
        )
        balance_translation()
        # Restore the six collective modes suppressed by the rod solve metric.
        # Only unanchored, independently supported islands are eligible. The
        # existing directional search evaluates the enriched correction.
        # Project f-B*delta with the original body metric: rod rows annihilate
        # infinitesimal collective rigid motion. Do not omit projected joint
        # terms for ordinary world-frame duals or anchored islands.
        if self.has_free_rigid_modes:

            def enrich_rigid_modes():
                self.rigid_mode_center.zero_()
                self.rigid_mode_matrix.zero_()
                self.rigid_mode_rhs.zero_()
                wp.launch(
                    accumulate_free_rigid_center,
                    self.graph_body_count,
                    inputs=[
                        self.graph_body_ids,
                        self.graph_body_island,
                        self.island_translation_free,
                        self.rigid_mode_candidate,
                        self.island_contact_state,
                        self.translation_system,
                        body_q,
                        body_com,
                        body_mass,
                    ],
                    outputs=[self.rigid_mode_center],
                    device=self.device,
                )
                has_c = int(contacts is not None and contacts.rigid_contact_max > 0 and contact_normal_rho is not None)
                wp.launch(
                    accumulate_free_rigid_system,
                    self.graph_body_count,
                    inputs=[
                        self.graph_body_ids,
                        self.graph_body_island,
                        self.island_contact_state,
                        self.rigid_mode_center,
                        self.island_step_scale,
                        self.body_correction,
                        dt,
                        body_q,
                        body_inertia_q,
                        body_disp,
                        body_disp_ang,
                        use_displacement,
                        body_q_prev,
                        body_com,
                        body_mass,
                        body_inv_mass,
                        body_inertia,
                        contact_hessian_ll,
                        contact_hessian_al,
                        contact_hessian_aa,
                        contact_forces,
                        contact_torques,
                        dynamic_contact_hessian,
                        has_c,
                        bool(contact_damping_bidirectional),
                        contact_normal_rho if has_c else None,
                        contact_material_ke if has_c else None,
                        contact_material_kd if has_c else None,
                        contact_material_mu if has_c else None,
                        contact_tangent_rho if has_c else None,
                        contact_torsional_rho if has_c else None,
                        contact_rolling_rho if has_c else None,
                        contact_lambda if has_c else None,
                        contact_lambda_angular if has_c else None,
                        contact_C0 if has_c else None,
                        float(contact_stab_alpha),
                        contacts.rigid_contact_count if has_c else None,
                        contacts.rigid_contact_shape0 if has_c else None,
                        contacts.rigid_contact_shape1 if has_c else None,
                        contacts.rigid_contact_point0 if has_c else None,
                        contacts.rigid_contact_point1 if has_c else None,
                        contacts.rigid_contact_surface_velocity if has_c else None,
                        contacts.rigid_contact_offset0 if has_c else None,
                        contacts.rigid_contact_offset1 if has_c else None,
                        contacts.rigid_contact_normal if has_c else None,
                        contacts.rigid_contact_margin0 if has_c else None,
                        contacts.rigid_contact_margin1 if has_c else None,
                        shape_body,
                        body_contact_buffer_size,
                        body_contact_counts,
                        body_contact_indices,
                    ],
                    outputs=[self.rigid_mode_matrix, self.rigid_mode_rhs],
                    device=self.device,
                )
                wp.launch(
                    solve_free_rigid_system,
                    self.island_count,
                    inputs=[self.rigid_mode_center, self.rigid_mode_matrix, self.rigid_mode_rhs],
                    outputs=[self.rigid_mode_shift],
                    device=self.device,
                )
                wp.launch(
                    enrich_free_rigid_correction,
                    self.graph_body_count,
                    inputs=[
                        self.graph_body_ids,
                        self.graph_body_island,
                        self.rigid_mode_center,
                        self.rigid_mode_shift,
                        self.island_step_scale,
                        body_q,
                        body_com,
                    ],
                    outputs=[self.body_correction],
                    device=self.device,
                )

            if self.device.is_cpu or (self.device.is_capturing and wp.is_conditional_graph_supported()):
                wp.capture_if(self.translation_active, enrich_rigid_modes)
            else:
                enrich_rigid_modes()

        def _run_directional_search():
            if use_displacement:
                wp.launch(
                    _snapshot_search_state,
                    body_q.shape[0],
                    inputs=[body_q, body_disp, body_disp_ang],
                    outputs=[self.line_search_pose, self.line_search_disp, self.line_search_disp_ang],
                    device=self.device,
                )
            else:
                wp.copy(self.line_search_pose, body_q)

            self.line_search_enabled.fill_(True)
            # Contact search reuses step-zero slope rows; contact-free limit search
            # evaluates step-zero joint slope only here when requested.
            directional_derivative(refresh=False, stored_joint_slope=contacts is not None)
            wp.launch(
                _begin_limit_search if contacts is None else _begin_directional_search,
                self.island_count,
                inputs=[self.island_contact_state, self.line_search_slope],
                outputs=[
                    *([self.limit_search_needed, self.limit_search_islands] if contacts is None else []),
                    self.line_search_enabled,
                    self.island_step_scale,
                    self.line_search_initial_slope,
                    self.line_search_pending,
                ],
                device=self.device,
            )

            def apply_trial(trial=None):
                wp.launch(
                    _apply_trial_correction,
                    self.graph_body_count,
                    inputs=[
                        self.graph_body_ids,
                        self.graph_body_island,
                        self.body_correction,
                        self.island_step_scale,
                        body_com,
                        self.line_search_pose,
                        self.line_search_disp,
                        self.line_search_disp_ang,
                        use_displacement,
                        body_inertia_q,
                    ],
                    outputs=[body_q, self.line_search_pending, body_disp, body_disp_ang],
                    device=self.device,
                )
                directional_derivative()
                if trial is None:
                    wp.launch(
                        _update_counted_directional_search_tolerant
                        if contacts is None
                        else _update_counted_directional_search,
                        self.island_count,
                        inputs=[
                            self.line_search_enabled,
                            self.line_search_slope,
                            self.line_search_initial_slope,
                            self.line_search_trial,
                        ],
                        outputs=[self.island_step_scale, self.line_search_pending],
                        device=self.device,
                    )
                    wp.launch(_advance_directional_trial, 1, outputs=[self.line_search_trial], device=self.device)
                    return
                wp.launch(
                    _update_directional_search_tolerant if contacts is None else _update_directional_search,
                    self.island_count,
                    inputs=[
                        self.line_search_enabled,
                        self.line_search_slope,
                        self.line_search_initial_slope,
                        trial == 4,
                    ],
                    outputs=[self.island_step_scale, self.line_search_pending],
                    device=self.device,
                )

            if self.device.is_cuda and self.device.is_capturing and wp.is_conditional_graph_supported():
                # Captured CUDA conditionals read the predicate on device. One loop
                # runs the same trials as five conditionals; an idle conditional
                # node still costs a graph launch.
                self.line_search_trial.zero_()
                wp.capture_while(self.line_search_pending, apply_trial)
            else:
                for trial in range(5):
                    if self.device.is_cpu:
                        wp.capture_if(self.line_search_pending, apply_trial, trial=trial)
                    else:
                        # Eager CUDA retains its fixed schedule without readback.
                        apply_trial(trial)
            wp.launch(
                _apply_trial_correction,
                self.graph_body_count,
                inputs=[
                    self.graph_body_ids,
                    self.graph_body_island,
                    self.body_correction,
                    self.island_step_scale,
                    body_com,
                    self.line_search_pose,
                    self.line_search_disp,
                    self.line_search_disp_ang,
                    use_displacement,
                    body_inertia_q,
                ],
                outputs=[body_q, self.line_search_pending, body_disp, body_disp_ang],
                device=self.device,
            )

        def _run_plain_correction():
            wp.launch(
                apply_global_correction,
                self.graph_body_count,
                inputs=[
                    self.graph_body_ids,
                    self.graph_body_island,
                    self.island_contact_state,
                    self.body_correction,
                    self.island_step_scale,
                    body_com,
                    use_displacement,
                    body_inertia_q,
                ],
                outputs=[body_q, body_disp, body_disp_ang],
                device=self.device,
            )

        if contacts is not None:
            if refresh_contacts is not None:
                _run_directional_search()
            else:
                _run_plain_correction()
        elif not self.has_limit_rows:
            _run_plain_correction()
        else:
            # Islands that did not ask for the search take the plain correction now.
            wp.launch(
                _apply_unflagged_correction,
                self.graph_body_count,
                inputs=[
                    self.graph_body_ids,
                    self.graph_body_island,
                    self.island_contact_state,
                    self.limit_search_islands,
                    self.body_correction,
                    self.island_step_scale,
                    body_com,
                    use_displacement,
                    body_inertia_q,
                ],
                outputs=[body_q, body_disp, body_disp_ang],
                device=self.device,
            )
            if self.device.is_cpu or (self.device.is_capturing and wp.is_conditional_graph_supported()):
                wp.capture_if(self.limit_search_needed, on_true=_run_directional_search)
            elif self.device.is_capturing:
                # Older CUDA graphs cannot branch on the device predicate. The
                # fixed search preserves already-corrected, unrequested islands.
                _run_directional_search()
            elif int(self.limit_search_needed.numpy()[0]) != 0:
                _run_directional_search()

        if sync_alm_duals and not translation_only:
            self.sync_resolved_tree_alm_duals(
                joint_type=joint_type,
                joint_parent=joint_parent,
                joint_child=joint_child,
                joint_X_p=joint_X_p,
                joint_X_c=joint_X_c,
                joint_axis=joint_axis,
                joint_qd_start=joint_qd_start,
                joint_dof_dim=joint_dof_dim,
                joint_constraint_start=joint_constraint_start,
                joint_material_k=joint_material_k,
                joint_penalty_kd=joint_penalty_kd,
                body_q=body_q,
                body_q_prev=body_q_prev,
                body_q_rest=body_q_rest,
                dt=dt,
                joint_lambda_lin=joint_lambda_lin,
                joint_lambda_ang=joint_lambda_ang,
                only_unflagged_islands=False,
            )

    def sync_resolved_tree_alm_duals(
        self,
        *,
        joint_type,
        joint_parent,
        joint_child,
        joint_X_p,
        joint_X_c,
        joint_axis,
        joint_qd_start,
        joint_dof_dim,
        joint_constraint_start,
        joint_material_k,
        joint_penalty_kd,
        body_q,
        body_q_prev,
        body_q_rest,
        dt: float,
        joint_lambda_lin,
        joint_lambda_ang,
        only_unflagged_islands: bool = False,
    ) -> None:
        """Write solved open-tree saddle multipliers ``mu_S`` into ALM ``joint_lambda``."""
        unflagged = int(only_unflagged_islands)
        for bucket in self.tree_buckets:
            if not bucket.supports_sqp_chord_resolve:
                continue
            wp.launch(
                sync_tree_saddle_to_alm_duals,
                bucket.size,
                inputs=[
                    bucket.joint_ids,
                    bucket.row_node,
                    bucket.row_scale,
                    bucket.row_active,
                    bucket.solution,
                    joint_type,
                    joint_parent,
                    joint_child,
                    joint_X_p,
                    joint_X_c,
                    joint_axis,
                    joint_qd_start,
                    joint_dof_dim,
                    joint_constraint_start,
                    joint_material_k,
                    joint_penalty_kd,
                    body_q,
                    body_q_prev,
                    body_q_rest,
                    self.body_slot_by_id,
                    self.graph_body_island,
                    self.island_step_scale,
                    self.limit_search_islands,
                    unflagged,
                    dt,
                ],
                outputs=[joint_lambda_lin, joint_lambda_ang],
                device=self.device,
                block_dim=self.row_block_dim,
            )


@wp.kernel
def _snapshot_search_state(
    body_q: wp.array[wp.transform],
    body_disp: wp.array[wp.vec3],
    body_disp_ang: wp.array[wp.vec3],
    pose_out: wp.array[wp.transform],
    disp_out: wp.array[wp.vec3],
    disp_ang_out: wp.array[wp.vec3],
):
    """Snapshot pose and displacement of every body before a directional search (one launch)."""
    tid = wp.tid()
    pose_out[tid] = body_q[tid]
    disp_out[tid] = body_disp[tid]
    disp_ang_out[tid] = body_disp_ang[tid]


@wp.kernel
def _begin_directional_search(
    contact_state: wp.array[int],
    merit: wp.array[wp.float64],
    enabled: wp.array[bool],
    scale: wp.array[float],
    initial_slope: wp.array[wp.float64],
    pending: wp.array[int],
):
    island = wp.tid()
    slope = merit[island]
    initial_slope[island] = slope
    if island == 0:
        pending[0] = 1
    good = contact_state[island] >= -1 and wp.isfinite(slope) and slope < wp.float64(0.0)
    enabled[island] = good
    if not good:
        scale[island] = 0.0


@wp.kernel
def _begin_limit_search(
    contact_state: wp.array[int],
    merit: wp.array[wp.float64],
    flag: wp.array[int],
    needed: wp.array[int],
    enabled: wp.array[bool],
    scale: wp.array[float],
    initial_slope: wp.array[wp.float64],
    pending: wp.array[int],
):
    """Begin the contact-free search on the islands whose linearization asked for it."""
    island = wp.tid()
    slope = merit[island]
    initial_slope[island] = slope
    if island == 0:
        pending[0] = 1
        flag[0] = 0
    wanted = needed[island] != 0
    good = wanted and contact_state[island] >= -1 and wp.isfinite(slope) and slope < wp.float64(0.0)
    enabled[island] = good
    if not good:
        scale[island] = 0.0


@wp.kernel
def _apply_unflagged_correction(
    body_ids: wp.array[wp.int32],
    graph_body_island: wp.array[wp.int32],
    island_contact_state: wp.array[wp.int32],
    needed: wp.array[int],
    correction: wp.array[wp.spatial_vector],
    island_step_scale: wp.array[float],
    body_com: wp.array[wp.vec3],
    use_displacement: int,
    body_target_q: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_disp: wp.array[wp.vec3],
    body_disp_ang: wp.array[wp.vec3],
):
    """:func:`apply_global_correction` for islands that did not request the limit search."""
    slot = wp.tid()
    body = body_ids[slot]
    island = graph_body_island[slot]
    if island_contact_state[island] < -1 or needed[island] != 0:
        return
    scale = island_step_scale[island]
    if scale == 0.0 or not _spatial_vector_nonzero(correction[slot]):
        return
    body_q[body] = _corrected_pose(body_q[body], correction[slot], body_com[body], scale)
    if use_displacement != 0:
        delta = correction[slot]
        if scale != 0.0 and _spatial_vector_nonzero(delta):
            body_disp[body] = body_disp[body] + scale * wp.spatial_top(delta)
            body_disp_ang[body] = rotation_vector_compose(scale * wp.spatial_bottom(delta), body_disp_ang[body])
            body_q[body] = pose_from_displacement(
                body_target_q[body], body_com[body], body_disp[body], wp.transform_get_rotation(body_q[body])
            )


@wp.func
def _directional_search_step(
    island: int,
    enabled: wp.array[bool],
    merit: wp.array[wp.float64],
    initial_slope: wp.array[wp.float64],
    last_trial: bool,
    tolerance: float,
    scale: wp.array[float],
    pending: wp.array[int],
):
    slope = merit[island]
    if enabled[island] and (not wp.isfinite(slope) or slope > wp.float64(-tolerance) * initial_slope[island]):
        # Interpolate the directional residual, whose value at zero is negative.
        # Safeguarding guarantees contraction even for nonquadratic forces.
        # A proposed step is always re-evaluated; interpolation is not acceptance.
        fraction = wp.float64(0.5)
        if wp.isfinite(slope):
            slope0 = initial_slope[island]
            fraction = wp.clamp(slope0 / (slope0 - slope), wp.float64(0.1), wp.float64(0.9))
        scale[island] = 0.0 if last_trial else float(fraction) * scale[island]
        if not last_trial:
            wp.atomic_max(pending, 0, 1)
    # Keep previously acceptable islands enabled: another island's shortened
    # trial can change a shared contact. Recheck all coupled endpoints together.


@wp.kernel
def _update_directional_search(
    enabled: wp.array[bool],
    merit: wp.array[wp.float64],
    initial_slope: wp.array[wp.float64],
    last_trial: bool,
    scale: wp.array[float],
    pending: wp.array[int],
):
    _directional_search_step(wp.tid(), enabled, merit, initial_slope, last_trial, 0.1, scale, pending)


@wp.kernel
def _update_directional_search_tolerant(
    enabled: wp.array[bool],
    merit: wp.array[wp.float64],
    initial_slope: wp.array[wp.float64],
    last_trial: bool,
    scale: wp.array[float],
    pending: wp.array[int],
):
    _directional_search_step(wp.tid(), enabled, merit, initial_slope, last_trial, 0.1, scale, pending)


@wp.kernel
def _update_counted_directional_search(
    enabled: wp.array[bool],
    merit: wp.array[wp.float64],
    initial_slope: wp.array[wp.float64],
    trial: wp.array[int],
    scale: wp.array[float],
    pending: wp.array[int],
):
    """:func:`_update_directional_search` with the trial index read on device."""
    _directional_search_step(wp.tid(), enabled, merit, initial_slope, trial[0] == 4, 0.1, scale, pending)


@wp.kernel
def _update_counted_directional_search_tolerant(
    enabled: wp.array[bool],
    merit: wp.array[wp.float64],
    initial_slope: wp.array[wp.float64],
    trial: wp.array[int],
    scale: wp.array[float],
    pending: wp.array[int],
):
    """:func:`_update_directional_search_tolerant` with the trial index read on device."""
    _directional_search_step(wp.tid(), enabled, merit, initial_slope, trial[0] == 4, 0.1, scale, pending)


@wp.kernel
def _advance_directional_trial(trial: wp.array[int]):
    trial[0] = trial[0] + 1


@wp.kernel
def _apply_trial_correction(
    body_ids: wp.array[int],
    body_island: wp.array[int],
    correction: wp.array[wp.spatial_vector],
    scale: wp.array[float],
    body_com: wp.array[wp.vec3],
    original: wp.array[wp.transform],
    original_disp: wp.array[wp.vec3],
    original_disp_ang: wp.array[wp.vec3],
    use_displacement: int,
    body_target_q: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    pending: wp.array[int],
    body_disp: wp.array[wp.vec3],
    body_disp_ang: wp.array[wp.vec3],
):
    slot = wp.tid()
    if slot == 0:
        pending[0] = 0
    body = body_ids[slot]
    alpha = scale[body_island[slot]]
    body_q[body] = original[body]
    if use_displacement != 0:
        body_disp[body] = original_disp[body]
        body_disp_ang[body] = original_disp_ang[body]
    if alpha > 0.0:
        body_q[body] = _corrected_pose(original[body], correction[slot], body_com[body], alpha)
        if use_displacement != 0:
            delta = correction[slot]
            if _spatial_vector_nonzero(delta):
                body_disp[body] = original_disp[body] + alpha * wp.spatial_top(delta)
                body_disp_ang[body] = rotation_vector_compose(alpha * wp.spatial_bottom(delta), original_disp_ang[body])
                body_q[body] = pose_from_displacement(
                    body_target_q[body], body_com[body], body_disp[body], wp.transform_get_rotation(body_q[body])
                )


# Resolve numeric overloads before graph capture.

wp.overload(
    resolve_tree_unique_leaves,
    [
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_vector],
    ],
)
wp.overload(
    resolve_tree_unique_leaves,
    [
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_vectord],
    ],
)
wp.overload(
    resolve_tree_messages,
    [
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_vector],
        wp.array[wp.spatial_vector],
    ],
)
wp.overload(
    resolve_tree_messages,
    [
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_vectord],
        wp.array[wp.spatial_vectord],
    ],
)

wp.overload(
    solve_block_ldlt_persistent,
    [int, wp.array[wp.spatial_matrix], wp.array[wp.spatial_vector]],
)
wp.overload(
    solve_block_ldlt_persistent,
    [int, wp.array[wp.spatial_matrixd], wp.array[wp.spatial_vectord]],
)
wp.overload(
    back_substitute_block_ldlt_lanes,
    [
        int,
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_vector],
        wp.array[wp.spatial_vector],
    ],
)
wp.overload(
    back_substitute_block_ldlt_lanes,
    [
        int,
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_vectord],
        wp.array[wp.spatial_vectord],
    ],
)
wp.overload(
    back_substitute_block_ldlt_serial,
    [int, wp.array[wp.spatial_matrix], wp.array[wp.spatial_vector]],
)
wp.overload(
    back_substitute_block_ldlt_serial,
    [int, wp.array[wp.spatial_matrixd], wp.array[wp.spatial_vectord]],
)
wp.overload(
    solve_block_ldlt_diagonal,
    [int, wp.array[wp.spatial_matrix], wp.array[wp.spatial_vector]],
)
wp.overload(
    solve_block_ldlt_diagonal,
    [int, wp.array[wp.spatial_matrixd], wp.array[wp.spatial_vectord]],
)
wp.overload(update_block_ldlt_lower_triangle, [int, int, int, wp.array[wp.spatial_matrix]])
wp.overload(
    update_block_ldlt_lower_triangle,
    [int, int, int, wp.array[wp.spatial_matrixd]],
)
wp.overload(
    factor_block_ldlt_pivot_rows,
    [int, int, wp.array[wp.spatial_matrix], wp.array[wp.spatial_vector]],
)
wp.overload(
    factor_block_ldlt_pivot_rows,
    [int, int, wp.array[wp.spatial_matrixd], wp.array[wp.spatial_vectord]],
)
wp.overload(invert_block_dense_pivot, [int, int, wp.array[wp.spatial_matrix]])
wp.overload(invert_block_dense_pivot, [int, int, wp.array[wp.spatial_matrixd]])
wp.overload(
    back_substitute_tree_response_level,
    [
        wp.array[wp.int32],
        int,
        wp.array[wp.int32],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_matrix],
    ],
)
wp.overload(
    back_substitute_tree_response_level,
    [
        wp.array[wp.int32],
        int,
        wp.array[wp.int32],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_matrixd],
    ],
)
wp.overload(
    accumulate_tree_response_messages,
    [
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.int32],
        int,
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_matrix],
    ],
)
wp.overload(
    accumulate_tree_response_messages,
    [
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.int32],
        int,
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_matrixd],
    ],
)
wp.overload(
    eliminate_tree_response_unique_leaves,
    [
        wp.array[wp.int32],
        int,
        wp.array[wp.int32],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_matrix],
    ],
)
wp.overload(
    eliminate_tree_response_unique_leaves,
    [
        wp.array[wp.int32],
        int,
        wp.array[wp.int32],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_matrixd],
    ],
)
wp.overload(
    back_substitute_tree_level,
    [
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_vector],
        wp.array[wp.spatial_vector],
    ],
)
wp.overload(
    back_substitute_tree_level,
    [
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_vectord],
        wp.array[wp.spatial_vectord],
    ],
)
wp.overload(
    accumulate_tree_messages,
    [
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_vector],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_vector],
    ],
)
wp.overload(
    accumulate_tree_messages,
    [
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_vectord],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_vectord],
    ],
)
wp.overload(
    eliminate_tree_unique_leaves,
    [
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_vector],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_vector],
    ],
)
wp.overload(
    eliminate_tree_unique_leaves,
    [
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_vectord],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_vectord],
    ],
)
wp.overload(invert_tree_leaves, [wp.array[wp.int32], wp.array[wp.spatial_matrix]])
wp.overload(invert_tree_leaves, [wp.array[wp.int32], wp.array[wp.spatial_matrixd]])
wp.overload(
    solve_tree_roots,
    [
        wp.array[wp.int32],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_vector],
        wp.array[wp.spatial_vector],
    ],
)
wp.overload(
    solve_tree_roots,
    [
        wp.array[wp.int32],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_vectord],
        wp.array[wp.spatial_vectord],
    ],
)
wp.overload(
    scatter_tree_body_correction_indexed,
    [
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.spatial_vector],
        wp.array[wp.spatial_vector],
        wp.array[wp.spatial_vector],
    ],
)
wp.overload(
    scatter_tree_body_correction_indexed,
    [
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.int32],
        wp.array[wp.spatial_vector],
        wp.array[wp.spatial_vectord],
        wp.array[wp.spatial_vector],
    ],
)

wp.overload(invert_cr_eliminated_in_place, [wp.array[wp.vec3i], wp.array[wp.spatial_matrix]])
wp.overload(invert_cr_eliminated_in_place, [wp.array[wp.vec3i], wp.array[wp.spatial_matrixd]])
wp.overload(
    reduce_cr_in_place,
    [
        int,
        int,
        wp.array[wp.vec3i],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_vector],
    ],
)
wp.overload(
    reduce_cr_in_place,
    [
        int,
        int,
        wp.array[wp.vec3i],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_vectord],
    ],
)
wp.overload(
    solve_strided_cr_coarse,
    [
        wp.array[wp.vec3i],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_vector],
    ],
)
wp.overload(
    solve_strided_cr_coarse,
    [
        wp.array[wp.vec3i],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_vectord],
    ],
)
wp.overload(
    back_substitute_cr_in_place,
    [
        int,
        wp.array[wp.vec3i],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_vector],
    ],
)
wp.overload(
    back_substitute_cr_in_place,
    [
        int,
        wp.array[wp.vec3i],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_vectord],
    ],
)
wp.overload(
    solve_cr_persistent,
    [
        wp.array[wp.vec2i],
        int,
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_vector],
    ],
)
wp.overload(
    solve_cr_persistent,
    [
        wp.array[wp.vec2i],
        int,
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_vectord],
    ],
)
wp.overload(
    solve_cr_tail_persistent,
    [
        wp.array[wp.vec3i],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_matrix],
        wp.array[wp.spatial_vector],
    ],
)
wp.overload(
    solve_cr_tail_persistent,
    [
        wp.array[wp.vec3i],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_matrixd],
        wp.array[wp.spatial_vectord],
    ],
)
for _sol_dtype in (wp.spatial_vector, wp.spatial_vectord):
    wp.overload(
        sync_tree_saddle_to_alm_duals,
        [
            wp.array[wp.int32],
            wp.array[wp.int32],
            wp.array[wp.spatial_vector],
            wp.array[wp.int32],
            wp.array[_sol_dtype],
            wp.array[wp.int32],
            wp.array[wp.int32],
            wp.array[wp.int32],
            wp.array[wp.transform],
            wp.array[wp.transform],
            wp.array[wp.vec3],
            wp.array[wp.int32],
            wp.array2d[wp.int32],
            wp.array[wp.int32],
            wp.array[float],
            wp.array[float],
            wp.array[wp.transform],
            wp.array[wp.transform],
            wp.array[wp.transform],
            wp.array[wp.int32],
            wp.array[wp.int32],
            wp.array[float],
            wp.array[wp.int32],
            int,
            float,
            wp.array[wp.vec3],
            wp.array[wp.vec3],
        ],
    )
