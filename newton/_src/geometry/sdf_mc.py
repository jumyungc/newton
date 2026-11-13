# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Marching Cubes utilities for SDF isosurface extraction.

Provides lookup tables and GPU functions for extracting triangular faces
from voxels that contain the zero-isosurface of an SDF. Used by hydroelastic
contact generation to find the contact surface between two colliding SDFs.

The marching cubes algorithm classifies each voxel by which of its 8 corners
are inside (negative SDF) vs outside (positive SDF), producing up to 5
triangles per voxel along the zero-crossing.
"""

import numpy as np
import warp as wp

from newton._src.core.types import MAXVAL

#: Corner values for a single voxel (8 corners)
vec8f = wp.types.vector(length=8, dtype=wp.float32)

# Marching cubes edge-interpolation constants.  Shared by sdf_mc.py and
# sdf_hydroelastic.py — import from here to keep values in one place.

MC_EDGE_VAL_DIFF_EPS = 1.0e-10
"""Minimum |val_diff| below which the edge midpoint is used instead of
interpolation.  Well below float32 ULP for typical SDF magnitudes, so this
branch only fires for nearly-identical corner values."""

MC_EDGE_CLAMP_MIN = 0.02
"""Lower t-clamp for edge interpolation.  Prevents vertex collapse when
corner values are near zero (e.g. at SDF ridge boundaries where both
shapes share the same nearest face).  Without the clamp, t close to 0 or 1
places multiple vertices at the same corner, producing degenerate
(zero-area) triangles."""

MC_EDGE_CLAMP_MAX = 0.98
"""Upper t-clamp (``1 - MC_EDGE_CLAMP_MIN``)."""

MC_DEGENERATE_N_SQ_EPS = 1.0e-20
"""Dimensionless squared-cross-product threshold for degenerate triangles.
Normalize edges by their largest absolute component before applying this
threshold so changing length units does not discard regular triangles."""


def _get_marching_cubes_class():
    """Find Warp's marching-cubes class without using its deprecated alias.

    Remove the fallback once Newton's minimum Warp version provides
    ``warp.geometry.IsoSurfaceMarchingCubes``.
    """
    try:
        from warp.geometry import IsoSurfaceMarchingCubes  # noqa: PLC0415 - Warp 1.17 lacks this module
    except ImportError:
        return wp.MarchingCubes
    return IsoSurfaceMarchingCubes


def get_mc_tables(device):
    """Create marching cubes lookup tables on the specified device.

    Returns:
        Tuple of 5 warp arrays:
        - tri_range_table: Start/end indices into triangle list per cube case (256 cases)
        - tri_local_inds_table: Edge indices for each triangle vertex
        - edge_to_verts_table: Corner vertex pairs for each of 12 edges
        - corner_offsets_table: 3D offsets for 8 cube corners
        - flat_edge_verts_table: Pre-flattened edge→vertex mapping for efficiency
    """
    # 12 edges of a cube, each connecting two corner vertices
    edge_to_verts = np.array(
        [
            [0, 1],  # 0
            [1, 2],  # 1
            [2, 3],  # 2
            [3, 0],  # 3
            [4, 5],  # 4
            [5, 6],  # 5
            [6, 7],  # 6
            [7, 4],  # 7
            [0, 4],  # 8
            [1, 5],  # 9
            [2, 6],  # 10
            [3, 7],  # 11
        ]
    )

    marching_cubes = _get_marching_cubes_class()
    tri_local_inds = np.asarray(marching_cubes.TRI_LOCAL_INDICES, dtype=np.int32)
    tri_range_table = wp.array(marching_cubes.CASE_TO_TRI_RANGE, dtype=wp.int32, device=device)
    tri_local_inds_table = wp.array(tri_local_inds, dtype=wp.int32, device=device)
    corner_offsets_table = wp.array(marching_cubes.CUBE_CORNER_OFFSETS, dtype=wp.vec3ub, device=device)
    edge_to_verts_table = wp.array(edge_to_verts, dtype=wp.vec2ub, device=device)

    # Create flattened table:
    # Instead of tri_local_inds_table[i] -> edge_to_verts_table[edge_idx, 0/1],
    # we directly map tri_local_inds_table[i] -> vec2i(v_from, v_to)
    flat_edge_verts = np.zeros((len(tri_local_inds), 2), dtype=np.uint8)

    for i, edge_idx in enumerate(tri_local_inds):
        flat_edge_verts[i, 0] = edge_to_verts[edge_idx, 0]
        flat_edge_verts[i, 1] = edge_to_verts[edge_idx, 1]

    flat_edge_verts_table = wp.array(flat_edge_verts, dtype=wp.vec2ub, device=device)

    return (
        tri_range_table,
        tri_local_inds_table,
        edge_to_verts_table,
        corner_offsets_table,
        flat_edge_verts_table,
    )


@wp.func
def int_to_vec3f(x: wp.int32, y: wp.int32, z: wp.int32):
    """Convert integer voxel coordinates to float vector."""
    return wp.vec3f(float(x), float(y), float(z))


@wp.func
def _triangle_clipped_area_and_centroid(vert_depths: wp.vec3f, num_inside: wp.int32) -> tuple[float, wp.vec3]:
    """Return clipped area fraction and its centroid's barycentric weights."""
    if num_inside == 3:
        return 1.0, wp.vec3(1.0 / 3.0)
    if num_inside == 0:
        return 0.0, wp.vec3(1.0 / 3.0)

    d0 = vert_depths[0]
    d1 = vert_depths[1]
    d2 = vert_depths[2]
    rotation = int(0)
    if num_inside == 1:
        if vert_depths[1] < 0.0:
            d0, d1, d2 = vert_depths[1], vert_depths[2], vert_depths[0]
            rotation = 1
        elif vert_depths[2] < 0.0:
            d0, d1, d2 = vert_depths[2], vert_depths[0], vert_depths[1]
            rotation = 2
    else:
        if vert_depths[1] >= 0.0:
            d0, d1, d2 = vert_depths[1], vert_depths[2], vert_depths[0]
            rotation = 1
        elif vert_depths[2] >= 0.0:
            d0, d1, d2 = vert_depths[2], vert_depths[0], vert_depths[1]
            rotation = 2

    scale = wp.max(wp.abs(d0), wp.max(wp.abs(d1), wp.abs(d2)))
    d0, d1, d2 = d0 / scale, d1 / scale, d2 / scale
    fraction = float(0.0)
    weights = wp.vec3(1.0 / 3.0)
    if num_inside == 1:
        if d0 != 0.0:
            t1 = d0 / (d0 - d1)
            t2 = d0 / (d0 - d2)
            fraction = t1 * t2
            weights = wp.vec3(1.0 - (t1 + t2) / 3.0, t1 / 3.0, t2 / 3.0)
    else:
        # Split the retained quadrilateral into positive-area triangles.
        # This avoids cancellation when the excluded triangle is almost full.
        t1 = float(0.0)
        t2 = float(0.0)
        if d1 != 0.0:
            t1 = -d1 / (d0 - d1)
        if d2 != 0.0:
            t2 = -d2 / (d0 - d2)
        f1 = t1
        f2 = t2 * (1.0 - t1)
        fraction = f1 + f2
        if fraction > 0.0:
            weights = (
                (f1 / fraction) * wp.vec3(t1, 2.0 - t1, 1.0) + (f2 / fraction) * wp.vec3(t1 + t2, 1.0 - t1, 2.0 - t2)
            ) / 3.0
    if rotation == 1:
        weights = wp.vec3(weights[2], weights[0], weights[1])
    elif rotation == 2:
        weights = wp.vec3(weights[1], weights[2], weights[0])
    return fraction, weights


@wp.func
def get_triangle_fraction(vert_depths: wp.vec3f, num_inside: wp.int32) -> wp.float32:
    """Compute the fraction of a triangle's area inside its linear zero contour."""
    fraction, _weights = _triangle_clipped_area_and_centroid(vert_depths, num_inside)
    return fraction


@wp.func
def _scaled_triangle_cross(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> tuple[wp.vec3, float]:
    """Return a dimensionless triangle cross product and its edge scale."""
    e0 = b - a
    e1 = c - a
    scale = wp.max(
        wp.max(wp.abs(e0[0]), wp.max(wp.abs(e0[1]), wp.abs(e0[2]))),
        wp.max(wp.abs(e1[0]), wp.max(wp.abs(e1[1]), wp.abs(e1[2]))),
    )
    if scale == 0.0:
        return wp.vec3(0.0), 0.0
    return wp.cross(e0 / scale, e1 / scale), scale


@wp.func
def mc_calc_face(
    flat_edge_verts_table: wp.array[wp.vec2ub],
    corner_offsets_table: wp.array[wp.vec3ub],
    tri_range_start: wp.int32,
    corner_vals: vec8f,
    sdf_a: wp.uint64,
    x_id: wp.int32,
    y_id: wp.int32,
    z_id: wp.int32,
    isovalue: wp.float32 = 0.0,
) -> tuple[float, wp.vec3, wp.vec3, float, wp.mat33f]:
    """Extract a triangle face from a marching cubes voxel.

    Interpolates vertex positions along cube edges where the SDF crosses zero,
    then computes face properties for contact generation.

    Returns:
        Tuple of (area, normal, center, penetration_depth, vertices):
        - area: Triangle area scaled by fraction inside the object
        - normal: Outward-facing unit normal
        - center: Triangle centroid in world space
        - penetration_depth: Average SDF depth (negative = penetrating, standard convention)
        - vertices: 3x3 matrix with vertex positions as rows
    """
    face_verts = wp.mat33f()
    vert_depths = wp.vec3f()
    num_inside = wp.int32(0)
    for vi in range(3):
        edge_verts = wp.vec2i(flat_edge_verts_table[tri_range_start + vi])
        v_idx_from = edge_verts[0]
        v_idx_to = edge_verts[1]
        val_0 = wp.float32(corner_vals[v_idx_from])
        val_1 = wp.float32(corner_vals[v_idx_to])

        p_0 = wp.vec3f(corner_offsets_table[v_idx_from])
        p_1 = wp.vec3f(corner_offsets_table[v_idx_to])
        val_diff = wp.float32(val_1 - val_0)
        if wp.abs(val_diff) < MC_EDGE_VAL_DIFF_EPS:
            p = 0.5 * (p_0 + p_1)
        else:
            # Clamp t away from cube corners to prevent vertex collapse when
            # corner values are near zero (e.g. at SDF ridge boundaries).
            # Without the clamp, t close to 0 or 1 places multiple vertices
            # at the same corner, producing degenerate (zero-area) triangles.
            t = wp.clamp((isovalue - val_0) / val_diff, MC_EDGE_CLAMP_MIN, MC_EDGE_CLAMP_MAX)
            p = p_0 + t * (p_1 - p_0)
        vol_idx = p + int_to_vec3f(x_id, y_id, z_id)
        p_scaled = wp.volume_index_to_world(sdf_a, vol_idx)
        face_verts[vi] = p_scaled
        depth = wp.volume_sample_f(sdf_a, vol_idx, wp.Volume.LINEAR)
        if depth >= wp.static(MAXVAL * 0.99) or wp.isnan(depth):
            depth = 0.0
        vert_depths[vi] = depth  # Keep SDF convention: negative = inside/penetrating
        if depth < 0.0:
            num_inside += 1

    n, edge_scale = _scaled_triangle_cross(face_verts[0], face_verts[1], face_verts[2])
    n_sq = wp.dot(n, n)
    if n_sq < MC_DEGENERATE_N_SQ_EPS:
        # Degenerate triangle — return zero area with a valid (non-NaN) normal.
        area = 0.0
        normal = wp.vec3(0.0, 0.0, 1.0)
    else:
        n_len = wp.sqrt(n_sq)
        normal = n / n_len
        area = (n_len * 0.5 * edge_scale) * edge_scale
    center = (face_verts[0] + face_verts[1] + face_verts[2]) / 3.0
    pen_depth = (vert_depths[0] + vert_depths[1] + vert_depths[2]) / 3.0
    fraction = float(num_inside == 3)
    if num_inside > 0 and num_inside < 3:
        fraction, weights = _triangle_clipped_area_and_centroid(vert_depths, num_inside)
        center = weights[0] * face_verts[0] + weights[1] * face_verts[1] + weights[2] * face_verts[2]
        pen_depth = wp.dot(weights, vert_depths)
    area *= fraction
    return area, normal, center, pen_depth, face_verts
