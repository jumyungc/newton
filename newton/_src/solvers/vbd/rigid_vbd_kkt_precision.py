# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tree/closure assembly and recovery with double-precision arithmetic.

The body and joint linearization retains its original float32 material data.
Equilibration, elimination, closure assembly and recovery use float64 until the
final body correction is stored. CPU uses a serial tree walk; CUDA shares the
paired parallel factorization kernels with the float32 open-tree path.
"""

import warp as wp

wp.set_module_options({"enable_backward": False})


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
def _scale_spatial_matrix(
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
def _scale_spatial_vector(scale: wp.spatial_vector, value: wp.spatial_vector):
    result = wp.spatial_vectord()
    for row in range(6):
        result[row] = wp.float64(scale[row]) * wp.float64(value[row])
    return result


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
    diagonal: wp.array[wp.spatial_matrixd],
    rhs: wp.array[wp.spatial_vectord],
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
    value = _scale_spatial_matrix(row_scale[row], jacobian, body_scale[body])
    coupling[node] = value if node_is_joint else wp.transpose(value)


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
    response_rhs: wp.array[wp.spatial_matrixd],
):
    """Build the six tree right-hand sides induced by every closure row."""
    index = wp.tid()
    node = index // closure_count
    batch = node // tree_node_count
    closure = batch * closure_count + index % closure_count
    value = wp.spatial_matrixd(0.0)
    if node == closure_parent_node[closure]:
        coupling = _scale_spatial_matrix(row_scale[closure], jacobian_parent[closure], body_scale[node_body[node]])
        value = value + wp.transpose(coupling)
    if node == closure_child_node[closure]:
        coupling = _scale_spatial_matrix(row_scale[closure], jacobian_child[closure], body_scale[node_body[node]])
        value = value + wp.transpose(coupling)
    response_rhs[index] = value


@wp.kernel
def eliminate_tree_nodes(
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
def solve_tree_roots(
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
def back_substitute_tree_nodes(
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
def solve_block_dense_serial(
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
def assemble_closure_rhs(
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
    response_solution: wp.array[wp.spatial_matrixd],
    schur: wp.array[wp.spatial_matrixd],
):
    index = wp.tid()
    row = index // closure_count
    column = index % closure_count
    local_row = row % closure_count
    value = wp.spatial_matrixd(0.0)
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
    correction = tree_solution[node]
    for closure in range(closure_count):
        correction = (
            correction
            - response_solution[node * closure_count + closure] * closure_multiplier[batch * closure_count + closure]
        )
    rounded = wp.spatial_vector()
    for axis in range(6):
        rounded[axis] = float(wp.float64(body_scale[slot][axis]) * correction[axis])
    body_correction[slot] = rounded
