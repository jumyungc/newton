# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Independent checks of shared rigid-global recovery and step safeguards."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.vbd import rigid_vbd_kkt as kkt
from newton._src.solvers.xpbd import rigid_xpbd_kkt as xpbd_kkt
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_refined_paired_scatter(test, device):
    """Preserve the rounded base-plus-defect result, batching, and odd tails."""
    rng = np.random.default_rng(240914)
    for nodes, batches, closures in ((1, 1, 0), (7, 3, 1), (8, 2, 6), (63, 2, 24)):
        pairs = (nodes + 1) // 2
        count = pairs * batches
        arrays, hosts = [], []
        for size, dtype in ((count, kkt._SpatialPairVector), (count * closures, kkt._SpatialPairResponse)) * 2:
            array = wp.empty(size, dtype=dtype, device=device)
            host = array.numpy()
            for field in ("v0", "v1"):
                host[field] = rng.standard_normal(host[field].shape).astype(np.float32)
            array.assign(host)
            arrays.append(array)
            hosts.append(host)
        indices = rng.permutation(nodes * batches).astype(np.int32)
        mapping = wp.array(indices, dtype=int, device=device)
        result = wp.full(nodes * batches + 1, 37.0, dtype=wp.spatial_vector, device=device)
        response = wp.full((nodes * batches + 1) * closures, 37.0, dtype=wp.spatial_matrix, device=device)
        expected, expected_response = result.numpy(), response.numpy()
        for batch in range(batches):
            for node in range(nodes):
                source = batch * pairs + node // 2
                target = indices[batch * nodes + node]
                field = "v0" if node % 2 == 0 else "v1"
                expected[target] = hosts[0][field][source] + hosts[2][field][source]
                for closure in range(closures):
                    row = source * closures + closure
                    expected_response[target * closures + closure] = hosts[1][field][row] + hosts[3][field][row]
        wp.launch(
            kkt.scatter_refined_paired_tree_backbone_solution,
            count,
            inputs=[pairs, nodes, closures, mapping, *arrays],
            outputs=[result, response],
            device=device,
        )
        np.testing.assert_array_equal(result.numpy(), expected)
        np.testing.assert_array_equal(response.numpy(), expected_response)


def test_directional_step_guard(test, device):
    """Reject ascent, nonfinite slopes and islands assigned to local fallback."""
    terms = np.asarray([-2, -8, 2, -2, np.nan, 0, -np.inf], dtype=np.float64)
    contact_state = wp.array([0, 1, -1, -2, 0, 0, 0], dtype=int, device=device)
    enabled = wp.empty(7, dtype=bool, device=device)
    slope = wp.array(terms, dtype=wp.float64, device=device)
    scale = wp.ones(7, dtype=float, device=device)
    wp.launch(kkt._begin_directional_search, 7, inputs=[contact_state, slope], outputs=[enabled, scale], device=device)
    np.testing.assert_array_equal(enabled.numpy(), [True, True, False, False, False, False, False])
    np.testing.assert_array_equal(scale.numpy(), [1, 1, 0, 0, 0, 0, 0])


def test_directional_derivative_matches_dense_model(test, device):
    """Use original primal blocks, including body rhs and both joint endpoints."""
    rng = np.random.default_rng(240915)
    rhs = rng.standard_normal((2, 6)).astype(np.float32)
    delta = rng.standard_normal((2, 6)).astype(np.float32)
    jp, jc = rng.standard_normal((2, 1, 6, 6)).astype(np.float32)
    compliance = np.diag(np.geomspace(1.0e-6, 1.0, 6)).astype(np.float32)[None]
    residual = rng.standard_normal((1, 6)).astype(np.float32)

    def array(value, dtype):
        return wp.array(value, dtype=dtype, device=device)

    island = array([0, 0], int)
    enabled = array([True], bool)
    correction = array(delta, wp.spatial_vector)
    merit = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(
        kkt.accumulate_body_directional_derivative,
        2,
        inputs=[island, enabled, array(rhs, wp.spatial_vector), correction],
        outputs=[merit],
        device=device,
    )
    wp.launch(
        kkt.accumulate_joint_directional_derivative,
        1,
        inputs=[
            array([0], int),
            array([0], int),
            array([1], int),
            array([0, 1], int),
            island,
            array(jp, wp.spatial_matrix),
            array(jc, wp.spatial_matrix),
            array(compliance, wp.spatial_matrix),
            array(residual, wp.spatial_vector),
            correction,
        ],
        outputs=[merit],
        device=device,
    )
    d = delta.astype(float)
    jd = jp[0].astype(float) @ d[0] + jc[0].astype(float) @ d[1]
    inverse = 1.0 / np.diag(compliance[0].astype(float))
    linear = -np.sum(rhs.astype(float) * d) + (jd * inverse) @ residual[0].astype(float)
    np.testing.assert_allclose(merit.numpy()[0], linear, rtol=2.0e-14)


def test_quadratic_step_guard(test, device):
    """Accept the exact minimizer, shorten overshoots, reject ascent/nonfinite."""
    terms = np.asarray([[-2, 2], [-8, 32], [2, 2], [-2, 0], [np.nan, 1], [2, 3]], dtype=np.float64)
    enabled = wp.array([True, True, True, True, True, False], dtype=bool, device=device)
    merit = wp.array(terms, dtype=wp.vec2d, device=device)
    scale = wp.empty(6, dtype=float, device=device)
    wp.launch(xpbd_kkt.minimize_quadratic_step, 6, inputs=[enabled, merit], outputs=[scale], device=device)
    np.testing.assert_array_equal(scale.numpy(), [1, 0.25, 0, 0, 0, 1])
    steps = scale.numpy()[:4]
    test.assertTrue(np.all(steps * terms[:4, 0] + 0.5 * steps**2 * terms[:4, 1] <= 0.0))


def test_quadratic_merit_matches_dense_model(test, device):
    """Use original primal blocks, including body rhs and both joint endpoints."""
    rng = np.random.default_rng(240915)
    blocks = rng.standard_normal((2, 6, 6)).astype(np.float32)
    blocks = blocks @ blocks.transpose(0, 2, 1) + np.eye(6, dtype=np.float32)
    rhs = rng.standard_normal((2, 6)).astype(np.float32)
    delta = rng.standard_normal((2, 6)).astype(np.float32)
    jp, jc = rng.standard_normal((2, 1, 6, 6)).astype(np.float32)
    compliance = np.diag(np.geomspace(1.0e-6, 1.0, 6)).astype(np.float32)[None]
    residual = rng.standard_normal((1, 6)).astype(np.float32)

    def array(value, dtype):
        return wp.array(value, dtype=dtype, device=device)

    island = array([0, 0], int)
    enabled = array([True], bool)
    correction = array(delta, wp.spatial_vector)
    merit = wp.zeros(1, dtype=wp.vec2d, device=device)
    wp.launch(
        xpbd_kkt.accumulate_body_quadratic_merit,
        2,
        inputs=[island, enabled, array(blocks, wp.spatial_matrix), array(rhs, wp.spatial_vector), correction],
        outputs=[merit],
        device=device,
    )
    wp.launch(
        xpbd_kkt.accumulate_joint_quadratic_merit,
        1,
        inputs=[
            array([0], int),
            array([0], int),
            array([1], int),
            array([0, 1], int),
            island,
            array(jp, wp.spatial_matrix),
            array(jc, wp.spatial_matrix),
            array(compliance, wp.spatial_matrix),
            array(residual, wp.spatial_vector),
            correction,
        ],
        outputs=[merit],
        device=device,
    )
    d = delta.astype(float)
    jd = jp[0].astype(float) @ d[0] + jc[0].astype(float) @ d[1]
    inverse = 1.0 / np.diag(compliance[0].astype(float))
    linear = -np.sum(rhs.astype(float) * d) + (jd * inverse) @ residual[0].astype(float)
    quadratic = np.einsum("bi,bij,bj->", d, blocks.astype(float), d) + (jd * inverse) @ jd
    np.testing.assert_allclose(merit.numpy()[0], [linear, quadratic], rtol=2.0e-14)


class TestRigidKKTPrimitives(unittest.TestCase):
    pass


for _test in (
    test_quadratic_step_guard,
    test_quadratic_merit_matches_dense_model,
    test_refined_paired_scatter,
    test_directional_step_guard,
    test_directional_derivative_matches_dense_model,
):
    add_function_test(TestRigidKKTPrimitives, _test.__name__, _test, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main(verbosity=2)
