# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Independent checks of shared rigid-global recovery and step safeguards."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd import rigid_vbd_kkt as kkt
from newton._src.solvers.xpbd import rigid_xpbd_kkt as xpbd_kkt
from newton.tests.test_rigid_vbd_kkt import _linearize_bucket
from newton.tests.test_tendon_vbd import build_single_span_tendon
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
    initial_slope = wp.empty(7, dtype=wp.float64, device=device)
    pending = wp.zeros(1, dtype=int, device=device)
    wp.launch(
        kkt._begin_directional_search,
        7,
        inputs=[contact_state, slope],
        outputs=[enabled, scale, initial_slope, pending],
        device=device,
    )
    np.testing.assert_array_equal(enabled.numpy(), [True, True, False, False, False, False, False])
    np.testing.assert_array_equal(scale.numpy(), [1, 1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(initial_slope.numpy(), terms)
    np.testing.assert_array_equal(pending.numpy(), [1])


@wp.kernel
def _accumulate_joint_slope_control(
    parent: wp.array[int],
    child: wp.array[int],
    slots: wp.array[int],
    islands: wp.array[int],
    jp: wp.array[wp.spatial_matrix],
    jc: wp.array[wp.spatial_matrix],
    compliance: wp.array[wp.spatial_matrix],
    residual: wp.array[wp.spatial_vector],
    force: wp.array[wp.spatial_vector],
    diagonal_rows: bool,
    correction: wp.array[wp.spatial_vector],
    merit: wp.array[wp.float64],
):
    row = wp.tid()
    kkt._accumulate_joint_slope(
        parent[row],
        child[row],
        slots,
        islands,
        jp[row],
        jc[row],
        compliance[row],
        residual[row],
        force[row],
        diagonal_rows,
        correction,
        merit,
    )


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
    pose = array(np.tile([0, 0, 0, 0, 0, 0, 1], (2, 1)), wp.transform)
    # Coincident inertial/current poses isolate the supplied body force. The
    # current slope kernel evaluates that force from body data, not a saved rhs.
    wp.launch(
        kkt.accumulate_body_directional_derivative,
        2,
        inputs=[
            array([0, 1], int),
            island,
            enabled,
            pose,
            pose,
            array([1, 1], float),
            array([1, 1], float),
            array(np.tile(np.eye(3), (2, 1, 1)), wp.mat33),
            array(np.zeros((2, 3)), wp.vec3),
            array(rhs[:, :3], wp.vec3),
            array(rhs[:, 3:], wp.vec3),
            0.5,
            correction,
        ],
        outputs=[merit],
        device=device,
    )
    d = delta.astype(float)
    jd = jp[0].astype(float) @ d[0] + jc[0].astype(float) @ d[1]
    body_slope = -np.sum(rhs.astype(float) * d)
    np.testing.assert_allclose(merit.numpy()[0], body_slope, rtol=2.0e-14)
    for diagonal_rows in (True, False):
        with test.subTest(diagonal_rows=diagonal_rows):
            matrix = compliance.copy()
            if not diagonal_rows:
                rotation, _ = np.linalg.qr(rng.standard_normal((6, 6)))
                matrix[0] = rotation @ np.diag(np.geomspace(0.01, 1.0, 6)) @ rotation.T
            force = np.linalg.solve(matrix[0].astype(float), residual[0]).astype(np.float32)[None]
            merit.zero_()
            wp.launch(
                _accumulate_joint_slope_control,
                1,
                inputs=[
                    array([0], int),
                    array([1], int),
                    array([0, 1], int),
                    island,
                    array(jp, wp.spatial_matrix),
                    array(jc, wp.spatial_matrix),
                    array(matrix, wp.spatial_matrix),
                    array(residual, wp.spatial_vector),
                    array(force, wp.spatial_vector),
                    diagonal_rows,
                    correction,
                ],
                outputs=[merit],
                device=device,
            )
            if diagonal_rows:
                inverse = 1.0 / np.diag(matrix[0].astype(float))
                expected = (jd * inverse) @ residual[0].astype(float)
            else:
                # General rows consume the actual force, not diag(C)^-1*c.
                expected = jd @ force[0].astype(float)
            np.testing.assert_allclose(merit.numpy()[0], expected, rtol=2.0e-14)


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


def test_global_d6_preserves_authored_equilibrium(test, device):
    """Keep multi-axis D6 drive equilibria consistent with forward kinematics."""
    for axis_count in (2, 3):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, -1.0)), 0.3)
        axes = [wp.quat_rotate(rotation, axis) for axis in (wp.vec3(1, 0, 0), wp.vec3(0, 1, 0), wp.vec3(0, 0, 1))]
        targets = (0.45, -0.35, 0.2)[:axis_count]
        config = newton.ModelBuilder.JointDofConfig
        joint = builder.add_joint_d6(
            -1,
            body,
            angular_axes=[
                config(axis=axis, target_pos=target, target_ke=1.0e4, target_kd=0.0)
                for axis, target in zip(axes[:axis_count], targets, strict=True)
            ],
        )
        builder.add_articulation([joint])
        builder.color()
        model = builder.finalize(device=device)
        coordinates = wp.array(targets, dtype=float, device=device)
        for solver_type in (newton.solvers.SolverVBD, newton.solvers.SolverXPBD):
            for global_count in (0, 1):
                with test.subTest(axes=axis_count, solver=solver_type.__name__, global_count=global_count):
                    options = {"iterations": 2, "rigid_joint_global_iterations": global_count}
                    if solver_type is newton.solvers.SolverVBD:
                        options["rigid_compliant_alm"] = True
                    solver = solver_type(model, **options)
                    state_in, state_out = model.state(), model.state()
                    newton.eval_fk(model, coordinates, model.joint_qd, state_in)
                    initial = state_in.body_q.numpy()
                    solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)
                    np.testing.assert_allclose(state_out.body_q.numpy(), initial, rtol=0.0, atol=2.0e-5)
                    if solver_type is newton.solvers.SolverVBD and global_count:
                        solver.body_q_prev.assign(state_in.body_q)
                        bucket = solver._structural_graph_kkt.buckets[0]
                        _linearize_bucket(model, solver, bucket, state_in.body_q, 1.0 / 60.0)
                        np.testing.assert_allclose(bucket.residual.numpy(), 0.0, rtol=0.0, atol=2.0e-5)


def test_global_rejects_unrepresented_tendons(test, device):
    """Reject G1 when routed-tendon forces are absent from its objective."""
    with wp.ScopedDevice(device):
        model, _ = build_single_span_tendon()
        for solver_type in (newton.solvers.SolverVBD, newton.solvers.SolverXPBD):
            options = {"iterations": 2}
            if solver_type is newton.solvers.SolverVBD:
                options["rigid_compliant_alm"] = True
            # The same physical model remains supported through upstream's local path.
            solver_type(model, rigid_joint_global_iterations=0, **options)
            with test.assertRaisesRegex(ValueError, "models containing tendons"):
                solver_type(model, rigid_joint_global_iterations=1, **options)


class TestRigidKKTPrimitives(unittest.TestCase):
    pass


for _test in (
    test_quadratic_step_guard,
    test_quadratic_merit_matches_dense_model,
    test_refined_paired_scatter,
    test_directional_step_guard,
    test_directional_derivative_matches_dense_model,
    test_global_d6_preserves_authored_equilibrium,
    test_global_rejects_unrepresented_tendons,
):
    add_function_test(TestRigidKKTPrimitives, _test.__name__, _test, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main(verbosity=2)
