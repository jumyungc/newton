# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check host-only path incidence ordering independently of device kernels."""

import unittest
from types import SimpleNamespace

import numpy as np

from newton import JointType
from newton._src.solvers.vbd.rigid_vbd_kkt import _build_closed_trees, _build_paths, _build_trees, _make_path


def _host_array(values):
    values = np.asarray(values)
    return SimpleNamespace(numpy=lambda: values)


class TestRigidKKTPathPlanner(unittest.TestCase):
    def test_prescribed_endpoint_path_keeps_rows_without_static_unknowns(self):
        """Two boundary rows sharing a fixed body do not create a dynamic cycle."""
        parent = np.array([0, 1, 2])
        child = np.array([1, 2, 0])
        self.assertIsNone(_make_path({0, 1, 2}, parent, child))
        path = _make_path({0, 1, 2}, parent, child, dynamic_body_mask=np.array([False, True, True]))
        self.assertIsNotNone(path)
        self.assertEqual(path.joints, [0, 1, 2])
        self.assertEqual(path.bodies, [1, 2])
        self.assertEqual(path.body_rows, [[(0, 1), (1, 0)], [(1, 1), (2, 0)]])

    def test_static_branch_retains_all_dynamic_boundary_rows(self):
        """A prescribed branching body separates independent dynamic systems."""
        model = SimpleNamespace(
            joint_parent=_host_array([0, 1, 0, 3, 0, 5, 0]),
            joint_child=_host_array([1, 2, 3, 4, 5, 6, 7]),
            joint_type=_host_array([JointType.ROD] * 7),
            joint_enabled=_host_array([True] * 7),
        )
        inverse_mass = np.array([0, 1, 1, 1, 1, 1, 1, 0], dtype=float)
        paths = _build_paths(model, inverse_mass)
        self.assertEqual(paths, [])
        trees = _build_trees(model, set(), inverse_mass)
        self.assertEqual([tree.joints for tree in trees], [[0, 1], [2, 3], [4, 5]])
        self.assertEqual([tree.bodies for tree in trees], [[1, 2], [3, 4], [5, 6]])
        handled = {frozenset(tree.joints) for tree in trees}
        self.assertEqual(_build_closed_trees(model, handled, inverse_mass), [])
        self.assertEqual(_build_trees(model, set(), np.zeros(8)), [])

    def test_static_boundary_separates_tree_and_cycle_without_dropping_rows(self):
        """A cyclic part and a tree sharing a static anchor are independent."""
        model = SimpleNamespace(
            joint_parent=_host_array([0, 1, 2, 3, 0, 4]),
            joint_child=_host_array([1, 2, 3, 1, 4, 5]),
            joint_type=_host_array([JointType.ROD] * 6),
            joint_enabled=_host_array([True] * 6),
        )
        inverse_mass = np.array([0, 1, 1, 1, 1, 1], dtype=float)
        trees = _build_trees(model, set(), inverse_mass)
        self.assertEqual([tree.joints for tree in trees], [[4, 5]])
        handled = {frozenset(tree.joints) for tree in trees}
        closed = _build_closed_trees(model, handled, inverse_mass)
        self.assertEqual([tree.joints for tree in closed], [[0, 1, 2, 3]])
        self.assertEqual(closed[0].bodies, [1, 2, 3])
        self.assertEqual(closed[0].closure_joints, [3])
        rows = [j for component in [*trees, *closed] for j in component.joints]
        self.assertEqual(sorted(rows), list(range(6)))

    def test_incidence_preserves_body_row_and_side_order(self):
        """Preserve first-encounter body order for shuffled, world-anchored paths."""
        model = SimpleNamespace(
            joint_parent=_host_array([2, -1, 1, 3, 6]),
            joint_child=_host_array([3, 1, 2, 4, 7]),
            joint_type=_host_array([JointType.ROD, JointType.FIXED, JointType.ROD, JointType.ROD, JointType.ROD]),
            joint_enabled=_host_array([True] * 5),
        )
        paths = _build_paths(model, np.ones(8))
        self.assertEqual(len(paths), 2)
        self.assertEqual(paths[0].joints, [1, 2, 0, 3])
        self.assertEqual(paths[0].bodies, [1, 2, 3, 4])
        self.assertEqual(paths[0].body_rows, [[(0, 1), (1, 0)], [(1, 1), (2, 0)], [(2, 1), (3, 0)], [(3, 1)]])
        self.assertEqual(paths[1].joints, [4])
        self.assertEqual(paths[1].bodies, [6, 7])
        self.assertEqual(paths[1].body_rows, [[(0, 0)], [(0, 1)]])

    def test_incidence_rejects_cycles_branches_and_static_components(self):
        """Keep non-path and wholly static components out of the path route."""
        for parents, children, mass in [
            ([0, 1, 2], [1, 2, 0], np.ones(3)),
            ([0, 0, 0], [1, 2, 3], np.ones(4)),
            ([0, 1], [1, 2], np.zeros(3)),
            ([], [], np.ones(1)),
        ]:
            with self.subTest(parents=parents, children=children):
                model = SimpleNamespace(
                    joint_parent=_host_array(parents),
                    joint_child=_host_array(children),
                    joint_type=_host_array([JointType.ROD] * len(parents)),
                    joint_enabled=_host_array([True] * len(parents)),
                )
                self.assertEqual(_build_paths(model, mass), [])


if __name__ == "__main__":
    unittest.main()
