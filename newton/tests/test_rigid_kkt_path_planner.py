# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check host-only path incidence ordering independently of device kernels."""

import unittest
from types import SimpleNamespace

import numpy as np

from newton import JointType
from newton._src.solvers.vbd.rigid_vbd_kkt import _build_paths


def _host_array(values):
    values = np.asarray(values)
    return SimpleNamespace(numpy=lambda: values)


class TestRigidKKTPathPlanner(unittest.TestCase):
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
