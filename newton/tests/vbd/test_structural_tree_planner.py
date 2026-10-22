# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check host-only KKT tree schedules independently of numerical device kernels."""

import unittest

import numpy as np

from newton._src.solvers.vbd.rigid_vbd_kkt import _make_tree


class TestStructuralTreePlanner(unittest.TestCase):
    def test_root_and_leaf_schedule(self):
        """Preserve the central dynamic root, sorted leaf rounds and coupling orientation."""
        rng = np.random.default_rng(20261022)
        for count in (1, 2, 7, 32):
            for topology in ("path", "star", "random"):
                for static_root in (False, True):
                    with self.subTest(count=count, topology=topology, static_root=static_root):
                        parent = np.array(
                            [-1]
                            + [
                                i - 1 if topology == "path" else 0 if topology == "star" else rng.integers(i)
                                for i in range(1, count)
                            ]
                        )
                        child = np.arange(count)
                        inverse_mass = np.ones(count)
                        inverse_mass[0] = 0.0 if static_root else 1.0
                        # Add distinct world leaves, then scramble row/body numbering.
                        parent = np.append(parent, [-1, -1])
                        child = np.append(child, [count - 1, count // 2])
                        permutation = rng.permutation(count)
                        parent = np.array([permutation[i] if i >= 0 else -1 for i in parent])
                        child = permutation[child]
                        inverse_mass[permutation] = inverse_mass.copy()
                        joints = rng.permutation(len(parent)).tolist()
                        # A row attached only to static endpoints is outside this tree's domain.
                        joints = [
                            j for j in joints if any(b >= 0 and inverse_mass[b] > 0 for b in (parent[j], child[j]))
                        ]
                        tree = _make_tree(joints, parent, child, inverse_mass)
                        if not joints:
                            self.assertIsNone(tree)
                            continue
                        bodies = sorted(
                            {b for j in joints for b in (parent[j], child[j]) if b >= 0 and inverse_mass[b] > 0}
                        )
                        body_node = {body: i for i, body in enumerate(bodies)}
                        adjacency = [set() for _ in range(len(bodies) + len(joints))]
                        for row, joint in enumerate(joints):
                            node = len(bodies) + row
                            for body in (parent[joint], child[joint]):
                                if body in body_node:
                                    neighbor = body_node[body]
                                    adjacency[node].add(neighbor)
                                    adjacency[neighbor].add(node)
                        reached, queue = {0}, [0]
                        for node in queue:
                            for neighbor in adjacency[node] - reached:
                                reached.add(neighbor)
                                queue.append(neighbor)
                        if len(reached) != len(adjacency):
                            # Removing a static branch hub leaves independent dynamic islands.
                            self.assertIsNone(tree)
                            continue
                        self.assertIsNotNone(tree)
                        self.assertEqual(tree.bodies, bodies)
                        self.assertEqual(tree.joints, joints)
                        self.assertEqual(tree.node_body, bodies + [-1] * len(joints))
                        self.assertEqual(tree.node_row, [-1] * len(bodies) + list(range(len(joints))))
                        eccentricity = []
                        for start in range(len(bodies)):
                            distance, queue = {start: 0}, [start]
                            for node in queue:
                                for neighbor in adjacency[node]:
                                    if neighbor not in distance:
                                        distance[neighbor] = distance[node] + 1
                                        queue.append(neighbor)
                            self.assertEqual(len(distance), len(adjacency))
                            eccentricity.append(max(distance.values()))
                        root = min(range(len(bodies)), key=lambda n: (eccentricity[n], bodies[n]))
                        self.assertEqual(tree.root, root)
                        self.assertEqual(tree.parent_node[root], -1)
                        self.assertEqual(tree.coupling_side[root], -1)
                        active = set(range(len(adjacency)))
                        for actual in tree.elimination_levels:
                            leaves = sorted(n for n in active if n != root and len(adjacency[n] & active) == 1)
                            self.assertEqual(actual, leaves)
                            for node in leaves:
                                neighbor = next(iter(adjacency[node] & active))
                                self.assertEqual(tree.parent_node[node], neighbor)
                                row_node = node if node >= len(bodies) else neighbor
                                body_index = neighbor if node >= len(bodies) else node
                                joint = joints[row_node - len(bodies)]
                                side = 0 if parent[joint] == bodies[body_index] else 1
                                self.assertEqual(tree.coupling_side[node], side)
                            active.difference_update(leaves)
                        self.assertEqual(active, {root})

    def test_reject_invalid_graphs(self):
        """Reject cycles, disconnected graphs, self/duplicate edges and no dynamic bodies."""
        cases = (
            ([-1, 0, 1, 2], [0, 1, 2, 0], [1, 1, 1]),
            ([-1, -1], [0, 1], [1, 1]),
            ([-1, 0], [0, 0], [1]),
            ([0, 0], [1, 1], [1, 1]),
            ([-1], [0], [0]),
            ([], [], []),
        )
        for parent, child, inverse_mass in cases:
            with self.subTest(parent=parent, child=child):
                self.assertIsNone(
                    _make_tree(list(range(len(parent))), np.array(parent), np.array(child), np.array(inverse_mass))
                )


if __name__ == "__main__":
    unittest.main()
