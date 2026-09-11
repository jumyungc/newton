# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
import newton.solvers
from newton.tests.unittest_utils import add_function_test, get_test_devices


class _ContactFrictionScene:
    """Provide one settled sphere contact with an isolated friction channel."""

    radius = 0.2  # [m]
    dt = 1.0 / 240.0  # [s]

    def __init__(self, device, channel, *, support="free", history=True, matching="sticky"):
        self.channel = channel
        self.support = support
        self.coefficient = 0.2 if channel == "sliding" else 0.01
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        cfg = builder.default_shape_cfg
        cfg.ke = 5.0e5
        cfg.kd = 1000.0
        cfg.mu = self.coefficient if channel == "sliding" else 0.0
        cfg.mu_torsional = self.coefficient if channel == "torsional" else 0.0
        cfg.mu_rolling = self.coefficient if channel == "rolling" else 0.0
        builder.add_ground_plane()
        center = wp.vec3(0.0, 0.0, self.radius - 1.0e-5)
        add_body = builder.add_body if support == "free" else builder.add_link
        self.body = add_body(xform=wp.transform(center, wp.quat_identity()))
        builder.add_shape_sphere(self.body, radius=self.radius)
        self.angular_axis = 2 if channel == "torsional" else 0
        joint = None
        if support != "free":
            config = newton.ModelBuilder.JointDofConfig
            if channel == "sliding":
                linear_axes = [config(axis=newton.Axis.X, target_ke=1.0 if support == "anisotropic" else 0.0)]
                if support == "anisotropic":
                    linear_axes.append(config(axis=newton.Axis.Y))
                linear_axes.append(config(axis=newton.Axis.Z))
                angular_axes = []
                drive_offset = 0
            else:
                linear_axes = [config(axis=newton.Axis.Z)]
                angular_axes = [
                    config(axis=newton.Axis.Z if channel == "torsional" else newton.Axis.X, target_ke=1.0),
                    config(axis=newton.Axis.Y),
                ]
                drive_offset = 1
            joint = builder.add_joint_d6(
                -1,
                self.body,
                linear_axes=linear_axes,
                angular_axes=angular_axes,
                parent_xform=wp.transform(center, wp.quat_identity()),
                child_xform=wp.transform_identity(),
            )
            builder.add_articulation([joint])
        builder.color()
        self.model = builder.finalize(device=device)
        mass = float(self.model.body_mass.numpy()[self.body])
        inertia = float(self.model.body_inertia.numpy()[self.body, self.angular_axis, self.angular_axis])
        self.coordinate_mass = mass if channel == "sliding" else inertia
        self.drive_stiffness = 0.0
        if support == "anisotropic":
            self.drive_stiffness = 100.0 * self.coordinate_mass / (0.9 * self.dt**2)
            stiffnesses = self.model.joint_target_ke.numpy()
            stiffnesses[int(self.model.joint_qd_start.numpy()[joint]) + drive_offset] = self.drive_stiffness
            self.model.joint_target_ke.assign(stiffnesses)
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=32,
            rigid_compliant_alm=True,
            rigid_contact_history=history,
            rigid_joint_linear_ke=1.0e8,
            rigid_joint_angular_ke=1.0e8,
        )
        self.pipeline = newton.CollisionPipeline(self.model, contact_matching=matching)
        self.contacts = self.pipeline.contacts()
        self.state, self.next_state = self.model.state(), self.model.state()
        self.control = self.model.control()
        for _ in range(60):
            self.step(0.0)
        self.origin = self.state.body_q.numpy()[self.body].copy()
        self.normal_load, _, _ = self.reactions()
        self.bound = self.coefficient * self.normal_load

    def step(self, effort):
        """Apply a force [N] or torque [N m] and advance one timestep."""
        forces = np.zeros((self.model.body_count, 6), dtype=np.float32)
        if self.channel == "sliding":
            forces[self.body, 0] = effort
            if self.support != "structural":
                # Apply the wrench at the contact point so a sticking contact
                # can balance it without deflecting the joint's angular spring.
                forces[self.body, 4] = -self.radius * effort
        else:
            forces[self.body, 3 + self.angular_axis] = effort
        self.state.body_f.assign(forces)
        self.pipeline.collide(self.state, self.contacts)
        self.solver.step(self.state, self.next_state, self.control, self.contacts, self.dt)
        self.state, self.next_state = self.next_state, self.state

    def reactions(self):
        """Return normal load [N], selected reaction, and inactive reactions."""
        normal = self.contacts.rigid_contact_normal.numpy()[0]
        force = self.solver.body_body_contact_lambda.numpy()[0]
        torque = self.solver.body_body_contact_lambda_angular.numpy()[0]
        normal_load = max(float(force @ normal), 0.0)
        tangent = force - float(force @ normal) * normal
        torsion = float(torque @ normal)
        rolling = torque - torsion * normal
        if self.channel == "sliding":
            return normal_load, float(np.linalg.norm(tangent)), torque
        if self.channel == "torsional":
            return normal_load, abs(torsion), np.concatenate((tangent, rolling))
        return normal_load, float(np.linalg.norm(rolling)), np.append(tangent, torsion)

    def displacement_and_speed(self):
        """Return accumulated contact slip [m or rad] and speed [m/s or rad/s]."""
        pose = self.state.body_q.numpy()[self.body]
        velocity = self.state.body_qd.numpy()[self.body]
        if self.channel == "sliding":
            angle = 2.0 * np.arctan2(pose[4], pose[6])
            origin_angle = 2.0 * np.arctan2(self.origin[4], self.origin[6])
            # The COM may translate to compensate finite structural joint
            # rotation while the contact still sticks: test material slip.
            slip = float(pose[0] - self.origin[0]) - self.radius * float(angle - origin_angle)
            speed = float(velocity[0]) - self.radius * float(velocity[4])
            return slip, speed
        angle = float(2.0 * np.arctan2(pose[3 + self.angular_axis], pose[6]))
        return angle, float(velocity[3 + self.angular_axis])

    def set_coefficient(self, coefficient):
        """Update both shapes' coefficient, dimensionless for sliding or [m] for angular friction."""
        attribute = "shape_material_mu" if self.channel == "sliding" else f"shape_material_mu_{self.channel}"
        getattr(self.model, attribute).fill_(coefficient)
        self.solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)


def test_vbd_sliding_friction_structural_hold(test, device, iterations):
    """Hold a sub-limit load and its reversal despite stiff transverse joints."""
    scene = _ContactFrictionScene(device, "sliding", support="structural")
    test.assertEqual(int(scene.contacts.rigid_contact_count.numpy()[0]), 1)
    test.assertGreater(scene.normal_load, 0.0)
    scene.solver.iterations = iterations
    tolerance = 1.0e-7 if iterations == 10 else 5.0e-10
    for step in range(60):
        scene.step((0.2 if step < 30 else -0.2) * scene.bound)
        slip, _ = scene.displacement_and_speed()
        load, reaction, inactive = scene.reactions()
        test.assertLess(abs(slip), tolerance, msg=f"step={step}, slip={slip} m")
        test.assertLessEqual(reaction, scene.coefficient * load * 1.00001)
        np.testing.assert_array_equal(inactive, 0.0)
    _, speed = scene.displacement_and_speed()
    test.assertLess(abs(speed), 1.0e-6)


def test_vbd_contact_friction_anisotropic_hold(test, device):
    """Recover a holding load when a disk has stiff and free directions."""
    for channel in ("sliding", "torsional", "rolling"):
        with test.subTest(channel=channel):
            scene = _ContactFrictionScene(device, channel, support="anisotropic")
            scene.solver.iterations = 24
            effort = 0.2 * scene.bound
            # The displacement tolerance permits at most 1% of the applied
            # effort to be spuriously absorbed by the driven-axis spring.
            tolerance = 0.01 * effort / (scene.coordinate_mass / scene.dt**2 + scene.drive_stiffness)
            for sign in (1.0, -1.0):
                scene.step(sign * effort)
                slip, _ = scene.displacement_and_speed()
                load, reaction, inactive = scene.reactions()
                test.assertGreaterEqual(reaction / effort, 0.99)
                test.assertLessEqual(reaction / effort, 1.01)
                test.assertLess(abs(slip), tolerance)
                test.assertLessEqual(reaction, scene.coefficient * load * 1.00001)
                np.testing.assert_array_equal(inactive, 0.0)


def test_vbd_contact_friction_channel_toggle(test, device):
    """Respect each channel's bound and clear stale reactions on material toggles."""
    for channel in ("sliding", "torsional", "rolling"):
        for history in (False, True):
            with test.subTest(channel=channel, history=history):
                scene = _ContactFrictionScene(device, channel, history=history, matching="latest")
                for _ in range(3):
                    scene.step(2.0 * scene.bound)
                    load, reaction, inactive = scene.reactions()
                    test.assertGreater(reaction, 0.9 * scene.coefficient * load)
                    test.assertLessEqual(reaction, 1.00001 * scene.coefficient * load)
                    np.testing.assert_array_equal(inactive, 0.0)
                scene.set_coefficient(0.0)
                scene.step(0.0)
                _, reaction, inactive = scene.reactions()
                test.assertEqual(reaction, 0.0)
                np.testing.assert_array_equal(inactive, 0.0)
                _, unresisted_speed = scene.displacement_and_speed()
                test.assertGreater(abs(unresisted_speed), 1.0e-5)
                scene.set_coefficient(scene.coefficient)
                for _ in range(8):
                    scene.step(0.0)
                    load, reaction, inactive = scene.reactions()
                    test.assertLessEqual(reaction, 1.00001 * scene.coefficient * load)
                    np.testing.assert_array_equal(inactive, 0.0)
                _, resisted_speed = scene.displacement_and_speed()
                test.assertLess(abs(resisted_speed), 0.05 * abs(unresisted_speed))


def test_vbd_contact_friction_ignores_regularization_speed(test, device):
    """The ALM Coulomb solution must not depend on legacy regularization speed."""
    for channel in ("sliding", "torsional", "rolling"):
        with test.subTest(channel=channel):
            observations = []
            for epsilon in (1.0e-6, 1.0e6):
                scene = _ContactFrictionScene(device, channel, support="anisotropic")
                scene.solver.friction_epsilon = epsilon
                scene.solver.iterations = 10
                trajectory = []
                for fraction in (0.2, 2.0, -2.0, 0.0):
                    scene.step(fraction * scene.bound)
                    slip, speed = scene.displacement_and_speed()
                    normal, reaction, _inactive = scene.reactions()
                    trajectory.append((slip, speed, normal, reaction))
                observations.append(trajectory)
            np.testing.assert_array_equal(observations[0], observations[1])


def test_vbd_contact_friction_combined_hold(test, device):
    """Balance a combined point-force and angular load with all channels active."""
    scene = _ContactFrictionScene(device, "sliding")
    scene.model.shape_material_mu_torsional.fill_(0.01)
    scene.model.shape_material_mu_rolling.fill_(0.01)
    scene.solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    force = 0.1 * scene.coefficient * scene.normal_load
    torque = 0.1 * 0.01 * scene.normal_load
    for sign in (1.0, -1.0):
        for _ in range(4):
            # Force at the contact plus two independent angular couples.
            wrench = np.array([[force, 0.0, 0.0, torque, -scene.radius * force, torque]], dtype=np.float32)
            scene.state.body_f.assign(sign * wrench)
            scene.pipeline.collide(scene.state, scene.contacts)
            scene.solver.step(scene.state, scene.next_state, scene.control, scene.contacts, scene.dt)
            scene.state, scene.next_state = scene.next_state, scene.state
            normal = scene.contacts.rigid_contact_normal.numpy()[0]
            reaction = scene.solver.body_body_contact_lambda.numpy()[0]
            angular = scene.solver.body_body_contact_lambda_angular.numpy()[0]
            load = max(float(reaction @ normal), 0.0)
            tangential = reaction - float(reaction @ normal) * normal
            torsional = float(angular @ normal)
            rolling = angular - torsional * normal
            test.assertLessEqual(np.linalg.norm(tangential), 1.00001 * scene.coefficient * load)
            test.assertLessEqual(abs(torsional), 1.00001 * 0.01 * load)
            test.assertLessEqual(np.linalg.norm(rolling), 1.00001 * 0.01 * load)
        test.assertLess(np.max(np.abs(scene.state.body_qd.numpy())), 1.0e-5)


class TestSolverVBDFrictionConsistency(unittest.TestCase):
    pass


devices = get_test_devices(mode="basic")
for iterations in (10, 32):
    add_function_test(
        TestSolverVBDFrictionConsistency,
        f"test_vbd_sliding_friction_structural_hold_{iterations}",
        test_vbd_sliding_friction_structural_hold,
        devices=devices,
        iterations=iterations,
    )
add_function_test(
    TestSolverVBDFrictionConsistency,
    "test_vbd_contact_friction_anisotropic_hold",
    test_vbd_contact_friction_anisotropic_hold,
    devices=devices,
)
add_function_test(
    TestSolverVBDFrictionConsistency,
    "test_vbd_contact_friction_channel_toggle",
    test_vbd_contact_friction_channel_toggle,
    devices=devices,
)
add_function_test(
    TestSolverVBDFrictionConsistency,
    "test_vbd_contact_friction_ignores_regularization_speed",
    test_vbd_contact_friction_ignores_regularization_speed,
    devices=devices,
)
add_function_test(
    TestSolverVBDFrictionConsistency,
    "test_vbd_contact_friction_combined_hold",
    test_vbd_contact_friction_combined_hold,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
