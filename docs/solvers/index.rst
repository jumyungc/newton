.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Solvers
========

Solvers are used to integrate the dynamics of a Newton model.
The typical workflow is to construct a :class:`~newton.Model` and a :class:`~newton.State` object, then use a solver to advance the state forward in time
via the :meth:`~newton.solvers.SolverBase.step` method:

.. mermaid::
  :config: {"theme": "forest", "themeVariables": {"lineColor": "#76b900"}}

  flowchart LR
      subgraph Input["Input Data"]
          M[newton.Model]
          S[newton.State]
          C[newton.Control]
          K[newton.Contacts]
          DT[Time step dt]
      end

      STEP["solver.step()"]

      subgraph Output["Output Data"]
          SO["newton.State (updated)"]
      end

      %% Connections
      M --> STEP
      S --> STEP
      C --> STEP
      K --> STEP
      DT --> STEP
      STEP --> SO

See the :doc:`solver API reference </api/newton_solvers>` for constructor and
method details. For symptom-driven diagnosis and parameter tuning, start with
:ref:`Simulation Tuning`.

Choosing a Solver
-----------------

MuJoCo, Kamino, and VBD currently have dedicated backend guides:

- :doc:`MuJoCo <mujoco>` — generalized-coordinate rigid-body simulation and
  MuJoCo or MJCF workflows.
- :doc:`Kamino <kamino>` — constrained rigid mechanisms with kinematic loops
  and hard frictional contacts; experimental.
- :doc:`VBD <vbd>` — unified simulation of cloth, soft bodies, and rigid
  bodies in a single VBD-based solver; experimental.

The other solver backends are documented through their linked API references
and the comparison tables below.

For a rigid articulated robot, first decide which coordinate representation
fits the application. :class:`~newton.solvers.SolverMuJoCo`,
:class:`~newton.solvers.SolverFeatherstone`, and
:class:`~newton.solvers.SolverSteel` use generalized coordinates, while
:class:`~newton.solvers.SolverXPBD`,
:class:`~newton.solvers.SolverSemiImplicit`, and
:class:`~newton.solvers.SolverKamino` use maximal coordinates. For deformable
bodies, particles, or differentiable simulation, use the feature matrix below
to narrow the choice, then follow the linked API documentation.

.. _Supported Features:

Supported Features
------------------

.. list-table::
   :header-rows: 1
   :widths: auto
   :stub-columns: 0

   * - Solver
     - :abbr:`Integration (Available methods for integrating the dynamics)`
     - Rigid bodies
     - :ref:`Articulations <Articulations>`
     - Particles
     - Cloth
     - Soft bodies
     - Differentiable
     - Deterministic :sup:`3`
   * - :class:`~newton.solvers.SolverFeatherstone`
     - Semi-implicit
     - ✅
     - ✅ generalized coordinates
     - ✅
     - 🟨 no self-collision
     - ✅
     - 🟨 basic :sup:`2`
     - ✅
   * - :class:`~newton.solvers.SolverImplicitMPM`
     - Implicit
     - ❌
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
   * - :class:`~newton.solvers.SolverKamino`
     - Semi-implicit: Euler, Moreau-Jean
     - ✅ maximal coordinates
     - ✅ maximal coordinates
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
   * - :class:`~newton.solvers.SolverMuJoCo`
     - Explicit, Semi-implicit, Implicit-in-velocity
     - ✅ :sup:`1`
     - ✅ generalized coordinates
     - ❌
     - ❌
     - ❌
     - ❌
     - 🟨 CPU only
   * - :class:`~newton.solvers.SolverSemiImplicit`
     - Semi-implicit
     - ✅
     - ✅ maximal coordinates
     - ✅
     - 🟨 no self-collision
     - ✅
     - 🟨 basic :sup:`2`
     - ✅
   * - :class:`~newton.solvers.SolverSteel`
     - Semi-implicit, implicit drives
     - ✅
     - ✅ generalized coordinates
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅
   * - :class:`~newton.solvers.SolverStyle3D`
     - Implicit
     - ❌
     - ❌
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
   * - :class:`~newton.solvers.SolverVBD`
     - Implicit
     - ✅
     - 🟨 :ref:`limited joint support <Joint feature support>`
     - ✅
     - ✅
     - ✅
     - ❌
     - ✅
   * - :class:`~newton.solvers.SolverXPBD`
     - Implicit
     - ✅
     - ✅ maximal coordinates
     - ✅
     - 🟨 no self-collision
     - 🟨 experimental
     - ❌
     - ✅

| :sup:`1` Uses MuJoCo's built-in collision pipeline by default.
  Set ``use_mujoco_contacts=False`` to use Newton's collision pipeline.
| :sup:`2` ``basic`` means Newton includes several examples that use these solvers in diffsim workflows,
  see :ref:`Differentiability` for further details.
| :sup:`3` Run-to-run determinism with ``deterministic=wp.DeterministicMode.RUN_TO_RUN``
  (MuJoCo's CPU backend is deterministic regardless of that argument);
  see :doc:`/concepts/determinism` for limitations and performance trade-offs.

.. experimental::
    :class:`~newton.solvers.SolverSteel`'s public API and behavior may change without prior notice.
    Its matrix-free solve runs on CPU and CUDA; contact torsion, contact compliance, joint friction, the non-default ``friction_mode`` values and sleeping with ``solve_mode="colored"`` require CUDA, and ``solve_mode="colored"`` runs the serial propagation sweep on CPU.

.. experimental::
    :class:`~newton.solvers.SolverKamino`'s public API and behavior may change without prior notice.

.. experimental::
    :class:`~newton.solvers.SolverVBD`'s public API and behavior may change without prior notice.

.. _Contact material support:

Contact Material Support
------------------------

:class:`~newton.ModelBuilder.ShapeConfig` and the matching :class:`~newton.Model`
shape material arrays store solver-neutral contact data. This section documents
which fields are currently used by Newton's built-in solvers. External solvers
may use different subsets or interpret these fields according to their own
formulation.

- ``mu``: :class:`~newton.solvers.SolverSteel` (arithmetic mean of the two
  shapes, scaled by ``contact_friction_scale``), :class:`~newton.solvers.SolverFeatherstone`,
  :class:`~newton.solvers.SolverSemiImplicit`,
  :class:`~newton.solvers.SolverXPBD`, :class:`~newton.solvers.SolverMuJoCo`,
  :class:`~newton.solvers.SolverVBD`, :class:`~newton.solvers.SolverKamino`,
  :class:`~newton.solvers.SolverStyle3D`, and
  :class:`~newton.solvers.SolverImplicitMPM`.
- ``ke`` / ``kd``: :class:`~newton.solvers.SolverFeatherstone`,
  :class:`~newton.solvers.SolverSemiImplicit`,
  :class:`~newton.solvers.SolverMuJoCo`, and
  :class:`~newton.solvers.SolverVBD`.
- ``kf`` / ``ka``: :class:`~newton.solvers.SolverFeatherstone` and
  :class:`~newton.solvers.SolverSemiImplicit`; ``kf`` is also used by
  :class:`~newton.solvers.SolverMuJoCo`
  (see :ref:`mujoco-contact-friction-solreffriction`).
- ``restitution``: :class:`~newton.solvers.SolverSteel` (arithmetic mean of
  the two shapes, for contacts closing faster than ``restitution_velocity_threshold``
  at the start of the step, unless ``enable_restitution=False``),
  :class:`~newton.solvers.SolverXPBD` when ``enable_restitution=True``, and
  :class:`~newton.solvers.SolverKamino`.
- ``mu_torsional`` / ``mu_rolling``: :class:`~newton.solvers.SolverXPBD` and
  :class:`~newton.solvers.SolverMuJoCo`.
- ``mu_static``: stored as :attr:`~newton.Model.shape_material_mu_static` and
  carried into :class:`~newton.solvers.SolverKamino`'s material tables, but no
  built-in solver distinguishes static from dynamic friction in its contact
  model yet; all of them use ``mu``.
- ``kh``: consumed by hydroelastic contact generation for Newton-generated
  contacts used by :class:`~newton.solvers.SolverFeatherstone`,
  :class:`~newton.solvers.SolverSemiImplicit`, and
  :class:`~newton.solvers.SolverMuJoCo` when ``use_mujoco_contacts=False``. See
  :ref:`Hydroelastic Contacts`.

.. _Joint feature support:

Joint Feature Support
---------------------

Not every solver supports every joint type or joint property.
The tables below document which joint features each solver handles.

:class:`~newton.solvers.SolverFeatherstone`, :class:`~newton.solvers.SolverSteel`, and
:class:`~newton.solvers.SolverMuJoCo` operate on :ref:`articulations <Articulations>` (generalized/reduced coordinates).
The maximal-coordinate solvers (:class:`~newton.solvers.SolverSemiImplicit`,
:class:`~newton.solvers.SolverXPBD`, and :class:`~newton.solvers.SolverKamino`)
enforce joints as pairwise body constraints but do not use the articulation kinematic-tree structure.
:class:`~newton.solvers.SolverVBD` supports a subset of joint types through maximal-coordinate
constraints, with unified compliant ALM by default and a deprecated legacy AVBD path.
:class:`~newton.solvers.SolverStyle3D` and :class:`~newton.solvers.SolverImplicitMPM` do not support joints.

**Joint types**

.. list-table::
   :header-rows: 1
   :widths: auto
   :stub-columns: 1

   * - Joint type
     - :class:`~newton.solvers.SolverFeatherstone`
     - :class:`~newton.solvers.SolverSteel`
     - :class:`~newton.solvers.SolverSemiImplicit`
     - :class:`~newton.solvers.SolverXPBD`
     - :class:`~newton.solvers.SolverMuJoCo`
     - :class:`~newton.solvers.SolverVBD`
     - :class:`~newton.solvers.SolverKamino`
   * - PRISMATIC
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
   * - REVOLUTE
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
   * - BALL
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
   * - FIXED
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
   * - FREE
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
   * - DISTANCE
     - 🟨 :sup:`1`
     - 🟨 :sup:`1`
     - 🟨 :sup:`1`
     - |yes|
     - |no|
     - |no|
     - |no|
   * - D6
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |no|
   * - ROD
     - |no|
     - |no|
     - |no|
     - |no|
     - |no|
     - |yes|
     - |no|

| :sup:`1` DISTANCE joints are treated as FREE (no distance constraint enforcement).

**Joint properties**

.. list-table::
   :header-rows: 1
   :widths: auto
   :stub-columns: 1

   * - Property
     - :class:`~newton.solvers.SolverFeatherstone`
     - :class:`~newton.solvers.SolverSteel`
     - :class:`~newton.solvers.SolverSemiImplicit`
     - :class:`~newton.solvers.SolverXPBD`
     - :class:`~newton.solvers.SolverMuJoCo`
     - :class:`~newton.solvers.SolverVBD`
     - :class:`~newton.solvers.SolverKamino`
   * - :attr:`~newton.Model.joint_enabled`
     - |no|
     - |no| :sup:`8`
     - |yes|
     - |yes|
     - |no|
     - |yes|
     - |no|
   * - :attr:`~newton.Model.joint_armature`
     - |yes|
     - |yes|
     - |no|
     - |no|
     - |yes|
     - |no|
     - |yes|
   * - :attr:`~newton.Model.joint_stiffness` / :attr:`~newton.Model.joint_rest_q`
     - REVOLUTE, PRISMATIC, D6
     - REVOLUTE, PRISMATIC, D6
     - REVOLUTE, PRISMATIC, D6
     - |no|
     - Scalar joints and isotropic BALL springs
     - ROD ``joint_rest_q`` only
     - |no|
   * - :attr:`~newton.Model.joint_friction`
     - |no|
     - |yes| :sup:`9`
     - |no|
     - |no|
     - |yes|
     - |no|
     - |yes|
   * - :attr:`~newton.Model.joint_limit_lower` / :attr:`~newton.Model.joint_limit_upper`
     - |yes|
     - |yes| :sup:`10`
     - |yes| :sup:`2`
     - |yes|
     - |yes|
     - |yes|
     - |yes|
   * - :attr:`~newton.Model.joint_limit_ke` / :attr:`~newton.Model.joint_limit_kd`
     - |yes|
     - |no|
     - |yes| :sup:`2`
     - |no|
     - |yes|
     - |yes| :sup:`7`
     - |no|
   * - :attr:`~newton.Model.joint_effort_limit`
     - |no|
     - |yes| :sup:`11`
     - |no|
     - |no|
     - |yes|
     - |no|
     - |yes|
   * - :attr:`~newton.Model.joint_velocity_limit`
     - |no|
     - |yes| :sup:`12`
     - |no|
     - |no|
     - |no|
     - |no|
     - |no|

| :sup:`2` Not enforced for BALL joints in SemiImplicit.

SemiImplicit angular springs use principal coordinates in [-pi, pi). Crossing
that boundary produces a force jump of ``2*pi*stiffness`` and can destabilize
motion near it. Use Featherstone or MuJoCo for boundary-crossing or multi-turn
motion; see :ref:`passive-joint-springs` for details.

**Actuation and control**

.. list-table::
   :header-rows: 1
   :widths: auto
   :stub-columns: 1

   * - Feature
     - :class:`~newton.solvers.SolverFeatherstone`
     - :class:`~newton.solvers.SolverSteel`
     - :class:`~newton.solvers.SolverSemiImplicit`
     - :class:`~newton.solvers.SolverXPBD`
     - :class:`~newton.solvers.SolverMuJoCo`
     - :class:`~newton.solvers.SolverVBD`
     - :class:`~newton.solvers.SolverKamino`
   * - :attr:`~newton.Model.joint_target_ke` / :attr:`~newton.Model.joint_target_kd`
     - |yes|
     - |yes|
     - |yes| :sup:`2`
     - |yes|
     - |yes|
     - |yes| :sup:`7`
     - |yes|
   * - :attr:`~newton.Model.joint_target_mode`
     - |no|
     - |no|
     - |no|
     - |no|
     - |yes|
     - |no|
     - |yes|
   * - :attr:`~newton.Control.joint_f` (feedforward forces)
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|

**Constraints**

.. list-table::
   :header-rows: 1
   :widths: auto
   :stub-columns: 1

   * - Feature
     - :class:`~newton.solvers.SolverFeatherstone`
     - :class:`~newton.solvers.SolverSteel`
     - :class:`~newton.solvers.SolverSemiImplicit`
     - :class:`~newton.solvers.SolverXPBD`
     - :class:`~newton.solvers.SolverMuJoCo`
     - :class:`~newton.solvers.SolverVBD`
     - :class:`~newton.solvers.SolverKamino`
   * - Equality constraints (CONNECT, WELD, JOINT)
     - |no|
     - |no|
     - |no|
     - |no|
     - |yes|
     - |no|
     - |no|
   * - Mimic joints
     - |yes| :sup:`3`
     - |yes| :sup:`13`
     - |yes| :sup:`4`
     - |yes| :sup:`5`
     - |yes| :sup:`6`
     - |yes| :sup:`5`
     - |no|
   * - Particle-body attachments
     - |no|
     - |no|
     - |no|
     - |no|
     - |no|
     - |yes| :sup:`14`
     - |no|

| :sup:`3` Featherstone eliminates follower degrees of freedom from its reduced dynamics and transfers follower forces and inertia to the reference joint.
| :sup:`4` SemiImplicit enforces joint-owned mimic relationships with penalty springs configured by ``joint_mimic_ke`` and ``joint_mimic_kd``.
| :sup:`5` XPBD and VBD enforce joint-owned mimic relationships through coupled maximal-coordinate corrections. Both apply one mimic correction per solver iteration.
| :sup:`6` MuJoCo lowers each joint-owned relationship to joint equality constraints. Multi-axis D6 relationships produce one equality constraint per axis.
| :sup:`7` VBD interprets ``joint_target_kd`` and ``joint_limit_kd`` as absolute damping coefficients in physical units.
| :sup:`8` Steel rejects models with disabled articulation joints instead of ignoring them; a disabled loop-closing joint starts as a released closure.
| :sup:`9` Steel enforces dry joint friction of PRISMATIC, REVOLUTE, BALL, and D6 DOFs as one bounded constraint row per DOF when constructed with ``enable_joint_friction=True`` on CUDA (off by default, experimental); ``steel:joint_friction_load_coefficient`` adds the load-proportional joint friction of PhysX articulations before PhysX 5.6.
| :sup:`10` Steel enforces joint limits as hard unilateral constraint rows when constructed with ``enable_joint_limits=True`` (off by default).
| :sup:`11` Steel clamps the explicit joint drive force to the effort limit. With the default ``drive_mode="augmented"`` the implicit stiffness and damping response is unbounded, so under a large external load the drive reaction can exceed the limit; the PGS drive rows of ``drive_mode="rows"`` bound the complete reaction.
| :sup:`12` Steel enforces velocity limits of PRISMATIC, REVOLUTE, and D6 DOFs when constructed with ``enable_joint_velocity_limits=True``. With ``drive_mode="rows"``, driven DOFs are clamped at the end of every solver iteration instead of using velocity-limit rows (``fuse_joint_velocity_limits``).
| :sup:`13` Steel enforces each mimic relationship within one articulation as one bilateral constraint row per follower coordinate; mimics of BALL, FREE, and DISTANCE joints and mimics across articulations are rejected. It also enforces loop-closing BALL joints as point constraints, loop-closing REVOLUTE joints as point constraints plus two hinge-axis constraints, and loop-closing FIXED joints as point constraints plus three rotation constraints. The propagation contact response (``articulated_contact_response="propagation"``) rejects loop-closing joints and solves mimic rows iteratively.
| :sup:`14` See :ref:`Particle-body attachments` for authoring and semantics. Attachments spanning two
  solvers are coupled by :class:`~newton.solvers.experimental.coupled.SolverCoupledADMM` instead.

.. _Particle-body attachments:

Particle-Body Attachments
-------------------------

A particle-body attachment ties one cloth or solid particle to a point in a
rigid body's local frame. Author attachments with
:meth:`newton.ModelBuilder.add_attachment_particle_body`. They are stored on the
:class:`~newton.Model` and do not require a coupled solver.

:class:`~newton.solvers.SolverVBD` applies an attachment whenever it integrates
both endpoints. It solves every attachment as a compliant augmented-Lagrangian
(ALM) row, with the same formulation as the VBD joint rows: a reaction
multiplier carries the attachment force, and a numerical penalty chosen from the
endpoint masses and the particle's surrounding material helps the iterations
converge. The penalty is chosen like the joint limit rows, so a finite
attachment reaches its spring over several iterations. The penalty affects convergence only, not physical compliance.
A finite ``stiffness`` [N/m] converges to a spring with ``damping`` [N·s/m], so a
loaded attachment keeps a physical offset. Set ``stiffness=math.inf`` to request
a hard point constraint; damping is ignored for hard attachments. Multipliers are
retained across steps, scaled by ``rigid_avbd_gamma``. A finite iteration budget
can leave constraint error, especially for heavy free bodies at large time steps;
more iterations or smaller steps reduce it. Incompatible hard attachments cannot
all be satisfied. Separated initial anchors can cause large correction impulses.
The constraint is translational, similar to the positional part of a ball joint;
a single particle has no orientation, so there is no angular counterpart. Forces
are equal and opposite, including the torque about the body's center of mass, and
the path is independent of contact, so it stays active without penetration.

Both endpoints must be integrated by the same solver, so
:class:`~newton.solvers.SolverVBD` rejects attachments when it is constructed
with ``integrate_with_external_rigid_solver=True``.

When the endpoints of a finite-stiffness attachment belong to two different solvers in a coupled simulation,
:class:`~newton.solvers.experimental.coupled.SolverCoupledADMM` turns the same
model rows into cross-solver interface constraints; see :ref:`ADMM coupling`.
The two cases are mutually exclusive, so no attachment is applied twice, and a
row whose body or particle is owned by no solver applies no force at all. The
coupler also warns when both endpoints belong to one entry whose solver does not
support attachments. Hard attachments require both endpoints in one VBD entry;
the coupler raises an error for other ownership arrangements.

VBD Cable Relaxation
--------------------

.. experimental::

   The ``visco_*`` custom attributes and their behavior may change without prior notice.

:class:`~newton.solvers.SolverVBD` can add a standard-linear-solid (Zener) branch to rod
bending and rod stretch, so a held cable relaxes its force and a loaded cable creeps beyond
its instantaneous response. For a row of authored stiffness :math:`k`, a Maxwell element of
stiffness :math:`r k` and relaxation time :math:`\tau` acts in parallel; the response to a
held strain decays as

.. math::

   k(t) = k \left(1 + r\, e^{-t/\tau}\right),

so the authored stiffness is the equilibrium (long-time) stiffness and :math:`k (1 + r)` is the
instantaneous stiffness. :math:`\tau` is the relaxation time under held deformation; under a
held force a pure SLS material (no other damping) creeps with time constant :math:`(1 + r)\tau`. Calibration must state
which stiffness a measurement represents. The dimensionless ratio :math:`r` follows the authored
stiffness when the segment length changes. Register the VBD custom attributes before building,
then author positive pairs before constructing the solver.

For one material across all rods, fill the Model arrays as below. Only the bend pair is
needed for bending relaxation; the stretch pair is optional. This uses the same custom-attribute
registration as VBD's Dahl model and leaves the normal State and stepping workflow unchanged.

.. code-block:: python

   builder = newton.ModelBuilder()
   newton.solvers.SolverVBD.register_custom_attributes(builder)
   builder.add_rod(rod=rod, stretch_stiffness=stretch_k, bend_stiffness=bend_k)
   builder.color()
   model = builder.finalize()
   model.vbd.visco_bend_ratio.fill_(9.9)  # bending (twist is not relaxed)
   model.vbd.visco_bend_tau.fill_(3.5)
   model.vbd.visco_stretch_ratio.fill_(2.0)  # axial stretch
   model.vbd.visco_stretch_tau.fill_(0.5)
   solver = newton.solvers.SolverVBD(model)

For different materials on different cables, author the joint indices returned by
:meth:`~newton.ModelBuilder.add_rod` before finalizing the builder instead. Update the
existing override map so assigning a second cable preserves the first cable's values:

.. code-block:: python

   bodies, joints = builder.add_rod(rod=rod, stretch_stiffness=stretch_k, bend_stiffness=bend_k)
   for name, value in {"visco_bend_ratio": 3.0, "visco_bend_tau": 0.5}.items():
       builder.custom_attributes[f"vbd:{name}"].values.update(dict.fromkeys(joints, value))

Repeat this block for each cable, then call ``builder.color()``, ``builder.finalize()``,
and construct the solver. Unspecified joints retain their zero defaults. Model-wide
``fill_()`` is appropriate only when every rod should use that value. For manually built
rod joints, :meth:`~newton.ModelBuilder.add_joint_rod` also accepts the existing
``custom_attributes={"vbd:visco_bend_ratio": 3.0, "vbd:visco_bend_tau": 0.5}`` argument.

In USD, author ``newton:vbd:visco_bend_ratio`` and the other attributes on the cable curve;
they apply to every rod joint generated from it. After construction, edited values take
effect through ``solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)``.

With no SLS branch, an elastic row with ordinary viscous damping has a Kelvin--Voigt
response. A pure three-parameter SLS instead has an equilibrium spring in parallel
with a spring--dashpot series branch: set the existing damping of the enabled mode
(``bend_damping`` or ``stretch_damping``) to zero to obtain this response. Retaining that
damping adds a separate parallel dashpot; it is not the SLS branch's internal viscosity.
The latter is :math:`\eta_1 = r k \tau` in the row's deformation coordinates.
All four parameters must be finite and non-negative. Enabling SLS on a solver constructed
without any active pair requires a new solver and recapturing any CUDA graphs that use it;
edits to an already-enabled solver preserve
buffer addresses for CUDA graph replay. SLS requires VBD to integrate the rod bodies;
``integrate_with_external_rigid_solver=True`` is not supported for active SLS.
An SLS-enabled VBD solver also cannot be an entry of ``SolverCoupled``: the coupler
does not synchronize this State-owned material history between global and entry states.
Construction rejects that combination; use :class:`~newton.solvers.SolverVBD` directly.

Each step uses backward Euler: with relaxation history :math:`u_v` from the input state, the
branch adds :math:`g k (u - u_v)` with :math:`g = r\tau / (\tau + \Delta t)` to the row as a
frozen stiffness and offset, outside the compliant-ALM multiplier, and the output state
receives :math:`u_v \leftarrow u_v + \tfrac{\Delta t}{\tau + \Delta t}(u - u_v)` once per step.
Computing this small increment directly preserves relaxation when :math:`\Delta t/\tau`
is smaller than float32 precision near one; forming ``1 - tau/(tau + dt)`` can round it to zero.
While a parameter pair is inactive, its history tracks the current strain; re-enabling it
after an inactive step starts that branch relaxed, independently of other active channels.
This differs from a newly created state's virgin-at-rest history. Editing positive parameters
preserves existing history and can change stored energy; such an edit is not passive material evolution.
The history ``state.vbd.visco_history`` (bend x, bend y, axial extension) lives on the
:class:`~newton.State`; zero means relaxed in the rest configuration, and
:meth:`~newton.solvers.SolverVBD.reset` restores it for the selected worlds. Legacy AVBD
hard rows are not affected, and models without positive pairs run exactly as before.
An initially deformed state with zero history is a newly strained material, not a cable
already relaxed in that deformed pose. For a cable that has been held in its initial pose,
for example a measured shape before release, call
:meth:`~newton.solvers.SolverVBD.relax_rod_history` after authoring that pose and before the
first step. To restart a trajectory, copy its full State,
including the material history, instead of only body poses and velocities.
The bending history uses dimensionless geometric bend coordinates in the parent material frame;
axial history is an anchor extension [m], not engineering strain. Ratios are dimensionless and
relaxation times are in seconds. A ratio parameterization cannot represent a pure Maxwell material
with zero equilibrium stiffness. Existing viscous damping and Dahl hysteresis remain separate,
additive mechanisms; they are not replaced by SLS. Dahl's ``dahl_tau`` is a deformation-memory
scale, not a time. History is measured from the rest configuration and keeps its values when the
rest shape is edited; such an edit is not a stress-preserving remapping.

SLS is linear viscoelasticity in these rod coordinates, combined with the existing nonlinear
rod geometry. It does not add plastic flow, arbitrary nonlinear stiffness laws, twist/shear
relaxation, or a general material-model registry. USD support uses registered Newton custom
attributes on curves and joint prims, with ratios dimensionless and times in seconds;
the importer's existing SI-stage requirements still apply.
Rod bending and axial stretch share the same constitutive coefficients and history update.
Routed-tendon SLS is outside this feature's scope.

There is no SLS preparation pass: trial forces read the input State, and one final kernel
commits the history once per step. With no active pair at construction, the solve excludes
SLS array reads and the history kernel is omitted. Registering the attributes still allocates
their standard Model/State storage, even when their values are zero. A solver initially
constructed with SLS retains its buffers and history update when all pairs are later disabled,
so subsequent re-enabling has well-defined history.

The transient branch increases stiffness and can require more iterations or smaller timesteps.
In compliant-ALM mode, contact conditioning uses inertial support and a structural summary of
the authored equilibrium stretch/shear stiffness; the SLS transient stiffness is not included.
A stiff contact loaded along a rod with stretch relaxation can therefore require more iterations.
Bending relaxation does not directly change this structural summary.
Check iteration convergence at a fixed timestep separately from timestep convergence; discrete
convergence alone does not establish time accuracy. For pure SLS at a constant timestep small
relative to the material time constants, backward Euler makes held-deformation relaxation and
quasistatic creep time constants approximately :math:`\Delta t / 2` longer than :math:`\tau`
and :math:`(1 + r)\tau`, respectively.

The ``cable_viscoelastic_release`` example holds a cable tip, then releases it:

.. code-block:: console

   python -m newton.examples cable_viscoelastic_release

.. _Differentiability:

Differentiability
-----------------

Differentiable simulation in Newton typically runs a forward rollout inside
``wp.Tape()``, computes a scalar loss from the simulated state, and then calls
``tape.backward(loss)`` to populate gradients on differentiable state,
control, or model arrays. In practice, this starts by calling
:meth:`~newton.ModelBuilder.finalize` with ``requires_grad=True``.

.. testcode::

    import warp as wp
    import newton

    @wp.kernel
    def loss_kernel(particle_q: wp.array[wp.vec3], target: wp.vec3, loss: wp.array[float]):
        delta = particle_q[0] - target
        loss[0] = wp.dot(delta, delta)

    builder = newton.ModelBuilder()
    builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.0), vel=wp.vec3(1.0, 0.0, 0.0), mass=1.0)

    model = builder.finalize(requires_grad=True)
    solver = newton.solvers.SolverSemiImplicit(model)

    state_in = model.state(requires_grad=True)
    state_out = model.state(requires_grad=True)
    control = model.control()
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    target = wp.vec3(0.25, 0.0, 0.0)

    tape = wp.Tape()
    with tape:
        state_in.clear_forces()
        solver.step(state_in, state_out, control, None, 1.0 / 60.0)
        wp.launch(
            loss_kernel,
            dim=1,
            inputs=[state_out.particle_q, target],
            outputs=[loss],
        )

    tape.backward(loss)
    initial_velocity_grad = state_in.particle_qd.grad.numpy()
    assert float(initial_velocity_grad[0, 0]) < 0.0

See the `DiffSim examples on GitHub`_ for the current reference workflows.

.. _DiffSim examples on GitHub: https://github.com/newton-physics/newton/tree/main/newton/examples/diffsim

.. |yes| unicode:: U+2705
.. |no| unicode:: U+274C
