Preserve rigid VBD linear velocity when small displacements round away in float32 poses,
including through joint-mimic projection. For compliant body solves, retain the
final Newton update lost to position rounding when reconstructing velocity;
discard this remainder after subsequent pose projection.

Preserve pair-contact normal forces across uniform mass/material scales.
Clamp pair-contact friction to zero when its Coulomb disk has zero radius,
including for sub-epsilon tangential forces.

Use the same non-attractive spring-damper law for compliant body and pair contacts,
including release damping and a friction bound based on the total normal reaction.
