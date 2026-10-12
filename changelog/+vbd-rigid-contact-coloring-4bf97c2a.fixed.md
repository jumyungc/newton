Preserve rigid VBD linear velocity when small displacements round away in float32 poses,
including through joint-mimic projection and free-body pair-contact impulses.
Preserve distinct normal and tangential moment arms in pair-contact warm starts.
Evaluate pair-contact damping from step-start poses, independently of collision-frame refreshes.
Clamp pair-coloring reads to allocated contact capacity.
Keep legacy penalty-only contacts, and legacy hard contacts with in-iteration collision detection,
on a complete per-body schedule when pair state is unavailable.
Preserve requested run-to-run determinism for pair-contact force gathering.
