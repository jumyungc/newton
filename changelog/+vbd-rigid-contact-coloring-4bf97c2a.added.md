Add deterministic rigid VBD contact scheduling compatible with CUDA graph capture.
Expose `rigid_contact_coloring` (`"auto"` by default, `"model"`, or `"refined"`) and
`rigid_body_contact_color_budget` (default 32) on `SolverVBD` for per-body contact scheduling.
Joint separation is always preserved, and these options do not select the contact formulation.
Collision-isolated worlds without structural joints, tendons, or particle attachments solve contacts
per body-pair manifold with exact edge colors when compliant ALM has numeric contact history;
other worlds and deprecated legacy modes use per-body solves. Models with tendons retain authored
body colors. Per-body contact lists are sorted, and `enable_fast_math` also applies to pair-contact solves.
Repeatable results require deterministic collision generation, sufficient per-body list capacity, and
identical solver/contact history; color refinement can change finite-iteration Gauss-Seidel order.
Gather applied joint forces without atomics in deterministic modes, and avoid ordered reductions
for body-disjoint joint mimics in run-to-run mode.
Pair-manifold rows retain projected tangential warm starts with sticky matching,
independently of per-body contact rows in other worlds.
Use bounded, division-free comparisons in the pair-friction root solve without changing the Coulomb disk objective.
Evaluate pair-contact inertial responses inside the manifold solve and retain
body-local geometry that remains valid after center-of-mass updates. Accumulate
first-order pose increments within each manifold, then write each pose once.
Run all sweeps of a one-block world in one launch. For pair-routed steps, snapshot
rigid contact history in the body-velocity update launch. The contact equations and solve order
are unchanged; finite-sweep results differ from per-sweep preparation and pose updates.
