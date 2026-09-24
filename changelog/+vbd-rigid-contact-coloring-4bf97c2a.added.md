Add deterministic rigid VBD contact scheduling compatible with CUDA graph capture.
Expose `rigid_contact_coloring` (`auto`, `model`, or `refined`) and
`rigid_body_contact_color_budget` (default 32) on `SolverVBD` for per-body scheduling,
while preserving mandatory joint separation. These options do not select the contact formulation
or bound pair-manifold coloring.
With compliant ALM, collision-isolated joint-free worlds with numeric contact history use
pair-manifold solves; structural worlds and history-disabled contacts use per-body solves.
Require deterministic collision generation and sufficient per-body contact-list capacity
for repeatable contact results.
