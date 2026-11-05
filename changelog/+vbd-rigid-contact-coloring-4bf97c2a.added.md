Add deterministic rigid VBD contact scheduling compatible with CUDA graph capture.
Expose `rigid_contact_coloring` (`auto`, `model`, or `refined`) and
`rigid_body_contact_color_budget` (default 32) on `SolverVBD` for per-body scheduling,
while preserving mandatory joint separation. These options do not select the contact formulation
or bound pair-manifold coloring.
With compliant ALM, collision-isolated worlds without structural joints, tendons, or particle attachments
use pair-manifold solves when numeric contact history is enabled; other worlds and
history-disabled contacts use per-body solves.
Models with tendons retain authored body colors, including dynamic-bypass endpoint separation.
Require deterministic collision generation and sufficient per-body contact-list capacity
for repeatable contact results.
Use warp-cooperative sorting for short CUDA contact lists while retaining ordered long-row and CPU fallbacks.
Use cooperative tile stores and avoid redundant scratch clears before uncoupled body-contact gathering.
Avoid ordered integer-reservation overhead where sorted lists or independent pair colors preserve deterministic results.
Parallelize color retention and pack only used colors in ascending body-ID order; skip empty refined sweeps.
Empty a contact color the repair creates by bounded Kempe interchanges where possible; this changes the
per-body Gauss-Seidel order on scenes where it applies.
Stop redundant neighbor scans once a retained color conflicts or an unfinished dependency blocks body-color repair.
Scan pending pair groups cooperatively while preserving descending serial completion order.
Use previous pair colors to prioritize parallel GPU coloring. This scheduling history changes the finite-iteration
pair Gauss-Seidel order; replay requires identical solver history as well as identical contact inputs.
Reuse pair-contact material and step-start anchor constants between collision refreshes within one solver step.
Avoid redundant quaternion normalization in pair-contact pose updates.
