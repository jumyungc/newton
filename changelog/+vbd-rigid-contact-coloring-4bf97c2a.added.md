Add deterministic rigid VBD contact scheduling compatible with CUDA graph capture.
Expose `rigid_contact_coloring` (`auto`, `model`, or `refined`) and
`rigid_body_contact_color_budget` (default 32) on `SolverVBD` for per-body scheduling,
while preserving mandatory joint separation. These options do not select the contact
formulation or bound pair-manifold coloring.
With compliant ALM, collision-isolated worlds without structural joints, tendons, or
particle attachments use pair-manifold solves when numeric contact history is enabled;
other worlds and history-disabled contacts use per-body solves. Models with tendons
retain authored body colors, including dynamic-bypass endpoint separation.
Use sorted contact lists, GPU body-color refinement and cached exact pair-color schedules;
dispatch only needed refined sweeps and parallelize independent pair-overflow work while
preserving every shared body's original update order.
Repeatable results require deterministic collision generation, sufficient per-body list
capacity, and identical solver/contact history. Color refinement and previous-color
priorities can change finite-iteration Gauss-Seidel order.
Preload rigid pair-manifold body properties and contact inputs before serial pose updates.
Avoid repeated endpoint scratch clears when building dense pair-overflow schedules.
Apply the existing `enable_fast_math` option to pair-contact solves as well as body solves.
