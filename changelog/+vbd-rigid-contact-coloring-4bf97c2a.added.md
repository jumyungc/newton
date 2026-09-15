Add `SolverVBD.rigid_contact_coloring` with deterministic pair-manifold scheduling for joint-free worlds using
persistent contact history and joint-preserving contact refinement for structural worlds.
Add `SolverVBD.rigid_body_contact_color_budget` to bound per-body contact refinement while preserving the
mandatory joint-color count. The default budget remains 32 and does not change the pair-manifold palette.
Preserve distinct normal and friction moment arms when applying retained pair-contact forces, and fuse
the serial pair-color overflow tail into the existing CUDA color sweep.
Use populated late pair colors to select a parallel CUDA-graph tail without changing contact order,
choosing once per substep when a pair-only contact graph stays fixed through the iterations.
Avoid deterministic scatter-reduction overhead for joint forces only when fixed topology proves all
body-force writers disjoint; preserve the original kernel for shared bodies and stronger determinism.
