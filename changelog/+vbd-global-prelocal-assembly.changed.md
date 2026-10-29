Apply contact-free rigid VBD global joint corrections before their selected local sweeps, retaining exactly the configured local iteration budget even when every iteration includes a global correction. Contact scheduling and local-only VBD are unchanged.

Parallelize CUDA rigid VBD path-system assembly across scalar rows while preserving the existing Schur equations, in-place buffers, and CPU assembly path.

Apply the joint-stress metric during linearization when every selected row is a parented rod, omitting the separate metric launch; disabled rows clear their owned stress slots. Mixed and world-anchored graphs keep the separate pass and a linearization kernel compiled without the fused code. Overlap disjoint persistent and long-chain CUDA factorization routes with caller-stream synchronization.

Represent reproducible CUDA contact-translation reconciliation with a bounded conditional loop, retaining the existing maximum of sixteen evaluations and roundoff stopping policy without duplicating every conditional body in the captured graph. Include the loop counter in structural memory-budget accounting.
