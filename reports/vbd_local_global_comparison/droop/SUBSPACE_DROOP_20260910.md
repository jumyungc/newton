# Compliance-KKT: a strong droop result, and a rejected contact subspace

2026-09-10. [Web report and figure](droop_20260910.html),
[source-checked raw manifest](droop_20260910_manifest.json), and
[frozen pre-experiment protocol](subspace_droop_20260910_protocol.md).
This continues the [merged-main assessment](COMPLIANCE_KKT_20260910.md).

## Decision

Keep the existing optional structural G1 path and I4/G1 as a strong working
configuration for the quoted dynamic cable. It substantially removes the
spurious stretch-dependent droop, at about 3.7-3.8x lower captured substep cost
than main I20 in paired measurements. This is a new validation of existing
production code, **not a newly implemented solver acceleration**.

Do not promote the proposed small stock-anchored residual-Krylov contact
subspace. It fails to improve important cases even in a CPU float64 oracle;
the frozen protocol therefore stops it before CUDA integration.

Neither result establishes universal optimality. The wider screen includes a
case where main is more accurate, and high-stiffness raw force balance is still
a limitation. This round does not certify a cable SysID pipeline or gradients.

No production source changed. No full unit suite, commit, or push was performed.
Newton's review procedure guided the separation of material correctness,
convergence diagnostics, state ownership, and measured timing claims.

## Exact reproduction and independent reference

Feature HEAD: `d711ddf377d642ff040ed887df4adebe16707504`.
Verbatim main: `31f585713a631d9a87acb5bb332b6a8ba61e2410`.
External branch: `718599ce6bda2566358fcdbee10c5f747c694843`;
[pinned script](https://github.com/jenv-nv/newton/blob/718599ce6bda2566358fcdbee10c5f747c694843/newton/scripts/droop_figure.py).
RTX 5090, Warp 1.17.0. No contacts in this fixture.

The reproduction retains the author's conventions: ten 0.05-m segments,
radius 0.005 m, the entire first segment fixed, and nine moving masses of
0.0022 kg. Actual moving mass is 0.0198 kg, not the 0.022-kg parameter.
The capsule-derived inertia is retained even though masses are overwritten.
Shear defaults to stretch: this varies both, not axial stiffness alone.

For bend 0.1 and damping 0.01, dynamic routes run 6000 frames, ten substeps per
1/60-s frame, dt=1/600 s. The external static route runs its actual single
momentum-free I6000 solve at dt=1/60 s. Main has no such static API; no imitation
was added. All actual solver trajectories/static solves use CUDA graphs.

The [independent discrete planar oracle](cable_droop_equilibrium_oracle.md)
is CPU float64 and is not timed as a simulation route. Exact translational
elimination gives `c_i = -W_i e_z / ks`; all stretch-dependent energy remaining
after elimination is constant in the angles. Thus, for this particular
isotropic-spring fixture,

```text
droop(ks) = droop(infinity) + 0.97119 / ks   [metres; nominal masses].
```

The independent equilibrium droops are 318.426 and 318.330 mm for ks=1e4/1e6.
The softer cable should droop only 0.09614781 mm more. The oracle uses the
nonlinear discrete curvature law, not small-deflection beam theory. Its energy
gradient is checked by finite differences; material forces and torques are
cross-checked against production CUDA functions at a nonstationary pose.
The oracle's uniqueness argument applies to the downward planar branch, not
arbitrary 3D equilibria. Nominal segment geometry differs from stored float32
anchors at rounding scale; do not treat submicrometre errors as an exact
bit-level production-geometry certificate.

## Quoted case: much better shape at lower cost

Maximum Euclidean node error against the independent reference after 6000
frames, except the explicitly labeled single static solve:

| Route | ks=1e4 droop / max error (mm) | ks=1e6 droop / max error (mm) |
|---|---:|---:|
| Independent equilibrium | 318.426 / 0 | 318.330 / 0 |
| Main I20, ALM on | 337.016 / 29.563 | 446.720 / 308.583 |
| Main I20, ALM off | 340.748 / 35.800 | 447.099 / 309.877 |
| Existing G1, I20 | 318.514 / 0.129 | 318.320 / 0.033 |
| Existing G1, I4 | 318.445 / 0.027 | 318.296 / 0.054 |
| External static I6000, ALM on | 28.184 / 325.485 | 3.407 / 348.030 |
| External static I6000, ALM off | 25.726 / 327.745 | 3.297 / 348.130 |

External dynamic model arrays, final poses, and every recorded checkpoint match
main exactly for each ALM setting. The branch G0 also matches main pose and
velocity bitwise after ten substeps in the four paired cost checks (two poses
times two stiffnesses). G0 equality is not a contact-determinism claim.

Increasing main to I80 at ks=1e6 still leaves **276.349 mm** maximum node error
after 6000 frames; condensed angular residual is 0.796887 in units of mgL.
Its final-state cost is 2766.73 us/substep in that separate process. More local
iterations help some metrics but do not resolve this case at that budget.

For the quoted I4/G1 rows, condensed angular residuals are approximately
5.12e-5 and 5.82e-5, versus main I20's 0.01616 and 1.93966. This measures
bending equilibrium after exact translational elimination; it is deliberately
reported separately from raw force balance at the actual returned positions.

The independent final-state timing runs measure about 187 us/substep for
I4/G1 versus 699-701 us for main I20. A separate randomized paired run from
identical rest or equilibrium poses confirms the cost advantage: main medians
751.5-761.0 us, I4/G1 medians 201.2-202.4 us. Paired median ratios are about
3.72-3.76; the individual within-process bootstrap intervals span roughly
3.708-3.769. Absolute clocks differ between runs, so the paired result is the
preferred cost comparison. It includes complete solver steps, not just the
global kernel. See [paired raw](droop_20260910_paired_cost_v3.json).

Each graph contains ten substeps. Thirty-one samples follow seven warmups;
route order is randomized and all mutable solver/state/control arrays are
restored before every sample. Events exclude restoring, collision detection,
host readback, and plotting. Quality comes from the separate evolving
trajectories. We did **not** measure an independently certified minimum
time-to-quality across all schedules, nor an end-to-end collision pipeline.

## ALM, settling, and the user's observation

The ALM flag reaches stretch, shear, bend and twist rows. These soft rows retain
the same finite authored elastic target with ALM on or off; the iteration and
auxiliary variables change. Both remain local without G. Similar unconverged
plots are therefore possible, but the outputs are not identical here.
The pinned plotting script hard-codes ALM on in both drivers; this harness
explicitly tests both settings instead.

Main I20/ALM on at ks=1e6 has only **0.326 mm** last-100-frame swing, below the
quoted 0.5-mm settling tolerance, while its node error is **308.583 mm**.
That is direct evidence that "settled" is not an equilibrium certificate.
The static routine sets velocities to zero by construction; its zero reported
swing from one solve is not evidence of convergence either.

Our global path removes the dominant false droop trend in this dynamic example.
It does not implement or validate a momentum-free global static solver.

## Broader screen and remaining limitations

The completed 800-frame screen compares main I20 with I4/G1 at bend 0.001 and
10, each with stretch/shear 1e2, 1e4, 1e6, 1e8: sixteen configurations.

| Bend | ks | Main I20 max node error (mm) | I4/G1 max node error (mm) |
|---:|---:|---:|---:|
| 0.001 | 1e2 | 0.1142 | 0.1132 |
| 0.001 | 1e4 | 1.4307 | 0.1117 |
| 0.001 | 1e6 | 21.7527 | 0.1214 |
| 0.001 | 1e8 | 38.1510 | 0.1063 |
| 10 | 1e2 | 0.0043 | 0.0190 |
| 10 | 1e4 | 0.9303 | 0.0014 |
| 10 | 1e6 | 168.0868 | 0.0063 |
| 10 | 1e8 | 709.7731 | 0.0004 |

The soft-stretch/high-bend row is a counterexample to uniform accuracy
dominance. Some other main rows are still moving substantially at frame 800;
these are equal-horizon errors, not claims that every route has settled.
Selected low-bend G1 and high-bend main/G1 rows were extended to 6000 frames;
low-bend G1 errors become 0.0092/0.0307 mm at ks=1e4/1e6. At bend 10,
ks=1e6, main's error is 223.873 mm versus G1's 0.0049 mm. Main's swing there
is only 0.428 mm: another settled-looking, inaccurate result. Complete values
are in the web report and raw manifest.

At quoted ks=1e6, I4/G1's raw force residual remains **0.185 times total
weight**. Merely rounding the independent equilibrium to float32 yields about
0.441 in that diagnostic. The latter illustrates cancellation sensitivity, not
a proved minimum achievable residual. At ks=1e8 the wider G1 screen has raw
force residuals around 31-35 times total weight despite small shape errors.
Do not conflate condensed angular convergence with full force convergence.

Most importantly for SysID, the two quoted G1/I4 final droops differ by about
**0.1493 mm**, versus the independent physical signal **0.09615 mm**. The
two-point signal error is therefore about 55% in these final samples. We have
not evaluated gradients, identification loss, repeated-run noise, or a fitting
pipeline. A visually excellent forward solve is not yet a validated parameter
identification solve. More iterations do not uniformly improve the floating-
point residuals; simply increasing I is not the recommended solution.

## Recommended contact subspace: tested and rejected for general use

At CUDA-generated current production states after three local sweeps, form a
block-whitened residual Krylov basis U, then solve the float64 CPU oracle

```text
min_alpha Q(s + U alpha),   subject to A U alpha <= 0.
```

Here s is the stock direction and A retains every original normal constraint.
Alpha=0 is stock-feasible. The basis uses current residual and operator only;
it does not contain an oracle optimum or case-specific hand-selected direction.
The original linear and nonlinear quality gates remain unchanged.

Fifteen fixture cases were attempted: twelve 16-body axes (two friction values,
two mass ratios, three timesteps), and 2/4/8-body prefixes. The 2/4-body
prefixes failed baseline self-Hessian reconstruction at 5.65e-5 and 3.56e-5
against the original 2e-5 gate. They are excluded before candidate evaluation,
not counted as subspace failures or repaired by loosening the tolerance.

| Basis dimension | Quality passes / 13 valid cases | Zero-change directions |
|---:|---:|---:|
| 2 | 0 / 13 | 13 |
| 4 | 0 / 13 | 13 |
| 8 | 5 / 13 | 8 |
| 16 | 6 / 13 | 7 |

All six mass-ratio-100 cases give no improvement at every tested dimension.
The passing equal-mass cases show substantial stationarity-proxy gains, but
only 11/52 candidates pass every gate. There is no general-use quality win.

The rejection has mathematical evidence: after projected-Hessian whitening,
the objective is `0.5 z^T z - g^T z`, with `D z <= 0`. If `g=D^T lambda`
for lambda>=0, then z=0 satisfies convex-QP KKT conditions and is optimal.
The numerical polar-cone fit has tiny relative residual in all 26 dimension-
2/4 cases and thirteen additional zero-change cases at dimensions 8/16.
Large fit residuals in the other cases are **not** infeasibility proofs.

This says the tested small residual subspace often contains no useful contact-
admissible descent. It does not refute all subspaces or the full contact-envelope
formulation. Under the frozen kill gate, no CUDA subspace prototype was run and
no production contact path was changed. See [all valid raw evidence](contact_subspace_20260910_v2.json).

## Next recommendation: precision-aware convergence, not more local sweeps

1. Keep G1 as the structural long-range correction and retain local nonlinear
   polishing. Use this droop invariant as a regression case, not a cable-only
   solver branch. Preserve authored finite compliance and main's G0 default.
2. Audit small relative joint deformations independently of world-space float32
   pose subtraction. Test local-frame/compensated deformation storage or mixed-
   precision residual replacement with iterative refinement. Merely factoring
   the same noisy residual in higher precision cannot fix quantized geometry.
   A useful dimensionless warning is `ks * position_uncertainty / weight`.
   Any implementation must preserve the same material equation and prove both
   force accuracy and complete captured-step cost, including new storage and
   conversions. This is a proposed experiment, not an implemented fix.
3. For SysID, test response differences against the analytic compliance signal
   and vary bend independently; include nonplanar and unequal shear/stretch
   holdouts. Use physical stationarity, not only velocity or shape, as a gate.
4. If pursuing the contact envelope, first construct contact-admissible descent
   directions or a bounded working-set solve in the full coupled space. A new
   residual-only basis with the same cone obstruction is unlikely to help.
   Require the unchanged nonlinear penetration/friction/history gates and an
   exact full-state stock fallback; the oracle still does not supply these.

KKT, Krylov, Schur elimination, active sets, and iterative refinement are not
new inventions. A possible research contribution would be a demonstrated
precision-aware, contact-admissible global/local method with preservation
guarantees and topology-spanning GPU time-to-quality evidence. No novelty or
paper-readiness claim is established by this round.

## Validation failures, artifacts, and reproduction

The first high-bend material probe failed the unchanged 5e-4 relative check:
force error 1.24e-4, torque error 2.86e-3. The original probe perturbed only
translations while retaining almost-equilibrium rotations. A separate
[range adapter](../../benchmarks/vbd_cable_droop_range_screen.py) adds a known
nonstationary rotational perturbation **only to the derivative-check pose**.
The same derivative formulas and tolerance then pass, without changing the
oracle, simulation, material, or measured outputs. The completed first two
high-bend rows reproduce exactly. The failed raw is retained as
`droop_20260910_main_highbend_i20.json`; v2 is the valid complete screen.

Two initial paired-harness attempts exited with segmentation faults before
writing results. A diagnostic retry located the fault in `wp.capture_launch`
during the timed loop. The harness retained arrays but not all solver/graph
owners. Retaining solver, states, control and class alongside each graph fixed
the lifetime error; v3 completes all checks. These are harness failures, not
solver divergence or valid timings. Logs remain under `/tmp/vbd_droop_20260910_paired_cost*.log`.
The initial contact-subspace raws likewise retain failed baseline/serialization
attempts; use only `contact_subspace_20260910_v2.json` for complete results.

Harnesses: [droop reproduction](../../benchmarks/vbd_cable_droop_screen.py),
[paired cost](../../benchmarks/vbd_cable_droop_paired.py),
[contact subspace](../../benchmarks/vbd_contact_subspace_screen.py), and
[web generator](../../benchmarks/build_vbd_droop_report.py).
The valid manifest contains 37 trajectory/static rows, four paired cost cases,
and 52 subspace candidates, with input hashes and source verification. All
solver runs are CUDA-captured; the independent equilibria and subspace QPs are
explicitly CPU oracles. No contact GPU speed result is claimed from the latter.

Final focused verification passed:

- `test_cable_kkt_contact_capture_is_finite_cuda_0` and
  `test_structural_kkt_path_matches_float64_oracle_cuda_0`: two selected tests,
  0.536 seconds of reported test time. This does not rerun or clear the known
  redundant-cycle history failure from the preceding assessment.
- Ruff lint and format checks on all five new Python files; `git diff --check`.
- All input source hashes, external/main equality checks, finite outputs and
  four paired G0 equality checks in the web generator; 97 local report/handoff
  link checks. The generated numerical figure was visually inspected.
- No tracked production `_src` difference against HEAD; against main, only
  `solver_vbd.py` and `rigid_vbd_kkt.py` differ. GPU compute inventory is empty
  after completion. No experiment is left running.

Example focused reruns, from the repo root with exclusive GPU access:

```bash
uv run --extra dev python benchmarks/vbd_cable_droop_screen.py --route branch --iterations 4 --output /tmp/droop_g1_recheck.json
uv run --extra dev python benchmarks/vbd_cable_droop_screen.py --route main --globals 0 --output /tmp/droop_main_recheck.json
uv run --extra dev python benchmarks/vbd_cable_droop_paired.py --output /tmp/droop_paired_recheck.json
uv run --extra dev python benchmarks/vbd_cable_droop_range_screen.py --route branch --iterations 4 --bend 10 --stretch 1e2 1e4 1e6 1e8 --frames 800 --output /tmp/droop_range_recheck.json
uv run --extra dev python benchmarks/build_vbd_droop_report.py
```

For external reproduction, clone the named branch separately, verify HEAD
equals the pinned SHA, and pass `--route external --external-root PATH
--globals 0 --alm true` (or false). Static adds `--static --iterations 6000`.
The original isolated checkout here is
`/tmp/newton_droop_source_20260910_9WbU5U/newton`, unmodified. It is temporary;
retain its pinned commit or a separate source copy when moving machines.
Raw source paths are absolute for droop runs; the generator verifies those
paths on this machine. On another machine, preserve or explicitly remap source
roots before regenerating. The already-generated HTML, PNG and JSON are portable.
Copy uncommitted research material separately; a normal clone does not carry it.
