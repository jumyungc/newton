"""Immutable, setup-inclusive lockstep pile harness for contact KKT v3."""

from __future__ import annotations

import argparse
from dataclasses import replace
import datetime
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import newton
import numpy as np
import warp as wp

from bench.global_cable import codex_bgn_contact_kkt as v1
from bench.global_cable import codex_bgn_contact_kkt_pile as base_harness
from bench.global_cable import codex_bgn_dense_pile as pile
from bench.global_cable import codex_contact_kkt_v2 as v2
from bench.global_cable import codex_contact_kkt_v2_pile as v2_harness
from bench.global_cable import codex_contact_kkt_v3 as v3
from bench.global_cable import codex_contact_safe_predictor as predictor


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/_workspace/codex_contact_kkt_v3_pile"
SCHEMA = "codex-contact-kkt-v3-pile-lockstep/v1"


def _same_groups(left: list[np.ndarray], right: list[np.ndarray]) -> bool:
    return bool(
        len(left) == len(right)
        and all(
            np.array_equal(a, b)
            for a, b in zip(left, right, strict=True)
        )
    )


def _exception_checkpoint(
    *,
    step: int,
    completed_steps: int,
    certified_steps: int,
    exc: Exception,
    failed_call_ms: float,
    source_hash_before: str,
    source_hash_after: str,
    last_certified: dict[str, Any] | None,
) -> dict[str, Any]:
    """Create the immutable information required after a failed step."""
    return {
        "kind": "solver_exception",
        "attempted_physical_step": step,
        "completed_physical_steps_before_failure": completed_steps,
        "certified_physical_steps_before_failure": certified_steps,
        "type": type(exc).__name__,
        "message": str(exc),
        "failed_call_wall_ms": failed_call_ms,
        "caller_state_sha256_before": source_hash_before,
        "caller_state_sha256_after": source_hash_after,
        "caller_unchanged_during_failed_call": (
            source_hash_before == source_hash_after
        ),
        "failed_step_time_advance_count": 0,
        "last_certified": last_certified,
    }


def run(
    case_name: str,
    steps: int | None,
    outer_iters: int,
    dual_iters: int,
) -> dict[str, Any]:
    """Run one source-identical v3/VBD10 trajectory and fail at first bad step."""
    v3.owner_guard("KKT v3 pile run start")
    case = pile.CASE_SPECS[case_name]
    if case_name == "smoke":
        case = replace(case, iterations=10)
    requested_steps = (
        case.frames * case.substeps if steps is None else int(steps)
    )
    if requested_steps <= 0 or requested_steps > case.frames * case.substeps:
        raise ValueError("steps must be in [1, frames*substeps]")
    dt = 1.0 / (case.fps * case.substeps)

    # Candidate timing begins before model/pipeline construction and ends only
    # after plan, state, geometry, and monitor buffers are ready.
    candidate_setup_started = time.perf_counter_ns()
    candidate_built = pile.build_pile(case)
    candidate_scene = candidate_built.scene
    candidate_initial = v1.state_sha256(candidate_scene.state_0)
    candidate_plan = v1.prepare_contact(candidate_scene, dt)
    candidate_state = candidate_scene.model.state()
    candidate_state.assign(candidate_scene.state_0)
    candidate_geometry = pile.precompute_geometry(
        candidate_state, candidate_built.cable_groups
    )
    candidate_monitor = candidate_scene.model.contacts(
        collision_pipeline=candidate_scene.collision_pipeline
    )
    wp.synchronize_device(candidate_scene.model.device)
    candidate_setup_ms = (
        time.perf_counter_ns() - candidate_setup_started
    ) * 1.0e-6

    vbd_setup_started = time.perf_counter_ns()
    vbd_built = pile.build_pile(case)
    vbd_scene = vbd_built.scene
    vbd_initial = v1.state_sha256(vbd_scene.state_0)
    vbd_state = vbd_scene.model.state()
    vbd_state.assign(vbd_scene.state_0)
    vbd_out = vbd_scene.model.state()
    vbd_geometry = pile.precompute_geometry(
        vbd_state, vbd_built.cable_groups
    )
    vbd_contacts = vbd_scene.model.contacts(
        collision_pipeline=vbd_scene.collision_pipeline
    )
    vbd_solver = newton.solvers.SolverVBD(
        vbd_scene.model,
        iterations=10,
        rigid_body_contact_buffer_size=256,
        rigid_contact_history=True,
        rigid_contact_hard=True,
    )
    vbd_control = vbd_scene.model.control()
    vbd_monitor = vbd_scene.model.contacts(
        collision_pipeline=vbd_scene.collision_pipeline
    )
    wp.synchronize_device(vbd_scene.model.device)
    vbd_setup_ms = (
        time.perf_counter_ns() - vbd_setup_started
    ) * 1.0e-6

    if candidate_initial != vbd_initial:
        raise RuntimeError("lockstep builders produced different initial states")
    if not _same_groups(
        candidate_built.cable_groups, vbd_built.cable_groups
    ):
        raise RuntimeError("lockstep builders produced different cable groups")

    candidate_pairs: set[int] | None = None
    vbd_pairs: set[int] | None = None
    candidate_peak_penetration = 0.0
    vbd_peak_penetration = 0.0
    candidate_peak_active = 0
    vbd_peak_active = 0
    candidate_ground_observed = False
    candidate_cable_observed = False
    vbd_ground_observed = False
    vbd_cable_observed = False
    candidate_step_ms = 0.0
    candidate_call_wall_ms = 0.0
    candidate_predictor_ms = 0.0
    vbd_step_ms = 0.0
    candidate_solver_pass = True
    all_one_dt = True
    all_caller_unchanged = True
    all_predictor_targets_unchanged = True
    all_predictor_endpoints_safe = True
    all_certificates_commit_aware = True
    rows: list[dict[str, Any]] = []
    completed_steps = 0
    certified_steps = 0
    last_certified: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None

    for step in range(1, requested_steps + 1):
        frame_scene = replace(candidate_scene, state_0=candidate_state)
        source_hash_before = v1.state_sha256(candidate_state)
        candidate_call_started = time.perf_counter_ns()
        try:
            candidate_result = v3.solve_contact_frame(
                frame_scene,
                dt,
                plan=candidate_plan,
                max_outer_iters=outer_iters,
                max_dual_iters=dual_iters,
            )
        except Exception as exc:
            failed_call_ms = (
                time.perf_counter_ns() - candidate_call_started
            ) * 1.0e-6
            candidate_call_wall_ms += failed_call_ms
            source_hash_after = v1.state_sha256(candidate_state)
            failed_caller_unchanged = (
                source_hash_before == source_hash_after
            )
            candidate_solver_pass = False
            all_caller_unchanged &= failed_caller_unchanged
            failure = _exception_checkpoint(
                step=step,
                completed_steps=completed_steps,
                certified_steps=certified_steps,
                exc=exc,
                failed_call_ms=failed_call_ms,
                source_hash_before=source_hash_before,
                source_hash_after=source_hash_after,
                last_certified=last_certified,
            )
            rows.append({
                "checkpoint_type": "solver_exception",
                "physical_step_attempted": step,
                "completed_physical_steps_before_failure": completed_steps,
                "certified_physical_steps_before_failure": certified_steps,
                "failure": failure,
            })
            break
        candidate_call_wall_ms += (
            time.perf_counter_ns() - candidate_call_started
        ) * 1.0e-6
        candidate_step_ms += float(candidate_result["elapsed_ms"])
        candidate_predictor_ms += float(
            candidate_result["predictor_elapsed_ms"]
        )
        candidate_solver_pass &= bool(candidate_result["pass"])
        all_one_dt &= bool(
            candidate_result["time_advance_count"] == 1
            and not candidate_result["outer_iterations_are_time_substeps"]
            and candidate_result["dt"] == dt
        )
        all_caller_unchanged &= bool(candidate_result["caller_unchanged"])
        predictor_info = candidate_result["predictor"]
        all_predictor_targets_unchanged &= bool(
            predictor_info["physical_target_unchanged"]
            and predictor_info["predictor_time_advance_count"] == 0
            and predictor_info["physical_dt_fraction"] == 1.0
        )
        all_predictor_endpoints_safe &= bool(
            predictor_info["decision"]["endpoint_safe"]
        )
        all_certificates_commit_aware &= bool(
            predictor_info["commit_aware_endpoint_certification"]
            and candidate_result["collision_gate_manifest"][
                "float32_commit_aware_certification"
            ]
        )
        candidate_state = candidate_result["state"]

        vbd_started = time.perf_counter_ns()
        vbd_scene.collision_pipeline.collide(vbd_state, vbd_contacts)
        vbd_solver.step(
            vbd_state, vbd_out, vbd_control, vbd_contacts, dt
        )
        wp.synchronize_device(vbd_scene.model.device)
        vbd_step_ms += (time.perf_counter_ns() - vbd_started) * 1.0e-6
        vbd_state, vbd_out = vbd_out, vbd_state

        candidate_contact, candidate_pairs = base_harness._contact_at_state(
            candidate_scene,
            candidate_state,
            candidate_monitor,
            candidate_pairs,
        )
        vbd_contact, vbd_pairs = base_harness._contact_at_state(
            vbd_scene, vbd_state, vbd_monitor, vbd_pairs
        )
        candidate_peak_penetration = max(
            candidate_peak_penetration,
            float(candidate_contact["maximum_penetration_m"]),
        )
        vbd_peak_penetration = max(
            vbd_peak_penetration,
            float(vbd_contact["maximum_penetration_m"]),
        )
        candidate_peak_active = max(
            candidate_peak_active, int(candidate_contact["active_rows"])
        )
        vbd_peak_active = max(
            vbd_peak_active, int(vbd_contact["active_rows"])
        )
        candidate_ground_observed |= bool(
            candidate_contact["active_ground_rows"]
        )
        candidate_cable_observed |= bool(
            candidate_contact["active_cable_cable_rows"]
        )
        vbd_ground_observed |= bool(vbd_contact["active_ground_rows"])
        vbd_cable_observed |= bool(
            vbd_contact["active_cable_cable_rows"]
        )
        completed_steps = step

        if (
            step == 1 or step == requested_steps
            or step % case.substeps == 0
            or not candidate_result["pass"]
        ):
            final_iteration = candidate_result["iterations"][-1]
            rows.append({
                "physical_step": step,
                "display_frame": step / case.substeps,
                "candidate": {
                    "pass": bool(candidate_result["pass"]),
                    "converged": bool(candidate_result["converged"]),
                    "elapsed_ms": float(candidate_result["elapsed_ms"]),
                    "predictor_elapsed_ms": float(
                        candidate_result["predictor_elapsed_ms"]
                    ),
                    "predictor": predictor_info,
                    "collision_gate_manifest": candidate_result[
                        "collision_gate_manifest"
                    ],
                    "time_advance_count": candidate_result[
                        "time_advance_count"
                    ],
                    "outer_iterations_are_time_substeps": candidate_result[
                        "outer_iterations_are_time_substeps"
                    ],
                    "caller_unchanged": candidate_result[
                        "caller_unchanged"
                    ],
                    "outer_iterations": len(candidate_result["iterations"]),
                    "final_globalization": final_iteration[
                        "collision_safe_globalization"
                    ],
                    "velocity_reconstruction_inf": candidate_result[
                        "velocity_reconstruction_inf"
                    ],
                    "state_finite": candidate_result["state_finite"],
                    "quaternion_norm_max_error": candidate_result[
                        "quaternion_norm_max_error"
                    ],
                    "final_kkt": candidate_result["final_kkt"],
                    "internal_final_contact": candidate_result[
                        "final_contact"
                    ],
                    "committed_contact": candidate_result[
                        "committed_contact"
                    ],
                    "contact": candidate_contact,
                    "cable": pile.geometric_cable_metrics(
                        candidate_state, candidate_geometry
                    ),
                    "state_sha256": v1.state_sha256(candidate_state),
                },
                "vbd10": {
                    "contact": vbd_contact,
                    "cable": pile.geometric_cable_metrics(
                        vbd_state, vbd_geometry
                    ),
                    "state_sha256": v1.state_sha256(vbd_state),
                },
            })
        if candidate_result["pass"]:
            certified_steps = step
            last_certified = {
                "physical_step": step,
                "monitor_penetration_m": candidate_contact[
                    "maximum_penetration_m"
                ],
                "committed_internal_penetration_m": candidate_result[
                    "committed_contact"
                ]["maximum_penetration_m"],
                "state_sha256": v1.state_sha256(candidate_state),
                "time_advance_count": candidate_result[
                    "time_advance_count"
                ],
                "outer_iterations_are_time_substeps": candidate_result[
                    "outer_iterations_are_time_substeps"
                ],
                "caller_unchanged": candidate_result[
                    "caller_unchanged"
                ],
                "physical_target_unchanged": predictor_info[
                    "physical_target_unchanged"
                ],
                "float32_commit_aware_certification": predictor_info[
                    "commit_aware_endpoint_certification"
                ],
                "predictor_internal_target_m": candidate_result[
                    "collision_gate_manifest"
                ]["predictor_internal_target_m"],
                "public_solver_penetration_gate_m": candidate_result[
                    "collision_gate_manifest"
                ]["public_solver_penetration_gate_m"],
            }
        else:
            failure = {
                "kind": "solver_gate_failure",
                "attempted_physical_step": step,
                "completed_physical_steps_before_failure": step - 1,
                "completed_physical_steps_including_failed_output": step,
                "certified_physical_steps_before_failure": certified_steps,
                "last_certified": last_certified,
                "failed_output_penetration_m": candidate_result[
                    "committed_contact"
                ]["maximum_penetration_m"],
                "failed_output_pass": False,
            }
        if not candidate_result["pass"]:
            break

    candidate_final_cable = pile.geometric_cable_metrics(
        candidate_state, candidate_geometry
    )
    vbd_final_cable = pile.geometric_cable_metrics(
        vbd_state, vbd_geometry
    )
    penetration_limit = case.radius
    stretch_limit = 0.25 * case.segment_length
    full = completed_steps == requested_steps

    def trajectory_gates(
        peak_penetration: float,
        final_cable: dict[str, Any],
        ground: bool,
        cable: bool,
    ) -> dict[str, Any]:
        return {
            "full_frozen_trajectory_completed": full,
            "penetration_below_frozen_limit": (
                peak_penetration <= penetration_limit
            ),
            "stretch_below_frozen_limit": (
                final_cable["stretch_max_m"] <= stretch_limit
            ),
            "real_ground_contact_observed": ground,
            "real_cable_cable_contact_observed": cable,
            "finite": bool(final_cable["finite"]),
        }

    candidate_gates = trajectory_gates(
        candidate_peak_penetration,
        candidate_final_cable,
        candidate_ground_observed,
        candidate_cable_observed,
    )
    vbd_gates = trajectory_gates(
        vbd_peak_penetration,
        vbd_final_cable,
        vbd_ground_observed,
        vbd_cable_observed,
    )
    execution_certificates = {
        "one_physical_dt_per_step": all_one_dt,
        "caller_state_isolated_every_step": all_caller_unchanged,
        "full_dt_target_unchanged_every_step": (
            all_predictor_targets_unchanged
        ),
        "predictor_endpoint_safe_every_step": (
            all_predictor_endpoints_safe
        ),
        "predictor_and_line_search_commit_aware_every_step": (
            all_certificates_commit_aware
        ),
    }
    candidate_setup_inclusive_ms = (
        candidate_setup_ms + candidate_call_wall_ms
    )
    vbd_setup_inclusive_ms = vbd_setup_ms + vbd_step_ms
    candidate_pass = bool(
        candidate_solver_pass
        and all(candidate_gates.values())
        and all(execution_certificates.values())
    )
    v3.owner_guard("KKT v3 pile run end")
    return {
        "case": {
            "name": case.name,
            "body_count": case.body_count,
            "frames": case.frames,
            "substeps": case.substeps,
            "fps": case.fps,
            "dt": dt,
            "radius_m": case.radius,
            "segment_length_m": case.segment_length,
        },
        "requested_physical_steps": requested_steps,
        "completed_physical_steps": completed_steps,
        "certified_physical_steps": certified_steps,
        "failure": failure,
        "last_certified": last_certified,
        "source_identical_initial_state": True,
        "initial_state_sha256": candidate_initial,
        "candidate": {
            "method": (
                "KKT v3: endpoint-safe initial predictor, unchanged full-dt "
                "target, v2 contact algebra"
            ),
            "timing": {
                "scene_plan_state_geometry_setup_ms": candidate_setup_ms,
                "physical_step_solver_ms": candidate_step_ms,
                "physical_step_call_wall_ms_including_failed_call": (
                    candidate_call_wall_ms
                ),
                "predictor_ms_subset_of_step_solver": candidate_predictor_ms,
                "setup_inclusive_ms": candidate_setup_inclusive_ms,
                "audit_monitor_collision_excluded": True,
            },
            "peak_penetration_m": candidate_peak_penetration,
            "peak_active_contacts": candidate_peak_active,
            "final_cable": candidate_final_cable,
            "gates": candidate_gates,
            "execution_certificates": execution_certificates,
            "executed_solver_pass": candidate_solver_pass,
            "pass": candidate_pass,
        },
        "vbd10": {
            "method": "production SolverVBD, requested iterations=10",
            "timing": {
                "scene_solver_state_geometry_setup_ms": vbd_setup_ms,
                "physical_step_solver_ms": vbd_step_ms,
                "setup_inclusive_ms": vbd_setup_inclusive_ms,
                "audit_monitor_collision_excluded": True,
            },
            "peak_penetration_m": vbd_peak_penetration,
            "peak_active_contacts": vbd_peak_active,
            "final_cable": vbd_final_cable,
            "gates": vbd_gates,
            "pass": bool(all(vbd_gates.values())),
        },
        "frozen_gates": {
            "penetration_limit_m": penetration_limit,
            "stretch_limit_m": stretch_limit,
            "changed_from_dense_pile_harness": False,
        },
        "collision_gate_manifest": {
            "predictor_internal_target_m": (
                v3.PREDICTOR_HEADROOM_TARGET_M
            ),
            "public_solver_penetration_gate_m": (
                v3.PUBLIC_SOLVER_PENETRATION_GATE_M
            ),
            "public_gate_relaxed": False,
            "float32_commit_aware_certification": True,
        },
        "samples": rows,
        "claim_boundary": (
            "full trajectory claims require all 60 physical steps, both real "
            "contact classes, unchanged frozen gates, v2 algebraic gates, "
            "caller isolation, and one-dt certificates; predictor and line "
            "search certify endpoints only, not swept paths"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", choices=("smoke", "moderate"), default="smoke"
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--outer-iters", type=int, default=v1.DEFAULT_OUTER_ITERS
    )
    parser.add_argument(
        "--dual-iters", type=int, default=v1.DEFAULT_DUAL_ITERS
    )
    args = parser.parse_args()
    error = None
    try:
        result = run(
            args.case, args.steps, args.outer_iters, args.dual_iters
        )
        status = (
            "PASS" if result["candidate"]["pass"] else
            "PENDING_PARTIAL" if args.steps is not None
            and result["candidate"]["executed_solver_pass"] else "FAIL"
        )
    except Exception as exc:  # Save an immutable error artifact before failing.
        status = "ERROR"
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        result = None

    source = Path(__file__).resolve()
    solver = Path(v3.__file__).resolve()
    test = solver.with_name("test_codex_contact_kkt_v3.py")
    predictor_source = Path(predictor.__file__).resolve()
    predictor_test = predictor_source.with_name(
        "test_codex_contact_safe_predictor.py"
    )
    payload = {
        "schema": SCHEMA,
        "status": status,
        "generated_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "host": platform.node(),
        "python": sys.version,
        "owner": v3.owner_guard("KKT v3 artifact metadata"),
        "arguments": {
            "case": args.case,
            "steps": args.steps,
            "outer_iters": args.outer_iters,
            "dual_iters": args.dual_iters,
        },
        "source_sha256": v2_harness.sha256(source),
        "solver_sha256": v2_harness.sha256(solver),
        "solver_test_sha256": v2_harness.sha256(test),
        "predictor_sha256": v2_harness.sha256(predictor_source),
        "predictor_test_sha256": v2_harness.sha256(predictor_test),
        "v2_solver_sha256": v2_harness.sha256(Path(v2.__file__).resolve()),
        "dense_pile_source_sha256": v2_harness.sha256(
            Path(pile.__file__).resolve()
        ),
        "error": error,
        "result": result,
    }
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    path = OUT / f"{args.case}_{stamp}_{os.getpid()}.json"
    v2_harness.atomic_json(path, payload)
    print(path)
    print(f"status={status}")
    print(f"sha256={v2_harness.sha256(path)}")
    if error is not None:
        print(f"error={error['type']}: {error['message']}")
    return 0 if status in {"PASS", "PENDING_PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
