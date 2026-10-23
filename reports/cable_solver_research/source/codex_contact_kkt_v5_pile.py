"""Immutable canonical pile harness for certificate-aware KKT v5."""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
import platform
from types import SimpleNamespace
import sys
import types
from typing import Any

from bench.global_cable import codex_bgn_contact_kkt as v1
from bench.global_cable import codex_bgn_dense_pile as pile
from bench.global_cable import codex_contact_kkt_v2_pile as v2_harness
from bench.global_cable import codex_contact_kkt_v3 as v3
from bench.global_cable import codex_contact_kkt_v3_pile as v3_harness
from bench.global_cable import codex_contact_kkt_v4 as v4
from bench.global_cable import codex_contact_kkt_v5 as v5
from bench.global_cable import codex_contact_safe_predictor as predictor
from bench.global_cable import codex_desaxce as desaxce


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/_workspace/codex_contact_kkt_v5_pile"
SCHEMA = "codex-contact-kkt-v5-pile-lockstep/v1"


def _bound_v3_harness_run() -> Any:
    facade = SimpleNamespace(
        owner_guard=v5.owner_guard,
        solve_contact_frame=v5.solve_contact_frame,
        PREDICTOR_HEADROOM_TARGET_M=v3.PREDICTOR_HEADROOM_TARGET_M,
        PUBLIC_SOLVER_PENETRATION_GATE_M=(
            v3.PUBLIC_SOLVER_PENETRATION_GATE_M
        ),
    )
    globals_copy = dict(v3_harness.run.__globals__)
    globals_copy["v3"] = facade
    bound = types.FunctionType(
        v3_harness.run.__code__,
        globals_copy,
        name="run_v5_pile_bound",
        argdefs=v3_harness.run.__defaults__,
        closure=v3_harness.run.__closure__,
    )
    bound.__kwdefaults__ = dict(v3_harness.run.__kwdefaults__ or {})
    return bound


def run(
    case_name: str,
    steps: int | None,
    outer_iters: int,
    dual_iters: int,
) -> dict[str, Any]:
    v5.owner_guard("KKT v5 pile run start")
    result = _bound_v3_harness_run()(
        case_name, steps, outer_iters, dual_iters
    )
    result["candidate"]["method"] = (
        "KKT v5: commit-aware v3 geometry + certified high-mu sticking + "
        "certificate-aware stick-condensed De Saxce refinement"
    )
    result["claim_boundary"] += (
        "; De Saxce solve tightens until normalized natural-map and every "
        "absolute 1e-8 law certificate pass; no associated fallback"
    )
    v5.owner_guard("KKT v5 pile run end")
    return result


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
    except Exception as exc:
        status = "ERROR"
        error = {"type": type(exc).__name__, "message": str(exc)}
        result = None

    source = Path(__file__).resolve()
    solver = Path(v5.__file__).resolve()
    test = solver.with_name("test_codex_contact_kkt_v5.py")
    payload = {
        "schema": SCHEMA,
        "status": status,
        "generated_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "host": platform.node(),
        "python": sys.version,
        "owner": v5.owner_guard("KKT v5 artifact metadata"),
        "arguments": {
            "case": args.case,
            "steps": args.steps,
            "outer_iters": args.outer_iters,
            "dual_iters": args.dual_iters,
        },
        "source_sha256": v2_harness.sha256(source),
        "solver_sha256": v2_harness.sha256(solver),
        "solver_test_sha256": v2_harness.sha256(test),
        "v4_solver_sha256": v2_harness.sha256(
            Path(v4.__file__).resolve()
        ),
        "v3_solver_sha256": v2_harness.sha256(
            Path(v3.__file__).resolve()
        ),
        "v3_harness_sha256": v2_harness.sha256(
            Path(v3_harness.__file__).resolve()
        ),
        "predictor_sha256": v2_harness.sha256(
            Path(predictor.__file__).resolve()
        ),
        "desaxce_sha256": v2_harness.sha256(
            Path(desaxce.__file__).resolve()
        ),
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
