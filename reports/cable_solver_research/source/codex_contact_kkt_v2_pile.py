"""Immutable lockstep pile harness for contact KKT v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import replace
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any

from bench.global_cable import codex_bgn_contact_kkt as v1
from bench.global_cable import codex_bgn_contact_kkt_pile as harness
from bench.global_cable import codex_bgn_dense_pile as pile
from bench.global_cable import codex_contact_kkt_v2 as v2


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/_workspace/codex_contact_kkt_v2_pile"
SCHEMA = "codex-contact-kkt-v2-pile-lockstep/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict(item) for item in value]
    if hasattr(value, "tolist"):
        return strict(value.tolist())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON value: {value}")
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    v2.owner_guard(f"create {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    encoded = (
        json.dumps(strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        v2.owner_guard(f"publish {path.name}")
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(case_name: str, steps: int | None, outer_iters: int,
        dual_iters: int) -> dict[str, Any]:
    v2.owner_guard("KKT v2 pile run start")
    case = pile.CASE_SPECS[case_name]
    if case_name == "smoke":
        case = replace(case, iterations=10)
    old_guard = v1.owner_guard
    old_solver = v1.solve_contact_frame
    v1.owner_guard = v2.owner_guard
    v1.solve_contact_frame = v2.solve_contact_frame
    try:
        result = harness.run_lockstep(
            case, steps=steps, outer_iters=outer_iters, dual_iters=dual_iters
        )
    finally:
        v1.owner_guard = old_guard
        v1.solve_contact_frame = old_solver
    result["candidate"]["method"] = (
        "contact KKT v2: certified direct sticking active set + cone fallback"
    )
    result["claim_boundary"] += (
        "; direct sticking is used only when its equality impulse is inside "
        "the authored Coulomb cone"
    )
    v2.owner_guard("KKT v2 pile run end")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("smoke", "moderate"), default="smoke")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--outer-iters", type=int, default=v1.DEFAULT_OUTER_ITERS)
    parser.add_argument("--dual-iters", type=int, default=v1.DEFAULT_DUAL_ITERS)
    args = parser.parse_args()
    result = run(args.case, args.steps, args.outer_iters, args.dual_iters)
    source = Path(__file__).resolve()
    solver = Path(v2.__file__).resolve()
    test = solver.with_name("test_codex_contact_kkt_v2.py")
    payload = {
        "schema": SCHEMA,
        "status": (
            "PASS" if result["candidate"]["pass"] else
            ("PENDING_PARTIAL" if args.steps is not None
             and result["candidate"]["executed_solver_pass"] else "FAIL")
        ),
        "generated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "host": platform.node(),
        "python": sys.version,
        "owner": v2.owner_guard("KKT v2 artifact metadata"),
        "source_sha256": sha256(source),
        "solver_sha256": sha256(solver),
        "solver_test_sha256": sha256(test),
        "result": result,
    }
    stamp = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    path = OUT / f"{args.case}_{stamp}_{os.getpid()}.json"
    atomic_json(path, payload)
    print(path)
    print(f"status={payload['status']}")
    print(f"sha256={sha256(path)}")
    return 0 if payload["status"] in {"PASS", "PENDING_PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
