"""Immutable small-oracle artifact for CCDSN constrained contact coupling."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time
from typing import Any

import numpy as np
from scipy import sparse

from bench.global_cable import codex_ccdsn as ccdsn
from bench.global_cable import codex_desaxce as desaxce


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/_workspace/codex_ccdsn_reference"
SCHEMA = "codex-ccdsn-small-reference/v1"
DT = 1.0 / 600.0


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
    if isinstance(value, np.ndarray):
        return strict(value.tolist())
    if isinstance(value, np.generic):
        return strict(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value: {value}")
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    ccdsn.owner_guard(f"create {path.name}")
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
        ccdsn.owner_guard(f"publish {path.name}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def timed_solve(kwargs: dict[str, Any], repeats: int = 11):
    ccdsn.solve_coupled_quadratic(**kwargs)
    times = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = ccdsn.solve_coupled_quadratic(**kwargs)
        times.append((time.perf_counter_ns() - started) * 1.0e-6)
    assert result is not None
    ordered = sorted(times)
    return result, {
        "repeats": repeats,
        "median_ms": statistics.median(times),
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
        "p90_ms": ordered[math.ceil(0.9 * len(ordered)) - 1],
        "claim_boundary": "warm in-process CPU diagnostic; not production timing",
    }


def selected_info(info: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "stationarity_inf",
        "stretch_residual_inf_m",
        "natural_map_inf",
        "natural_map_relative_inf",
        "primal_cone_violation_N",
        "dual_cone_violation_m",
        "complementarity_inf_Nm",
        "friction_work_Nm",
        "physical_friction_work_Nm",
        "compliant_contact_value",
        "active_contact_indices",
        "friction_law",
    )
    return {key: info[key] for key in keys if key in info}


def run() -> dict[str, Any]:
    ccdsn.owner_guard("CCDSN reference benchmark start")
    analytic_kwargs = dict(
        H=sparse.eye(4, format="csc"),
        rhs=np.zeros(4),
        A=sparse.csr_matrix([[1.0, 0.0, 0.0, -1.0]]),
        stretch_offset=np.zeros(1),
        stretch_compliance=np.zeros(1),
        J=sparse.csr_matrix(np.hstack((np.eye(3), np.zeros((3, 1))))),
        contact_offset=np.asarray([-1.0, 2.0, 0.0]),
        contact_mode=ccdsn.DESAXCE,
        contact_compliance=0.0,
        friction_mu=np.asarray([0.5]),
        dt=DT,
    )
    analytic, analytic_timing = timed_solve(analytic_kwargs)

    mixed_J = np.zeros((6, 8), dtype=np.float64)
    mixed_J[:3, :3] = np.eye(3)
    mixed_J[3:, 3:6] = np.eye(3)
    mixed_J[3, 0] = 0.2
    mixed_kwargs = dict(
        H=sparse.eye(8, format="csc"),
        rhs=np.zeros(8),
        A=sparse.csr_matrix([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
        ]),
        stretch_offset=np.zeros(2),
        stretch_compliance=np.zeros(2),
        J=sparse.csr_matrix(mixed_J),
        contact_offset=np.asarray([-1.0, 0.2, 0.0, -1.0, 2.0, 0.0]),
        contact_mode=ccdsn.DESAXCE,
        contact_compliance=0.0,
        friction_mu=np.asarray([1.0e4, 0.5]),
        dt=DT,
    )
    mixed, mixed_timing = timed_solve(mixed_kwargs)

    analytic_expected = np.asarray([2.0, -1.0, 0.0])
    analytic_error = float(np.max(np.abs(analytic.contact_force_N - analytic_expected)))
    gates = {
        "analytic_impulse_error_le_1e_9": analytic_error <= 1.0e-9,
        "analytic_natural_map_le_1e_9": analytic.info["natural_map_relative_inf"] <= 1.0e-9,
        "mixed_natural_map_le_1e_9": mixed.info["natural_map_relative_inf"] <= 1.0e-9,
        "both_primal_cones_le_1e_9": max(
            analytic.info["primal_cone_violation_N"],
            mixed.info["primal_cone_violation_N"],
        ) <= 1.0e-9,
        "both_dual_cones_le_1e_9": max(
            analytic.info["dual_cone_violation_m"],
            mixed.info["dual_cone_violation_m"],
        ) <= 1.0e-9,
        "both_complementarity_le_1e_9": max(
            analytic.info["complementarity_inf_Nm"],
            mixed.info["complementarity_inf_Nm"],
        ) <= 1.0e-9,
        "both_stationarity_le_1e_9": max(
            analytic.info["stationarity_inf"], mixed.info["stationarity_inf"]
        ) <= 1.0e-9,
        "both_stretch_residual_le_1e_9": max(
            analytic.info["stretch_residual_inf_m"],
            mixed.info["stretch_residual_inf_m"],
        ) <= 1.0e-9,
        "both_friction_work_nonpositive": max(
            analytic.info["friction_work_Nm"], mixed.info["friction_work_Nm"]
        ) <= 1.0e-12,
    }
    payload = {
        "schema": SCHEMA,
        "generated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "host": platform.node(),
        "owner": ccdsn.owner_guard("CCDSN reference metadata"),
        "source_sha256": sha256(Path(__file__).resolve()),
        "solver_sha256": sha256(Path(ccdsn.__file__).resolve()),
        "desaxce_sha256": sha256(Path(desaxce.__file__).resolve()),
        "test_sha256": sha256(Path(ccdsn.__file__).with_name("test_codex_ccdsn.py")),
        "analytic_sliding": {
            "expected_contact_force_N": analytic_expected,
            "contact_force_N": analytic.contact_force_N,
            "maximum_force_error_N": analytic_error,
            "info": selected_info(analytic.info),
            "timing": analytic_timing,
        },
        "mixed_stick_slide": {
            "contact_force_N": mixed.contact_force_N,
            "info": selected_info(mixed.info),
            "timing": mixed_timing,
        },
        "gates": gates,
        "pass": all(gates.values()),
        "claim_boundary": (
            "small dense exact-law and constrained-response oracle only; no "
            "collision discovery, swept safety, long-chain scale, pile, or GPU claim"
        ),
    }
    return payload


def main() -> int:
    payload = run()
    stamp = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    directory = OUT / f"run_{stamp}_{os.getpid()}"
    path = directory / "result.json"
    atomic_json(path, payload)
    print(path)
    print(f"pass={payload['pass']}")
    print(f"sha256={sha256(path)}")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
