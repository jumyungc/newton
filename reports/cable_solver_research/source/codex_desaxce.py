"""Dense reference solver for the non-associated De Saxce Coulomb law.

This is a small-oracle implementation for research and unit tests.  It solves
the natural map

    F(r) = r - projection_K(r - gamma * D(u)) = 0,
    u = b + W r,

where ``D(u) = (u_n + mu ||u_t||, u_t)`` is the De Saxce correction and
``K = {(r_n,r_t): r_n >= 0, ||r_t|| <= mu r_n}``.  The generalized Jacobian
is analytic away from cone apexes and zero tangential velocity.  A monotone
line search and projected fixed-point recovery make failure explicit.

The routine is intentionally dense and bounded.  It is the correctness oracle
for a future matrix-free contact solve, not the large-pile implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np


DOC = Path(__file__).with_name("cable_research.md")
EXPECTED_OWNER = (
    "OWNER: Team Codex-Remote \u2014 2026-07-05T11:40Z \u2014 "
    "coupled long-chain/contact solver invention round"
)
DEFAULT_DIRECT_CONTACT_LIMIT = 256


@dataclass(frozen=True)
class DeSaxceResult:
    impulse: np.ndarray
    velocity: np.ndarray
    info: dict[str, Any]


def owner_guard(where: str) -> str:
    line = DOC.read_text(encoding="utf-8").splitlines()[0]
    if line != EXPECTED_OWNER:
        raise RuntimeError(f"owner mismatch before {where}: {line!r}")
    return line


def _validate_mu(mu: np.ndarray, contacts: int) -> np.ndarray:
    value = np.asarray(mu, dtype=np.float64).reshape(-1)
    if len(value) != contacts:
        raise ValueError(f"expected {contacts} friction values, got {len(value)}")
    if not np.all(np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError("friction values must be finite and nonnegative")
    return value


def project_coulomb_with_jacobian(
    values: np.ndarray,
    mu: np.ndarray,
    *,
    apex_epsilon: float = 1.0e-14,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Project product Coulomb cones and return a generalized Jacobian."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(flat) % 3:
        raise ValueError("contact vector length must be divisible by three")
    contacts = len(flat) // 3
    mu = _validate_mu(mu, contacts)
    z = flat.reshape(contacts, 3)
    out = np.zeros_like(z)
    jacobian = np.zeros((3 * contacts, 3 * contacts), dtype=np.float64)
    regions: list[str] = []

    for contact, (value, friction) in enumerate(zip(z, mu, strict=True)):
        normal = float(value[0])
        tangent = value[1:]
        tangent_norm = float(np.linalg.norm(tangent))
        block = slice(3 * contact, 3 * contact + 3)
        if friction <= apex_epsilon:
            if normal > 0.0:
                out[contact, 0] = normal
                jacobian[block, block] = np.diag((1.0, 0.0, 0.0))
                regions.append("frictionless_positive")
            else:
                regions.append("polar")
            continue

        if normal >= 0.0 and tangent_norm <= friction * normal:
            out[contact] = value
            jacobian[block, block] = np.eye(3)
            regions.append("inside")
            continue
        if normal + friction * tangent_norm <= 0.0:
            regions.append("polar")
            continue

        denominator = 1.0 + friction * friction
        projected_normal = (normal + friction * tangent_norm) / denominator
        out[contact, 0] = projected_normal
        if tangent_norm <= apex_epsilon:
            # A valid Clarke element at the apex.  Newton may need the
            # projected recovery path here; no arbitrary tangent direction is
            # introduced.
            jacobian[block, block] = np.diag((1.0 / denominator, 0.0, 0.0))
            regions.append("boundary_apex")
            continue

        tangent_direction = tangent / tangent_norm
        out[contact, 1:] = friction * projected_normal * tangent_direction
        projector_tangent = np.eye(2) - np.outer(tangent_direction, tangent_direction)
        derivative = np.zeros((3, 3), dtype=np.float64)
        derivative[0, 0] = 1.0 / denominator
        derivative[0, 1:] = friction * tangent_direction / denominator
        derivative[1:, 0] = friction * tangent_direction / denominator
        derivative[1:, 1:] = (
            friction * friction / denominator
            * np.outer(tangent_direction, tangent_direction)
            + friction * projected_normal / tangent_norm * projector_tangent
        )
        jacobian[block, block] = derivative
        regions.append("boundary")

    return out.reshape(-1), jacobian, regions


def desaxce_correction_with_jacobian(
    velocity: np.ndarray,
    mu: np.ndarray,
    *,
    tangent_epsilon: float = 1.0e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Return De Saxce corrected velocity and one generalized Jacobian."""
    flat = np.asarray(velocity, dtype=np.float64).reshape(-1)
    if len(flat) % 3:
        raise ValueError("contact vector length must be divisible by three")
    contacts = len(flat) // 3
    mu = _validate_mu(mu, contacts)
    value = flat.reshape(contacts, 3)
    corrected = value.copy()
    jacobian = np.eye(3 * contacts, dtype=np.float64)
    for contact, friction in enumerate(mu):
        tangent = value[contact, 1:]
        tangent_norm = float(np.linalg.norm(tangent))
        corrected[contact, 0] += friction * tangent_norm
        if tangent_norm > tangent_epsilon:
            jacobian[3 * contact, 3 * contact + 1:3 * contact + 3] = (
                friction * tangent / tangent_norm
            )
    return corrected.reshape(-1), jacobian


def natural_map(
    impulse: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    mu: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Evaluate the natural map and an analytic generalized Jacobian."""
    impulse = np.asarray(impulse, dtype=np.float64).reshape(-1)
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    velocity = b + W @ impulse
    corrected, correction_jacobian = desaxce_correction_with_jacobian(velocity, mu)
    projection_input = impulse - gamma * corrected
    projected, projection_jacobian, regions = project_coulomb_with_jacobian(
        projection_input, mu
    )
    residual = impulse - projected
    jacobian = np.eye(len(impulse)) - projection_jacobian @ (
        np.eye(len(impulse)) - gamma * correction_jacobian @ W
    )
    return residual, {
        "jacobian": jacobian,
        "velocity": velocity,
        "corrected_velocity": corrected,
        "projection_input": projection_input,
        "projected": projected,
        "regions": regions,
    }


def cone_violation(impulse: np.ndarray, mu: np.ndarray) -> float:
    value = np.asarray(impulse, dtype=np.float64).reshape(-1, 3)
    mu = _validate_mu(mu, len(value))
    if not len(value):
        return 0.0
    violation = np.maximum(
        -value[:, 0], np.linalg.norm(value[:, 1:], axis=1) - mu * value[:, 0]
    )
    return float(np.max(np.maximum(violation, 0.0), initial=0.0))


def dual_cone_violation(corrected_velocity: np.ndarray, mu: np.ndarray) -> float:
    value = np.asarray(corrected_velocity, dtype=np.float64).reshape(-1, 3)
    mu = _validate_mu(mu, len(value))
    if not len(value):
        return 0.0
    violation = mu * np.linalg.norm(value[:, 1:], axis=1) - value[:, 0]
    return float(np.max(np.maximum(violation, 0.0), initial=0.0))


def solve_desaxce_dense(
    W: np.ndarray,
    b: np.ndarray,
    mu: np.ndarray,
    *,
    initial_impulse: np.ndarray | None = None,
    gamma: float | None = None,
    tolerance: float = 1.0e-10,
    max_iterations: int = 80,
    max_line_search: int = 24,
    direct_contact_limit: int = DEFAULT_DIRECT_CONTACT_LIMIT,
    fail_closed: bool = True,
) -> DeSaxceResult:
    """Solve a bounded dense De Saxce contact problem by semismooth Newton."""
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if W.shape != (len(b), len(b)) or len(b) % 3:
        raise ValueError("W must be square and match a 3-vector per contact")
    contacts = len(b) // 3
    if contacts > direct_contact_limit:
        raise ValueError(
            f"{contacts} contacts exceed dense limit {direct_contact_limit}"
        )
    mu = _validate_mu(mu, contacts)
    if not np.all(np.isfinite(W)) or not np.all(np.isfinite(b)):
        raise ValueError("W and b must be finite")
    W = 0.5 * (W + W.T)
    eigenvalues = np.linalg.eigvalsh(W)
    if len(eigenvalues) and eigenvalues[0] <= 0.0:
        raise ValueError("W must be symmetric positive definite")
    spectral_radius = float(eigenvalues[-1]) if len(eigenvalues) else 1.0
    if gamma is None:
        gamma = 1.0 / max(spectral_radius, 1.0e-12)
    if not (math.isfinite(gamma) and gamma > 0.0):
        raise ValueError("gamma must be finite and positive")
    if not (math.isfinite(tolerance) and tolerance > 0.0):
        raise ValueError("tolerance must be finite and positive")

    if initial_impulse is None:
        unconstrained = np.linalg.solve(W, -b) if len(b) else np.empty(0)
        impulse, _, _ = project_coulomb_with_jacobian(unconstrained, mu)
    else:
        impulse = np.asarray(initial_impulse, dtype=np.float64).reshape(-1).copy()
        if len(impulse) != len(b) or not np.all(np.isfinite(impulse)):
            raise ValueError("initial impulse must be finite and match b")

    history: list[dict[str, Any]] = []
    converged = False
    recovery_steps = 0
    linear_solve_fallbacks = 0
    final_aux: dict[str, Any] = {}
    for iteration in range(max_iterations + 1):
        residual, aux = natural_map(impulse, W, b, mu, gamma)
        final_aux = aux
        residual_inf = float(np.max(np.abs(residual), initial=0.0))
        residual_l2 = float(np.linalg.norm(residual))
        scale = max(
            1.0,
            float(np.max(np.abs(impulse), initial=0.0)),
            gamma * float(np.max(np.abs(aux["corrected_velocity"]), initial=0.0)),
        )
        relative = residual_inf / scale
        record: dict[str, Any] = {
            "iteration": iteration,
            "residual_inf": residual_inf,
            "relative_residual_inf": relative,
            "residual_l2": residual_l2,
            "regions": list(aux["regions"]),
        }
        history.append(record)
        if residual_inf <= tolerance * scale:
            converged = True
            break
        if iteration == max_iterations:
            break

        jacobian = np.asarray(aux["jacobian"], dtype=np.float64)
        try:
            direction = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            direction, *_ = np.linalg.lstsq(jacobian, -residual, rcond=1.0e-12)
            linear_solve_fallbacks += 1
        if not np.all(np.isfinite(direction)):
            direction = aux["projected"] - impulse
            recovery_steps += 1

        merit = 0.5 * residual_l2 * residual_l2
        accepted = False
        best: tuple[float, np.ndarray, float] | None = None
        alpha = 1.0
        for line_iteration in range(max_line_search):
            trial = impulse + alpha * direction
            trial_residual, _ = natural_map(trial, W, b, mu, gamma)
            trial_merit = 0.5 * float(trial_residual @ trial_residual)
            if math.isfinite(trial_merit):
                candidate = (trial_merit, trial, alpha)
                if best is None or candidate[0] < best[0]:
                    best = candidate
                if trial_merit <= (1.0 - 1.0e-4 * alpha) * merit:
                    impulse = trial
                    record["accepted_alpha"] = alpha
                    record["step"] = "semismooth_newton"
                    accepted = True
                    break
            alpha *= 0.5
        if accepted:
            continue

        # Projected fixed-point recovery is the exact natural-map iteration.
        # Use the best of it and any finite Newton trial, but require strict
        # residual decrease; otherwise fail rather than report a false root.
        projected = np.asarray(aux["projected"], dtype=np.float64)
        projected_residual, _ = natural_map(projected, W, b, mu, gamma)
        projected_merit = 0.5 * float(projected_residual @ projected_residual)
        candidates = [(projected_merit, projected, 0.0)]
        if best is not None:
            candidates.append(best)
        chosen = min(candidates, key=lambda item: item[0])
        if not math.isfinite(chosen[0]) or chosen[0] >= merit:
            record["step"] = "stagnated"
            break
        impulse = chosen[1]
        record["accepted_alpha"] = chosen[2]
        record["step"] = "projected_recovery"
        recovery_steps += 1

    velocity = b + W @ impulse
    corrected, _ = desaxce_correction_with_jacobian(velocity, mu)
    final_residual, final_aux = natural_map(impulse, W, b, mu, gamma)
    residual_inf = float(np.max(np.abs(final_residual), initial=0.0))
    scale = max(
        1.0,
        float(np.max(np.abs(impulse), initial=0.0)),
        gamma * float(np.max(np.abs(corrected), initial=0.0)),
    )
    relative = residual_inf / scale
    impulse3 = impulse.reshape(-1, 3)
    velocity3 = velocity.reshape(-1, 3)
    corrected3 = corrected.reshape(-1, 3)
    complementarity = (
        float(np.max(np.abs(np.sum(impulse3 * corrected3, axis=1)), initial=0.0))
        if contacts else 0.0
    )
    friction_work = float(np.sum(impulse3[:, 1:] * velocity3[:, 1:]))
    info = {
        "method": "dense De Saxce natural-map semismooth Newton",
        "contacts": contacts,
        "iterations": max(0, len(history) - 1),
        "gamma": gamma,
        "spectral_radius": spectral_radius,
        "converged": converged,
        "natural_map_inf": residual_inf,
        "natural_map_relative_inf": relative,
        "cone_violation": cone_violation(impulse, mu),
        "dual_cone_violation": dual_cone_violation(corrected, mu),
        "complementarity_inf": complementarity,
        "friction_work": friction_work,
        "recovery_steps": recovery_steps,
        "linear_solve_fallbacks": linear_solve_fallbacks,
        "final_regions": list(final_aux.get("regions", [])),
        "history": history,
        "claim_boundary": (
            "small dense correctness oracle; no swept-contact or large-pile claim"
        ),
    }
    if fail_closed and not converged:
        raise RuntimeError(
            "De Saxce solve did not converge: "
            f"natural-map relative inf={relative:.3e}"
        )
    return DeSaxceResult(impulse, velocity, info)
