# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver-independent, single-branch standard-linear-solid constitutive functions.

An equilibrium spring ``k`` acts in parallel with a Maxwell branch of stiffness
``ratio * k`` and viscosity ``ratio * k * tau``. Given input viscous deformation
``z``, backward Euler adds ``k * sls_gain(ratio, tau, dt) * (e - z)`` to the
equilibrium force. Keep ``z`` fixed during a solve and commit it once afterward.

The caller supplies deformation coordinates, equilibrium stiffness, and State
storage, and decides how an inactive channel tracks the pose.
These functions contain no geometry, solver multipliers, or allocation.
Existing velocity damping is a separate parallel dashpot; set it to zero for a
pure three-parameter SLS. This module does not implement a multi-branch Maxwell model.
"""

import warp as wp


@wp.func
def sls_step_fraction(tau: float, dt: float) -> float:
    """Return the backward-Euler relaxation fraction for ``tau >= 0`` and ``dt > 0`` [s]."""
    return dt / (tau + dt)


@wp.func
def sls_gain(ratio: float, tau: float, dt: float) -> float:
    """Return transient step stiffness per unit equilibrium stiffness; ``ratio >= 0``.

    With input history ``z`` frozen, the branch force is ``gain * k * (e - z)``
    and its tangent with respect to ``e`` is ``gain * k``. A zero ratio or time
    disables this force contribution. Parameters must be finite and validated by
    the authoring layer; this function does not clamp invalid material inputs.
    """
    return ratio * (tau / (tau + dt))


@wp.func
def sls_update_history(deformation: float, history: float, step_fraction: float) -> float:
    """Commit viscous deformation from the accepted pose and a fixed relaxation fraction.

    ``deformation`` and ``history`` have the same units (strain, extension, or a
    bending coordinate). Use :func:`sls_step_fraction` for an active branch. A caller may
    pass one to initialize/track the accepted deformation for an inactive channel;
    that lifecycle policy is distinct from advancing an active material in time.
    """
    if step_fraction == 1.0:
        return deformation
    # Compute the small increment directly; 1 - decay can round to zero for long relaxation times.
    return history + step_fraction * (deformation - history)
