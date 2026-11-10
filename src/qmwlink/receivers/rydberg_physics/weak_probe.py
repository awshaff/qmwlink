"""Weak-probe engine: tridiagonal coherence model, closed form over velocity. (FR-LIN-2, TSD §14.4)

For ``Omega_p << Gamma_2``, ``rho_11 ~ 1`` and ``x = (rho_21, rho_31, rho_41)`` obeys
``dx/dt = i(M x - s)``. Everything is evaluated in closed form (3x3 symmetric tridiagonal
inverse) and broadcast over velocity classes; there is no Python loop over ``v``.

This engine is written independently of ``lindblad.py`` (agent.md §3): their agreement in the
weak-probe limit is the main correctness check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qmwlink.receivers.rydberg_physics.scheme import Drive, LadderScheme

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

DOPPLER_SCALE = 1e-6
"""rad/m x m/s -> rad/us (TSD §14.2)."""


@dataclass(frozen=True)
class _Diag:
    """Diagonal of ``M`` per velocity class and the two couplings ``b1 = -Oc/2``, ``b2 = -Om/2``."""

    a1: ComplexArray
    a2: ComplexArray
    a3: ComplexArray
    b1: float
    b2: float


def _diag(scheme: LadderScheme, drive: Drive, v: ArrayLike) -> _Diag:
    vv = np.asarray(v, dtype=np.float64)
    d2 = drive.delta_p - DOPPLER_SCALE * drive.k_p * vv
    d3 = d2 + drive.delta_c + DOPPLER_SCALE * drive.k_c * vv
    a1 = (d2 + 1j * scheme.coherence_rate(2)).astype(np.complex128)
    a2 = (d3 + 1j * scheme.coherence_rate(3)).astype(np.complex128)
    if scheme.n_levels == 4:
        a3 = (d3 + drive.delta_m + 1j * scheme.coherence_rate(4)).astype(np.complex128)
        b2 = -0.5 * drive.omega_m
    else:  # drop the last row/column: decoupled dummy level
        a3 = np.ones_like(a2)
        b2 = 0.0
    return _Diag(a1, a2, a3, -0.5 * drive.omega_c, b2)


def m_matrix(scheme: LadderScheme, drive: Drive, v: float = 0.0) -> ComplexArray:
    """The matrix ``M`` of TSD §14.4 for one velocity (3x3; 2x2 for a 3-level scheme)."""
    d = _diag(scheme, drive, v)
    m = np.array([[d.a1, d.b1, 0.0], [d.b1, d.a2, d.b2], [0.0, d.b2, d.a3]], dtype=np.complex128)
    return m if scheme.n_levels == 4 else m[:2, :2]


def steady_state(scheme: LadderScheme, drive: Drive, v: ArrayLike) -> ComplexArray:
    """Steady-state coherences ``x_0 = M^{-1} s``, shape ``v.shape + (3,)``. (TSD §14.4)

    ``x_0[..., 0] = rho_21`` is the continued fraction of the TSD; ``rho_41 = 0`` for 3 levels.
    """
    d = _diag(scheme, drive, v)
    s1 = 0.5 * drive.omega_p
    if d.b1 == 0.0:  # Omega_c = 0: the probe transition decouples (two-level atom)
        zero = np.zeros_like(d.a1)
        return np.stack([s1 / d.a1, zero, zero], axis=-1)
    minor = d.a2 * d.a3 - d.b2**2
    det = d.a1 * minor - d.b1**2 * d.a3
    x = np.stack([s1 * minor / det, -s1 * d.b1 * d.a3 / det, s1 * d.b1 * d.b2 / det], axis=-1)
    return np.asarray(x, dtype=np.complex128)


def rho21(scheme: LadderScheme, drive: Drive, v: ArrayLike) -> ComplexArray:
    """``rho_21`` per velocity class: the continued fraction of TSD §14.4.

    Evaluated as ``s_1 (a_2 a_3 - b_2^2) / det M``, the same fraction cleared of its inner
    denominators, so ideal EIT (``a_2 a_3 = b_2^2``) gives exactly 0 instead of ``1/inf``.
    """
    d = _diag(scheme, drive, v)
    s1 = 0.5 * drive.omega_p
    if d.b1 == 0.0:
        return np.asarray(s1 / d.a1, dtype=np.complex128)
    minor = d.a2 * d.a3 - d.b2**2
    return np.asarray(s1 * minor / (d.a1 * minor - d.b1**2 * d.a3), dtype=np.complex128)


def transfer_function(
    scheme: LadderScheme, drive: Drive, v: ArrayLike, omega: ArrayLike
) -> ComplexArray:
    """Small-signal response of ``rho_21`` to ``Omega_m``: ``H(i w)``. (TSD §14.4)

    ``H(i w) = e_1^T (i w I - i M)^{-1} (i M_1 x_0) = -e_1^T (M - w)^{-1} M_1 x_0`` with
    ``M_1 = dM/dOmega_m``. Returns shape ``omega.shape + v.shape``; ``H(0) = d rho_21/d Omega_m``.
    """
    if scheme.n_levels != 4:
        raise ValueError("the microwave transfer function needs n_levels = 4")
    d = _diag(scheme, drive, v)
    x0 = steady_state(scheme, drive, v)
    w = np.asarray(omega, dtype=np.float64)
    if d.b1 == 0.0:  # rho_21 does not depend on Omega_m without the coupling field
        return np.zeros(w.shape + d.a1.shape, dtype=np.complex128)
    w = w.reshape(w.shape + (1,) * d.a1.ndim)
    a1, a2, a3 = d.a1 - w, d.a2 - w, d.a3 - w
    det = a1 * (a2 * a3 - d.b2**2) - d.b1**2 * a3
    inv12 = -d.b1 * a3 / det
    inv13 = d.b1 * d.b2 / det
    # M_1 x_0 = (0, -x_3/2, -x_2/2)
    return np.asarray(0.5 * (inv12 * x0[..., 2] + inv13 * x0[..., 1]), dtype=np.complex128)


def evolve(
    scheme: LadderScheme,
    drive: Drive,
    t: ArrayLike,
    v: float = 0.0,
    x_init: ArrayLike | None = None,
) -> ComplexArray:
    """Solution of ``dx/dt = i(M x - s)`` at times ``t`` [us] for one velocity. (TSD §14.4)

    ``x(t) = x_0 + exp(i M t)(x(0) - x_0)`` by eigendecomposition; returns shape ``t.shape + (n,)``
    with ``n = n_levels - 1``. Stable: the eigenvalues of ``i M`` have negative real part.
    """
    m = m_matrix(scheme, drive, v)
    n = m.shape[0]
    x0 = steady_state(scheme, drive, v)[..., :n].reshape(n)
    start = np.zeros(n, dtype=np.complex128) if x_init is None else np.asarray(x_init, complex)
    if start.shape != (n,):
        raise ValueError(f"x_init must have shape ({n},)")
    lam, vec = np.linalg.eig(1j * m)
    coef = np.linalg.solve(vec, start - x0)
    tt = np.asarray(t, dtype=np.float64)
    phases = np.exp(tt[..., None] * lam)
    return np.asarray(x0 + (phases * coef) @ vec.T, dtype=np.complex128)
