"""Two-mode Gaussian states: TMSV and TMST covariance matrices. (FR-GAU-1, TSD §5.1)

Vacuum variance 1, ordering (x_A, p_A, x_B, p_B), symplectic form Omega = (+) [[0,1],[-1,0]].
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

PHYSICALITY_TOL = 1e-12
"""Smallest eigenvalue of V + i Omega must be >= -PHYSICALITY_TOL (TSD §5.1)."""

OMEGA: FloatArray = np.kron(np.eye(2), np.array([[0.0, 1.0], [-1.0, 0.0]]))
_Z = np.diag([1.0, -1.0])


def two_mode_ac(r: ArrayLike, n_jpa: ArrayLike = 0.0) -> tuple[FloatArray, FloatArray]:
    """Return ``(a, c) = ((2 n+1) cosh 2r, (2 n+1) sinh 2r)``; TMSV when ``n = 0``.

    (FR-GAU-1, TSD §5.1) Note ``a^2 - c^2 = (2 n + 1)^2``.
    """
    rr = np.asarray(r, dtype=np.float64)
    n = np.asarray(n_jpa, dtype=np.float64)
    if np.any(rr < 0):
        raise ValueError("r must be >= 0")
    if np.any(n < 0):
        raise ValueError("n_jpa must be >= 0")
    s = 2.0 * n + 1.0
    return np.asarray(s * np.cosh(2.0 * rr)), np.asarray(s * np.sinh(2.0 * rr))


def _cm(a: FloatArray, c: FloatArray) -> FloatArray:
    a, c = np.broadcast_arrays(a, c)
    out = np.zeros((*a.shape, 4, 4))
    out[..., 0:2, 0:2] = a[..., None, None] * np.eye(2)
    out[..., 2:4, 2:4] = a[..., None, None] * np.eye(2)
    out[..., 0:2, 2:4] = c[..., None, None] * _Z
    out[..., 2:4, 0:2] = c[..., None, None] * _Z
    return out


def tmsv_cm(r: ArrayLike) -> FloatArray:
    """Two-mode squeezed vacuum CM, shape ``(..., 4, 4)``. (FR-GAU-1, TSD §5.1)"""
    a, c = two_mode_ac(r)
    return _cm(a, c)


def tmst_cm(r: ArrayLike, n_jpa: ArrayLike) -> FloatArray:
    """Two-mode squeezed thermal CM with JPA occupation ``n_jpa``. (FR-GAU-1, TSD §5.1)"""
    a, c = two_mode_ac(r, n_jpa)
    return _cm(a, c)


def min_physicality_eigenvalue(V: ArrayLike) -> FloatArray:
    """Smallest eigenvalue of the Hermitian matrix ``V + i Omega``. (FR-GAU-1, TSD §5.1)"""
    v = np.asarray(V, dtype=np.float64)
    if v.shape[-2:] != (4, 4):
        raise ValueError("V must have shape (..., 4, 4)")
    return np.asarray(np.linalg.eigvalsh(v + 1j * OMEGA)[..., 0], dtype=np.float64)


def is_physical(V: ArrayLike, tol: float = PHYSICALITY_TOL) -> NDArray[np.bool_]:
    """Physicality check ``V + i Omega >= 0`` (up to ``tol``). (FR-GAU-1, TSD §5.1)"""
    return np.asarray(min_physicality_eigenvalue(V) >= -tol)
