"""Thermal occupation and critical transmissivity. (FR-PHY-1, FR-PHY-2, TSD §3)"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qmwlink.constants import K_B, H

FloatArray = NDArray[np.float64]


def _validate(f_hz: ArrayLike, T_env_k: ArrayLike) -> tuple[FloatArray, FloatArray]:
    f = np.asarray(f_hz, dtype=np.float64)
    T = np.asarray(T_env_k, dtype=np.float64)
    if not np.all(np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("f_hz must be finite and > 0 (frequency in Hz)")
    if not np.all(np.isfinite(T)) or np.any(T < 0):
        raise ValueError("T_env_k must be finite and >= 0 (temperature in K)")
    return f, T


def n_th(f_hz: ArrayLike, T_env_k: ArrayLike) -> FloatArray:
    """Bose-Einstein occupation ``1/(exp(hf/kT) - 1)``; exactly 0 at T = 0. (FR-PHY-1, TSD §3.1)"""
    f, T = _validate(f_hz, T_env_k)
    f, T = np.broadcast_arrays(f, T)
    out = np.zeros(f.shape, dtype=np.float64)
    warm = T > 0
    # Denormal T makes K_B*T underflow to 0 (x = inf) and large x overflows expm1; both
    # correctly give occupation 0.
    with np.errstate(over="ignore", divide="ignore"):
        x = H * f[warm] / (K_B * T[warm])
        out[warm] = 1.0 / np.expm1(x)
    return out


def n_th_rj(f_hz: ArrayLike, T_env_k: ArrayLike) -> FloatArray:
    """Rayleigh-Jeans approximation ``k_B T / (h f)``. (FR-PHY-1, TSD §3.1)"""
    f, T = _validate(f_hz, T_env_k)
    return np.asarray(K_B * T / (H * f), dtype=np.float64)


def eta_star(f_hz: ArrayLike, T_env_k: ArrayLike) -> FloatArray:
    """Critical transmissivity ``n/(1+n)``; entanglement-breaking for eta <= eta*.

    (FR-PHY-2, TSD §3.2)

    Prefer :func:`one_minus_eta_star` when the distance to 1 matters (TSD §13 rule 3).
    """
    n = n_th(f_hz, T_env_k)
    return np.asarray(n / (1.0 + n), dtype=np.float64)


def one_minus_eta_star(f_hz: ArrayLike, T_env_k: ArrayLike) -> FloatArray:
    """Window ``1 - eta* = 1/(1+n)`` computed directly, without subtraction. (FR-PHY-2, TSD §3.2)"""
    n = n_th(f_hz, T_env_k)
    return np.asarray(1.0 / (1.0 + n), dtype=np.float64)
