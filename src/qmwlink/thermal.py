"""Thermal occupation, critical transmissivity, BBR-equivalent field.

(FR-PHY-1, FR-PHY-2, FR-PHY-3, TSD §3, §7.4)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qmwlink.constants import EPS0, ETA0, K_B, C, H

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


def effective_aperture_m2(f_hz: ArrayLike) -> FloatArray:
    """Short-dipole effective aperture ``A_e = 3 lambda^2 / (8 pi)`` [m^2]. (FR-PHY-3, TSD §7.4)

    Antenna-equivalence convention (FSD §4): a modeling convention for comparing receivers,
    not derived physics.
    """
    f, _ = _validate(f_hz, 0.0)
    lam = C / f
    return np.asarray(3.0 * lam**2 / (8.0 * np.pi), dtype=np.float64)


def s_e_bbr(f_hz: ArrayLike, T_env_k: ArrayLike, *, planck: bool = True) -> FloatArray:
    """BBR-equivalent rms single-sided field density ``sqrt(n hf eta0 / A_e)`` [V/m/sqrt(Hz)].

    Mode-occupation route. ``planck=True`` uses the Bose-Einstein ``n``; ``planck=False`` the
    Rayleigh-Jeans ``k_B T / (h f)``. (FR-PHY-3, TSD §7.4)
    """
    n = n_th(f_hz, T_env_k) if planck else n_th_rj(f_hz, T_env_k)
    f, _ = _validate(f_hz, T_env_k)
    return np.asarray(np.sqrt(n * H * f * ETA0 / effective_aperture_m2(f)), dtype=np.float64)


def s_e_bbr_energy_density_rj(f_hz: ArrayLike, T_env_k: ArrayLike) -> FloatArray:
    """RJ single-component BBR field density ``sqrt(8 pi f^2 k_B T / (3 eps0 c^3))``.

    Independent energy-density route; equals ``s_e_bbr(..., planck=False)`` algebraically.
    (FR-PHY-3, TSD §7.4)
    """
    f, T = _validate(f_hz, T_env_k)
    return np.asarray(np.sqrt(8.0 * np.pi * f**2 * K_B * T / (3.0 * EPS0 * C**3)), dtype=np.float64)
