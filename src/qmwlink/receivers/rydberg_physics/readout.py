"""Readout: absorption, transmission, slope, sensitivity, linear range. (FR-LIN-6, TSD §14.7-14.9)

All functions here are SI in, SI out (the unit conversion from the core rad/us happens in
``model.py``). ``S_E`` is an **rms** single-sided field density; ``E_1dB`` is a **peak** field.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from qmwlink.constants import EPS0, H

FloatArray = NDArray[np.float64]

HBAR = H / (2.0 * np.pi)


def absorption_prefactor(k_p: float, n0_per_m3: float, d12_cm: float) -> float:
    """``2 k_p N_0 d_12^2/(eps_0 hbar)`` [rad/s/m]: ``alpha = prefactor (-Im rho_21)/Omega_p``.

    (TSD §14.7; ``Omega_p`` in rad/s.)
    """
    if k_p <= 0 or n0_per_m3 < 0 or d12_cm <= 0:
        raise ValueError("k_p and d12 must be > 0 and n0 >= 0")
    return 2.0 * k_p * n0_per_m3 * d12_cm**2 / (EPS0 * HBAR)


def absorption_coefficient(
    im_rho21: float, omega_p_rad_s: float, k_p: float, n0_per_m3: float, d12_cm: float
) -> float:
    """Probe intensity absorption coefficient ``alpha`` [1/m]. (TSD §14.7)

    ``alpha = 2 k_p N_0 d_12^2 / (eps_0 hbar Omega_p) (-Im <rho_21>)`` with the Doppler-averaged
    ``rho_21 = <2|rho|1>``.
    """
    if omega_p_rad_s <= 0:
        raise ValueError("omega_p must be > 0")
    return absorption_prefactor(k_p, n0_per_m3, d12_cm) * (-im_rho21) / omega_p_rad_s


def cycling_dipole_cm(gamma2_rad_s: float, k_p: float) -> float:
    """``d_12 = sqrt(3 pi eps_0 hbar Gamma_2 / k_p^3)``: closed two-level cycling transition.

    With it the resonant weak-probe cross-section is ``3 lambda^2 / (2 pi)`` (TSD §14.7 check).
    """
    return float(np.sqrt(3.0 * np.pi * EPS0 * HBAR * gamma2_rad_s / k_p**3))


def transmitted_power(p_in_w: float, alpha_per_m: float, length_m: float) -> float:
    """Beer-Lambert ``P_out = P_in exp(-alpha L)`` (weak probe, thin uniform cell)."""
    if p_in_w < 0 or length_m < 0:
        raise ValueError("p_in_w and length_m must be >= 0")
    return float(p_in_w * np.exp(-alpha_per_m * length_m))


def slope_kappa(p_out_w: float, length_m: float, dalpha_domega_m: float, d34_cm: float) -> float:
    """``kappa = dP_out/dE_sig = -P_out L (d alpha/d Omega_m) d_34/hbar`` [W per V/m]. (TSD §14.7)

    ``dalpha_domega_m`` in 1/m per rad/s. ``kappa`` is linear, so it is the same per peak or per
    rms field.
    """
    return float(-p_out_w * length_m * dalpha_domega_m * d34_cm / HBAR)


def power_noise_psd(p_out_w: float, nu_p_hz: float, eta_d: float, nep_w_per_rthz: float) -> float:
    """Single-sided optical power noise PSD ``S_P = 2 h nu_p P_out/eta_d + NEP^2`` [W^2/Hz]."""
    if not 0 < eta_d <= 1:
        raise ValueError("eta_d must be in (0, 1]")
    if p_out_w < 0 or nu_p_hz <= 0 or nep_w_per_rthz < 0:
        raise ValueError("p_out_w, nep must be >= 0 and nu_p_hz > 0")
    return float(2.0 * H * nu_p_hz * p_out_w / eta_d + nep_w_per_rthz**2)


def sensitivity_rms(s_p_w2_per_hz: float, kappa: float) -> float:
    """Noise-equivalent field ``S_E = sqrt(S_P)/|kappa|`` [V/m/sqrt(Hz)], **rms**. (TSD §14.8)

    The output at the IF has rms amplitude ``|kappa| E_sig,rms``; SNR = 1 in 1 Hz gives
    ``kappa^2 E_rms^2 = S_P``. No factor ``sqrt 2``.
    """
    if kappa == 0.0 or not np.isfinite(kappa):
        raise ValueError("kappa must be finite and non-zero (no small-signal response)")
    return float(np.sqrt(s_p_w2_per_hz) / abs(kappa))


# ---- linear range (TSD §14.9) -------------------------------------------------------------------
_GL_NODES = 64
_XG, _WG = np.polynomial.legendre.leggauss(_GL_NODES)
_THETA = np.pi * (_XG + 1.0)  # nodes on [0, 2 pi]
_WTHETA = np.pi * _WG
_HALF = _GL_NODES // 2  # nodes are symmetric about pi and cos(2 pi - t) = cos t
_DB_CAP = 300.0


def first_harmonic(
    p_ss: Callable[[FloatArray], FloatArray], omega_lo: float, omega_sig: float
) -> float:
    """``A_1 = (1/pi) int_0^{2 pi} P_ss(O_LO + O_sig cos t) cos t dt`` (Gauss-Legendre, 64 nodes).

    The integrand is symmetric about ``t = pi``, so ``P_ss`` is evaluated at the 32 distinct
    arguments only. ``p_ss`` must be vectorized over ``Omega_m``.
    """
    c = np.cos(_THETA[:_HALF])
    vals = np.asarray(p_ss(omega_lo + omega_sig * c), dtype=np.float64)
    return float(2.0 * np.sum(_WTHETA[:_HALF] * vals * c) / np.pi)


def compression(
    p_ss: Callable[[FloatArray], FloatArray], omega_lo: float, omega_sig: float, kappa_omega: float
) -> float:
    """``c = A_1 / (kappa_Omega Omega_sig)`` with ``kappa_Omega = dP_out/dOmega_m`` at the LO."""
    return first_harmonic(p_ss, omega_lo, omega_sig) / (kappa_omega * omega_sig)


@dataclass(frozen=True)
class LinearRange:
    """1 dB point of the first-harmonic response. ``omega_sig_1db`` is ``NaN`` with a ``reason``
    when ``|20 log10 c|`` stays below 1 dB for every signal amplitude up to the LO."""

    omega_sig_1db: float
    case: Literal["compression", "expansion", "none"]
    reason: str = ""


def linear_range(
    p_ss: Callable[[FloatArray], FloatArray],
    omega_lo: float,
    kappa_omega: float,
    *,
    rel_min: float = 1e-3,
    n_scan: int = 25,
) -> LinearRange:
    """Smallest ``Omega_sig`` with ``|20 log10 c| = 1 dB`` (compression or expansion). (TSD §14.9)

    Quasi-static (valid for ``delta << B_3dB``; the caller asserts that). The search covers
    ``Omega_sig`` in ``[rel_min, 1] x Omega_LO``: the envelope picture ``Omega_m(t) = Omega_LO +
    Omega_sig cos(.)`` of TSD §14.7 assumes ``E_sig << E_LO`` and has no meaning beyond the LO.
    Log scan for the first crossing, then ``brentq`` on ``log Omega_sig``. Units of ``Omega`` are
    the caller's (the same for ``omega_lo``, the result and ``1/kappa_omega``).
    """
    if omega_lo <= 0 or kappa_omega == 0.0 or not np.isfinite(kappa_omega):
        raise ValueError("omega_lo must be > 0 and kappa_omega finite and non-zero")

    def excess_db(log_w: float) -> float:
        c = compression(p_ss, omega_lo, float(np.exp(log_w)), kappa_omega)
        if not c > 0:  # first harmonic vanished or changed sign: far beyond 1 dB compression
            return _DB_CAP
        return min(abs(20.0 * float(np.log10(c))) - 1.0, _DB_CAP)

    grid = np.log(np.geomspace(rel_min * omega_lo, omega_lo, n_scan))
    if excess_db(float(grid[0])) >= 0:
        return LinearRange(
            float("nan"), "none", "already beyond 1 dB at the smallest scanned signal"
        )
    for lo, hi in itertools.pairwise(grid):
        if excess_db(float(hi)) >= 0:
            root = brentq(excess_db, float(lo), float(hi), xtol=1e-10, rtol=1e-10)
            w = float(np.exp(root))
            c = compression(p_ss, omega_lo, w, kappa_omega)
            return LinearRange(w, "compression" if c < 1.0 else "expansion")
    return LinearRange(float("nan"), "none", "no 1 dB point for signal amplitudes up to the LO")
