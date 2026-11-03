"""Link budget: Fresnel product, aperture coupling, absorption. (FR-CH-6, TSD §6.5)

Absorption enters only through ``Psi0`` (large-scale law), never in ``eta_bar``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from qmwlink.constants import C

CouplingModel = Literal["goubau", "friis_clipped"]

GAMMA_ABS_5GHZ = 1.44e-6
"""Clear-air power absorption at 5 GHz [1/m]. Source: Gonzalez-Raya et al., PRApplied 2022
(``GCF:22`` in FSD FR-CH-6). Other bands: supply ITU-R P.676 values with a ``source:``."""


@dataclass(frozen=True)
class LinkBudget:
    """Result of :func:`link_budget`."""

    lam_m: float
    a_t_m2: float
    a_r_m2: float
    tau2: float
    """Fresnel product ``A_t A_r / (lambda d)^2`` (the Friis power ratio)."""
    eta_geo: float
    """Geometric collection efficiency ``eta_bar_s``."""
    psi0: float
    """Deterministic absorption log-loss ``gamma_abs * d``."""
    coupling_model: CouplingModel
    physical: bool
    """False for the flagged ``friis_clipped`` baseline."""


def aperture_from_gain(gain_linear: float, lam_m: float) -> float:
    """Transmit aperture ``A_t = G_t lambda^2 / (4 pi)``. (FR-CH-6, TSD §6.5)"""
    if gain_linear <= 0 or lam_m <= 0:
        raise ValueError("gain_linear and lam_m must be > 0")
    return gain_linear * lam_m**2 / (4.0 * np.pi)


def fresnel_product(a_t_m2: float, a_r_m2: float, lam_m: float, d_m: float) -> float:
    """``tau^2 = A_t A_r / (lambda d)^2``. (FR-CH-6, TSD §6.5)"""
    if min(a_t_m2, a_r_m2, lam_m, d_m) <= 0:
        raise ValueError("a_t_m2, a_r_m2, lam_m and d_m must be > 0")
    return a_t_m2 * a_r_m2 / (lam_m * d_m) ** 2


def goubau_efficiency(tau2: float) -> float:
    """``1 - exp(-tau^2)`` [L] (Goubau & Schwering 1961); in ``[0, 1)`` with no clipping.

    (FR-CH-6, TSD §6.5) Uses ``expm1`` for accuracy at small ``tau^2``.
    """
    if tau2 < 0:
        raise ValueError("tau2 must be >= 0")
    return float(-np.expm1(-tau2))


def link_budget(
    f_hz: float,
    d_m: float,
    a_t_m2: float,
    a_r_m2: float,
    *,
    gamma_abs_per_m: float,
    coupling_model: CouplingModel = "goubau",
    g_enh_tx: float = 1.0,
    g_enh_rx: float = 1.0,
) -> LinkBudget:
    """Mean collection efficiency and absorption for one link. (FR-CH-6, TSD §6.5)

    ``g_enh_tx`` / ``g_enh_rx`` (``>= 1``) multiply the corresponding aperture for
    field-confining structures. ``gamma_abs_per_m`` is required (no hidden default).
    """
    if f_hz <= 0:
        raise ValueError("f_hz must be > 0")
    if g_enh_tx < 1 or g_enh_rx < 1:
        raise ValueError("enhancement factors must be >= 1")
    if gamma_abs_per_m < 0:
        raise ValueError("gamma_abs_per_m must be >= 0")
    lam = C / f_hz
    at, ar = a_t_m2 * g_enh_tx, a_r_m2 * g_enh_rx
    tau2 = fresnel_product(at, ar, lam, d_m)
    if coupling_model == "goubau":
        eta, physical = goubau_efficiency(tau2), True
    elif coupling_model == "friis_clipped":
        eta, physical = min(tau2, 1.0), False
    else:
        raise ValueError(f"unknown coupling_model {coupling_model!r}")
    return LinkBudget(lam, at, ar, tau2, eta, gamma_abs_per_m * d_m, coupling_model, physical)


def tau2_for_efficiency(eta_geo: float) -> float:
    """Inverse Goubau: ``tau^2 = -ln(1 - eta_geo)``. (FR-CH-6, TSD §6.5)"""
    if not (0.0 <= eta_geo < 1.0):
        raise ValueError("eta_geo must be in [0, 1)")
    return float(-np.log1p(-eta_geo))


def symmetric_aperture_for_efficiency(eta_geo: float, f_hz: float, d_m: float) -> float:
    """Equal ``A_t = A_r`` giving ``eta_geo`` at distance ``d_m``: ``A = sqrt(tau^2) lambda d``."""
    return float(np.sqrt(tau2_for_efficiency(eta_geo)) * (C / f_hz) * d_m)


def absorption_range_m(gamma_abs_per_m: float, n_th: float) -> float:
    """Distance where ``exp(-gamma d) = eta*``: ``d = ln(1 + 1/n_th) / gamma``. (TSD §6.5)

    Uses ``log1p`` (``-ln eta* = ln(1 + 1/n_th)``), no subtraction near 1.
    """
    if gamma_abs_per_m <= 0 or n_th <= 0:
        raise ValueError("gamma_abs_per_m and n_th must be > 0")
    return float(np.log1p(1.0 / n_th) / gamma_abs_per_m)
