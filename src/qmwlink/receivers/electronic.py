"""Electronic receiver from a noise temperature. (FR-RX-3, TSD §7.3)"""

from __future__ import annotations

from typing import Literal

import numpy as np

from qmwlink.constants import K_B, H
from qmwlink.receivers.base import N_AMP_MIN, MeasuringReceiver, Sideband
from qmwlink.thermal import n_th

TnConvention = Literal["rj", "planck", "callen_welton"]
CONVENTIONS = ("rj", "planck", "callen_welton")


def n_amp_from_tn(t_n_k: float, f_hz: float, convention: TnConvention) -> float:
    """``N_amp`` from ``T_n`` under a *required* convention (no default). (TSD §7.3)

    ``rj``: ``k_B T_n/(h f)``; ``planck``: ``n_bar(f, T_n)``; ``callen_welton``:
    ``n_bar(f, T_n) + 1/2``. Raises below the quantum limit (``N_amp < 1/2``).
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"tn_convention must be one of {CONVENTIONS}, got {convention!r}")
    if not (np.isfinite(t_n_k) and t_n_k >= 0):
        raise ValueError("t_n_k must be finite and >= 0")
    if f_hz <= 0:
        raise ValueError("f_hz must be > 0")
    if convention == "rj":
        n = K_B * t_n_k / (H * f_hz)
    else:
        n = float(n_th(f_hz, t_n_k)) if t_n_k > 0 else 0.0
        if convention == "callen_welton":
            n += 0.5
    if n < N_AMP_MIN * (1.0 - 1e-12):
        raise ValueError(
            f"T_n = {t_n_k:g} K gives N_amp = {n:.6g} < 1/2 under '{convention}' "
            "(below the quantum limit)"
        )
    return max(n, N_AMP_MIN)


def electronic_receiver(
    t_n_k: float,
    f_hz: float,
    *,
    tn_convention: TnConvention,
    sideband: Sideband = "ssb",
    bandwidth_hz: float | None = None,
) -> MeasuringReceiver:
    """Electronic receiver (default ``ssb``) from ``T_n`` with a required convention. (FR-RX-3)"""
    return MeasuringReceiver(n_amp_from_tn(t_n_k, f_hz, tn_convention), sideband, bandwidth_hz)


def quantum_limit_tn(f_hz: float, convention: TnConvention) -> float:
    """``T_n`` at which ``N_amp = 1/2``: ``hf/2k`` (rj), ``hf/(k ln3)`` (planck), 0 (CW)."""
    if convention == "rj":
        return float(H * f_hz / (2.0 * K_B))
    if convention == "planck":
        return float(H * f_hz / (K_B * np.log(3.0)))
    if convention == "callen_welton":
        return 0.0
    raise ValueError(f"tn_convention must be one of {CONVENTIONS}, got {convention!r}")
