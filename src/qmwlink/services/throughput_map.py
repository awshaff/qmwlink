"""RAR-vs-LNA throughput map (RQ5). (FR-SVC-C5, TSD §8.4)

Both receivers sit behind the same aperture (same ``eta`` per realization) and share the TX power
``P_t``; ``N_s = P_t/(h f B)`` differs with each receiver's bandwidth.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from qmwlink.channel.compound import CompoundPDT
from qmwlink.receivers.base import MeasuringReceiver
from qmwlink.services.classical import ClassicalService, c_rx, tx_photons_per_mode


def interp_loglog(f_hz: float, f_pts: Sequence[float], v_pts: Sequence[float]) -> float:
    """Linear interpolation of ``log v`` in ``log f``; ``NaN`` outside ``[min f, max f]``
    (no extrapolation, TSD §8.4)."""
    f_arr = np.asarray(f_pts, dtype=np.float64)
    v_arr = np.asarray(v_pts, dtype=np.float64)
    if f_arr.size < 1 or f_arr.size != v_arr.size or np.any(f_arr <= 0) or np.any(v_arr <= 0):
        raise ValueError("need matching positive f and value points")
    order = np.argsort(f_arr)
    f_arr, v_arr = f_arr[order], v_arr[order]
    if f_hz < f_arr[0] * (1 - 1e-12) or f_hz > f_arr[-1] * (1 + 1e-12):
        return math.nan
    if f_arr.size == 1:
        return float(v_arr[0])
    return float(np.exp(np.interp(np.log(f_hz), np.log(f_arr), np.log(v_arr))))


@dataclass(frozen=True)
class ThroughputPoint:
    rate_rar_bps: float
    rate_lna_bps: float
    rho_db: float
    n_s_rar: float
    n_s_lna: float


def ergodic_rate_bps(
    compound: CompoundPDT, rx: MeasuringReceiver, n_th_val: float, p_t_w: float, f_hz: float
) -> tuple[float, float]:
    """``(B E[C_rx], N_s)`` for ``rx`` (must carry ``bandwidth_hz``)."""
    if rx.bandwidth_hz is None:
        raise ValueError("receiver needs bandwidth_hz for throughput")
    n_s = tx_photons_per_mode(p_t_w, f_hz, rx.bandwidth_hz)
    svc = ClassicalService(compound, lambda e: c_rx(e, n_s, n_th_val, rx))
    return svc.throughput(rx.bandwidth_hz), n_s


def throughput_ratio(
    compound: CompoundPDT,
    rar: MeasuringReceiver,
    lna: MeasuringReceiver,
    n_th_val: float,
    p_t_w: float,
    f_hz: float,
) -> ThroughputPoint:
    """``rho = 10 log10(R_RAR / R_LNA)`` on one ``(f, d, K)`` cell."""
    r_rar, ns_rar = ergodic_rate_bps(compound, rar, n_th_val, p_t_w, f_hz)
    r_lna, ns_lna = ergodic_rate_bps(compound, lna, n_th_val, p_t_w, f_hz)
    rho = math.nan if r_rar <= 0 or r_lna <= 0 else 10.0 * math.log10(r_rar / r_lna)
    return ThroughputPoint(r_rar, r_lna, rho, ns_rar, ns_lna)
