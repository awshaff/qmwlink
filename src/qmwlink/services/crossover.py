"""Crossover sensitivity: required receiver noise / ``S_E`` for a dB penalty. (FR-SVC-C4, TSD §8.3)

Penalty is relative to ideal SSB heterodyne at a reference transmissivity ``eta_ref``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from qmwlink.constants import ETA0, H
from qmwlink.receivers.base import N_AMP_MIN, MeasuringReceiver, Sideband, check_sideband
from qmwlink.services.classical import ClassicalService, c_het, c_rx
from qmwlink.thermal import effective_aperture_m2, n_th, s_e_bbr


@dataclass(frozen=True)
class Crossover:
    """Result: ``NaN`` fields come with a ``reason``. ``ratio_to_bbr = S_E^req / S_E,BBR``."""

    n_amp_req: float
    s_e_req: float
    ratio_to_bbr: float
    reason: str | None = None

    @property
    def attainable(self) -> bool:
        return self.reason is None


def _nan(reason: str) -> Crossover:
    return Crossover(math.nan, math.nan, math.nan, reason)


def s_e_from_n_amp(n_amp: float, f_hz: float) -> float:
    """rms ``S_E = sqrt(N_amp eta0 h f / A_e)`` [V/m/sqrt(Hz)]. (TSD §8.3)"""
    return float(np.sqrt(n_amp * ETA0 * H * f_hz / effective_aperture_m2(f_hz)))


def dsb_image_floor_db(n_th_val: float, eta_ref: float = 0.0) -> float:
    """Minimum penalty of a ``dsb`` receiver (``N_amp = 1/2``): ``10 log10[(N_out+1+N_img)/
    (N_out+1)]``; at ``eta_ref -> 0`` this is ``(2 n+1)/(n+1)``."""
    n_out = (1.0 - eta_ref) * n_th_val
    return float(10.0 * np.log10((2.0 * n_out + 1.0) / (n_out + 1.0)))


def required_n_amp(
    delta_db: float, n_th_val: float, sideband: Sideband, eta_ref: float = 0.0
) -> tuple[float, str | None]:
    """``N_amp^req = (10^{D/10} - 1)(N_out + 1) + 1/2 - N_img`` and a reason if ``< 1/2``."""
    check_sideband(sideband)
    if not 0.0 <= eta_ref <= 1.0:
        raise ValueError("eta_ref must be in [0, 1]")
    n_out = (1.0 - eta_ref) * n_th_val
    n_img = n_out if sideband == "dsb" else 0.0
    n_req = float(np.expm1(delta_db * np.log(10.0) / 10.0) * (n_out + 1.0) + 0.5 - n_img)
    if n_req < N_AMP_MIN:
        if sideband == "dsb":
            floor = dsb_image_floor_db(n_th_val, eta_ref)
            return math.nan, (
                f"target {delta_db:g} dB is below the DSB image floor of {floor:.3f} dB "
                f"(N_amp_req = {n_req:.4g} < 1/2)"
            )
        return math.nan, f"target {delta_db:g} dB needs N_amp = {n_req:.4g} < 1/2 (Caves bound)"
    return n_req, None


def crossover_closed_form(
    delta_db: float,
    f_hz: float,
    t_env_k: float,
    sideband: Sideband,
    eta_ref: float = 0.0,
) -> Crossover:
    """Closed-form required ``S_E`` (rms) for a ``delta_db`` penalty vs ideal SSB heterodyne."""
    nth = float(n_th(f_hz, t_env_k))
    n_req, reason = required_n_amp(delta_db, nth, sideband, eta_ref)
    if reason is not None:
        return _nan(reason)
    s_req = s_e_from_n_amp(n_req, f_hz)
    return Crossover(n_req, s_req, s_req / float(s_e_bbr(f_hz, t_env_k)))


def crossover_numeric(
    delta_db: float, f_hz: float, t_env_k: float, sideband: Sideband, eta_ref: float = 0.0
) -> Crossover:
    """Same quantity by ``brentq`` on the penalty ratio (cross-check of the closed form)."""
    check_sideband(sideband)
    nth = float(n_th(f_hz, t_env_k))
    n_out = (1.0 - eta_ref) * nth

    def penalty_db(n_amp: float) -> float:
        d = MeasuringReceiver(n_amp, sideband).noise_photons(eta_ref, nth)
        return float(10.0 * np.log10(d / (n_out + 1.0)))

    if penalty_db(N_AMP_MIN) > delta_db:
        return crossover_closed_form(delta_db, f_hz, t_env_k, sideband, eta_ref)  # NaN + reason
    hi = 1.0
    while penalty_db(hi) < delta_db:
        hi *= 2.0
    n_req = float(brentq(lambda n: penalty_db(n) - delta_db, N_AMP_MIN, hi, xtol=1e-12, rtol=1e-14))
    s_req = s_e_from_n_amp(n_req, f_hz)
    return Crossover(n_req, s_req, s_req / float(s_e_bbr(f_hz, t_env_k)))


def crossover_capacity(
    service_het: ClassicalService,
    n_s: float,
    n_th_val: float,
    sideband: Sideband,
    delta_loss: float,
    f_hz: float,
    t_env_k: float,
) -> Crossover:
    """Capacity-based variant: the *largest* ``N_amp`` with ``C_rx >= (1 - delta) C_het``.

    ``C_rx`` is non-increasing in ``N_amp``, so the tolerable noise is an upper bound on
    ``N_amp`` (TSD §8.3 says "smallest"; this is the reading that makes the set non-trivial,
    see QUESTIONS Q-10). ``service_het`` must wrap the compound with ``C_het``.
    """
    check_sideband(sideband)
    if not 0.0 < delta_loss < 1.0:
        raise ValueError("delta_loss must be in (0, 1)")
    comp = service_het.compound
    target = (1.0 - delta_loss) * service_het.ergodic()

    def cbar(log_n: float) -> float:
        rx = MeasuringReceiver(float(np.exp(log_n)), sideband)
        return ClassicalService(comp, lambda e: c_rx(e, n_s, n_th_val, rx)).ergodic() - target

    lo = float(np.log(N_AMP_MIN))
    if cbar(lo) < 0.0:
        return _nan(
            f"even N_amp = 1/2 ({sideband}) misses {100 * (1 - delta_loss):g}% of the "
            "heterodyne ergodic capacity" + (" (DSB image floor)" if sideband == "dsb" else "")
        )
    hi = lo + 1.0
    while cbar(hi) > 0.0 and hi < lo + 60.0:
        hi += 1.0
    log_n = float(brentq(cbar, lo, hi, xtol=1e-12, rtol=1e-12))
    n_req = float(np.exp(log_n))
    s_req = s_e_from_n_amp(n_req, f_hz)
    return Crossover(n_req, s_req, s_req / float(s_e_bbr(f_hz, t_env_k)))


__all__ = [
    "Crossover",
    "c_het",
    "crossover_capacity",
    "crossover_closed_form",
    "crossover_numeric",
    "dsb_image_floor_db",
    "required_n_amp",
    "s_e_from_n_amp",
]
