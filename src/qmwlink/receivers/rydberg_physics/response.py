"""Doppler-averaged small-signal response: ``H(i w)``, ``B_3dB``, peaking. (FR-LIN-5, TSD §14.7)"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, minimize_scalar

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class Bandwidth:
    """Instantaneous bandwidth of ``|H|`` and its peaking above the DC value."""

    b_3db_hz: float
    peaking: float
    """``max|H| / |H(0)|`` (1 if the response never exceeds its DC value)."""
    f_peak_hz: float
    reached_scan_limit: bool
    """True if ``|H|`` is still above ``|H(0)|/sqrt 2`` at the top of the scan (lower bound)."""


def bandwidth_3db(
    h_abs: Callable[[FloatArray], FloatArray],
    *,
    f_min_mhz: float = 1e-3,
    f_max_mhz: float = 1e3,
    n_grid: int = 121,
) -> Bandwidth:
    """Highest ``f`` with ``|H(i 2 pi f)| >= |H(0)|/sqrt 2``. (FR-LIN-5, TSD §14.7)

    ``h_abs`` maps angular frequencies [rad/us] (1D array, 0 allowed) to ``|H|``. A log grid from
    ``f_min_mhz`` to ``f_max_mhz`` locates the last crossing, which ``brentq`` then refines.
    ``|H|`` may peak above ``|H(0)|`` near the EIT dynamics; the peaking value is reported.
    """
    if not 0 < f_min_mhz < f_max_mhz or n_grid < 8:
        raise ValueError("need 0 < f_min_mhz < f_max_mhz and n_grid >= 8")
    f = np.geomspace(f_min_mhz, f_max_mhz, n_grid)
    vals = np.asarray(h_abs(np.concatenate(([0.0], TWO_PI * f))), dtype=np.float64)
    h0, hf = float(vals[0]), vals[1:]
    if not (np.isfinite(h0) and h0 > 0):
        raise ValueError("|H(0)| must be finite and > 0")
    level = h0 / np.sqrt(2.0)
    i_peak = int(np.argmax(hf))
    peaking, f_peak = 1.0, 0.0
    if hf[i_peak] > h0:  # refine the grid maximum between its neighbours
        lo, hi = np.log(f[max(i_peak - 1, 0)]), np.log(f[min(i_peak + 1, n_grid - 1)])
        opt = minimize_scalar(
            lambda lf: -float(h_abs(np.array([TWO_PI * np.exp(lf)]))[0]),
            bounds=(lo, hi), method="bounded", options={"xatol": 1e-9},
        )  # fmt: skip
        peaking = max(-float(opt.fun), float(hf[i_peak])) / h0
        f_peak = float(np.exp(opt.x)) if -opt.fun >= hf[i_peak] else float(f[i_peak])
    above = np.nonzero(hf >= level)[0]
    if above.size == 0:
        raise ValueError("bandwidth is below the scan range; lower f_min_mhz")
    last = int(above[-1])
    if last == n_grid - 1:
        return Bandwidth(float(f[-1]) * 1e6, peaking, f_peak * 1e6, True)

    def g(log_f: float) -> float:
        return float(h_abs(np.array([TWO_PI * np.exp(log_f)]))[0] - level)

    root = brentq(g, np.log(f[last]), np.log(f[last + 1]), xtol=1e-12, rtol=1e-12)
    return Bandwidth(float(np.exp(root)) * 1e6, peaking, f_peak * 1e6, False)
