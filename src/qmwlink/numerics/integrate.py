"""Checked quadrature with mpmath fallback. (TSD §6.4, §13 rule 6)"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable, Iterable

import mpmath
import numpy as np
from scipy import integrate

logger = logging.getLogger(__name__)

EPSREL = 1e-10
LIMIT = 500
ERR_FALLBACK = 1e-8
"""Absolute error estimate above which ``mpmath.quad`` takes over (TSD §6.4)."""


def _split(a: float, b: float, points: Iterable[float] | None) -> list[float]:
    inner = [] if points is None else [p for p in points if a < p < b and np.isfinite(p)]
    return sorted({a, b, *inner})


def checked_quad(
    func: Callable[[float], float],
    a: float,
    b: float,
    points: Iterable[float] | None = None,
    *,
    epsrel: float = EPSREL,
    limit: int = LIMIT,
) -> tuple[float, float]:
    """Integrate ``func`` over ``[a, b]`` split at ``points``; return ``(value, abserr)``.

    Uses ``scipy.integrate.quad`` with ``epsabs=0``. If any piece is non-finite or the summed
    error estimate exceeds 1e-8 (TSD §6.4), falls back to ``mpmath.quad`` and logs a warning;
    quadrature failures are never returned unchecked. (TSD §6.4, §13 rule 6)
    """
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError("integration limits must be finite")
    if b <= a:
        return 0.0, 0.0
    edges = _split(a, b, points)
    total, err, failed = 0.0, 0.0, False
    for lo, hi in itertools.pairwise(edges):
        out = integrate.quad(func, lo, hi, epsabs=0.0, epsrel=epsrel, limit=limit, full_output=1)
        total += out[0]
        err += out[1]
        if not np.isfinite(out[0]) or not np.isfinite(out[1]):
            failed = True
    if failed or err > ERR_FALLBACK:
        logger.warning("quad error %.3g (failed=%s); falling back to mpmath.quad", err, failed)
        mpmath.mp.dps = 20
        val = mpmath.quad(lambda t: func(float(t)), edges)
        return float(val), 0.0
    return float(total), float(err)
