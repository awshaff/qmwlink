"""MC-agree rule: Bonferroni pointwise bands and KS. (FSD §4.3, TSD §10.1)

A comparison of an analytic quantity with Monte Carlo passes iff, at each of ``m`` points,
``|p_hat - p| <= z_m * se`` with ``z_m = Phi^-1(1 - alpha/(2m))`` (family-wise ``alpha``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

ALPHA = 1e-3
"""Family-wise significance of the MC-agree rule."""

FloatArray = NDArray[np.float64]


def bonferroni_z(m: int, alpha: float = ALPHA) -> float:
    """``z_m = Phi^-1(1 - alpha/(2m))``; about 4.4 for ``m = 100``, ``alpha = 1e-3``."""
    if m < 1:
        raise ValueError("m must be >= 1")
    return float(stats.norm.isf(alpha / (2.0 * m)))


@dataclass(frozen=True)
class Agreement:
    """Outcome of an MC-agree check."""

    ok: bool
    z_crit: float
    max_z: float
    """Largest observed ``|p_hat - p| / se``."""


def ecdf_estimate(samples: ArrayLike, xs: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Empirical ``P(sample <= x)`` and binomial standard error at each ``x``.

    The estimate is ``k/n``; the standard error uses the adjusted proportion
    ``(k + 1/2)/(n + 1)`` so it stays positive at ``k = 0`` or ``k = n``.
    """
    s = np.sort(np.asarray(samples, dtype=np.float64).ravel())
    n = s.size
    x = np.asarray(xs, dtype=np.float64)
    k = np.searchsorted(s, x, side="right")
    p_hat = k / n
    p_adj = (k + 0.5) / (n + 1.0)
    return p_hat, np.sqrt(p_adj * (1.0 - p_adj) / n)


def check_probabilities(
    p_hat: ArrayLike, se: ArrayLike, p: ArrayLike, alpha: float = ALPHA
) -> Agreement:
    """Pointwise Bonferroni check of MC probabilities against analytic ``p``."""
    ph, s, pp = (np.asarray(v, dtype=np.float64).ravel() for v in (p_hat, se, p))
    if not (ph.shape == s.shape == pp.shape):
        raise ValueError("p_hat, se and p must have the same shape")
    if np.any(s <= 0):
        raise ValueError("standard errors must be > 0")
    z = bonferroni_z(ph.size, alpha)
    dev = float(np.max(np.abs(ph - pp) / s))
    return Agreement(ok=dev <= z, z_crit=z, max_z=dev)


def check_moment(samples: ArrayLike, expected: float, alpha: float = ALPHA) -> Agreement:
    """Single-point check of a sample mean against ``expected`` (sample standard error)."""
    s = np.asarray(samples, dtype=np.float64).ravel()
    se = float(np.std(s, ddof=1) / np.sqrt(s.size))
    if se <= 0:
        raise ValueError("samples have zero variance")
    z = bonferroni_z(1, alpha)
    dev = abs(float(np.mean(s)) - expected) / se
    return Agreement(ok=dev <= z, z_crit=z, max_z=dev)


def ks_agree(
    samples: ArrayLike, cdf: Callable[[FloatArray], FloatArray], alpha: float = ALPHA
) -> tuple[bool, float, float]:
    """One-sample KS vs an analytic cdf. Returns ``(ok, D, p_value)``, ``ok = p >= alpha``."""
    res = stats.kstest(np.asarray(samples, dtype=np.float64).ravel(), cdf)
    return bool(res.pvalue >= alpha), float(res.statistic), float(res.pvalue)
