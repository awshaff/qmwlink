"""``ncx2`` floor helper: the single entry point for chi-square cdf/sf/pdf. (TSD §6.1, §13 rule 10)

``scipy.stats.chi2``, ``gamma``, ``special.gammainc`` and ``ncx2(nc=0)`` share a code path that
loses accuracy in the lower tail at large shape (cancellation in ``a ln y - y - lgamma(a+1)``;
QUESTIONS Q-3). ``ncx2`` with ``lambda >= NC_FLOOR`` takes another path and is the same
distribution at double precision. The fix relies on a scipy implementation detail and is guarded
by the pinned regression test ``test_ncx2_floor_tail``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

FloatArray = NDArray[np.float64]

NC_FLOOR = 1e-300
"""Lower bound on the noncentrality passed to ``scipy.stats.ncx2``."""


@lru_cache(maxsize=512)
def _frozen_cached(nu: float, lam: float) -> stats.rv_continuous:
    return stats.ncx2(df=nu, nc=max(lam, NC_FLOOR))


def _frozen(nu: float, lam: float) -> stats.rv_continuous:
    """Validated, cached frozen distribution (construction costs ~0.6 ms; E3 profile)."""
    if not (np.isfinite(nu) and nu > 0):
        raise ValueError("nu must be finite and > 0")
    if not (np.isfinite(lam) and lam >= 0):
        raise ValueError("lam must be finite and >= 0")
    return _frozen_cached(float(nu), float(lam))


def ncx2_cdf(x: ArrayLike, nu: float, lam: float) -> FloatArray:
    """Noncentral chi-square cdf through the floor path (all ``lam >= 0``)."""
    return np.asarray(_frozen(nu, lam).cdf(x), dtype=np.float64)


def ncx2_sf(x: ArrayLike, nu: float, lam: float) -> FloatArray:
    """Noncentral chi-square survival function through the floor path."""
    return np.asarray(_frozen(nu, lam).sf(x), dtype=np.float64)


def ncx2_pdf(x: ArrayLike, nu: float, lam: float) -> FloatArray:
    """Noncentral chi-square pdf through the floor path (may be ``nan`` far from the mean)."""
    return np.asarray(_frozen(nu, lam).pdf(x), dtype=np.float64)


def ncx2_ppf(q: ArrayLike, nu: float, lam: float) -> FloatArray:
    """Noncentral chi-square quantile through the floor path."""
    return np.asarray(_frozen(nu, lam).ppf(q), dtype=np.float64)


def gamma_lower_regularized(a: float, u: ArrayLike) -> FloatArray:
    """``P(a, u) = F_{chi^2_{2a}}(2u)`` via the floor helper (large-shape safe). (TSD §6.2)"""
    return ncx2_cdf(2.0 * np.asarray(u, dtype=np.float64), 2.0 * a, 0.0)


def gamma_upper_regularized(a: float, u: ArrayLike) -> FloatArray:
    """``Q(a, u) = S_{chi^2_{2a}}(2u)`` via the floor helper. (TSD §6.2)"""
    return ncx2_sf(2.0 * np.asarray(u, dtype=np.float64), 2.0 * a, 0.0)
