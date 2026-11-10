"""N-element diversity: array outage, selection-combining tail, Chernoff and Cramer bounds,
design law and ``N_req``. (FR-DIV-1..4, TSD §9; ported from V0, exact forms.)

MRC array outage uses the compound PDT with ``SmallScale(n_elements=N)``. ``eta_bar_sigma`` is the
array scale ``eta_bar_Sigma``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from qmwlink.channel.compound import CompoundPDT
from qmwlink.channel.large_scale import DeterministicLoss, LargeScaleLaw
from qmwlink.channel.small_scale import N_MAX_ELEMENTS, SmallScale
from qmwlink.numerics.integrate import checked_quad
from qmwlink.numerics.special import ncx2_cdf, ncx2_pdf

FloatArray = NDArray[np.float64]

SC_MIN_X = 0.5
"""The exact SC tail needs at most one element above ``x``: valid only for ``x >= 1/2``."""


def array_outage(
    n: int, k: float, eta_bar_sigma: float, eta_star: float, large: LargeScaleLaw | None = None
) -> float:
    """``P_out,N = F_N(eta*/eta_l)`` averaged over the large-scale law (``eta_l = 1`` if None)."""
    small = SmallScale(k, eta_bar_sigma, n)
    if large is None:
        return float(small.cdf(eta_star))
    return float(CompoundPDT(small, large).cdf(eta_star))


# ---- selection combining ----------------------------------------------------------------------
def sc_exact_tail(x: float, k: float, eta_bar_sigma: float, n: int) -> float:
    """Exact ``P(eta_SC > x) = (N/Z) int_x^1 f_s(t) G_{N-1}(1-t) dt``, valid only for ``x >= 1/2``.

    ``f_s`` is the untruncated single-element pdf (scale ``eta_bar_Sigma/N``);
    ``G_{N-1}`` is the ``chi'^2_{2(N-1)}(2(N-1)K)`` cdf at ``y/sigma_N^2``. Raises
    ``ValueError`` for ``x < 1/2`` (never returns a value there).
    """
    if x < SC_MIN_X:
        raise ValueError(
            f"exact selection-combining tail is valid only for x >= 1/2 (got x = {x}); "
            "use sc_chernoff_bound for smaller x"
        )
    if x >= 1.0:
        return 0.0
    small = SmallScale(k, eta_bar_sigma, n)
    s2 = small.sigma2

    def integrand(t: float) -> float:
        fs = float(ncx2_pdf(t / s2, 2.0, 2.0 * k)) / s2
        g = 1.0 if n == 1 else float(ncx2_cdf((1.0 - t) / s2, 2.0 * (n - 1), 2.0 * (n - 1) * k))
        return fs * g

    return n / small.z * checked_quad(integrand, x, 1.0)[0]


def sc_chernoff_bound(x: float, k: float, eta_bar_sigma: float, n: int) -> float:
    """Chernoff/union bound on ``P(eta_SC > x)``, valid for all ``x``. (TSD §9)

    ``(N/Z) inf_{0<s<K+1} (K+1)/(K+1-s) exp(Ks/(K+1-s) - s N x/eta_bar_Sigma)``.
    """
    if not 0.0 < x < 1.0:
        raise ValueError("x must be in (0, 1)")
    z = SmallScale(k, eta_bar_sigma, n).z
    kp = k + 1.0

    def log_bound(s: float) -> float:
        return math.log(kp / (kp - s)) + k * s / (kp - s) - s * n * x / eta_bar_sigma

    res = minimize_scalar(log_bound, bounds=(0.0, kp * (1.0 - 1e-12)), method="bounded",
                          options={"xatol": 1e-12})  # fmt: skip
    best = min(float(res.fun), 0.0)  # s -> 0 gives log 1 = 0
    return float(min(math.exp(best) * n / z, n / z))


def sc_sample(
    rng: np.random.Generator, size: int, k: float, eta_bar_sigma: float, n: int,
    n_max: int = N_MAX_ELEMENTS,
) -> FloatArray:  # fmt: skip
    """Per-element SC samples ``eta_SC = max_i |h_i|^2`` (joint rejection ``sum <= 1``).

    ``N > n_max`` raises (FR-DIV-4 (d)); use ``SmallScale.sample_sum`` for MRC at large N.
    """
    g = SmallScale(k, eta_bar_sigma, n).sample_elements(rng, size, n_max=n_max)
    return np.asarray(g.max(axis=1), dtype=np.float64)


# ---- large-deviation machinery ----------------------------------------------------------------
def cramer_rate(a: float, k: float) -> float:
    """Cramer rate ``R_K(a)``; ``R_0(a) = a - 1 - ln a``, ``R_K(1) = 0``. (TSD §9)"""
    if a <= 0:
        raise ValueError("a must be > 0")
    if k == 0.0:
        return a - 1.0 - math.log(a)
    u = (1.0 + math.sqrt(1.0 + 4.0 * a * k * (k + 1.0))) / (2.0 * a)
    return (k + 1.0 - u) * (a - k / u) + math.log(u / (k + 1.0))


def log_mgf(s: float, k: float) -> float:
    """``ln M_K(s) = ln((K+1)/(K+1-s)) + K s/(K+1-s)`` for ``s < K+1``."""
    kp = k + 1.0
    return math.log(kp / (kp - s)) + k * s / (kp - s)


def cramer_rate_legendre(a: float, k: float) -> float:
    """``sup_{s<K+1} [s a - ln M_K(s)]`` numerically (test oracle for :func:`cramer_rate`)."""
    kp = k + 1.0
    res = minimize_scalar(lambda s: -(s * a - log_mgf(s, k)), bounds=(-1e5, kp * (1 - 1e-12)),
                          method="bounded", options={"xatol": 1e-13})  # fmt: skip
    return float(-res.fun)


def exponent_bound(n: int, k: float, a: float, eta_bar_sigma: float) -> float:
    """``P <= exp(-N R_K(a)) / Z_{K,N}`` with ``a = eta*/(eta_l eta_bar_Sigma)``."""
    return math.exp(-n * cramer_rate(a, k)) / SmallScale(k, eta_bar_sigma, n).z


def design_dimension(n: int, k: float) -> float:
    """``D_{N,K} = N (K+1)^2 / (1 + 2K)``."""
    return n * (k + 1.0) ** 2 / (1.0 + 2.0 * k)


def design_law_holds(n: int, k: float, a: float, eps: float, eta_bar_sigma: float) -> bool:
    """``D_{N,K} >= 2 ln(1/(eps Z)) / (1-a)^2`` (sufficient condition for ``F_N <= eps``)."""
    if not 0.0 < a < 1.0:
        raise ValueError("design law needs 0 < a < 1")
    z = SmallScale(k, eta_bar_sigma, n).z
    return design_dimension(n, k) >= 2.0 * math.log(1.0 / (eps * z)) / (1.0 - a) ** 2


# ---- N_req ------------------------------------------------------------------------------------
def n_req(
    eps: float, k: float, eta_bar_sigma: float, eta_star: float,
    large: LargeScaleLaw | None = None, n_cap: int = 10**10,
) -> int:  # fmt: skip
    """Smallest integer ``N`` with ``F_N(eta*) <= eps``: exponential search then bisection.

    ``F_N`` is the exact array cdf (compound with ``large``; ``eta_l = 1`` if None) and is
    assumed non-increasing in ``N``. Raises if no ``N <= n_cap`` works.
    """
    if not 0.0 < eps < 1.0:
        raise ValueError("eps must be in (0, 1)")
    cdf: Callable[[int], float] = lambda n: array_outage(n, k, eta_bar_sigma, eta_star, large)  # noqa: E731
    if isinstance(large, DeterministicLoss) and large.psi < 0:
        raise ValueError("deterministic loss must have psi >= 0")
    if cdf(1) <= eps:
        return 1
    hi = 2
    while cdf(hi) > eps:
        hi *= 2
        if hi > n_cap:
            raise ValueError(f"no N <= {n_cap} reaches outage {eps:g}")
    lo = hi // 2  # cdf(lo) > eps >= cdf(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if cdf(mid) > eps else (lo, mid)
    return hi
