"""Small-scale PDT: truncated noncentral chi-square for passive MRC. (FR-CH-1, TSD §6.1)

``X = eta / sigma_N^2 ~ chi'^2(nu = 2N, lambda = 2NK)`` truncated at ``x_max = 1/sigma_N^2``,
``sigma_N^2 = eta_bar / (2N(K+1))``. ``eta_bar`` is a *scale* parameter, not the mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from qmwlink.numerics.integrate import checked_quad
from qmwlink.numerics.sampling import rejection_sample

FloatArray = NDArray[np.float64]

CENTRAL_NC = 1e-300
"""Vanishing noncentrality standing in for the central chi-square (K = 0)."""

N_MAX_ELEMENTS = 64
"""Default cap for :meth:`SmallScale.sample_elements` (TSD §6.1)."""


@dataclass(frozen=True)
class SmallScale:
    """Truncated noncentral chi-square small-scale PDT. (FR-CH-1, TSD §6.1)

    Args:
        K: Rician factor, ``>= 0``.
        eta_bar: scale parameter ``eta_bar_s`` (``N = 1``) or ``eta_bar_Sigma`` (arrays),
            ``0 < eta_bar <= 1``.
        n_elements: number of passive-MRC elements ``N >= 1``.
    """

    K: float
    eta_bar: float
    n_elements: int = 1

    def __post_init__(self) -> None:
        if not np.isfinite(self.K) or self.K < 0:
            raise ValueError("K must be finite and >= 0")
        if not (0.0 < self.eta_bar <= 1.0):
            raise ValueError("eta_bar must satisfy 0 < eta_bar <= 1")
        if int(self.n_elements) != self.n_elements or self.n_elements < 1:
            raise ValueError("n_elements must be an integer >= 1")

    # ---- derived constants -------------------------------------------------------------
    @property
    def sigma2(self) -> float:
        """``sigma_N^2 = eta_bar / (2N(K+1))``."""
        return self.eta_bar / (2.0 * self.n_elements * (self.K + 1.0))

    @property
    def x_max(self) -> float:
        """Truncation point of the surrogate variable, ``1 / sigma_N^2``."""
        return 1.0 / self.sigma2

    @property
    def nu(self) -> float:
        """Degrees of freedom ``2N``."""
        return 2.0 * self.n_elements

    @property
    def nc(self) -> float:
        """Noncentrality ``2NK``."""
        return 2.0 * self.n_elements * self.K

    @cached_property
    def _dist(self) -> stats.rv_continuous:
        # K = 0: scipy's central chi2/gammainc is inaccurate in the far lower tail for
        # large N (rel. error 1.6e-6 at N=1e6, 2.3% at N=1e7 for the 1e-8 tail; QUESTIONS Q-3),
        # and ncx2 with nc = 0 silently delegates to it. ncx2 with a vanishing positive nc
        # is the same distribution (agrees to 1e-15 at small N) and is accurate to ~1e-13.
        return stats.ncx2(df=self.nu, nc=max(self.nc, CENTRAL_NC))

    @cached_property
    def z(self) -> float:
        """Truncation normalizer ``Z_{K,N} = F_{chi'^2}(x_max)`` (= sampler acceptance rate)."""
        return float(self._dist.cdf(self.x_max))

    # ---- distribution functions --------------------------------------------------------
    def pdf(self, x: ArrayLike) -> FloatArray:
        """Density on ``[0, 1]`` (zero elsewhere). (FR-CH-1)"""
        xx = np.asarray(x, dtype=np.float64)
        inside = (xx >= 0.0) & (xx <= 1.0)
        val = self._dist.pdf(np.where(inside, xx, 0.0) / self.sigma2) / (self.sigma2 * self.z)
        return np.asarray(np.where(inside, val, 0.0), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> FloatArray:
        """``F(x) = F_chi'^2(min(x, 1)/sigma^2)/Z``; exactly 1 for ``x >= 1``. (FR-CH-1)

        The clamp ``min(x, 1)`` is mandatory (FSD FR-CH-1).
        """
        xx = np.asarray(x, dtype=np.float64)
        xc = np.clip(xx, 0.0, 1.0)
        val = np.minimum(self._dist.cdf(xc / self.sigma2) / self.z, 1.0)
        return np.asarray(np.where(xx >= 1.0, 1.0, np.where(xx <= 0.0, 0.0, val)), dtype=np.float64)

    def sf(self, x: ArrayLike) -> FloatArray:
        """Survival ``P(eta_s > x)``, computed in the smaller tail. (FR-CH-1, TSD §6.1, §13 rule 1)

        For ``F(x) < 1/2`` it is ``1 - F``; otherwise
        ``[S_chi'^2(x/sigma^2) - S_chi'^2(x_max)] / Z`` (no ``1 - F`` near 1).
        """
        xx = np.asarray(x, dtype=np.float64)
        xc = np.clip(xx, 0.0, 1.0)
        f = self.cdf(xc)
        upper = (self._dist.sf(xc / self.sigma2) - self._dist.sf(self.x_max)) / self.z
        val = np.where(f < 0.5, 1.0 - f, np.maximum(upper, 0.0))
        return np.asarray(np.where(xx >= 1.0, 0.0, np.where(xx <= 0.0, 1.0, val)), dtype=np.float64)

    def ppf(self, q: ArrayLike) -> FloatArray:
        """Inverse cdf on ``q in [0, 1]``. (FR-CH-1)"""
        qq = np.asarray(q, dtype=np.float64)
        if np.any((qq < 0.0) | (qq > 1.0)):
            raise ValueError("q must be in [0, 1]")
        val = self._dist.ppf(qq * self.z) * self.sigma2
        return np.asarray(np.where(qq >= 1.0, 1.0, np.minimum(val, 1.0)), dtype=np.float64)

    # ---- moments -----------------------------------------------------------------------
    def moment(self, m: int | float) -> float:
        """Truncated moment ``E^QC[eta_s^m]`` by checked quadrature. (FR-CH-1, TSD §6.1)

        Integrates ``(x/x_max)^m f_chi'^2(x)`` over ``[0, x_max]`` with breakpoints at the
        surrogate mean +/- j std, then divides by the m = 0 integral (= ``Z``).
        """
        if m < 0:
            raise ValueError("m must be >= 0")
        mean = self.nu + self.nc
        sd = float(np.sqrt(2.0 * (self.nu + 2.0 * self.nc)))
        xm = self.x_max
        # ncx2.pdf can return nan ~27 sd from the mean for huge N or K; the true density there is
        # < 1e-80, so integrate over mean +/- 20 sd (clipped to [0, x_max]) instead of masking it.
        lo = max(0.0, mean - 20.0 * sd)
        hi = min(xm, mean + 20.0 * sd)  # x_max can be thousands of sd above the mean (large K)
        pts = [mean + j * sd for j in (-6, -3, -1, 0, 1, 3, 6)]

        def integral(power: float) -> float:
            def integrand(t: float) -> float:
                v = float((t / xm) ** power * self._dist.pdf(t))
                if not np.isfinite(v):
                    raise FloatingPointError(f"non-finite chi'^2 pdf at x = {t!r}")
                return v

            return checked_quad(integrand, lo, hi, points=pts)[0]

        # Normalising by the m = 0 integral of the same pdf cancels scipy's ~1e-8 relative
        # pdf error at df ~ 1e7 (log-gamma cancellation); equals Z up to that error.
        return integral(float(m)) / integral(0.0)

    # ---- samplers ----------------------------------------------------------------------
    def sample_sum(self, rng: np.random.Generator, size: int) -> FloatArray:
        """Exact passive-MRC sampler for any ``N``: draw the surrogate and reject ``eta > 1``.

        (FR-CH-1, TSD §6.1) Acceptance rate is ``Z``.
        """

        def draw(n: int) -> FloatArray:
            if self.K == 0.0:
                x = rng.chisquare(self.nu, size=n)
            else:
                x = rng.noncentral_chisquare(self.nu, self.nc, size=n)
            return np.asarray(x * self.sigma2, dtype=np.float64)

        return rejection_sample(draw, lambda eta: eta <= 1.0, size, self.z)

    def sample_elements(
        self,
        rng: np.random.Generator,
        size: int,
        *,
        n_max: int = N_MAX_ELEMENTS,
        phi0: ArrayLike = 0.0,
    ) -> FloatArray:
        """Per-element gains ``|h_i|^2``, shape ``(size, N)``; ``eta = sum_i |h_i|^2 <= 1``.

        ``h_i ~ CN(sqrt(K eta_bar/(N(K+1))) e^{i phi_i}, eta_bar/(N(K+1)))``; whole vectors with
        ``sum |h_i|^2 > 1`` are rejected. Needed only for selection combining. (TSD §6.1)
        """
        n = self.n_elements
        if n > n_max:
            raise ValueError(
                f"sample_elements supports N <= {n_max} (got N = {n}); use sample_sum for large N"
            )
        scale = self.eta_bar / (n * (self.K + 1.0))
        los = np.sqrt(self.K * scale) * np.exp(1j * np.asarray(phi0, dtype=np.float64))
        los = np.broadcast_to(los, (n,))
        s = np.sqrt(scale / 2.0)

        def draw(k: int) -> FloatArray:
            h = los + s * (rng.standard_normal((k, n)) + 1j * rng.standard_normal((k, n)))
            return np.asarray(np.abs(h) ** 2, dtype=np.float64)

        return rejection_sample(draw, lambda g: g.sum(axis=1) <= 1.0, size, self.z)
