"""Small-scale PDT: truncated noncentral chi-square for passive MRC. (FR-CH-1, TSD §6.1)

``X = eta / sigma_N^2 ~ chi'^2(nu = 2N, lambda = 2NK)`` truncated at ``x_max = 1/sigma_N^2``,
``sigma_N^2 = eta_bar / (2N(K+1))``. ``eta_bar`` is a *scale* parameter, not the mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qmwlink.numerics.integrate import checked_quad
from qmwlink.numerics.sampling import rejection_sample
from qmwlink.numerics.special import NC_FLOOR, ncx2_cdf, ncx2_pdf, ncx2_ppf, ncx2_sf

FloatArray = NDArray[np.float64]

CENTRAL_NC = NC_FLOOR
"""Alias of :data:`qmwlink.numerics.special.NC_FLOOR` (kept for backwards compatibility)."""

W_START = 20.0
W_CAP = 320.0
MASS_TOL = 1e-14
"""Adaptive moment window: start, cap (sd) and neglected-mass tolerance (TSD §6.1)."""

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

    def _scdf(self, x: ArrayLike) -> FloatArray:
        return ncx2_cdf(x, self.nu, self.nc)

    def _ssf(self, x: ArrayLike) -> FloatArray:
        return ncx2_sf(x, self.nu, self.nc)

    def _spdf(self, x: ArrayLike) -> FloatArray:
        return ncx2_pdf(x, self.nu, self.nc)

    @cached_property
    def z(self) -> float:
        """Truncation normalizer ``Z_{K,N} = F_{chi'^2}(x_max)`` (= sampler acceptance rate)."""
        return float(self._scdf(self.x_max))

    # ---- distribution functions --------------------------------------------------------
    def pdf(self, x: ArrayLike) -> FloatArray:
        """Density on ``[0, 1]`` (zero elsewhere). (FR-CH-1)"""
        xx = np.asarray(x, dtype=np.float64)
        inside = (xx >= 0.0) & (xx <= 1.0)
        val = self._spdf(np.where(inside, xx, 0.0) / self.sigma2) / (self.sigma2 * self.z)
        return np.asarray(np.where(inside, val, 0.0), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> FloatArray:
        """``F(x) = F_chi'^2(min(x, 1)/sigma^2)/Z``; exactly 1 for ``x >= 1``. (FR-CH-1)

        The clamp ``min(x, 1)`` is mandatory (FSD FR-CH-1).
        """
        xx = np.asarray(x, dtype=np.float64)
        xc = np.clip(xx, 0.0, 1.0)
        val = np.minimum(self._scdf(xc / self.sigma2) / self.z, 1.0)
        return np.asarray(np.where(xx >= 1.0, 1.0, np.where(xx <= 0.0, 0.0, val)), dtype=np.float64)

    def surrogate_cdf(self, x: ArrayLike) -> FloatArray:
        """Untruncated surrogate cdf ``F_chi'^2(x/sigma^2)``, no ``Z``, no clamp. NON-PHYSICAL.

        RQ1 only (``physical = False``): for a deterministic large-scale factor,
        ``F(x) = surrogate_cdf(x) / Z`` for ``x < 1``; the removed mass is ``1 - Z``. (TSD §6.1)
        """
        xx = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
        return self._scdf(xx / self.sigma2)

    surrogate_physical = False

    def sf(self, x: ArrayLike) -> FloatArray:
        """Survival ``P(eta_s > x)``, computed in the smaller tail. (FR-CH-1, TSD §6.1, §13 rule 1)

        For ``F(x) < 1/2`` it is ``1 - F``; otherwise
        ``[S_chi'^2(x/sigma^2) - S_chi'^2(x_max)] / Z`` (no ``1 - F`` near 1).
        """
        xx = np.asarray(x, dtype=np.float64)
        xc = np.clip(xx, 0.0, 1.0)
        f = self.cdf(xc)
        upper = (self._ssf(xc / self.sigma2) - self._ssf(self.x_max)) / self.z
        val = np.where(f < 0.5, 1.0 - f, np.maximum(upper, 0.0))
        return np.asarray(np.where(xx >= 1.0, 0.0, np.where(xx <= 0.0, 1.0, val)), dtype=np.float64)

    def ppf(self, q: ArrayLike) -> FloatArray:
        """Inverse cdf on ``q in [0, 1]``. (FR-CH-1)"""
        qq = np.asarray(q, dtype=np.float64)
        if np.any((qq < 0.0) | (qq > 1.0)):
            raise ValueError("q must be in [0, 1]")
        val = ncx2_ppf(qq * self.z, self.nu, self.nc) * self.sigma2
        return np.asarray(np.where(qq >= 1.0, 1.0, np.minimum(val, 1.0)), dtype=np.float64)

    # ---- moments -----------------------------------------------------------------------
    def neglected_mass(self, x_lo: float, x_hi: float) -> float:
        """``M_out``: surrogate mass inside ``[0, x_max]`` but outside ``[x_lo, x_hi]``."""
        out = float(self._scdf(x_lo)) if x_lo > 0.0 else 0.0
        if x_hi < self.x_max:
            out += float(self._ssf(x_hi) - self._ssf(self.x_max))
        return max(out, 0.0)

    def moment_window(self) -> tuple[float, float, float]:
        """Adaptive ``X``-space window ``(x_lo, x_hi, w)`` with ``M_out <= 1e-14 Z``. (TSD §6.1)

        Starts at ``w = 20`` surrogate sd about the mean and doubles ``w`` (cap 320, then raises)
        until the neglected mass is at most ``MASS_TOL * Z``.
        """
        mean = self.nu + self.nc
        sd = float(np.sqrt(2.0 * (self.nu + 2.0 * self.nc)))
        w = W_START
        while True:
            lo = max(0.0, mean - w * sd)
            hi = min(self.x_max, mean + w * sd)
            if self.neglected_mass(lo, hi) <= MASS_TOL * self.z:
                return lo, hi, w
            if w >= W_CAP:
                raise ValueError(
                    f"moment window: neglected mass above {MASS_TOL:g} Z at w = {w:g} sd"
                )
            w = min(2.0 * w, W_CAP)

    def moment(self, m: int | float) -> float:
        """Truncated moment ``E^QC[eta_s^m]`` by checked quadrature. (FR-CH-1, TSD §6.1)

        Integrates ``(x/x_max)^m f_chi'^2(x)`` over the adaptive window of
        :meth:`moment_window` (breakpoints at the surrogate mean +/- j sd) and divides by the
        ``m = 0`` integral. Any non-finite pdf value in the window raises; it is never masked.
        """
        if m < 0:
            raise ValueError("m must be >= 0")
        mean = self.nu + self.nc
        sd = float(np.sqrt(2.0 * (self.nu + 2.0 * self.nc)))
        xm = self.x_max
        lo, hi, _ = self.moment_window()
        pts = [mean + j * sd for j in (-6, -3, -1, 0, 1, 3, 6)]

        def integral(power: float) -> float:
            def integrand(t: float) -> float:
                v = float((t / xm) ** power * self._spdf(t))
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
