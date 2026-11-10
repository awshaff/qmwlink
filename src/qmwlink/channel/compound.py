"""Compound PDT ``eta = eta_l * eta_s`` (independent). (FR-CH-5, TSD §6.4)

Everything is evaluated in ``Psi``-space, ``eta_l = exp(-Psi)``:

``F(x) = P(eta_l <= x) + int_{eta_l > x} F_s(x e^Psi) f_Psi dPsi`` (small-scale cdf clamped),
``S(x) = int_{eta_l > x} S_s(x e^Psi) f_Psi dPsi``,
``f(x) = int_{eta_l > x} e^Psi f_s(x e^Psi) f_Psi dPsi``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from qmwlink.channel.large_scale import DeterministicLoss, LargeScaleLaw
from qmwlink.channel.small_scale import SmallScale
from qmwlink.numerics.integrate import checked_quad

FloatArray = NDArray[np.float64]

_SMALL_QUANTILES = (1e-8, 1e-4, 1e-2, 0.1, 0.5, 0.9, 0.99, 1.0 - 1e-4)
"""Small-scale quantiles that place extra breakpoints where ``F_s(x e^Psi)`` changes."""


class CompoundPDT:
    """Compound PDT of an independent small-scale and large-scale law. (FR-CH-5, TSD §6.4)"""

    def __init__(self, small: SmallScale, large: LargeScaleLaw) -> None:
        self.small = small
        self.large = large

    # ---- integration plumbing ----------------------------------------------------------
    def _bounds(self, x: float) -> tuple[float, float, list[float]]:
        """Integration interval in Psi (``eta_l > x``) and breakpoints (law, kink, small-scale)."""
        lo = self.large.psi_lo_eff()
        hi = -float(np.log(x))
        pts = list(self.large.psi_breakpoints())
        pts += [hi + float(np.log(q)) for q in self.small.ppf(np.array(_SMALL_QUANTILES)) if q > 0]
        return lo, hi, pts

    def _integrate(self, g: Callable[[float], float], x: float) -> float:
        lo, hi, pts = self._bounds(x)
        return checked_quad(g, lo, hi, points=pts)[0]

    def _deterministic(self) -> float | None:
        return self.large.psi if isinstance(self.large, DeterministicLoss) else None

    def _cdf_direct(self, x: float) -> float:
        if (psi := self._deterministic()) is not None:
            return float(self.small.cdf(x * np.exp(psi)))
        head = float(self.large.cdf(x))
        body = self._integrate(
            lambda p: float(self.small.cdf(x * np.exp(p))) * float(self.large.psi_pdf(p)), x
        )
        return min(head + body, 1.0)

    def _sf_direct(self, x: float) -> float:
        if (psi := self._deterministic()) is not None:
            return float(self.small.sf(x * np.exp(psi)))
        return self._integrate(
            lambda p: float(self.small.sf(x * np.exp(p))) * float(self.large.psi_pdf(p)), x
        )

    # ---- public distribution functions -------------------------------------------------
    def cdf(self, x: ArrayLike) -> FloatArray:
        """``P(eta <= x)``; uses ``1 - S`` when the cdf exceeds 1/2 (smaller tail primary)."""

        def one(v: float) -> float:
            if v <= 0.0:
                return 0.0
            c = self._cdf_direct(v)
            return c if c < 0.5 else 1.0 - self._sf_direct(v)

        return _vectorize(one, x)

    def sf(self, x: ArrayLike) -> FloatArray:
        """``P(eta > x)``; computed directly when it is the smaller tail."""

        def one(v: float) -> float:
            if v <= 0.0:
                return 1.0
            s = self._sf_direct(v)
            return s if s < 0.5 else 1.0 - self._cdf_direct(v)

        return _vectorize(one, x)

    def pdf(self, x: ArrayLike) -> FloatArray:
        """Density of ``eta``."""

        def one(v: float) -> float:
            if v <= 0.0:
                return 0.0
            if (psi := self._deterministic()) is not None:
                return float(np.exp(psi) * self.small.pdf(v * np.exp(psi)))
            return self._integrate(
                lambda p: (
                    float(np.exp(p))
                    * float(self.small.pdf(v * np.exp(p)))
                    * float(self.large.psi_pdf(p))
                ),
                v,
            )

        return _vectorize(one, x)

    def ppf(self, eps: ArrayLike) -> FloatArray:
        """Quantile ``F^{-1}(eps)`` by ``brentq`` in ``omega = 1 - eta`` on [0, 1].

        (TSD §6.4, §13 rules 3 and 5; ``xtol = 1e-14``.)
        """

        def one(q: float) -> float:
            if not (0.0 < q < 1.0):
                raise ValueError("eps must be in (0, 1)")
            f_hi = float(self._cdf_scalar(1.0))  # omega = 0
            if f_hi < q:
                raise ValueError("quantile above the support: cdf(1) < eps")

            def g(omega: float) -> float:
                return self._cdf_scalar(1.0 - omega) - q

            if g(1.0) >= 0.0 or g(0.0) <= 0.0:
                raise ValueError("no sign change for the quantile bracket (TSD §13 rule 5)")
            return 1.0 - float(brentq(g, 0.0, 1.0, xtol=1e-14, rtol=4 * np.finfo(float).eps))

        return _vectorize(one, eps)

    def _cdf_scalar(self, x: float) -> float:
        return float(self.cdf(x))

    def moment(self, m: int | float) -> float:
        """``E[eta^m] = E[eta_l^m] * E^QC[eta_s^m]`` using truncated small-scale moments."""
        return self.large.moment(m) * self.small.moment(m)

    # ---- sampling ----------------------------------------------------------------------
    def sample(self, rng: np.random.Generator, size: int) -> FloatArray:
        """Product of independent ``sample_sum`` and large-scale draws. (TSD §6.4)"""
        return np.asarray(
            self.small.sample_sum(rng, size) * self.large.sample_eta(rng, size), dtype=np.float64
        )


def _vectorize(func: Callable[[float], float], x: ArrayLike) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64)
    out = np.fromiter((func(float(v)) for v in arr.ravel()), dtype=np.float64, count=arr.size)
    return out.reshape(arr.shape)
