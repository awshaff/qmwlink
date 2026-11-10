"""Large-scale PDT: shifted log-gamma (primary) and log-normal baselines. (FR-CH-2, FR-CH-3)

All laws live in the log-loss variable ``Psi`` with ``eta_l = exp(-Psi)``. (TSD §6.2)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special, stats

from qmwlink.numerics.special import gamma_lower_regularized, gamma_upper_regularized

FloatArray = NDArray[np.float64]

LARGE_SHAPE = 1e5
"""Shape above which ``gammainc``/``gammaincc`` go through the ``ncx2`` floor helper (TSD §6.2)."""

BREAK_SIGMAS = (-6, -3, -1, 0, 1, 3, 6)
"""Quadrature breakpoints at ``center + j * sigma`` (TSD §6.4)."""


class LargeScaleLaw(ABC):
    """Interface shared by all large-scale laws (``Psi``-space primitives, ``eta``-space views)."""

    physical: bool = True
    """False for laws that put mass on ``eta_l > 1`` (V0 log-normal)."""

    # ---- Psi-space primitives ----------------------------------------------------------
    @abstractmethod
    def psi_pdf(self, psi: ArrayLike) -> FloatArray:
        """Density of ``Psi``."""

    @abstractmethod
    def psi_cdf(self, psi: ArrayLike) -> FloatArray:
        """``P(Psi <= psi)``."""

    @abstractmethod
    def psi_sf(self, psi: ArrayLike) -> FloatArray:
        """``P(Psi >= psi)`` (equals ``P(Psi > psi)`` for continuous laws)."""

    @abstractmethod
    def psi_lo_eff(self) -> float:
        """Finite lower end of the ``Psi`` integration domain (neglected mass < 1e-30)."""

    @abstractmethod
    def psi_breakpoints(self) -> list[float]:
        """Quadrature breakpoints in ``Psi``."""

    @abstractmethod
    def moment(self, m: float) -> float:
        """``E[eta_l^m]``."""

    @abstractmethod
    def sample_psi(self, rng: np.random.Generator, size: int) -> FloatArray:
        """Draw ``Psi``."""

    # ---- eta-space views ---------------------------------------------------------------
    def sample_eta(self, rng: np.random.Generator, size: int) -> FloatArray:
        """Draw ``eta_l = exp(-Psi)``."""
        return np.asarray(np.exp(-self.sample_psi(rng, size)), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> FloatArray:
        """``P(eta_l <= x) = P(Psi >= -ln x)``."""
        xx = np.asarray(x, dtype=np.float64)
        with np.errstate(divide="ignore"):
            val = self.psi_sf(-np.log(np.where(xx > 0, xx, 1.0)))
        return np.asarray(np.where(xx > 0, val, 0.0), dtype=np.float64)

    def sf(self, x: ArrayLike) -> FloatArray:
        """``P(eta_l > x) = P(Psi < -ln x)``."""
        xx = np.asarray(x, dtype=np.float64)
        val = self.psi_cdf(-np.log(np.where(xx > 0, xx, 1.0)))
        return np.asarray(np.where(xx > 0, val, 1.0), dtype=np.float64)

    def pdf(self, x: ArrayLike) -> FloatArray:
        """Density of ``eta_l``: ``f_Psi(-ln x) / x``."""
        xx = np.asarray(x, dtype=np.float64)
        safe = np.where(xx > 0, xx, 1.0)
        return np.asarray(
            np.where(xx > 0, self.psi_pdf(-np.log(safe)) / safe, 0.0), dtype=np.float64
        )


@dataclass(frozen=True)
class ShiftedLogGamma(LargeScaleLaw):
    """``Psi = Psi0 + G``, ``G ~ Gamma(k, theta)`` (shape, scale). (FR-CH-2, TSD §6.2 [D])

    ``eta_l = exp(-Psi)`` has support ``(0, exp(-Psi0)]``; ``Psi0 = gamma_abs * d >= 0``.
    """

    psi0: float
    k: float
    theta: float

    physical = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.psi0) or self.psi0 < 0:
            raise ValueError("psi0 must be finite and >= 0")
        if not (np.isfinite(self.k) and self.k > 0):
            raise ValueError("k must be finite and > 0")
        if not (np.isfinite(self.theta) and self.theta > 0):
            raise ValueError("theta must be finite and > 0")

    @classmethod
    def from_moments(
        cls, psi0: float, mu_g: float, var_g: float
    ) -> ShiftedLogGamma | DeterministicLoss:
        """Moment-match ``G``: ``k = mu^2/var``, ``theta = var/mu``. (FR-CH-2, TSD §6.2)

        ``var_g == 0`` returns the point mass :class:`DeterministicLoss` at ``psi0 + mu_g``.
        """
        if mu_g < 0 or var_g < 0:
            raise ValueError("mu_g and var_g must be >= 0")
        if var_g == 0.0 or mu_g == 0.0:
            if var_g != 0.0:
                raise ValueError("mu_g = 0 requires var_g = 0")
            return DeterministicLoss(psi0 + mu_g)
        return cls(psi0, mu_g**2 / var_g, var_g / mu_g)

    @property
    def mu_g(self) -> float:
        """Mean of ``G``."""
        return self.k * self.theta

    @property
    def sigma_g(self) -> float:
        """Standard deviation of ``G``."""
        return float(np.sqrt(self.k) * self.theta)

    @property
    def eta_max(self) -> float:
        """Upper end of the ``eta_l`` support, ``exp(-Psi0)``."""
        return float(np.exp(-self.psi0))

    def psi_pdf(self, psi: ArrayLike) -> FloatArray:
        """Log-domain gamma density of ``G = Psi - Psi0``; zero for ``G <= 0``."""
        u = np.asarray(psi, dtype=np.float64) - self.psi0
        pos = u > 0
        us = np.where(pos, u, 1.0)
        logf = (
            special.xlogy(self.k - 1.0, us)
            - us / self.theta
            - special.gammaln(self.k)
            - self.k * np.log(self.theta)
        )
        return np.asarray(np.where(pos, np.exp(logf), 0.0), dtype=np.float64)

    def psi_cdf(self, psi: ArrayLike) -> FloatArray:
        """``P(Psi <= psi) = P(k, u/theta)`` (regularized lower incomplete gamma)."""
        u = np.asarray(psi, dtype=np.float64) - self.psi0
        z = np.maximum(u, 0.0) / self.theta
        p = (
            gamma_lower_regularized(self.k, z)
            if self.k > LARGE_SHAPE
            else special.gammainc(self.k, z)
        )
        return np.asarray(np.where(u > 0, p, 0.0), dtype=np.float64)

    def psi_sf(self, psi: ArrayLike) -> FloatArray:
        """``P(Psi >= psi) = Q(k, u/theta)``; equals 1 for ``u <= 0`` (``gammaincc``)."""
        u = np.asarray(psi, dtype=np.float64) - self.psi0
        z = np.maximum(u, 0.0) / self.theta
        q = (
            gamma_upper_regularized(self.k, z)
            if self.k > LARGE_SHAPE
            else special.gammaincc(self.k, z)
        )
        return np.asarray(np.where(u > 0, q, 1.0), dtype=np.float64)

    def psi_lo_eff(self) -> float:
        return self.psi0

    def psi_breakpoints(self) -> list[float]:
        c = self.psi0 + self.mu_g
        return [c + j * self.sigma_g for j in BREAK_SIGMAS if c + j * self.sigma_g > self.psi0]

    def moment(self, m: float) -> float:
        """``E[eta_l^m] = exp(-m Psi0) (1 + m theta)^(-k)``. (FR-CH-2, TSD §6.2 [V])"""
        if m < 0:
            raise ValueError("m must be >= 0")
        return float(np.exp(-m * self.psi0 - self.k * np.log1p(m * self.theta)))

    def sample_psi(self, rng: np.random.Generator, size: int) -> FloatArray:
        return np.asarray(self.psi0 + rng.gamma(self.k, self.theta, size=size), dtype=np.float64)


@dataclass(frozen=True)
class DeterministicLoss(LargeScaleLaw):
    """Point mass ``Psi = psi`` (the ``sigma_G -> 0`` limit). (FR-CH-2, TSD §6.2)"""

    psi: float

    physical = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.psi) or self.psi < 0:
            raise ValueError("psi must be finite and >= 0")

    def psi_pdf(self, psi: ArrayLike) -> FloatArray:
        raise TypeError("a point mass has no density; handle DeterministicLoss explicitly")

    def psi_cdf(self, psi: ArrayLike) -> FloatArray:
        return np.asarray(np.asarray(psi, dtype=np.float64) >= self.psi, dtype=np.float64)

    def psi_sf(self, psi: ArrayLike) -> FloatArray:
        return np.asarray(np.asarray(psi, dtype=np.float64) <= self.psi, dtype=np.float64)

    # The step is compared in eta-space: exp(-ln x) round-trips are not exact in floating point.
    def cdf(self, x: ArrayLike) -> FloatArray:
        """Step at ``exp(-psi)``: ``P(eta_l <= x) = 1`` iff ``x >= exp(-psi)``."""
        return np.asarray(np.asarray(x, dtype=np.float64) >= np.exp(-self.psi), dtype=np.float64)

    def sf(self, x: ArrayLike) -> FloatArray:
        """``P(eta_l > x) = 1`` iff ``x < exp(-psi)``."""
        return np.asarray(np.asarray(x, dtype=np.float64) < np.exp(-self.psi), dtype=np.float64)

    def psi_lo_eff(self) -> float:
        return self.psi

    def psi_breakpoints(self) -> list[float]:
        return [self.psi]

    def moment(self, m: float) -> float:
        if m < 0:
            raise ValueError("m must be >= 0")
        return float(np.exp(-m * self.psi))

    def sample_psi(self, rng: np.random.Generator, size: int) -> FloatArray:
        return np.full(size, self.psi, dtype=np.float64)


def _check_lognormal(psi_bar: float, sigma_psi: float) -> None:
    if not np.isfinite(psi_bar) or psi_bar <= 0:
        raise ValueError("psi_bar must be finite and > 0")
    if not (np.isfinite(sigma_psi) and sigma_psi > 0):
        raise ValueError("sigma_psi must be finite and > 0")


@dataclass(frozen=True)
class LogNormalV0(LargeScaleLaw):
    """V0 baseline: ``Psi ~ N(psi_bar, sigma_psi^2)``, untruncated. NON-PHYSICAL. (FR-CH-3)

    Puts mass ``Phi(-psi_bar/sigma_psi)`` on ``eta_l > 1`` (``physical = False``).
    """

    psi_bar: float
    sigma_psi: float

    physical = False

    def __post_init__(self) -> None:
        _check_lognormal(self.psi_bar, self.sigma_psi)

    def mass_above_one(self) -> float:
        """``P(eta_l > 1) = Phi(-psi_bar / sigma_psi)``; 0.158655 for ``sigma = psi_bar``."""
        return float(stats.norm.cdf(-self.psi_bar / self.sigma_psi))

    def psi_pdf(self, psi: ArrayLike) -> FloatArray:
        return np.asarray(stats.norm.pdf(psi, self.psi_bar, self.sigma_psi), dtype=np.float64)

    def psi_cdf(self, psi: ArrayLike) -> FloatArray:
        return np.asarray(stats.norm.cdf(psi, self.psi_bar, self.sigma_psi), dtype=np.float64)

    def psi_sf(self, psi: ArrayLike) -> FloatArray:
        return np.asarray(stats.norm.sf(psi, self.psi_bar, self.sigma_psi), dtype=np.float64)

    def psi_lo_eff(self) -> float:
        return self.psi_bar - 12.0 * self.sigma_psi

    def psi_breakpoints(self) -> list[float]:
        return [self.psi_bar + j * self.sigma_psi for j in BREAK_SIGMAS]

    def moment(self, m: float) -> float:
        """``E[eta_l^m] = exp(-m psi_bar + m^2 sigma^2 / 2)``; can exceed 1 (non-physical)."""
        if m < 0:
            raise ValueError("m must be >= 0")
        return float(np.exp(-m * self.psi_bar + 0.5 * m**2 * self.sigma_psi**2))

    def sample_psi(self, rng: np.random.Generator, size: int) -> FloatArray:
        return np.asarray(rng.normal(self.psi_bar, self.sigma_psi, size=size), dtype=np.float64)

    def sample_eta_clipped(self, rng: np.random.Generator, size: int) -> tuple[FloatArray, float]:
        """Policy ``eta_gt1: clip``: ``min(eta_l, 1)`` and the affected mass. (TSD §6.2, FR-CH-3)

        Returns the clipped samples and the analytic clipped mass ``mass_above_one()``. Results
        built from these samples are ``physical = False`` and must report that mass.
        """
        return np.minimum(self.sample_eta(rng, size), 1.0), self.mass_above_one()


@dataclass(frozen=True)
class ConditionedLogNormal(LargeScaleLaw):
    """Log-normal conditioned on ``Psi >= 0`` (normalizer ``Phi(psi_bar/sigma_psi)``). (FR-CH-3)"""

    psi_bar: float
    sigma_psi: float

    physical = True

    def __post_init__(self) -> None:
        _check_lognormal(self.psi_bar, self.sigma_psi)

    @property
    def normalizer(self) -> float:
        """``P(Psi >= 0) = Phi(psi_bar / sigma_psi)``."""
        return float(stats.norm.cdf(self.psi_bar / self.sigma_psi))

    def psi_pdf(self, psi: ArrayLike) -> FloatArray:
        p = np.asarray(psi, dtype=np.float64)
        val = stats.norm.pdf(p, self.psi_bar, self.sigma_psi) / self.normalizer
        return np.asarray(np.where(p >= 0, val, 0.0), dtype=np.float64)

    def psi_sf(self, psi: ArrayLike) -> FloatArray:
        p = np.asarray(psi, dtype=np.float64)
        val = stats.norm.sf(p, self.psi_bar, self.sigma_psi) / self.normalizer
        return np.asarray(np.where(p >= 0, val, 1.0), dtype=np.float64)

    def psi_cdf(self, psi: ArrayLike) -> FloatArray:
        p = np.asarray(psi, dtype=np.float64)
        num = stats.norm.cdf(p, self.psi_bar, self.sigma_psi) - stats.norm.cdf(
            0.0, self.psi_bar, self.sigma_psi
        )
        return np.asarray(np.where(p >= 0, num / self.normalizer, 0.0), dtype=np.float64)

    def psi_lo_eff(self) -> float:
        return 0.0

    def psi_breakpoints(self) -> list[float]:
        pts = [self.psi_bar + j * self.sigma_psi for j in BREAK_SIGMAS]
        return [p for p in pts if p > 0]

    def moment(self, m: float) -> float:
        """``E[e^{-m Psi} | Psi >= 0]``: lognormal moment times a Phi-ratio (see code)."""
        if m < 0:
            raise ValueError("m must be >= 0")
        s = self.sigma_psi
        lead = np.exp(-m * self.psi_bar + 0.5 * m**2 * s**2)
        return float(lead * stats.norm.cdf((self.psi_bar - m * s**2) / s) / self.normalizer)

    def sample_psi(self, rng: np.random.Generator, size: int) -> FloatArray:
        a = (0.0 - self.psi_bar) / self.sigma_psi
        return np.asarray(
            stats.truncnorm.rvs(
                a, np.inf, loc=self.psi_bar, scale=self.sigma_psi, size=size, random_state=rng
            ),
            dtype=np.float64,
        )


def matched_laws(
    psi0: float, mu_g: float, sigma_g: float
) -> tuple[ShiftedLogGamma | DeterministicLoss, LogNormalV0, ConditionedLogNormal]:
    """RQ2 law set sharing the first two moments of ``Psi`` and the same ``Psi0``. (FR-CH-3 (a))

    ``Psi_bar = psi0 + mu_g`` and ``sigma_Psi = sigma_g`` for all three laws. The conditioned
    log-normal matches only *before* conditioning; its post-conditioning moments differ.
    ``sigma_g = 0`` is the deterministic limit, where only the log-gamma point mass exists, so it
    raises here (use ``DeterministicLoss`` directly; the baselines need ``sigma_psi > 0``).
    """
    if sigma_g <= 0:
        raise ValueError("sigma_g must be > 0 (deterministic limit: use DeterministicLoss)")
    psi_bar = psi0 + mu_g
    return (
        ShiftedLogGamma.from_moments(psi0, mu_g, sigma_g**2),
        LogNormalV0(psi_bar, sigma_g),
        ConditionedLogNormal(psi_bar, sigma_g),
    )
