"""Optional obstacle generator: compound-Poisson log-loss matched to a Gamma. (FR-CH-4, TSD §6.3)

``M ~ Poisson(lambda_o d)`` obstacles, each with log-loss ``psi_j ~ Gamma(k_o, theta_o)`` i.i.d.,
``G = sum_j psi_j``. The Gamma fit of §6.2 matches the first two moments of ``G`` exactly but cannot
represent the atom ``P(G = 0) = exp(-lambda_o d)``. Diagnostic only: there is no pass/fail
threshold on the histogram comparison, and no parameter value is defaulted (all need ``source:``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from qmwlink.channel.large_scale import ShiftedLogGamma

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ObstacleModel:
    """Compound-Poisson obstacle log-loss ``G``. (FR-CH-4, TSD §6.3) [A]

    Args:
        rate_per_m: obstacle density ``lambda_o`` along the path [1/m], ``> 0``.
        k_o: Gamma shape of one obstacle's log-loss, ``> 0``.
        theta_o: Gamma scale of one obstacle's log-loss, ``> 0``.
        d_m: path length [m], ``> 0``.
    """

    rate_per_m: float
    k_o: float
    theta_o: float
    d_m: float

    def __post_init__(self) -> None:
        for name in ("rate_per_m", "k_o", "theta_o", "d_m"):
            v = getattr(self, name)
            if not (np.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be finite and > 0")

    @property
    def mean_count(self) -> float:
        """``E[M] = lambda_o d``."""
        return self.rate_per_m * self.d_m

    @property
    def mu_g(self) -> float:
        """``mu_G = lambda_o d k_o theta_o``. (TSD §6.3)"""
        return self.mean_count * self.k_o * self.theta_o

    @property
    def var_g(self) -> float:
        """``sigma_G^2 = lambda_o d k_o (k_o + 1) theta_o^2``. (TSD §6.3)"""
        return self.mean_count * self.k_o * (self.k_o + 1.0) * self.theta_o**2

    @property
    def atom_at_zero(self) -> float:
        """``P(G = 0) = exp(-lambda_o d)``; not representable by the Gamma fit."""
        return float(np.exp(-self.mean_count))

    def matched_loggamma(self, psi0: float = 0.0) -> ShiftedLogGamma:
        """Shifted log-gamma (§6.2) with the same mean and variance of ``G``. (FR-CH-4)"""
        law = ShiftedLogGamma.from_moments(psi0, self.mu_g, self.var_g)
        assert isinstance(law, ShiftedLogGamma)
        return law

    def sample_g(self, rng: np.random.Generator, size: int) -> FloatArray:
        """Exact compound-Poisson samples of ``G`` (sum of ``M`` Gammas is ``Gamma(M k_o)``)."""
        if size < 0:
            raise ValueError("size must be >= 0")
        m = rng.poisson(self.mean_count, size=size)
        g = np.zeros(size, dtype=np.float64)
        hit = m > 0
        g[hit] = rng.gamma(self.k_o * m[hit], self.theta_o)
        return g


@dataclass(frozen=True)
class ObstacleDiagnostic:
    """Histogram of exact ``G`` samples against the fitted Gamma density (diagnostic only)."""

    bin_centers: FloatArray
    hist_density: FloatArray
    gamma_density: FloatArray
    atom_at_zero: float
    sample_fraction_zero: float
    max_abs_cdf_diff: float


def diagnose(
    model: ObstacleModel, rng: np.random.Generator, size: int = 10**6, bins: int = 100
) -> ObstacleDiagnostic:
    """MC histogram of the compound-Poisson ``G`` vs the fitted Gamma. (FR-CH-4 AC)

    Reported, not gated: the Kolmogorov distance includes the atom at 0.
    """
    g = model.sample_g(rng, size)
    law = model.matched_loggamma(0.0)
    hi = float(np.quantile(g, 0.999)) if np.any(g > 0) else 1.0
    edges = np.linspace(0.0, max(hi, np.finfo(float).tiny), bins + 1)
    hist, _ = np.histogram(g, bins=edges, density=False)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    xs = np.sort(g)
    ecdf = np.arange(1, size + 1) / size
    diff = float(np.max(np.abs(ecdf - law.psi_cdf(xs)))) if size else float("nan")
    return ObstacleDiagnostic(
        centers,
        hist / (size * widths),
        law.psi_pdf(centers),
        model.atom_at_zero,
        float(np.mean(g == 0.0)),
        diff,
    )
