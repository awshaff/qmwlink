"""Doppler averaging: velocity grid, Maxwell-Boltzmann weights, refinement. (FR-LIN-4, TSD §14.6)

Velocity ``v`` along the probe; uniform grid on ``[-n_sigma sigma_v, n_sigma sigma_v]`` with
normalized weights. The grid starts at ``n_v_start`` classes and doubles its number of intervals
until the relative change of the averaged quantity is below ``rtol``; the final ``N_v`` is recorded.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from qmwlink.constants import K_B

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def sigma_v(t_vapor_k: float, mass_kg: float) -> float:
    """1D thermal velocity spread ``sqrt(k_B T / m)`` [m/s] (169.4 m/s for 87Rb at 300 K)."""
    if t_vapor_k <= 0 or mass_kg <= 0:
        raise ValueError("t_vapor_k and mass_kg must be > 0")
    return float(np.sqrt(K_B * t_vapor_k / mass_kg))


@dataclass(frozen=True)
class DopplerSettings:
    """Grid and convergence settings (TSD §12 ``doppler`` block, §14.6)."""

    n_sigma: float = 5.0
    n_v_start: int = 2**12 + 1
    rtol: float = 1e-5
    n_v_max: int = 2**20 + 1

    def __post_init__(self) -> None:
        if self.n_sigma <= 0:
            raise ValueError("n_sigma must be > 0")
        if self.n_v_start < 3 or self.n_v_max < self.n_v_start:
            raise ValueError("need 3 <= n_v_start <= n_v_max")
        if not 0 < self.rtol < 1:
            raise ValueError("rtol must be in (0, 1)")


def velocity_grid(sigma: float, n_v: int, n_sigma: float = 5.0) -> tuple[FloatArray, FloatArray]:
    """Uniform velocities on ``[-n_sigma sigma, n_sigma sigma]`` and weights summing to 1."""
    if sigma <= 0 or n_v < 3:
        raise ValueError("sigma must be > 0 and n_v >= 3")
    v = np.linspace(-n_sigma * sigma, n_sigma * sigma, n_v)
    w = np.exp(-0.5 * (v / sigma) ** 2)
    return v, w / w.sum()


@dataclass(frozen=True)
class DopplerAverage:
    """Converged average, the final ``N_v`` and the last relative change."""

    value: ComplexArray
    n_v: int
    rel_change: float
    converged: bool


def doppler_average(
    fn: Callable[[FloatArray], ComplexArray],
    sigma: float,
    settings: DopplerSettings | None = None,
    *,
    n_v: int | None = None,
) -> DopplerAverage:
    """Maxwell-Boltzmann average of ``fn(v)`` with automatic grid refinement. (FR-LIN-4)

    ``fn`` maps velocities of shape ``(N_v,)`` to an array whose *last* axis is the velocity
    axis. With ``n_v`` given the grid is fixed (no refinement). The loop doubles the number of
    intervals until ``max|new - old| <= rtol max|new|``; reaching ``n_v_max`` without that raises.
    """
    st = settings or DopplerSettings()

    def avg(n: int) -> ComplexArray:
        v, w = velocity_grid(sigma, n, st.n_sigma)
        out = np.asarray(fn(v))
        if not np.all(np.isfinite(out)):
            raise FloatingPointError("non-finite value in the Doppler integrand")
        return np.asarray(out @ w, dtype=np.complex128)

    if n_v is not None:
        return DopplerAverage(avg(n_v), n_v, float("nan"), True)
    n = st.n_v_start
    old = avg(n)
    while True:
        n_new = 2 * (n - 1) + 1
        if n_new > st.n_v_max:
            raise RuntimeError(
                f"Doppler average not converged to rtol={st.rtol:g} at N_v={n} (n_v_max reached)"
            )
        new = avg(n_new)
        scale = float(np.max(np.abs(new)))
        change = float(np.max(np.abs(new - old))) / scale if scale > 0 else 0.0
        n, old = n_new, new
        if change <= st.rtol:
            return DopplerAverage(new, n, change, True)
