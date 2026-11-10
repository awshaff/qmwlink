"""Ladder level scheme, decay channels and coherence rates. (FR-LIN-1, TSD §14.3)

Levels are numbered from 1 as in the TSD: ``|1>`` ground, ``|2>`` intermediate, ``|3>`` Rydberg,
``|4>`` adjacent Rydberg (``n_levels = 4`` only). Rates and Rabi frequencies are in rad/us.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]

TWO_PI = 2.0 * np.pi
GAMMA2_RB87_D2 = TWO_PI * 6.0666
"""``Gamma_2`` of the 87Rb D2 line [rad/us] [L] (Steck, *Rubidium 87 D Line Data*, rev. 2.3)."""


@dataclass(frozen=True)
class Decay:
    """Spontaneous decay ``|from> -> |to>`` at ``rate`` [rad/us] (levels numbered from 1)."""

    from_level: int
    to_level: int
    rate: float


@dataclass(frozen=True)
class LadderScheme:
    """Ladder scheme with ``n_levels`` in {3, 4}. (FR-LIN-1, TSD §14.3)

    Args:
        n_levels: 3 (probe + coupling) or 4 (plus the microwave transition).
        decays: decay channels. ``None`` selects the TSD default ``2->1`` at ``gamma2``,
            ``3->2`` at ``gamma3`` and (4 levels) ``4->1`` at ``gamma4``; a direct ``4->2`` decay
            is parity-forbidden and is never a default.
        gamma_t: transit relaxation rate (reset to ``|1>``), ``>= 0``.
        dephasing: laser dephasing ``gamma_phi,k`` for levels ``2..n_levels`` (ground not
            dephased); empty means none.
    """

    n_levels: int
    gamma2: float = GAMMA2_RB87_D2
    gamma3: float = 0.0
    gamma4: float = 0.0
    gamma_t: float = 0.0
    decays: tuple[Decay, ...] | None = None
    dephasing: tuple[float, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.n_levels not in (3, 4):
            raise ValueError(f"n_levels must be 3 or 4, got {self.n_levels}")
        for name in ("gamma2", "gamma3", "gamma4", "gamma_t"):
            v = getattr(self, name)
            if not (np.isfinite(v) and v >= 0):
                raise ValueError(f"{name} must be finite and >= 0")
        if self.dephasing and len(self.dephasing) != self.n_levels - 1:
            raise ValueError("dephasing needs one rate per level 2..n_levels")
        if any((not np.isfinite(g)) or g < 0 for g in self.dephasing):
            raise ValueError("dephasing rates must be finite and >= 0")
        for d in self.channels:
            ok = 1 <= d.to_level <= self.n_levels and 2 <= d.from_level <= self.n_levels
            if not ok or d.to_level == d.from_level:
                raise ValueError(f"decay {d.from_level}->{d.to_level} is not a valid channel")
            if not (np.isfinite(d.rate) and d.rate >= 0):
                raise ValueError("decay rates must be finite and >= 0")

    @property
    def channels(self) -> tuple[Decay, ...]:
        """The decay channels in force (explicit list, or the TSD §14.3 defaults)."""
        if self.decays is not None:
            return self.decays
        out = [Decay(2, 1, self.gamma2), Decay(3, 2, self.gamma3)]
        if self.n_levels == 4:
            out.append(Decay(4, 1, self.gamma4))
        return tuple(out)

    def gamma_out(self, level: int) -> float:
        """Total spontaneous decay rate out of ``level`` (``Gamma_j^out``)."""
        self._check_level(level)
        return float(sum(d.rate for d in self.channels if d.from_level == level))

    def gamma_phi(self, level: int) -> float:
        """Laser dephasing rate of ``level`` (0 for the ground state)."""
        self._check_level(level)
        if level == 1 or not self.dephasing:
            return 0.0
        return float(self.dephasing[level - 2])

    def coherence_rate(self, level: int) -> float:
        """``gamma_j1 = Gamma_j^out/2 + gamma_t + gamma_phi,j`` for ``j >= 2``. (TSD §14.3)"""
        if level < 2:
            raise ValueError("coherence rates are defined for levels >= 2")
        return 0.5 * self.gamma_out(level) + self.gamma_t + self.gamma_phi(level)

    def _check_level(self, level: int) -> None:
        if not 1 <= level <= self.n_levels:
            raise ValueError(f"level must be in 1..{self.n_levels}")

    def collapse_operators(self) -> list[ComplexArray]:
        """Collapse operators ``L_k`` of the master equation. (TSD §14.3)

        Decays ``sqrt(rate)|to><from|``, transit reset ``sqrt(gamma_t)|1><k|`` for every ``k``
        (exactly ``gamma_t(|1><1| - rho)``), and dephasing ``sqrt(2 gamma_phi,k)|k><k|``.
        """
        n = self.n_levels
        ops: list[ComplexArray] = []

        def ket_bra(i: int, j: int, amp: float) -> ComplexArray:
            m = np.zeros((n, n), dtype=np.complex128)
            m[i - 1, j - 1] = amp
            return m

        for d in self.channels:
            if d.rate > 0:
                ops.append(ket_bra(d.to_level, d.from_level, np.sqrt(d.rate)))
        if self.gamma_t > 0:
            ops += [ket_bra(1, k, np.sqrt(self.gamma_t)) for k in range(1, n + 1)]
        for k in range(2, n + 1):
            g = self.gamma_phi(k)
            if g > 0:
                ops.append(ket_bra(k, k, np.sqrt(2.0 * g)))
        return ops


@dataclass(frozen=True)
class Drive:
    """Optical/microwave drive in core units: Rabi frequencies (peak-field, real, ``>= 0``) and
    detunings [rad/us], wavevectors [rad/m]. ``omega_m`` and ``delta_m`` are ignored for 3 levels.
    (TSD §14.3)"""

    omega_p: float
    omega_c: float
    omega_m: float = 0.0
    delta_p: float = 0.0
    delta_c: float = 0.0
    delta_m: float = 0.0
    k_p: float = 0.0
    k_c: float = 0.0

    def __post_init__(self) -> None:
        for name in ("omega_p", "omega_c", "omega_m"):
            v = getattr(self, name)
            if not (np.isfinite(v) and v >= 0):
                raise ValueError(f"{name} must be finite and >= 0")
        for name in ("delta_p", "delta_c", "delta_m", "k_p", "k_c"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")

    def with_omega_m(self, omega_m: float) -> Drive:
        """Copy with another microwave Rabi frequency."""
        return Drive(self.omega_p, self.omega_c, omega_m, self.delta_p, self.delta_c,
                     self.delta_m, self.k_p, self.k_c)  # fmt: skip
