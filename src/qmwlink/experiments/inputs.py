"""Receiver tables from the sourced inputs: literature RAR points and LNA datasheets (TSD §16).

Interpolation is log-log in frequency with no extrapolation: ``receiver_at`` returns ``None``
outside the covered range (the cell is masked).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qmwlink.experiments.schema import LiteratureEntry, LnaCfg, RarCfg, load_literature
from qmwlink.receivers.base import MeasuringReceiver, Sideband
from qmwlink.receivers.electronic import electronic_receiver
from qmwlink.receivers.rydberg import RydbergReceiver
from qmwlink.services.throughput_map import interp_loglog


@dataclass(frozen=True)
class RarTable:
    """Intrinsic rms ``S_E`` and bandwidth of the published RAR points."""

    entries: tuple[LiteratureEntry, ...]
    sideband_mode: str
    t_env_k: float = 300.0

    @classmethod
    def from_cfg(cls, cfg: RarCfg, t_env_k: float) -> RarTable:
        ent = load_literature(cfg.literature)
        if cfg.require_b_hz:
            ent = [e for e in ent if e.b_hz is not None]
        return cls(tuple(sorted(ent, key=lambda e: e.f_hz)), cfg.sideband, t_env_k)

    @property
    def f_range_hz(self) -> tuple[float, float] | None:
        return (self.entries[0].f_hz, self.entries[-1].f_hz) if self.entries else None

    def _nearest_sideband(self, f_hz: float) -> Sideband:
        i = int(np.argmin([abs(np.log(f_hz / e.f_hz)) for e in self.entries]))
        return self.entries[i].sideband

    def bandwidth_at(self, f_hz: float) -> float:
        """Log-log interpolated bandwidth; ``NaN`` outside the range (entries need ``b_hz``)."""
        pts = [e for e in self.entries if e.b_hz is not None]
        if not pts:
            return float("nan")
        return interp_loglog(f_hz, [e.f_hz for e in pts], [e.b_hz for e in pts if e.b_hz])

    def receiver_at(self, f_hz: float) -> RydbergReceiver | None:
        if not self.entries:
            return None
        s = interp_loglog(
            f_hz,
            [e.f_hz for e in self.entries],
            [e.s_e_intrinsic_rms(self.t_env_k) for e in self.entries],
        )
        b = self.bandwidth_at(f_hz)
        if np.isnan(s) or np.isnan(b):
            return None
        sb: Sideband = (
            self._nearest_sideband(f_hz) if self.sideband_mode == "from_entry"
            else self.sideband_mode  # type: ignore[assignment]
        )  # fmt: skip
        return RydbergReceiver.from_sensitivity(
            s, f_hz, field_convention="rms", sideband=sb, bandwidth_hz=b
        )


@dataclass(frozen=True)
class LnaTable:
    """Room-temperature LNA ``T_n`` and system channel bandwidth tables."""

    cfg: LnaCfg

    @property
    def f_range_hz(self) -> tuple[float, float]:
        f = [p.f_hz for p in self.cfg.bands]
        return min(f), max(f)

    def t_n_at(self, f_hz: float) -> float:
        return interp_loglog(
            f_hz, [p.f_hz for p in self.cfg.bands], [p.T_n_k for p in self.cfg.bands]
        )

    def bandwidth_at(self, f_hz: float) -> float:
        return interp_loglog(f_hz, [p.f_hz for p in self.cfg.B_hz], [p.B for p in self.cfg.B_hz])

    def receiver_at(
        self, f_hz: float, bandwidth_hz: float | None = None
    ) -> MeasuringReceiver | None:
        """LNA at ``f_hz``; ``bandwidth_hz`` overrides the system bandwidth (equal-B case)."""
        t_n = self.t_n_at(f_hz)
        b = self.bandwidth_at(f_hz) if bandwidth_hz is None else bandwidth_hz
        if np.isnan(t_n) or np.isnan(b):
            return None
        return electronic_receiver(
            t_n, f_hz, tn_convention=self.cfg.tn_convention,
            sideband=self.cfg.sideband, bandwidth_hz=b,
        )  # fmt: skip
