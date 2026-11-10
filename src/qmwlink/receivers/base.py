"""Receiver base types and ideal receivers. (FR-RX-1, FR-RX-2, TSD §7.1-7.2)

A *measuring* receiver produces classical samples and is characterised by the input-referred
added quanta ``N_amp >= 1/2`` and a sideband mode. A *quantum front end* is a
:class:`~qmwlink.gaussian.channels.PIChannel` inserted before quantum processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qmwlink.gaussian.channels import PIChannel

FloatArray = NDArray[np.float64]
Sideband = Literal["ssb", "dsb"]

QuantumFrontEnd = PIChannel
"""A quantum front end is just a phase-insensitive channel ``(tau_r, y_r)``. (TSD §7.1)"""

N_AMP_MIN = 0.5
"""Caves bound for phase-insensitive detection of both quadratures."""


def check_sideband(sideband: str) -> Sideband:
    """Validate the (required, no default) sideband string."""
    if sideband not in ("ssb", "dsb"):
        raise ValueError(f"sideband must be 'ssb' or 'dsb', got {sideband!r}")
    return sideband  # type: ignore[return-value]


@dataclass(frozen=True)
class MeasuringReceiver:
    """Measuring receiver: ``N_amp >= 1/2``, required ``sideband``, optional bandwidth ``B``.

    ``N_amp < 1/2`` raises; it is never clamped. (FR-RX-1 (a),(b))
    """

    n_amp: float
    sideband: Sideband
    bandwidth_hz: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.n_amp) or self.n_amp < N_AMP_MIN:
            raise ValueError(f"n_amp must be finite and >= 1/2 (Caves bound), got {self.n_amp!r}")
        check_sideband(self.sideband)
        if self.bandwidth_hz is not None and not (
            np.isfinite(self.bandwidth_hz) and self.bandwidth_hz > 0
        ):
            raise ValueError("bandwidth_hz must be finite and > 0")

    def n_img(
        self,
        eta: ArrayLike,
        n_th: float,
        n_tx_img: float = 0.0,
        *,
        one_minus_eta: ArrayLike | None = None,
    ) -> FloatArray:
        """Image-band occupation: 0 for ``ssb``; ``(1-eta) n_th + eta n_tx_img`` for ``dsb``.

        ``one_minus_eta`` may be given for precision near ``eta -> 1``. (TSD §7.1)
        """
        e = np.asarray(eta, dtype=np.float64)
        if self.sideband == "ssb":
            return np.zeros_like(e)
        if n_th < 0 or n_tx_img < 0:
            raise ValueError("n_th and n_tx_img must be >= 0")
        om = 1.0 - e if one_minus_eta is None else np.asarray(one_minus_eta, dtype=np.float64)
        return np.asarray(om * n_th + e * n_tx_img, dtype=np.float64)

    def noise_photons(
        self,
        eta: ArrayLike,
        n_th: float,
        n_tx_img: float = 0.0,
        *,
        one_minus_eta: ArrayLike | None = None,
    ) -> FloatArray:
        """Input-referred noise ``D = (1-eta) n_th + 1/2 + N_amp + N_img``. (TSD §8.2)"""
        e = np.asarray(eta, dtype=np.float64)
        om = 1.0 - e if one_minus_eta is None else np.asarray(one_minus_eta, dtype=np.float64)
        return np.asarray(
            om * n_th + 0.5 + self.n_amp + self.n_img(e, n_th, n_tx_img, one_minus_eta=om),
            dtype=np.float64,
        )


def ideal_heterodyne(bandwidth_hz: float | None = None) -> MeasuringReceiver:
    """Ideal heterodyne: ``N_amp = 1/2``, ``ssb`` (its 1/2 is the image-band vacuum). (FR-RX-2)"""
    return MeasuringReceiver(0.5, "ssb", bandwidth_hz)


@dataclass(frozen=True)
class IdealHomodyne:
    """Ideal homodyne marker: a separate capacity formula, not an ``N_amp`` value. (TSD §7.2)"""

    bandwidth_hz: float | None = None


def quantum_limited_amplifier(gain: ArrayLike) -> PIChannel:
    """Quantum-limited amplifier front end ``PIChannel(G, G - 1)``. (FR-RX-2, TSD §7.2)"""
    return PIChannel.amplifier(gain)
