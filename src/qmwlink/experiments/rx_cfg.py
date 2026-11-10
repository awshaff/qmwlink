"""Build capacity functions from :class:`RxSpec` (E5, E8)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from qmwlink.experiments.schema import RxSpec
from qmwlink.receivers.base import MeasuringReceiver, ideal_heterodyne
from qmwlink.receivers.electronic import electronic_receiver
from qmwlink.receivers.rydberg import RydbergReceiver
from qmwlink.services.classical import c_het, c_holevo, c_hom, c_rx

FloatArray = NDArray[np.float64]


def measuring_receiver(spec: RxSpec, f_hz: float) -> MeasuringReceiver | None:
    """The measuring receiver of ``spec``, or ``None`` for holevo/hom (no ``N_amp``)."""
    if spec.kind == "het":
        return ideal_heterodyne(spec.bandwidth_hz)
    if spec.kind == "electronic":
        assert spec.t_n_k is not None and spec.tn_convention is not None
        return electronic_receiver(
            spec.t_n_k, f_hz, tn_convention=spec.tn_convention,
            sideband=spec.sideband or "ssb", bandwidth_hz=spec.bandwidth_hz,
        )  # fmt: skip
    if spec.kind == "rydberg":
        assert spec.s_e is not None and spec.field_convention is not None
        return RydbergReceiver.from_sensitivity(
            spec.s_e, f_hz, field_convention=spec.field_convention,
            sideband=spec.sideband or "dsb", bandwidth_hz=spec.bandwidth_hz,
        )  # fmt: skip
    return None


def capacity_fn(
    spec: RxSpec, f_hz: float, n_s: float, n_th_val: float
) -> Callable[[FloatArray], FloatArray]:
    """``eta -> bit per mode`` for ``spec``."""
    if spec.kind == "holevo":
        return lambda e: c_holevo(e, n_s, n_th_val)
    if spec.kind == "hom":
        return lambda e: c_hom(e, n_s, n_th_val)
    if spec.kind == "het":
        return lambda e: c_het(e, n_s, n_th_val)
    rx = measuring_receiver(spec, f_hz)
    assert rx is not None
    return lambda e: c_rx(e, n_s, n_th_val, rx)
