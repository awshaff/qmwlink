"""Rydberg atomic receiver from field sensitivity. (FR-RX-4, TSD §7.4)

``N_amp = S_E^2 A_e / (eta_0 h f)`` with the bare-cell short-dipole aperture ``A_e``. Neither the
collecting aperture ``A_r`` nor the enhancement factor ``G_enh`` enter ``N_amp`` (they cancel); the
mapping therefore accepts no aperture argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from qmwlink.constants import ETA0, K_B, H
from qmwlink.receivers.base import N_AMP_MIN, MeasuringReceiver, Sideband, check_sideband
from qmwlink.thermal import effective_aperture_m2, s_e_bbr

if TYPE_CHECKING:
    from qmwlink.receivers.rydberg_physics.doppler import DopplerSettings
    from qmwlink.receivers.rydberg_physics.model import RARParams

FieldConvention = Literal["rms", "peak"]


def n_amp_from_s_e(s_e_rms: float, f_hz: float) -> float:
    """``N_amp = S_E^2 A_e/(eta_0 h f)`` for an rms, intrinsic ``S_E`` [V/m/sqrt(Hz)].

    Raises if the result is below 1/2; never clamps. No aperture argument by design (§7.4).
    """
    if not (np.isfinite(s_e_rms) and s_e_rms >= 0):
        raise ValueError("s_e must be finite and >= 0")
    if f_hz <= 0:
        raise ValueError("f_hz must be > 0")
    a_e = float(effective_aperture_m2(f_hz))
    n = s_e_rms**2 * a_e / (ETA0 * H * f_hz)
    if n < N_AMP_MIN:
        raise ValueError(f"S_E maps to N_amp = {n:.6g} < 1/2: below the quantum limit")
    return float(n)


@dataclass(frozen=True)
class RydbergReceiver(MeasuringReceiver):
    """Rydberg atomic receiver (a measuring receiver, default sideband ``dsb``). (FR-RX-4)"""

    f_hz: float = 0.0
    e_1db_pk_v_per_m: float | None = None

    @classmethod
    def from_sensitivity(
        cls,
        s_e: float,
        f_hz: float,
        *,
        field_convention: FieldConvention,
        sideband: Sideband = "dsb",
        bandwidth_hz: float | None = None,
        includes_bbr: bool = False,
        t_env_k: float = 300.0,
        e_1db_pk_v_per_m: float | None = None,
    ) -> RydbergReceiver:
        """Build from ``S_E`` with a *required* ``field_convention`` (``rms`` | ``peak``).

        A ``peak`` value is divided by ``sqrt(2)``. If ``includes_bbr``, the in-band BBR
        sensitivity is subtracted in power first (error if negative). (TSD §7.4)
        """
        if field_convention not in ("rms", "peak"):
            raise ValueError(f"field_convention must be 'rms' or 'peak', got {field_convention!r}")
        s = s_e / np.sqrt(2.0) if field_convention == "peak" else s_e
        if includes_bbr:
            diff = s**2 - float(s_e_bbr(f_hz, t_env_k)) ** 2
            if diff < 0:
                raise ValueError("S_E measured including BBR is below the BBR floor")
            s = float(np.sqrt(diff))
        return cls(
            n_amp_from_s_e(float(s), f_hz),
            check_sideband(sideband),
            bandwidth_hz,
            f_hz,
            e_1db_pk_v_per_m,
        )

    @classmethod
    def from_n_amp(
        cls,
        n_amp: float,
        f_hz: float,
        *,
        sideband: Sideband = "dsb",
        bandwidth_hz: float | None = None,
        e_1db_pk_v_per_m: float | None = None,
    ) -> RydbergReceiver:
        """Direct ``N_amp`` input."""
        return cls(n_amp, check_sideband(sideband), bandwidth_hz, f_hz, e_1db_pk_v_per_m)

    @classmethod
    def from_noise_temperature(
        cls,
        t_n_k: float,
        f_hz: float,
        *,
        sideband: Sideband = "dsb",
        bandwidth_hz: float | None = None,
        e_1db_pk_v_per_m: float | None = None,
    ) -> RydbergReceiver:
        """Direct ``T_n`` input (RJ convention): ``N_amp = k_B T_n/(h f)``."""
        return cls.from_n_amp(
            K_B * t_n_k / (H * f_hz),
            f_hz,
            sideband=sideband,
            bandwidth_hz=bandwidth_hz,
            e_1db_pk_v_per_m=e_1db_pk_v_per_m,
        )

    @classmethod
    def from_physics(
        cls,
        params: RARParams,
        *,
        engine: Literal["weak_probe", "full"] = "weak_probe",
        doppler: DopplerSettings | None = None,
        linear_range: bool = True,
    ) -> RydbergReceiver:
        """Model-derived receiver: ``S_E``, ``B_3dB`` and ``E_1dB`` from TSD §14. (FR-RX-6)

        ``S_E`` from the model is rms and intrinsic (photon shot noise and detector NEP only), the
        readout is envelope-sensitive, hence ``field_convention="rms"`` and ``sideband="dsb"``;
        ``E_1dB`` is a peak field (``None`` when the model finds no 1 dB point below the LO).
        """
        from qmwlink.receivers.rydberg_physics.model import LindbladRARModel

        res = LindbladRARModel(params, doppler).evaluate(engine, linear_range=linear_range)
        e_1db = res.e_1db_pk_v_per_m if np.isfinite(res.e_1db_pk_v_per_m) else None
        return cls.from_sensitivity(
            res.s_e_rms,
            params.f_mw_hz,
            field_convention="rms",
            sideband="dsb",
            bandwidth_hz=res.b_3db_hz,
            e_1db_pk_v_per_m=e_1db,
        )

    @property
    def t_n_k(self) -> float:
        """Equivalent noise temperature (RJ): ``N_amp h f / k_B``."""
        return float(self.n_amp * H * self.f_hz / K_B)

    @property
    def s_e_rms(self) -> float:
        """rms intrinsic sensitivity implied by ``N_amp`` (inverse of the mapping)."""
        a_e = float(effective_aperture_m2(self.f_hz))
        return float(np.sqrt(self.n_amp * ETA0 * H * self.f_hz / a_e))

    def e_rx_pk(self, p_rx_w: float, a_r_m2: float) -> float:
        """Received peak field ``sqrt(2 eta_0 P_rx / A_r)`` [V/m]. (TSD §7.4)"""
        if p_rx_w < 0 or a_r_m2 <= 0:
            raise ValueError("p_rx_w must be >= 0 and a_r_m2 > 0")
        return float(np.sqrt(2.0 * ETA0 * p_rx_w / a_r_m2))

    def linear_regime(self, p_rx_w: float, a_r_m2: float) -> bool:
        """False iff the peak field exceeds ``E_1dB,pk`` (peak vs peak only); numbers unchanged."""
        if self.e_1db_pk_v_per_m is None:
            return True
        return bool(self.e_rx_pk(p_rx_w, a_r_m2) <= self.e_1db_pk_v_per_m)
