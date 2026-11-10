"""Pydantic config schemas for E3-E6 and E8 (TSD §12). Extended by the M6 runner."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from qmwlink.channel.large_scale import LargeScaleLaw, ShiftedLogGamma
from qmwlink.thermal import s_e_bbr


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GridSpec(_Strict):
    """``num`` points from ``start`` to ``stop``, linear or log spaced."""

    start: float
    stop: float
    num: int = Field(ge=2)
    scale: Literal["lin", "log"] = "lin"

    def values(self) -> np.ndarray:
        if self.scale == "log":
            if self.start <= 0 or self.stop <= 0:
                raise ValueError("log grid needs positive endpoints")
            return np.geomspace(self.start, self.stop, self.num)
        return np.linspace(self.start, self.stop, self.num)


class Environment(_Strict):
    T_env_k: float = Field(gt=0)


class Stochastic(_Strict):
    """Large-scale stochastic loss ``G`` moments (mean, std)."""

    mean: float = Field(ge=0)
    std: float = Field(ge=0)


class LargeScale(_Strict):
    """Large-scale law inputs (TSD §6.2, §12).

    Default: ``stochastic_per_m`` per band, fully correlated path, so
    ``mu_G = m1 d`` and ``sigma_G = s1 d`` (``k`` independent of ``d``). The legacy per-link
    ``stochastic`` is accepted for tests and sweeps only and is flagged ``per_link``.
    """

    model: Literal["loggamma"] = "loggamma"
    psi0_from_absorption: bool = True
    gamma_abs_per_m: dict[float, float]
    """Per band in GHz [1/m]."""
    gamma_abs_source: str | dict[float, str]
    stochastic_per_m: dict[float, Stochastic] | None = None
    stochastic_source: str | None = None
    stochastic: Stochastic | None = None
    correlation: Literal["full"] = "full"
    """Partially correlated paths are not implemented and are rejected."""

    @model_validator(mode="after")
    def _check(self) -> LargeScale:
        if isinstance(self.gamma_abs_source, dict):
            missing = sorted(set(self.gamma_abs_per_m) - set(self.gamma_abs_source))
            if missing:
                raise ValueError(f"gamma_abs source missing for bands {missing} GHz")
        elif not self.gamma_abs_source.strip():
            raise ValueError("gamma_abs_source must not be empty")
        if (self.stochastic_per_m is None) == (self.stochastic is None):
            raise ValueError("give exactly one of stochastic_per_m and stochastic (per-link)")
        if self.stochastic_per_m is not None:
            if not self.stochastic_source:
                raise ValueError("stochastic_per_m needs stochastic_source")
            missing_b = sorted(set(self.gamma_abs_per_m) - set(self.stochastic_per_m))
            if missing_b:
                raise ValueError(f"stochastic_per_m missing for bands {missing_b} GHz")
        return self

    @property
    def per_link(self) -> bool:
        """True for the legacy per-link form (flagged in ``meta.json``)."""
        return self.stochastic is not None

    def gamma_abs_at(self, f_ghz: float) -> float:
        """Absorption at ``f_ghz``: exact at configured bands, linear in ``log f`` between."""
        bands = sorted(self.gamma_abs_per_m)
        vals = [self.gamma_abs_per_m[b] for b in bands]
        return float(np.interp(np.log(f_ghz), np.log(bands), vals))

    def moments_g(self, f_ghz: float, d_m: float) -> tuple[float, float]:
        """``(mu_G, sigma_G)`` at ``(f, d)``; per-metre values are log-log interpolated in ``f``."""
        if self.stochastic is not None:
            return self.stochastic.mean, self.stochastic.std
        assert self.stochastic_per_m is not None
        bands = sorted(self.stochastic_per_m)
        out = []
        for attr in ("mean", "std"):
            v = [getattr(self.stochastic_per_m[b], attr) for b in bands]
            if f_ghz in self.stochastic_per_m:
                out.append(v[bands.index(f_ghz)])
            else:
                out.append(float(np.exp(np.interp(np.log(f_ghz), np.log(bands), np.log(v)))))
        return out[0] * d_m, out[1] * d_m

    def law(self, f_ghz: float, d_m: float, psi0: float) -> LargeScaleLaw:
        """Shifted log-gamma at ``(f, d)`` with deterministic part ``psi0``."""
        mu, sd = self.moments_g(f_ghz, d_m)
        return ShiftedLogGamma.from_moments(psi0, mu, sd**2)


class LinkBudgetCfg(_Strict):
    coupling_model: Literal["goubau", "friis_clipped"] = "goubau"
    A_t_m2: float = Field(gt=0)
    A_r_m2: float = Field(gt=0)
    G_enh: float = Field(default=1.0, ge=1.0)
    sensitivity_models: list[Literal["goubau", "friis_clipped"]] = ["goubau", "friis_clipped"]


class SourceState(_Strict):
    type: Literal["tmsv"] = "tmsv"
    r: float = Field(gt=0)


class E3Config(_Strict):
    experiment: Literal["e3_entanglement_outage"]
    seed: int
    environment: Environment
    bands_ghz: list[float]
    distance_m: GridSpec
    small_scale: dict[Literal["K"], list[float]]
    large_scale: LargeScale
    link_budget: LinkBudgetCfg
    source_state: SourceState

    @model_validator(mode="after")
    def _bands_have_absorption(self) -> E3Config:
        for b in self.bands_ghz:
            if b not in self.large_scale.gamma_abs_per_m:
                raise ValueError(f"no gamma_abs entry for band {b} GHz")
        return self


class FrontEndClass(_Strict):
    """Published transducer: attenuator ``(eta_t, n_t)`` plus added ``N_t`` (TSD §7.5, §16.5).

    ``N_add_in = N_t / eta_t`` is the input-referred noise; only it sets the window.
    """

    label: str
    eta_t: float = Field(gt=0, le=1)
    n_t: float = Field(ge=0)
    N_t: float = Field(ge=0)
    N_add_in: float = Field(ge=0)
    source: str

    @model_validator(mode="after")
    def _consistent(self) -> FrontEndClass:
        if abs(self.N_t - self.eta_t * self.N_add_in) > 1e-3 * max(self.N_t, 1e-12) + 1e-12:
            raise ValueError(f"{self.label}: N_t must equal eta_t * N_add_in")
        return self


class E4Config(_Strict):
    experiment: Literal["e4_receiver_window"]
    environment: Environment
    bands_ghz: list[float]
    n_add: GridSpec
    """Input-referred noise sweep ``N_add_in``."""
    front_end_classes: list[FrontEndClass]
    references_outside_axis: list[dict[str, str | float]] = []


class RxSpec(_Strict):
    """Receiver spec. ``source`` is required for every non-ideal receiver (FR-SVC-C5 (d))."""

    label: str
    kind: Literal["holevo", "het", "hom", "electronic", "rydberg"]
    t_n_k: float | None = Field(default=None, gt=0)
    tn_convention: Literal["rj", "planck", "callen_welton"] | None = None
    s_e: float | None = Field(default=None, gt=0)
    field_convention: Literal["rms", "peak"] | None = None
    sideband: Literal["ssb", "dsb"] | None = None
    bandwidth_hz: float | None = Field(default=None, gt=0)
    source: str | None = None

    @model_validator(mode="after")
    def _complete(self) -> RxSpec:
        electronic_bad = self.t_n_k is None or self.tn_convention is None or not self.source
        rydberg_bad = self.s_e is None or self.field_convention is None or not self.source
        if self.kind == "electronic" and electronic_bad:
            raise ValueError(f"{self.label}: electronic needs t_n_k, tn_convention, source")
        if self.kind == "rydberg" and rydberg_bad:
            raise ValueError(f"{self.label}: rydberg needs s_e, field_convention, source")
        return self


class RarCfg(_Strict):
    """RAR inputs from the literature file; ``from_entry`` takes the nearest entry's sideband."""

    source: Literal["literature_rar"] = "literature_rar"
    literature: str = "configs/literature_rar.yaml"
    require_b_hz: bool = True
    sideband: Literal["from_entry", "ssb", "dsb"] = "from_entry"


class LnaPoint(_Strict):
    f_hz: float = Field(gt=0)
    T_n_k: float = Field(gt=0)
    source: str


class BwPoint(_Strict):
    f_hz: float = Field(gt=0)
    B: float = Field(gt=0)
    source: str


class LnaCfg(_Strict):
    bands: list[LnaPoint] = Field(min_length=1)
    B_hz: list[BwPoint] = Field(min_length=1)
    tn_convention: Literal["rj", "planck", "callen_welton"]
    sideband: Literal["ssb", "dsb"] = "ssb"


class ThroughputMapCfg(_Strict):
    P_t_w: float = Field(gt=0)
    f_ghz: GridSpec
    distance_m: GridSpec
    K: list[float]
    rar: RarCfg
    lna: LnaCfg
    interpolate: Literal["loglog"] = "loglog"
    equal_bandwidth_case: bool = True


class E5Config(_Strict):
    experiment: Literal["e5_classical_throughput"]
    environment: Environment
    n_s: list[float]
    """``N_s`` sweep for the bits-per-mode panel (a)."""
    bands_ghz: list[float]
    """Frequencies of panel (a); must lie inside both receiver ranges."""
    receivers: list[RxSpec]
    """Ideal receivers (holevo, het, hom); LNA and RAR come from ``throughput_map``."""
    large_scale: LargeScale
    link_budget: LinkBudgetCfg
    throughput_map: ThroughputMapCfg

    @model_validator(mode="after")
    def _ideal_only(self) -> E5Config:
        if any(r.kind in ("electronic", "rydberg") for r in self.receivers):
            raise ValueError("E5 receivers: ideal kinds only; LNA/RAR come from throughput_map")
        return self


class E6Config(_Strict):
    experiment: Literal["e6_bbr_crossover"]
    environment: Environment
    f_ghz: GridSpec
    delta_db: list[float]
    eta_ref: float = Field(default=0.0, ge=0.0, le=1.0)
    literature_rar: str = "configs/literature_rar.yaml"
    slope_fit_max_ghz: float = 30.0


class E8Config(_Strict):
    experiment: Literal["e8_outage_capacity"]
    seed: int
    environment: Environment
    band_ghz: float
    n_s: float = Field(gt=0)
    distance_m: list[float]
    eps: GridSpec
    small_scale: dict[Literal["K"], list[float]]
    large_scale: LargeScale
    link_budget: LinkBudgetCfg
    receivers: list[RxSpec]
    """Ideal receivers (holevo, het, hom)."""
    rar: RarCfg
    lna: LnaCfg

    @model_validator(mode="after")
    def _ideal_only(self) -> E8Config:
        if any(r.kind in ("electronic", "rydberg") for r in self.receivers):
            raise ValueError("E8 receivers: ideal kinds only; LNA/RAR come from rar/lna")
        return self


class LiteratureEntry(_Strict):
    """Published RAR point (FR-EXP-7, TSD §12, §16.3). Units: ``f_hz`` Hz, ``s_e`` V/m/sqrt(Hz)."""

    label: str
    f_hz: float = Field(gt=0)
    s_e: float = Field(gt=0)
    field_convention: Literal["rms", "peak"]
    sideband: Literal["ssb", "dsb"]
    includes_bbr: bool = False
    b_hz: float | None = Field(default=None, gt=0)
    source: str

    def s_e_rms(self) -> float:
        """rms value as published (peak divided by sqrt 2); BBR not removed."""
        return self.s_e / 2**0.5 if self.field_convention == "peak" else self.s_e

    def s_e_intrinsic_rms(self, t_env_k: float = 300.0) -> float:
        """rms ``S_E`` with the in-band BBR subtracted in power if ``includes_bbr``."""
        s = self.s_e_rms()
        if not self.includes_bbr:
            return s
        diff = s**2 - float(s_e_bbr(self.f_hz, t_env_k)) ** 2
        if diff <= 0:
            raise ValueError(f"{self.label}: S_E is at or below the BBR floor")
        return float(diff**0.5)


class LiteratureFile(_Strict):
    entries: list[LiteratureEntry] = []


def load_literature(path: str | Path) -> list[LiteratureEntry]:
    """Load ``literature_rar.yaml`` (FR-EXP-7); missing or empty file -> ``[]`` plus a warning."""
    import logging

    p = Path(path)
    if not p.is_file():
        logging.getLogger(__name__).warning("literature RAR file %s missing: no overlay", p)
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    entries = LiteratureFile.model_validate(data).entries
    if not entries:
        logging.getLogger(__name__).warning("literature RAR file %s empty: no overlay", p)
    return entries


def load_config(path: str | Path, model: type[BaseModel]) -> tuple[BaseModel, str]:
    """Load YAML and validate; return the model and a 12-hex-char content hash."""
    text = Path(path).read_text(encoding="utf-8")
    return model.model_validate(yaml.safe_load(text)), hashlib.sha256(text.encode()).hexdigest()[
        :12
    ]
