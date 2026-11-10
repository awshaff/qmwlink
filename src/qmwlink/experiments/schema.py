"""Pydantic config schemas for E3-E6 and E8 (TSD §12). Extended by the M6 runner."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    """Large-scale stochastic loss ``G`` (mean, std), a placeholder until set by the owner [A]."""

    mean: float = Field(ge=0)
    std: float = Field(ge=0)


class LargeScale(_Strict):
    model: Literal["loggamma"] = "loggamma"
    gamma_abs_per_m: dict[float, float]
    """Per band in GHz [1/m]."""
    gamma_abs_source: dict[float, str]
    stochastic: Stochastic

    @model_validator(mode="after")
    def _sources(self) -> LargeScale:
        missing = sorted(set(self.gamma_abs_per_m) - set(self.gamma_abs_source))
        if missing:
            raise ValueError(f"gamma_abs source missing for bands {missing} GHz")
        return self


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
    """Front-end class: attenuator ``(eta_t, n_t)`` then swept additive noise ``N_add``."""

    label: str
    eta_t: float = Field(gt=0, le=1)
    n_t: float = Field(ge=0)
    source: str


class E4Config(_Strict):
    experiment: Literal["e4_receiver_window"]
    environment: Environment
    bands_ghz: list[float]
    n_add: GridSpec
    classes: list[FrontEndClass]


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


class BandRx(_Strict):
    """Per-band receiver inputs for the E5 map (log-log interpolated, TSD §8.4)."""

    f_ghz: float = Field(gt=0)
    bandwidth_hz: float = Field(gt=0)
    value: float = Field(gt=0)
    """``S_E`` [V/m/sqrt(Hz)] for the RAR, ``T_n`` [K] for the LNA."""
    source: str


class E5Config(_Strict):
    experiment: Literal["e5_classical_throughput"]
    environment: Environment
    p_t_w: float = Field(gt=0)
    n_s: list[float]
    """``N_s`` for the bits-per-mode panel (a)."""
    bands_ghz: list[float]
    f_ghz: GridSpec
    distance_m: GridSpec
    small_scale: dict[Literal["K"], list[float]]
    large_scale: LargeScale
    link_budget: LinkBudgetCfg
    receivers: list[RxSpec]
    rar_bands: list[BandRx]
    rar_field_convention: Literal["rms", "peak"]
    rar_sideband: Literal["ssb", "dsb"] = "dsb"
    lna_bands: list[BandRx]
    lna_tn_convention: Literal["rj", "planck", "callen_welton"]
    lna_sideband: Literal["ssb", "dsb"] = "ssb"

    @model_validator(mode="after")
    def _bands(self) -> E5Config:
        for b in self.bands_ghz:
            if b not in self.large_scale.gamma_abs_per_m:
                raise ValueError(f"no gamma_abs entry for band {b} GHz")
        if not self.rar_bands or not self.lna_bands:
            raise ValueError("rar_bands and lna_bands must be non-empty")
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

    @model_validator(mode="after")
    def _band(self) -> E8Config:
        if self.band_ghz not in self.large_scale.gamma_abs_per_m:
            raise ValueError(f"no gamma_abs entry for band {self.band_ghz} GHz")
        return self


class LiteratureEntry(_Strict):
    label: str
    f_ghz: float = Field(gt=0)
    s_e: float = Field(gt=0)
    field_convention: Literal["rms", "peak"]
    sideband: Literal["ssb", "dsb"]
    includes_bbr: bool = False
    source: str

    def s_e_rms(self) -> float:
        return self.s_e / 2**0.5 if self.field_convention == "peak" else self.s_e


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
