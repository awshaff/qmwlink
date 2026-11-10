"""Pydantic config schemas for E3 and E4 (TSD §12). Extended by the M6 runner."""

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


def load_config(path: str | Path, model: type[BaseModel]) -> tuple[BaseModel, str]:
    """Load YAML and validate; return the model and a 12-hex-char content hash."""
    text = Path(path).read_text(encoding="utf-8")
    return model.model_validate(yaml.safe_load(text)), hashlib.sha256(text.encode()).hexdigest()[
        :12
    ]
