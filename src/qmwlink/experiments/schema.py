"""Pydantic config schemas for E1-E9 (FR-EXP-1, TSD §12). Unknown keys are rejected."""

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


class E7Config(_Strict):
    experiment: Literal["e7_diversity"]
    environment: Environment
    bands_ghz: list[float]
    K: list[float]
    eps: list[float]
    eta_l: Literal[1] = 1
    """Deterministic large-scale factor of the reference table (``eta_l = 1``)."""
    eta_bar_fraction: float = Field(default=0.9, gt=0, lt=1)
    """``eta_bar_Sigma = eta* + fraction (1 - eta*)`` (FSD FR-DIV-3 reference rule)."""
    n_grid: GridSpec
    """``N`` grid (log, integers) for the outage-vs-N curves."""


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


class E1Compound(_Strict):
    """Compound panel of E1: the E3 link over a distance grid (FSD §6, E1)."""

    bands_ghz: list[float] = Field(min_length=1)
    distance_m: GridSpec
    large_scale: LargeScale
    link_budget: LinkBudgetCfg

    @model_validator(mode="after")
    def _bands_have_absorption(self) -> E1Compound:
        for b in self.bands_ghz:
            if b not in self.large_scale.gamma_abs_per_m:
                raise ValueError(f"no gamma_abs entry for band {b} GHz")
        return self


class E1Config(_Strict):
    """E1: small-scale PDT shape and truncation effect (RQ1; FR-CH-1, FR-SVC-E2)."""

    experiment: Literal["e1_truncation"]
    environment: Environment
    K: list[float] = Field(min_length=1)
    eta_s_bar: GridSpec
    """Scale-parameter sweep for the removed-mass and outage panels (``0 < eta_s_bar <= 1``)."""
    band_ghz: float = Field(gt=0)
    """Band that sets ``eta*`` in the ``eta_s_bar`` sweep (deterministic ``eta_l = 1``)."""
    pdf_eta_s_bar: float = Field(gt=0, le=1)
    pdf_eta: GridSpec
    compound: E1Compound


class LawComparison(_Strict):
    """RQ2 comparison block (TSD §6.2, §12)."""

    match: Literal["psi_moments"] = "psi_moments"
    eta_gt1: Literal["clip"] = "clip"
    sigma_ratio: list[float] = Field(min_length=1)
    """Grid of ``sigma_Psi / Psi_bar`` (all ``> 0``)."""

    @model_validator(mode="after")
    def _positive(self) -> LawComparison:
        if any(r <= 0 for r in self.sigma_ratio):
            raise ValueError("sigma_ratio entries must be > 0")
        return self


class E2Config(_Strict):
    """E2: large-scale law comparison (RQ2; FR-CH-3).

    ``Psi_0`` and ``mu_G`` come from ``large_scale`` at ``(band_ghz, distance_m)``; the configured
    ``std`` only marks the sourced ratio in the figure, the sweep is ``law_comparison.sigma_ratio``.
    """

    experiment: Literal["e2_law_comparison"]
    environment: Environment
    band_ghz: float = Field(gt=0)
    distance_m: float = Field(gt=0)
    K: float = Field(ge=0)
    n_s: float = Field(gt=0)
    large_scale: LargeScale
    link_budget: LinkBudgetCfg
    source_state: SourceState
    law_comparison: LawComparison
    rar: RarCfg
    pdf_sigma_ratio: float = Field(gt=0)
    """``sigma_Psi/Psi_bar`` of the pdf panel."""
    pdf_eta: GridSpec


class BeamCfg(_Strict):
    """Probe or coupling beam (TSD §12 ``lindblad`` block). Dipoles in C m, fields peak."""

    wavelength_m: float = Field(gt=0)
    power_w: float | None = Field(default=None, gt=0)
    waist_m: float | None = Field(default=None, gt=0)
    d12_cm: float | None = Field(default=None, gt=0)
    """Probe only. ``null`` uses the closed cycling-transition dipole of TSD §14.7."""
    d23_cm: float | None = Field(default=None, gt=0)
    omega_c_over_2pi_mhz: float | None = Field(default=None, ge=0)
    """Coupling only: Rabi frequency given directly instead of power, waist and ``d23_cm``."""


class MwCfg(_Strict):
    f_hz: float = Field(gt=0)
    """Carrier of the ``|3> <-> |4>`` transition; needed by the §7.4 ``N_amp`` mapping."""
    d34_cm: float = Field(gt=0)
    e_lo_pk_v_per_m: float = Field(gt=0)


class DecayCfg(_Strict):
    """One decay channel; ``from``/``to`` are level numbers starting at 1 (TSD §14.3)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_level: int = Field(alias="from", ge=2, le=4)
    to_level: int = Field(alias="to", ge=1, le=3)
    rate_per_s: float = Field(ge=0)
    source: str

    @model_validator(mode="after")
    def _has_source(self) -> DecayCfg:
        if not self.source.strip():
            raise ValueError("decay entries need a non-empty source")
        return self


class DetectorCfg(_Strict):
    eta_d: float = Field(gt=0, le=1)
    nep_w_per_rthz: float = Field(default=0.0, ge=0)


class DopplerCfg(_Strict):
    n_sigma: float = Field(default=5.0, gt=0)
    n_v_start: int = Field(default=4097, ge=3)
    rtol: float = Field(default=1.0e-5, gt=0, lt=1)
    n_v_max: int = Field(default=1048577, ge=3)


class LindbladSweep(_Strict):
    omega_c_over_2pi_mhz: list[float] = Field(min_length=1)
    e_lo_pk_v_per_m: list[float] = Field(min_length=1)


LINDBLAD_SOURCE_KEYS = ("vapor", "probe", "coupling", "mw", "transit", "detector")


class LindbladCfg(_Strict):
    """Receiver physics model inputs (TSD §12, §14). Every block needs an entry in ``sources``."""

    method: Literal["auto", "weak_probe", "full"] = "auto"
    weak_probe_tol: float = Field(default=0.01, gt=0, lt=1)
    species: Literal["Rb87"] = "Rb87"
    n_levels: Literal[4] = 4
    """The receiver figures need the microwave transition; 3-level schemes are library-only."""
    T_vapor_k: float = Field(gt=0)
    n0_per_m3: float = Field(gt=0)
    cell_length_m: float = Field(gt=0)
    probe: BeamCfg
    coupling: BeamCfg
    mw: MwCfg
    decays: list[DecayCfg]
    """Rydberg decay channels; ``2 -> 1`` defaults to the 87Rb D2 rate when not listed."""
    gamma_t_per_s: float = Field(ge=0)
    detector: DetectorCfg
    doppler: DopplerCfg = DopplerCfg()
    sweep: LindbladSweep
    sources: dict[str, str]

    @model_validator(mode="after")
    def _complete(self) -> LindbladCfg:
        missing = [k for k in LINDBLAD_SOURCE_KEYS if not self.sources.get(k, "").strip()]
        if missing:
            raise ValueError(f"lindblad.sources missing for {missing}")
        if self.probe.power_w is None or self.probe.waist_m is None:
            raise ValueError("probe needs power_w and waist_m")
        c = self.coupling
        beam = c.power_w is not None and c.waist_m is not None and c.d23_cm is not None
        if c.omega_c_over_2pi_mhz is None and not beam:
            raise ValueError("coupling needs omega_c_over_2pi_mhz, or power_w, waist_m and d23_cm")
        return self


class E9Link(_Strict):
    """Link of panel (a): received peak field vs distance (TSD §7.4)."""

    P_t_w: float = Field(gt=0)
    A_t_m2: float = Field(gt=0)
    A_r_m2: float = Field(gt=0)
    coupling_model: Literal["goubau", "friis_clipped"] = "goubau"
    gamma_abs_per_m: float = Field(ge=0)
    gamma_abs_source: str
    distance_m: GridSpec


class E9Crossover(_Strict):
    """E6 map behind the model-derived points of panel (c)."""

    f_ghz: GridSpec
    delta_db: list[float]
    eta_ref: float = Field(default=0.0, ge=0.0, le=1.0)
    literature_rar: str = "configs/literature_rar.yaml"


class E9Config(_Strict):
    """E9: validity of the Gaussian RAR model and model-derived points (FR-RX-6, FR-LIN)."""

    experiment: Literal["e9_rar_validity"]
    seed: int
    environment: Environment
    lindblad: LindbladCfg
    link: E9Link
    crossover: E9Crossover


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
