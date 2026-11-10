"""E9: validity of the Gaussian RAR model and model-derived points (FR-RX-6, FR-LIN, TSD §14).

(a) ``E_1dB,pk`` of the model against the received peak field over distance: the region where the
    linear (Gaussian) receiver model holds;
(b) ``S_E``, ``N_amp`` and ``B_3dB`` vs ``Omega_c`` and ``E_LO``;
(c) the model-derived points on the E6 crossover map.

The engine used and the weak-probe/full discrepancy of the ``auto`` method go to ``meta.json``.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from qmwlink.channel.link_budget import link_budget
from qmwlink.constants import ETA0
from qmwlink.experiments import e6
from qmwlink.experiments.common import save_fig, write_csv, write_meta
from qmwlink.experiments.schema import (
    E6Config,
    E9Config,
    LindbladCfg,
    LiteratureEntry,
    load_config,
    load_literature,
)
from qmwlink.plotting.style import COLUMN_WIDTH_IN, OKABE_ITO, apply_style
from qmwlink.receivers.rydberg import RydbergReceiver
from qmwlink.receivers.rydberg_physics import readout
from qmwlink.receivers.rydberg_physics.doppler import DopplerSettings
from qmwlink.receivers.rydberg_physics.model import (
    GAMMA2_RB87_PER_S,
    SPECIES_MASS_KG,
    LindbladRARModel,
    RARModelResult,
    RARParams,
    SweepResult,
    evaluate_sweep,
    with_overrides,
)
from qmwlink.thermal import s_e_bbr

logger = logging.getLogger(__name__)

TWO_PI = 2.0 * math.pi


class SweepRow(NamedTuple):
    omega_c_over_2pi_mhz: float
    e_lo_pk_v_per_m: float
    omega_lo_over_2pi_mhz: float
    engine: str
    n_v: int
    s_e_rms: float
    n_amp: float
    b_3db_hz: float
    peaking: float
    b_3db_is_lower_bound: bool
    kappa_w_per_v_m: float
    p_out_w: float


class LinearRow(NamedTuple):
    d_m: float
    eta: float
    p_rx_w: float
    e_rx_pk_v_per_m: float
    e_1db_pk_v_per_m: float
    e_lo_pk_v_per_m: float
    linear_regime: bool


def build_params(cfg: LindbladCfg) -> RARParams:
    """SI model inputs from the config block (``d12_cm: null`` -> cycling-transition dipole)."""
    k_p = TWO_PI / cfg.probe.wavelength_m
    rates = {(d.from_level, d.to_level): d.rate_per_s for d in cfg.decays}
    decays = dict(rates)
    decays.setdefault((2, 1), GAMMA2_RB87_PER_S)
    gamma2 = sum(r for (a, _), r in decays.items() if a == 2)
    d12 = cfg.probe.d12_cm or readout.cycling_dipole_cm(gamma2, k_p)
    c = cfg.coupling
    assert cfg.probe.power_w is not None and cfg.probe.waist_m is not None
    return RARParams(
        mass_kg=SPECIES_MASS_KG[cfg.species],
        t_vapor_k=cfg.T_vapor_k,
        n0_per_m3=cfg.n0_per_m3,
        cell_length_m=cfg.cell_length_m,
        probe_wavelength_m=cfg.probe.wavelength_m,
        probe_power_w=cfg.probe.power_w,
        probe_waist_m=cfg.probe.waist_m,
        d12_cm=d12,
        coupling_wavelength_m=c.wavelength_m,
        f_mw_hz=cfg.mw.f_hz,
        d34_cm=cfg.mw.d34_cm,
        e_lo_pk_v_per_m=cfg.mw.e_lo_pk_v_per_m,
        gamma_t_per_s=cfg.gamma_t_per_s,
        gamma2_per_s=gamma2,
        gamma3_per_s=sum(r for (a, _), r in decays.items() if a == 3),
        gamma4_per_s=sum(r for (a, _), r in decays.items() if a == 4),
        omega_c_rad_s=(
            None if c.omega_c_over_2pi_mhz is None else TWO_PI * 1e6 * c.omega_c_over_2pi_mhz
        ),
        coupling_power_w=c.power_w,
        coupling_waist_m=c.waist_m,
        d23_cm=c.d23_cm,
        n_levels=cfg.n_levels,
        decays_per_s=tuple((a, b, r) for (a, b), r in sorted(decays.items())),
        eta_d=cfg.detector.eta_d,
        nep_w_per_rthz=cfg.detector.nep_w_per_rthz,
    )


def doppler_settings(cfg: LindbladCfg) -> DopplerSettings:
    d = cfg.doppler
    return DopplerSettings(d.n_sigma, d.n_v_start, d.rtol, d.n_v_max)


def compute_sweep(cfg: E9Config) -> tuple[list[SweepRow], SweepResult]:
    """Panel (b): the ``Omega_c x E_LO`` grid with ``method`` from the config (TSD §14.6)."""
    lc = cfg.lindblad
    base, dop = build_params(lc), doppler_settings(lc)
    grid = [(oc, e) for e in lc.sweep.e_lo_pk_v_per_m for oc in lc.sweep.omega_c_over_2pi_mhz]
    models = [
        LindbladRARModel(
            with_overrides(base, omega_c_rad_s=TWO_PI * 1e6 * oc, e_lo_pk_v_per_m=e), dop
        )
        for oc, e in grid
    ]
    sweep = evaluate_sweep(
        models, lc.method, weak_probe_tol=lc.weak_probe_tol,
        rng=np.random.default_rng(cfg.seed), bandwidth=True, linear_range=False,
    )  # fmt: skip
    rows = [
        SweepRow(oc, e, r.omega_lo_rad_s / TWO_PI / 1e6, r.engine, r.n_v, r.s_e_rms, r.n_amp,
                 r.b_3db_hz, r.peaking, r.b_3db_is_lower_bound, r.kappa_w_per_v_m, r.p_out_w)
        for (oc, e), r in zip(grid, sweep.results, strict=True)
    ]  # fmt: skip
    return rows, sweep


def compute_linear(cfg: E9Config, base: RARModelResult) -> tuple[list[LinearRow], RydbergReceiver]:
    """Panel (a): received peak field over distance against ``E_1dB,pk`` (peak vs peak, §7.4)."""
    lk = cfg.link
    e_1db = base.e_1db_pk_v_per_m
    rx = RydbergReceiver.from_sensitivity(
        base.s_e_rms, base.f_mw_hz, field_convention="rms", sideband="dsb",
        bandwidth_hz=base.b_3db_hz, e_1db_pk_v_per_m=e_1db if math.isfinite(e_1db) else None,
    )  # fmt: skip
    rows = []
    for d in lk.distance_m.values():
        lb = link_budget(
            base.f_mw_hz, float(d), lk.A_t_m2, lk.A_r_m2, gamma_abs_per_m=lk.gamma_abs_per_m,
            coupling_model=lk.coupling_model,
        )  # fmt: skip
        eta = lb.eta_geo * math.exp(-lb.psi0)
        p_rx = lk.P_t_w * eta
        rows.append(
            LinearRow(float(d), eta, p_rx, rx.e_rx_pk(p_rx, lk.A_r_m2), e_1db,
                      cfg.lindblad.mw.e_lo_pk_v_per_m, rx.linear_regime(p_rx, lk.A_r_m2))
        )  # fmt: skip
    return rows, rx


def crossover_rows(cfg: E9Config) -> tuple[list[e6.E6Row], list[LiteratureEntry]]:
    """Panel (c) background: the E6 map (FR-EXP-7 (b): a missing literature file only warns)."""
    c = cfg.crossover
    e6_cfg = E6Config(
        experiment="e6_bbr_crossover", environment=cfg.environment, f_ghz=c.f_ghz,
        delta_db=c.delta_db, eta_ref=c.eta_ref, literature_rar=c.literature_rar,
    )  # fmt: skip
    return e6.compute(e6_cfg), load_literature(c.literature_rar)


def _fig_linear(rows: list[LinearRow]) -> Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.5))
    d = [r.d_m for r in rows]
    ax.loglog(d, [r.e_rx_pk_v_per_m for r in rows], color=OKABE_ITO[0], label=r"$E_{\rm rx,pk}$")
    e_1db, e_lo = rows[0].e_1db_pk_v_per_m, rows[0].e_lo_pk_v_per_m
    if math.isfinite(e_1db):
        ax.axhline(e_1db, color=OKABE_ITO[1], ls="--", label=r"$E_{1\,\rm dB,pk}$")
    ax.axhline(e_lo, color="0.4", ls=":", label=r"$E_{\rm LO,pk}$")
    nonlin = [r.d_m for r in rows if not r.linear_regime]
    if nonlin:
        ax.axvspan(min(d), max(nonlin), color=OKABE_ITO[1], alpha=0.12, lw=0)
    ax.set_xlabel("distance (m)")
    ax.set_ylabel(r"peak field (V m$^{-1}$)")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _fig_sweep(rows: list[SweepRow]) -> Figure:
    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(COLUMN_WIDTH_IN, 4.8), sharex=True)
    cols = (("s_e_rms", r"$S_E$ (V m$^{-1}$ Hz$^{-1/2}$, rms)"), ("n_amp", r"$N_{\rm amp}$"),
            ("b_3db_hz", r"$B_{3\,\rm dB}$ (Hz)"))  # fmt: skip
    for i, e in enumerate(sorted({r.e_lo_pk_v_per_m for r in rows})):
        sel = [r for r in rows if r.e_lo_pk_v_per_m == e]
        lab = rf"$\Omega_{{\rm LO}}/2\pi={sel[0].omega_lo_over_2pi_mhz:.3g}$ MHz"
        for ax, (col, _) in zip(axes, cols, strict=True):
            x, y = [r.omega_c_over_2pi_mhz for r in sel], [getattr(r, col) for r in sel]
            ax.loglog(x, y, color=OKABE_ITO[i % len(OKABE_ITO)], marker="o", ms=2.5, label=lab)
    for ax, (_, ylabel) in zip(axes, cols, strict=True):
        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel(r"$\Omega_c/2\pi$ (MHz)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def _fig_overlay(
    e6_rows: list[e6.E6Row], lit: list[LiteratureEntry], sweep: list[SweepRow], f_mw_hz: float,
    cfg: E9Config,
) -> Figure:  # fmt: skip
    e6_cfg = E6Config(
        experiment="e6_bbr_crossover", environment=cfg.environment, f_ghz=cfg.crossover.f_ghz,
        delta_db=cfg.crossover.delta_db, eta_ref=cfg.crossover.eta_ref,
    )  # fmt: skip
    fig = e6._figure(e6_rows, e6_cfg, lit)
    ax = fig.axes[0]
    s = [r.s_e_rms for r in sweep]
    ax.plot([f_mw_hz / 1e9] * len(s), s, "_", color=OKABE_ITO[3], ms=5, alpha=0.6)
    ax.plot(f_mw_hz / 1e9, min(s), "*", color=OKABE_ITO[3], ms=7, label="model (best of sweep)")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    return fig


def run(config_path: str | Path, out_root: str | Path = "results") -> Path:
    """Compute E9 and write ``results/e9_rar_validity/<hash>/``."""
    cfg, h = load_config(config_path, E9Config)
    assert isinstance(cfg, E9Config)
    out = Path(out_root) / cfg.experiment / h
    out.mkdir(parents=True, exist_ok=True)
    lc = cfg.lindblad
    sweep_rows, sweep = compute_sweep(cfg)
    base = LindbladRARModel(build_params(lc), doppler_settings(lc)).evaluate(
        sweep.engine_used, bandwidth=True, linear_range=True
    )
    linear_rows, rx = compute_linear(cfg, base)
    e6_rows, lit = crossover_rows(cfg)
    write_csv(out / "data.csv", SweepRow._fields, sweep_rows)
    write_csv(out / "linear_range.csv", LinearRow._fields, linear_rows)
    write_csv(out / "crossover.csv", e6.COLUMNS, e6_rows)
    save_fig(_fig_linear(linear_rows), out / "fig_linear_range")
    save_fig(_fig_sweep(sweep_rows), out / "fig_sweep")
    save_fig(_fig_overlay(e6_rows, lit, sweep_rows, lc.mw.f_hz, cfg), out / "fig_crossover_overlay")
    linear_d = [r.d_m for r in linear_rows if r.linear_regime]
    best = min(sweep_rows, key=lambda r: r.s_e_rms)
    write_meta(
        out / "meta.json", cfg.experiment, h,
        {"seed": cfg.seed, "method": sweep.method, "engine_used": sweep.engine_used,
         "max_weak_probe_full_discrepancy": sweep.max_discrepancy,
         "weak_probe_tol": sweep.weak_probe_tol, "checkpoints": list(sweep.checkpoints),
         "recomputed_with_full": sweep.recomputed_with_full,
         "n_v_min": min(r.n_v for r in sweep_rows), "n_v_max": max(r.n_v for r in sweep_rows),
         "n_rows": len(sweep_rows), "n_literature_points": len(lit),
         "sideband": rx.sideband, "field_convention": "rms",
         "baseline": {"s_e_rms": base.s_e_rms, "n_amp": base.n_amp, "t_n_k": rx.t_n_k,
                      "b_3db_hz": base.b_3db_hz, "peaking": base.peaking,
                      "e_1db_pk_v_per_m": base.e_1db_pk_v_per_m, "e_1db_case": base.e_1db_case,
                      "e_1db_reason": base.e_1db_reason, "n_v": base.n_v,
                      "omega_p_over_2pi_mhz": base.omega_p_rad_s / TWO_PI / 1e6,
                      "s_e_over_bbr": base.s_e_rms / float(
                          s_e_bbr(base.f_mw_hz, cfg.environment.T_env_k))},
         "best_of_sweep": best._asdict(),
         "linear_from_d_m": min(linear_d) if linear_d else None,
         "e_rx_pk_formula": f"sqrt(2 eta0 P_rx / A_r), eta0 = {ETA0:.6g} ohm",
         "sources": lc.sources, "gamma_abs_source": cfg.link.gamma_abs_source,
         "placeholders": sorted(k for k, v in lc.sources.items() if "[A]" in v)},
    )  # fmt: skip
    logger.info("E9 written to %s", out)
    return out
