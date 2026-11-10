"""E2: large-scale law comparison (RQ2; FR-CH-3, TSD §6.2).

Shifted log-gamma (primary) against the V0 log-normal (``physical = False``) and the log-normal
conditioned on ``Psi >= 0``, all matched on the mean and variance of ``Psi`` with the same
``Psi_0``. Reports ``Delta P_out`` (absolute) and ``Delta C/C`` (relative; Holevo, ideal
heterodyne, RAR) of each baseline against the log-gamma, vs ``sigma_Psi / Psi_bar``.

V0 log-normal realizations with ``eta > 1`` are evaluated at ``min(eta, 1)`` (policy
``eta_gt1: clip``); the affected mass ``P(eta > 1)`` and the law-level ``P(eta_l > 1)`` are
written next to every number.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from qmwlink.channel.compound import CompoundPDT
from qmwlink.channel.large_scale import LargeScaleLaw, LogNormalV0, matched_laws
from qmwlink.channel.link_budget import link_budget
from qmwlink.channel.small_scale import SmallScale
from qmwlink.experiments.common import save_fig, write_csv, write_meta
from qmwlink.experiments.inputs import RarTable
from qmwlink.experiments.schema import E2Config, load_config
from qmwlink.gaussian.channels import PIChannel
from qmwlink.numerics.integrate import checked_quad
from qmwlink.plotting.style import COLUMN_WIDTH_IN, OKABE_ITO, apply_style
from qmwlink.services.classical import ClassicalService, c_het, c_holevo, c_rx
from qmwlink.services.entanglement import EntanglementService
from qmwlink.thermal import n_th

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
LAWS = ("loggamma", "lognormal_v0", "lognormal_conditioned")
CAPS = ("c_holevo", "c_het", "c_rx")
CapMap = dict[str, Callable[[FloatArray], FloatArray]]


class E2Row(NamedTuple):
    sigma_ratio: float
    law: str
    physical: bool
    mass_eta_l_above_one: float
    clipped_mass: float
    p_out: float
    c_holevo: float
    c_het: float
    c_rx: float
    delta_p_out: float
    rel_delta_c_holevo: float
    rel_delta_c_het: float
    rel_delta_c_rx: float


class PdfRow(NamedTuple):
    sigma_ratio: float
    law: str
    eta_l: float
    pdf: float


def clipped_mass(small: SmallScale, law: LargeScaleLaw) -> float:
    """``P(eta_l eta_s > 1)``: the mass evaluated at ``min(eta, 1)``. (TSD §6.2, ``eta_gt1: clip``)

    Non-zero only for laws with ``eta_l > 1`` (``Psi < 0``), i.e. the V0 log-normal.
    """
    lo = law.psi_lo_eff()
    if lo >= 0.0 or float(law.psi_cdf(0.0)) < 1e-300:
        return 0.0
    val = checked_quad(
        lambda p: float(small.sf(np.exp(p))) * float(law.psi_pdf(p)),
        lo, 0.0, points=[p for p in law.psi_breakpoints() if lo < p < 0.0],
    )[0]  # fmt: skip
    return float(val)


def _clip(cap: Callable[[FloatArray], FloatArray]) -> Callable[[FloatArray], FloatArray]:
    """``eta -> cap(min(eta, 1))``: the sanctioned baseline clip (TSD §13 rule 9)."""
    return lambda e: cap(np.minimum(e, 1.0))


def evaluate_law(
    small: SmallScale,
    law: LargeScaleLaw,
    n_th_val: float,
    r: float,
    caps: dict[str, Callable[[FloatArray], FloatArray]],
) -> tuple[float, dict[str, float]]:
    """``P_out`` and ergodic capacities under one large-scale law (clip applied if non-physical).

    The outage needs no clip: ``min(eta, 1) <= eta* < 1`` iff ``eta <= eta*``.
    """
    comp = CompoundPDT(small, law)
    p_out = EntanglementService(comp, n_th_val, PIChannel.identity(), r).p_out()
    clip = not law.physical
    erg = {
        name: ClassicalService(comp, _clip(cap) if clip else cap).ergodic()
        for name, cap in caps.items()
    }
    return p_out, erg


def _setup(cfg: E2Config) -> tuple[SmallScale, float, float, float, CapMap]:
    f_hz = cfg.band_ghz * 1e9
    nth = float(n_th(f_hz, cfg.environment.T_env_k))
    lbc = cfg.link_budget
    lb = link_budget(
        f_hz, cfg.distance_m, lbc.A_t_m2, lbc.A_r_m2,
        gamma_abs_per_m=cfg.large_scale.gamma_abs_per_m[cfg.band_ghz],
        coupling_model=lbc.coupling_model, g_enh=lbc.G_enh,
    )  # fmt: skip
    mu_g, _ = cfg.large_scale.moments_g(cfg.band_ghz, cfg.distance_m)
    rar = RarTable.from_cfg(cfg.rar, cfg.environment.T_env_k).receiver_at(f_hz)
    if rar is None:
        raise ValueError(f"no RAR input at {cfg.band_ghz:g} GHz (outside the literature range)")
    caps: CapMap = {
        "c_holevo": lambda e: c_holevo(e, cfg.n_s, nth),
        "c_het": lambda e: c_het(e, cfg.n_s, nth),
        "c_rx": lambda e: c_rx(e, cfg.n_s, nth, rar),
    }
    return SmallScale(cfg.K, lb.eta_geo), lb.psi0, mu_g, nth, caps


def compute(cfg: E2Config) -> list[E2Row]:
    small, psi0, mu_g, nth, caps = _setup(cfg)
    if mu_g <= 0:
        raise ValueError("E2 needs a positive stochastic mean mu_G")
    psi_bar = psi0 + mu_g
    rows: list[E2Row] = []
    for ratio in cfg.law_comparison.sigma_ratio:
        laws = dict(zip(LAWS, matched_laws(psi0, mu_g, ratio * psi_bar), strict=True))
        res = {
            n: evaluate_law(small, law, nth, cfg.source_state.r, caps) for n, law in laws.items()
        }
        p_ref, c_ref = res["loggamma"]
        for name, law in laws.items():
            p, c = res[name]
            above = law.mass_above_one() if isinstance(law, LogNormalV0) else 0.0
            rows.append(
                E2Row(
                    ratio, name, bool(law.physical), above, clipped_mass(small, law), p,
                    c["c_holevo"], c["c_het"], c["c_rx"], p - p_ref,
                    *(c[k] / c_ref[k] - 1.0 for k in CAPS),
                )
            )  # fmt: skip
    return rows


def compute_pdf(cfg: E2Config) -> list[PdfRow]:
    """Densities of ``eta_l`` under the three matched laws at ``pdf_sigma_ratio``."""
    _, psi0, mu_g, _, _ = _setup(cfg)
    eta = cfg.pdf_eta.values()
    laws = matched_laws(psi0, mu_g, cfg.pdf_sigma_ratio * (psi0 + mu_g))
    return [
        PdfRow(cfg.pdf_sigma_ratio, name, float(e), float(v))
        for name, law in zip(LAWS, laws, strict=True)
        for e, v in zip(eta, law.pdf(eta), strict=True)
    ]


def _figure(rows: list[E2Row]) -> Figure:
    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(COLUMN_WIDTH_IN, 4.6), sharex=True)
    styles = {"lognormal_v0": ("--", "V0 log-normal (clipped)"),
              "lognormal_conditioned": ("-", r"log-normal, $\Psi\geq0$")}  # fmt: skip
    for i, (name, (ls, label)) in enumerate(styles.items()):
        sel = [r for r in rows if r.law == name]
        x = [r.sigma_ratio for r in sel]
        axes[0].plot(x, [r.delta_p_out for r in sel], ls=ls, color=OKABE_ITO[i], label=label)
        for j, (col, lab) in enumerate(
            (("rel_delta_c_holevo", r"$C_\chi$"), ("rel_delta_c_het", r"$C_{\rm het}$"),
             ("rel_delta_c_rx", r"$C_{\rm rx}$ (RAR)"))
        ):  # fmt: skip
            axes[1].plot(x, [getattr(r, col) for r in sel], ls=ls, color=OKABE_ITO[2 + j],
                         label=lab if i == 0 else None)  # fmt: skip
    v0 = [r for r in rows if r.law == "lognormal_v0"]
    x = [r.sigma_ratio for r in v0]
    axes[2].semilogy(x, [max(r.mass_eta_l_above_one, 1e-300) for r in v0], color=OKABE_ITO[0],
                     label=r"$P(\eta_\ell>1)$")  # fmt: skip
    axes[2].semilogy(x, [max(r.clipped_mass, 1e-300) for r in v0], color=OKABE_ITO[1], ls="--",
                     label=r"clipped, $P(\eta>1)$")  # fmt: skip
    axes[2].set_ylim(bottom=1e-12)
    axes[0].set_ylabel(r"$\Delta P_{\rm out}$")
    axes[1].set_ylabel(r"$\Delta\bar C/\bar C$")
    axes[2].set_ylabel("V0 mass above 1")
    axes[2].set_xlabel(r"$\sigma_\Psi/\bar\Psi$")
    for ax in axes:
        ax.set_xscale("log")
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _fig_pdf(rows: list[PdfRow]) -> Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.4))
    labels = {"loggamma": "shifted log-gamma", "lognormal_v0": "V0 log-normal",
              "lognormal_conditioned": r"log-normal, $\Psi\geq0$"}  # fmt: skip
    for i, name in enumerate(LAWS):
        sel = [r for r in rows if r.law == name]
        ax.plot([r.eta_l for r in sel], [r.pdf for r in sel], color=OKABE_ITO[i],
                ls="--" if name == "lognormal_v0" else "-", label=labels[name])  # fmt: skip
    ax.axvline(1.0, color="0.5", lw=0.6)
    ax.set_xlabel(r"large-scale transmissivity $\eta_\ell$")
    ax.set_ylabel("probability density")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def run(config_path: str | Path, out_root: str | Path = "results") -> Path:
    """Compute E2 and write ``results/e2_law_comparison/<hash>/``."""
    cfg, h = load_config(config_path, E2Config)
    assert isinstance(cfg, E2Config)
    out = Path(out_root) / cfg.experiment / h
    out.mkdir(parents=True, exist_ok=True)
    rows, pdf_rows = compute(cfg), compute_pdf(cfg)
    write_csv(out / "data.csv", E2Row._fields, rows)
    write_csv(out / "pdf.csv", PdfRow._fields, pdf_rows)
    save_fig(_figure(rows), out / "fig_law_comparison")
    save_fig(_fig_pdf(pdf_rows), out / "fig_pdf")
    mu_g, sd_g = cfg.large_scale.moments_g(cfg.band_ghz, cfg.distance_m)
    psi0 = cfg.large_scale.gamma_abs_per_m[cfg.band_ghz] * cfg.distance_m
    write_meta(
        out / "meta.json", cfg.experiment, h,
        {"n_rows": len(rows), "match": cfg.law_comparison.match,
         "eta_gt1": cfg.law_comparison.eta_gt1, "lognormal_v0_physical": False,
         "max_clipped_mass": max(r.clipped_mass for r in rows),
         "max_mass_eta_l_above_one": max(r.mass_eta_l_above_one for r in rows),
         "sourced_sigma_ratio": sd_g / (psi0 + mu_g),
         "per_link": cfg.large_scale.per_link,
         "gamma_abs_source": cfg.large_scale.gamma_abs_source,
         "stochastic_source": cfg.large_scale.stochastic_source},
    )  # fmt: skip
    logger.info("E2 written to %s", out)
    return out
