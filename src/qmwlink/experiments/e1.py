"""E1: small-scale PDT shape and truncation effect (RQ1; FR-CH-1, FR-SVC-E2, TSD §6.1, §8.1).

Three outputs:

- pdf of the truncated law next to the untruncated surrogate (``physical = False``), per ``K``;
- removed mass ``1 - Z_K`` and the outage pair ``(P_out, P_out^surr)`` vs ``eta_s_bar`` for a
  deterministic ``eta_l = 1`` (where ``P_out/P_out^surr = 1/Z_K`` exactly);
- the same pair over the E3 link (compound PDT) vs distance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from qmwlink.channel.compound import CompoundPDT
from qmwlink.channel.large_scale import DeterministicLoss
from qmwlink.channel.link_budget import link_budget
from qmwlink.channel.small_scale import SmallScale
from qmwlink.experiments.common import save_fig, write_csv, write_meta
from qmwlink.experiments.schema import E1Config, load_config
from qmwlink.gaussian.channels import PIChannel
from qmwlink.numerics.special import ncx2_pdf
from qmwlink.plotting.style import COLUMN_WIDTH_IN, OKABE_ITO, apply_style
from qmwlink.services.entanglement import EntanglementService
from qmwlink.thermal import n_th

logger = logging.getLogger(__name__)

_R_ANY = 1.0
"""Squeezing used to build the service; the TMSV threshold and outage do not depend on it."""


class PdfRow(NamedTuple):
    K: float
    eta_s_bar: float
    eta: float
    pdf_truncated: float
    pdf_surrogate: float


class MassRow(NamedTuple):
    K: float
    eta_s_bar: float
    z: float
    removed_mass: float
    eta_star: float
    p_out: float
    p_out_surr: float
    delta_p_out: float
    ratio: float


class CompoundRow(NamedTuple):
    band_ghz: float
    K: float
    d_m: float
    eta_geo: float
    removed_mass: float
    eta_star: float
    p_out: float
    p_out_surr: float
    delta_p_out: float
    ratio: float


def surrogate_pdf(small: SmallScale, eta: np.ndarray) -> np.ndarray:
    """Untruncated surrogate density ``f_chi'^2(eta/sigma^2)/sigma^2`` (NON-PHYSICAL, RQ1 only)."""
    return ncx2_pdf(np.asarray(eta) / small.sigma2, small.nu, small.nc) / small.sigma2


def compute_pdf(cfg: E1Config) -> list[PdfRow]:
    eta = cfg.pdf_eta.values()
    rows: list[PdfRow] = []
    for k in cfg.K:
        s = SmallScale(k, cfg.pdf_eta_s_bar)
        trunc, surr = s.pdf(eta), surrogate_pdf(s, eta)
        rows += [
            PdfRow(k, cfg.pdf_eta_s_bar, float(e), float(t), float(u))
            for e, t, u in zip(eta, trunc, surr, strict=True)
        ]
    return rows


def compute_mass(cfg: E1Config) -> list[MassRow]:
    """``eta_s_bar`` sweep with ``eta_l = 1`` (deterministic), TMSV, identity front end."""
    nth = float(n_th(cfg.band_ghz * 1e9, cfg.environment.T_env_k))
    rows: list[MassRow] = []
    for k in cfg.K:
        for eb in cfg.eta_s_bar.values():
            small = SmallScale(k, float(eb))
            svc = EntanglementService(
                CompoundPDT(small, DeterministicLoss(0.0)), nth, PIChannel.identity(), _R_ANY
            )
            t = svc.truncation_effect()
            rows.append(
                MassRow(k, float(eb), small.z, t.removed_mass, svc.eta_star, t.p_out,
                        t.p_out_surr, t.delta, t.ratio)
            )  # fmt: skip
    return rows


def compute_compound(cfg: E1Config) -> list[CompoundRow]:
    """Truncation effect over the E3 link: compound PDT at each ``(band, d, K)``."""
    c = cfg.compound
    rows: list[CompoundRow] = []
    for band in c.bands_ghz:
        f_hz = band * 1e9
        nth = float(n_th(f_hz, cfg.environment.T_env_k))
        gamma = c.large_scale.gamma_abs_per_m[band]
        for d in c.distance_m.values():
            lb = link_budget(
                f_hz, float(d), c.link_budget.A_t_m2, c.link_budget.A_r_m2,
                gamma_abs_per_m=gamma, coupling_model=c.link_budget.coupling_model,
                g_enh=c.link_budget.G_enh,
            )  # fmt: skip
            law = c.large_scale.law(band, float(d), lb.psi0)
            for k in cfg.K:
                small = SmallScale(k, lb.eta_geo)
                svc = EntanglementService(
                    CompoundPDT(small, law), nth, PIChannel.identity(), _R_ANY
                )
                t = svc.truncation_effect()
                rows.append(
                    CompoundRow(band, k, float(d), lb.eta_geo, t.removed_mass, svc.eta_star,
                                t.p_out, t.p_out_surr, t.delta, t.ratio)
                )  # fmt: skip
    return rows


def _fig_pdf(rows: list[PdfRow], cfg: E1Config) -> Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.5))
    for i, k in enumerate(cfg.K):
        sel = [r for r in rows if k == r.K]
        col = OKABE_ITO[i % len(OKABE_ITO)]
        ax.plot([r.eta for r in sel], [r.pdf_truncated for r in sel], color=col,
                label=f"$K={k:g}$")  # fmt: skip
        ax.plot([r.eta for r in sel], [r.pdf_surrogate for r in sel], color=col, ls="--")
    ax.plot([], [], "k-", label="truncated")
    ax.plot([], [], "k--", label="untruncated")
    ax.axvline(1.0, color="0.5", lw=0.6)
    ax.set_xlabel(r"small-scale transmissivity $\eta_s$")
    ax.set_ylabel("probability density")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return fig


def _fig_mass(rows: list[MassRow], cfg: E1Config) -> Figure:
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH_IN, 3.6), sharex=True)
    for i, k in enumerate(cfg.K):
        sel = [r for r in rows if k == r.K]
        col = OKABE_ITO[i % len(OKABE_ITO)]
        x = [r.eta_s_bar for r in sel]
        axes[0].plot(x, [r.removed_mass for r in sel], color=col, label=f"$K={k:g}$")
        axes[1].plot(x, [r.delta_p_out for r in sel], color=col)
    axes[0].set_ylabel(r"removed mass $1-\mathcal{Z}_K$")
    axes[1].set_ylabel(r"$\Delta P_{\rm out}$")
    axes[1].set_xlabel(r"scale parameter $\bar\eta_s$")
    axes[0].legend(frameon=False, ncol=2)
    fig.tight_layout()
    return fig


def _fig_compound(rows: list[CompoundRow], cfg: E1Config) -> Figure:
    apply_style()
    bands = cfg.compound.bands_ghz
    fig, axes = plt.subplots(
        2, len(bands), figsize=(COLUMN_WIDTH_IN, 3.4), sharex=True, squeeze=False
    )
    for j, band in enumerate(bands):
        for i, k in enumerate(cfg.K):
            sel = [r for r in rows if r.band_ghz == band and k == r.K]
            col = OKABE_ITO[i % len(OKABE_ITO)]
            d = [r.d_m for r in sel]
            axes[0, j].plot(d, [r.delta_p_out for r in sel], color=col,
                            label=f"$K={k:g}$" if j == 0 else None)  # fmt: skip
            axes[1, j].plot(d, [r.ratio for r in sel], color=col)
        axes[0, j].set_xscale("log")
        axes[0, j].text(0.05, 0.86, f"{band:g} GHz", transform=axes[0, j].transAxes, fontsize=7)
        axes[1, j].set_xlabel("distance (m)")
    axes[0, 0].set_ylabel(r"$\Delta P_{\rm out}$")
    axes[1, 0].set_ylabel(r"$P_{\rm out}/P_{\rm out}^{\rm surr}$")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def run(config_path: str | Path, out_root: str | Path = "results") -> Path:
    """Compute E1 and write ``results/e1_truncation/<hash>/``."""
    cfg, h = load_config(config_path, E1Config)
    assert isinstance(cfg, E1Config)
    out = Path(out_root) / cfg.experiment / h
    out.mkdir(parents=True, exist_ok=True)
    pdf_rows, mass_rows, comp_rows = compute_pdf(cfg), compute_mass(cfg), compute_compound(cfg)
    write_csv(out / "pdf.csv", PdfRow._fields, pdf_rows)
    write_csv(out / "data.csv", MassRow._fields, mass_rows)
    write_csv(out / "compound.csv", CompoundRow._fields, comp_rows)
    save_fig(_fig_pdf(pdf_rows, cfg), out / "fig_pdf")
    save_fig(_fig_mass(mass_rows, cfg), out / "fig_removed_mass")
    save_fig(_fig_compound(comp_rows, cfg), out / "fig_compound")
    ls = cfg.compound.large_scale
    write_meta(
        out / "meta.json", cfg.experiment, h,
        {"n_rows": len(mass_rows) + len(comp_rows), "surrogate_physical": False,
         "max_removed_mass": max(r.removed_mass for r in mass_rows),
         "max_abs_delta_p_out_compound": max(abs(r.delta_p_out) for r in comp_rows),
         "per_link": ls.per_link, "gamma_abs_source": ls.gamma_abs_source,
         "stochastic_source": ls.stochastic_source},
    )  # fmt: skip
    logger.info("E1 written to %s", out)
    return out
