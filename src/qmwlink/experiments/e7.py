"""E7: diversity hardening and ``N_req`` (FR-DIV-1..3, TSD §9).

``N_req`` (smallest ``N`` with ``F_N(eta*) <= eps`` under passive MRC) per band, ``K`` and ``eps``,
and the array outage ``F_N(eta*)`` vs ``N``. Reference rule: ``eta_l = 1``,
``eta_bar_Sigma = eta* + 0.9 (1 - eta*)``.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from qmwlink.channel.small_scale import SmallScale
from qmwlink.experiments.common import save_fig, write_csv, write_meta
from qmwlink.experiments.schema import E7Config, load_config
from qmwlink.plotting.style import COLUMN_WIDTH_IN, OKABE_ITO, apply_style
from qmwlink.services.diversity import array_outage, design_dimension, exponent_bound, n_req
from qmwlink.thermal import n_th

logger = logging.getLogger(__name__)


class NreqRow(NamedTuple):
    band_ghz: float
    K: float
    eps: float
    eta_star: float
    eta_bar_sigma: float
    n_req: int
    outage_at_n_req: float
    outage_at_n_req_minus_1: float
    design_dimension: float
    exponent_bound: float


class OutageRow(NamedTuple):
    band_ghz: float
    K: float
    n: int
    outage: float


def _eta_star(cfg: E7Config, band: float) -> tuple[float, float]:
    n = float(n_th(band * 1e9, cfg.environment.T_env_k))
    es = n / (1.0 + n)
    return es, es + cfg.eta_bar_fraction * (1.0 - es)


def compute_nreq(cfg: E7Config) -> list[NreqRow]:
    rows: list[NreqRow] = []
    for band in cfg.bands_ghz:
        es, eb = _eta_star(cfg, band)
        for k in cfg.K:
            for eps in cfg.eps:
                n = n_req(eps, k, eb, es)
                below = array_outage(n - 1, k, eb, es) if n > 1 else float("nan")
                rows.append(
                    NreqRow(band, k, eps, es, eb, n, array_outage(n, k, eb, es), below,
                            design_dimension(n, k), exponent_bound(n, k, es / eb, eb))
                )  # fmt: skip
    return rows


def compute_outage(cfg: E7Config) -> list[OutageRow]:
    ns = np.unique(np.round(cfg.n_grid.values()).astype(int))
    rows: list[OutageRow] = []
    for band in cfg.bands_ghz:
        es, eb = _eta_star(cfg, band)
        for k in cfg.K:
            rows += [OutageRow(band, k, int(n), array_outage(int(n), k, eb, es)) for n in ns]
    return rows


def _figure(rows: list[OutageRow], nreq: list[NreqRow], cfg: E7Config) -> Figure:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(COLUMN_WIDTH_IN, 3.3), sharey=True)
    for ax, band in zip(axes.ravel(), cfg.bands_ghz, strict=False):
        for i, k in enumerate(cfg.K):
            sel = [r for r in rows if r.band_ghz == band and k == r.K and r.outage > 0]
            ax.plot([r.n for r in sel], [r.outage for r in sel], color=OKABE_ITO[i],
                    label=f"$K={k:g}$" if band == cfg.bands_ghz[0] else None)  # fmt: skip
            for r in nreq:
                if r.band_ghz == band and k == r.K and r.eps == cfg.eps[0]:
                    ax.plot(r.n_req, r.eps, "o", color=OKABE_ITO[i], ms=3)
        for eps in cfg.eps:
            ax.axhline(eps, color="k", lw=0.5, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-12, 1.0)
        ax.text(0.05, 0.08, f"{band:g} GHz", transform=ax.transAxes, fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("elements $N$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$F_N(\eta^*)$")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def run(config_path: str | Path, out_root: str | Path = "results") -> Path:
    cfg, h = load_config(config_path, E7Config)
    assert isinstance(cfg, E7Config)
    out = Path(out_root) / cfg.experiment / h
    out.mkdir(parents=True, exist_ok=True)
    nreq, outage = compute_nreq(cfg), compute_outage(cfg)
    write_csv(out / "nreq.csv", NreqRow._fields, nreq)
    write_csv(out / "outage_vs_n.csv", OutageRow._fields, outage)
    save_fig(_figure(outage, nreq, cfg), out / "fig_outage_vs_n")
    z = {f"{r.band_ghz:g}/{r.K:g}/{r.eps:g}": SmallScale(r.K, r.eta_bar_sigma, r.n_req).z
         for r in nreq}  # fmt: skip
    write_meta(
        out / "meta.json",
        cfg.experiment,
        h,
        {
            "n_nreq_rows": len(nreq),
            "eta_l": cfg.eta_l,
            "eta_bar_fraction": cfg.eta_bar_fraction,
            "min_Z_at_nreq": min(z.values()),
            "all_finite": all(math.isfinite(r.outage) for r in outage),
        },
    )
    logger.info("E7 written to %s", out)
    return out
