"""E3: entanglement outage vs distance (FR-SVC-E2, TSD §12).

For each band, ``K`` and distance: link budget -> compound PDT (shifted log-gamma large scale,
truncated noncentral chi-square small scale) -> ``P_out`` and ``E[E_N]``. Runs both coupling
models (``goubau`` default, ``friis_clipped`` flagged non-physical) for the sensitivity panel.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from qmwlink.channel.compound import CompoundPDT
from qmwlink.channel.large_scale import ShiftedLogGamma
from qmwlink.channel.link_budget import link_budget
from qmwlink.channel.small_scale import SmallScale
from qmwlink.experiments.common import save_fig, write_csv, write_meta
from qmwlink.experiments.schema import E3Config, load_config
from qmwlink.gaussian.channels import PIChannel
from qmwlink.gaussian.entanglement import log_negativity
from qmwlink.plotting.style import COLUMN_WIDTH_IN, OKABE_ITO, apply_style
from qmwlink.services.entanglement import EntanglementService
from qmwlink.thermal import n_th

logger = logging.getLogger(__name__)


class E3Row(NamedTuple):
    band_ghz: float
    K: float
    coupling_model: str
    physical: bool
    d_m: float
    tau2: float
    eta_geo: float
    psi0: float
    eta_star: float
    p_out: float
    e_log_neg: float


COLUMNS = E3Row._fields


def compute(cfg: E3Config) -> list[E3Row]:
    """All grid points. ``E[E_N]`` is skipped (0) when the outage is exactly 1."""
    rows: list[E3Row] = []
    d_grid = cfg.distance_m.values()
    st, lbc = cfg.large_scale.stochastic, cfg.link_budget
    for band in cfg.bands_ghz:
        f_hz = band * 1e9
        gamma = cfg.large_scale.gamma_abs_per_m[band]
        nth = float(n_th(f_hz, cfg.environment.T_env_k))
        for model in lbc.sensitivity_models:
            for d in d_grid:
                lb = link_budget(
                    f_hz, float(d), lbc.A_t_m2, lbc.A_r_m2,
                    gamma_abs_per_m=gamma, coupling_model=model, g_enh=lbc.G_enh,
                )  # fmt: skip
                law = ShiftedLogGamma.from_moments(lb.psi0, st.mean, st.std**2)
                for k in cfg.small_scale["K"]:
                    svc = EntanglementService(
                        CompoundPDT(SmallScale(k, lb.eta_geo), law),
                        nth, PIChannel.identity(), cfg.source_state.r,
                    )  # fmt: skip
                    p_out = svc.p_out()
                    e_n = 0.0
                    if p_out < 1.0:
                        e_n = svc.expect_entangled(
                            lambda e, s=svc: log_negativity(s.nu_minus(e))  # type: ignore[misc]
                        )
                    rows.append(
                        E3Row(band, k, model, lb.physical, float(d), lb.tau2, lb.eta_geo, lb.psi0,
                              svc.eta_star, p_out, e_n)
                    )  # fmt: skip
    return rows


def _panel_figure(
    rows: list[E3Row], cfg: E3Config, col: str, ylabel: str, log_y: bool, *, one_minus: bool = False
) -> Figure:
    apply_style()
    bands, ks = cfg.bands_ghz, cfg.small_scale["K"]
    fig, axes = plt.subplots(2, 2, figsize=(COLUMN_WIDTH_IN, 3.3), sharex=True)
    for ax, band in zip(axes.ravel(), bands, strict=False):
        for i, k in enumerate(ks):
            for model, ls in (("goubau", "-"), ("friis_clipped", "--")):
                sel = [
                    r for r in rows if r.band_ghz == band and k == r.K and r.coupling_model == model
                ]
                if not sel:
                    continue
                label = f"$K={k:g}$, {model}" if band == bands[0] else None
                ys = [(1.0 - getattr(r, col)) if one_minus else getattr(r, col) for r in sel]
                ax.plot([r.d_m for r in sel], ys, ls=ls,
                        color=OKABE_ITO[i % len(OKABE_ITO)], label=label)  # fmt: skip
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.text(0.05, 0.08, f"{band:g} GHz", transform=ax.transAxes, fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("distance (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    if log_y:
        for ax in axes.ravel():
            ax.set_ylim(bottom=1e-12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def run(config_path: str | Path, out_root: str | Path = "results") -> Path:
    """Compute E3 and write ``results/e3_entanglement_outage/<hash>/``."""
    cfg, h = load_config(config_path, E3Config)
    assert isinstance(cfg, E3Config)
    out = Path(out_root) / cfg.experiment / h
    out.mkdir(parents=True, exist_ok=True)
    rows = compute(cfg)
    write_csv(out / "data.csv", COLUMNS, rows)
    save_fig(_panel_figure(rows, cfg, "p_out", r"$P_{\rm out}$", False), out / "fig_p_out")
    save_fig(
        _panel_figure(rows, cfg, "p_out", r"$1-P_{\rm out}$", True, one_minus=True),
        out / "fig_p_ent",
    )
    ylab = r"$\mathbb{E}[E_N]$ (ebit)"
    save_fig(_panel_figure(rows, cfg, "e_log_neg", ylab, True), out / "fig_e_n")
    write_meta(
        out / "meta.json", cfg.experiment, h,
        {"seed": cfg.seed, "n_rows": len(rows),
         "n_rows_with_entanglement": sum(r.e_log_neg > 0 for r in rows),
         "friis_clipped_physical": False,
         "gamma_abs_sources": {str(k): v for k, v in cfg.large_scale.gamma_abs_source.items()}},
    )  # fmt: skip
    logger.info("E3 written to %s", out)
    return out
