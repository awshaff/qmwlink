"""E6: BBR crossover map (RQ4, FR-SVC-C4, TSD §8.3).

``S_E,BBR(f)`` and the required rms ``S_E`` for each target penalty, as separate SSB and DSB curves.
Unattainable targets (e.g. 3 dB for DSB) are NaN with a reason and are marked in the figure.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from qmwlink.experiments.common import save_fig, write_csv, write_meta
from qmwlink.experiments.schema import E6Config, LiteratureEntry, load_config, load_literature
from qmwlink.plotting.style import COLUMN_WIDTH_IN, OKABE_ITO, apply_style
from qmwlink.services.crossover import crossover_closed_form, dsb_image_floor_db
from qmwlink.thermal import n_th, s_e_bbr

logger = logging.getLogger(__name__)


class E6Row(NamedTuple):
    f_ghz: float
    s_e_bbr: float
    sideband: str
    delta_db: float
    n_amp_req: float
    s_e_req: float
    ratio_to_bbr: float
    dsb_floor_db: float
    reason: str


COLUMNS = E6Row._fields


def compute(cfg: E6Config) -> list[E6Row]:
    rows: list[E6Row] = []
    t = cfg.environment.T_env_k
    for fg in cfg.f_ghz.values():
        f = float(fg) * 1e9
        sbb = float(s_e_bbr(f, t))
        floor = dsb_image_floor_db(float(n_th(f, t)), cfg.eta_ref)
        for sb in ("ssb", "dsb"):
            for d in cfg.delta_db:
                r = crossover_closed_form(d, f, t, sb, cfg.eta_ref)
                rows.append(
                    E6Row(float(fg), sbb, sb, d, r.n_amp_req, r.s_e_req, r.ratio_to_bbr, floor,
                          r.reason or "")
                )  # fmt: skip
    return rows


def rj_slope(rows: list[E6Row], f_max_ghz: float) -> float:
    """Log-log slope of ``S_E,BBR`` vs ``f`` for ``f <= f_max_ghz`` (RJ regime: 1)."""
    seen = {r.f_ghz: r.s_e_bbr for r in rows if r.f_ghz <= f_max_ghz}
    f, s = np.array(sorted(seen)), np.array([seen[k] for k in sorted(seen)])
    return float(np.polyfit(np.log(f), np.log(s), 1)[0])


def _figure(rows: list[E6Row], cfg: E6Config, lit: list[LiteratureEntry]) -> Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.9))
    fs = sorted({r.f_ghz for r in rows})
    bbr = [next(r.s_e_bbr for r in rows if r.f_ghz == f) for f in fs]
    ax.loglog(fs, bbr, "k", lw=1.6, label=r"$S_{E,\mathrm{BBR}}$")
    for i, d in enumerate(cfg.delta_db):
        for sb, ls in (("ssb", "-"), ("dsb", "--")):
            sel = [r for r in rows if r.sideband == sb and r.delta_db == d]
            ys = [r.s_e_req for r in sel]
            bad = [r for r in sel if math.isnan(r.s_e_req)]
            if sb == "dsb" and bad:
                ok = [r.f_ghz for r in sel if not math.isnan(r.s_e_req)]
                where = f"below {min(ok):.3g} GHz" if ok else "at all f"
                note = f"DSB {d:g} dB unattainable {where} (image floor)"
                ax.text(0.03, 0.93 - 0.07 * i, note, transform=ax.transAxes, fontsize=5.5,
                        color=OKABE_ITO[i + 1])  # fmt: skip
                if not ok:
                    continue
            ax.loglog([r.f_ghz for r in sel], ys, ls=ls, color=OKABE_ITO[i + 1],
                      label=f"{sb.upper()}, {d:g} dB")  # fmt: skip
    for e in lit:
        ax.plot(e.f_ghz, e.s_e_rms(), "^" if e.sideband == "dsb" else "o", color="k", ms=4)
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel(r"$S_E$ (V m$^{-1}$ Hz$^{-1/2}$, rms)")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()
    return fig


def run(config_path: str | Path, out_root: str | Path = "results") -> Path:
    cfg, h = load_config(config_path, E6Config)
    assert isinstance(cfg, E6Config)
    out = Path(out_root) / cfg.experiment / h
    out.mkdir(parents=True, exist_ok=True)
    rows = compute(cfg)
    lit = load_literature(cfg.literature_rar)
    write_csv(out / "data.csv", COLUMNS, rows)
    save_fig(_figure(rows, cfg, lit), out / "fig_crossover")
    write_meta(
        out / "meta.json", cfg.experiment, h,
        {"n_rows": len(rows), "n_literature_points": len(lit),
         "bbr_slope_rj": rj_slope(rows, cfg.slope_fit_max_ghz),
         "slope_fit_max_ghz": cfg.slope_fit_max_ghz},
    )  # fmt: skip
    logger.info("E6 written to %s", out)
    return out
