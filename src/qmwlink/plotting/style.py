"""IEEE single-column figure style: 3.5 in wide, serif, colorblind-safe, no in-figure titles."""

from __future__ import annotations

import matplotlib.pyplot as plt
from cycler import cycler

COLUMN_WIDTH_IN = 3.5
OKABE_ITO = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
"""Okabe-Ito colorblind-safe cycle."""


def apply_style() -> None:
    """Set global rcParams (serif, small fonts, vector-friendly)."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.prop_cycle": cycler(color=OKABE_ITO),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 1.1,
        }
    )
