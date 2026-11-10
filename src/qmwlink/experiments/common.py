"""Output helpers shared by experiments: CSV, metadata, figure export."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.figure import Figure


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        w.writerows(rows)


def write_meta(path: Path, experiment: str, cfg_hash: str, extra: dict[str, Any]) -> None:
    meta = {
        "experiment": experiment,
        "config_hash": cfg_hash,
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        **extra,
    }
    path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str), encoding="utf-8")


def save_fig(fig: Figure, stem: Path) -> None:
    """Vector PDF and PNG."""
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
