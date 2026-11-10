"""LaTeX (booktabs) export of the parameter table and reference tables. (FR-EXP-5)

Tables are plain ``table`` floats with ``tabular`` + booktabs rules and compile in IEEEtran with
``\\usepackage{booktabs}``. Free text is escaped; cells wrapped in :class:`Raw` are emitted as is.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from qmwlink.experiments.schema import load_literature
from qmwlink.receivers.rydberg import n_amp_from_s_e
from qmwlink.services.crossover import dsb_image_floor_db
from qmwlink.thermal import n_th, one_minus_eta_star, s_e_bbr

_ESCAPES = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}  # fmt: skip


@dataclass(frozen=True)
class Raw:
    """A cell that is already LaTeX (math, units); not escaped."""

    tex: str


def escape(text: str) -> str:
    """Escape LaTeX special characters in plain text."""
    return "".join(_ESCAPES.get(ch, ch) for ch in text)


def fmt_number(x: float, sig: int = 4) -> str:
    """Number as LaTeX math: fixed notation in ``[1e-3, 1e5)``, ``a\\times10^{b}`` otherwise."""
    if isinstance(x, bool):
        raise TypeError("bool is not a number here")
    if not math.isfinite(x):
        return "--"
    if x == 0:
        return "$0$"
    if float(x).is_integer() and abs(x) < 10**9:
        return f"${int(x):,}$".replace(",", r"\,")
    exp = math.floor(math.log10(abs(x)))
    if -3 <= exp < 5:
        return f"${x:.{sig}g}$"
    mant = f"{x / 10**exp:.{sig}g}"
    return rf"${mant}\times10^{{{exp}}}$"


def _cell(v: Any) -> str:
    if isinstance(v, Raw):
        return v.tex
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int | float):
        return fmt_number(float(v))
    if isinstance(v, list | tuple):
        return ", ".join(_cell(i) for i in v)
    return escape(str(v))


def booktabs_table(
    columns: Sequence[Any],
    rows: Sequence[Sequence[Any]],
    *,
    caption: str,
    label: str,
    align: str | None = None,
    star: bool = False,
) -> str:
    """A ``table`` float with a booktabs ``tabular``. ``star`` gives the two-column ``table*``."""
    if any(len(r) != len(columns) for r in rows):
        raise ValueError("every row must have one cell per column")
    env = "table*" if star else "table"
    spec = align or "l" * len(columns)
    lines = [
        rf"\begin{{{env}}}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        " & ".join(_cell(c) for c in columns) + r" \\",
        r"\midrule",
        *(" & ".join(_cell(c) for c in r) + r" \\" for r in rows),
        r"\bottomrule",
        r"\end{tabular}",
        rf"\end{{{env}}}",
    ]
    return "\n".join(lines) + "\n"


def parameter_table(defaults_path: str | Path = "configs/defaults.yaml") -> str:
    """The paper's parameter table from ``configs/defaults.yaml``; each row keeps its source."""
    data = yaml.safe_load(Path(defaults_path).read_text(encoding="utf-8")) or {}
    params = data.get("parameters", [])
    rows = []
    for p in params:
        missing = {"symbol", "description", "value", "unit", "source"} - set(p)
        if missing:
            raise ValueError(f"defaults entry {p.get('key', '?')!r} lacks {sorted(missing)}")
        rows.append(
            (Raw(p["symbol"]), Raw(p["description"]), p["value"], Raw(p["unit"]), p["source"])
        )
    return booktabs_table(
        ("Symbol", "Parameter", "Value", "Unit", "Source"), rows,
        caption="Simulation parameters.", label="tab:params", align="llllp{0.36\\textwidth}",
        star=True,
    )  # fmt: skip


def thermal_reference_table(
    bands_ghz: Sequence[float] = (5, 30, 100, 300), t_env_k: float = 300.0
) -> str:
    """TSD §11 reference table, recomputed from the code (never typed in)."""
    rows = []
    for b in bands_ghz:
        f = b * 1e9
        nth = float(n_th(f, t_env_k))
        rows.append(
            (
                Raw(f"${b:g}$"), Raw(f"${nth:.3f}$"), Raw(f"${1 - 1 / (1 + nth):.5f}$"),
                float(one_minus_eta_star(f, t_env_k)),
                Raw(f"${float(s_e_bbr(f, t_env_k)) * 1e7:.3f}$"),
                Raw(f"${dsb_image_floor_db(nth):.3f}$"),
            )
        )  # fmt: skip
    cols = (
        Raw("$f$ (GHz)"), Raw(r"$\bar n_{\rm th}$"), Raw(r"$\eta^\ast$"), Raw(r"$1-\eta^\ast$"),
        Raw(r"$S_{E,\rm BBR}$ (nV\,cm$^{-1}$\,Hz$^{-1/2}$)"), Raw("DSB floor (dB)"),
    )  # fmt: skip
    return booktabs_table(
        cols, rows, align="rrrrrr", label="tab:thermal",
        caption=rf"Thermal reference values at $T={t_env_k:g}$\,K.",
    )  # fmt: skip


def n_req_table(nreq_csv: str | Path, eps: float = 1e-3) -> str:
    """``N_req`` table (FR-DIV-3) from the E7 ``nreq.csv`` at one target ``eps``."""
    with Path(nreq_csv).open(encoding="utf-8", newline="") as fh:
        recs = [r for r in csv.DictReader(fh) if math.isclose(float(r["eps"]), eps, rel_tol=1e-9)]
    if not recs:
        raise ValueError(f"no rows with eps = {eps:g} in {nreq_csv}")
    ks = sorted({float(r["K"]) for r in recs})
    bands = sorted({float(r["band_ghz"]) for r in recs})
    table = {(float(r["band_ghz"]), float(r["K"])): int(float(r["n_req"])) for r in recs}
    rows = [(Raw(f"${b:g}$"), *(table[(b, k)] for k in ks)) for b in bands]
    cols = (Raw("$f$ (GHz)"), *(Raw(f"$K={k:g}$") for k in ks))
    return booktabs_table(
        cols, rows, align="r" * len(cols), label="tab:nreq",
        caption=rf"Required number of elements $N_{{\rm req}}$ for an array outage of "
                rf"{fmt_number(eps)}.",
    )  # fmt: skip


def literature_table(
    path: str | Path = "configs/literature_rar.yaml", t_env_k: float = 300.0
) -> str:
    """Published RAR points with the derived ``N_amp`` (TSD §16.3)."""
    rows = []
    for e in load_literature(path):
        s = e.s_e_intrinsic_rms(t_env_k)
        rows.append(
            (e.source, Raw(f"${e.f_hz / 1e9:.4g}$"), s, e.field_convention, e.sideband,
             n_amp_from_s_e(s, e.f_hz))
        )  # fmt: skip
    cols = ("Source", Raw("$f$ (GHz)"), Raw(r"$S_E$ rms, intrinsic (V\,m$^{-1}$\,Hz$^{-1/2}$)"),
            "Quoted as", "Sideband", Raw(r"$N_{\rm amp}$"))  # fmt: skip
    return booktabs_table(
        cols, rows, align="p{0.34\\textwidth}rrllr", label="tab:rar-lit", star=True,
        caption="Published Rydberg atomic receiver sensitivities and the mapped added noise.",
    )  # fmt: skip


def write_tables(
    out_dir: str | Path,
    *,
    defaults_path: str | Path = "configs/defaults.yaml",
    literature_path: str | Path = "configs/literature_rar.yaml",
    nreq_csv: str | Path | None = None,
) -> list[Path]:
    """Write all ``.tex`` tables to ``out_dir`` and return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "params.tex": parameter_table(defaults_path),
        "thermal.tex": thermal_reference_table(),
    }
    if Path(literature_path).is_file() and load_literature(literature_path):
        tables["rar_literature.tex"] = literature_table(literature_path)
    if nreq_csv is not None and Path(nreq_csv).is_file():
        tables["n_req.tex"] = n_req_table(nreq_csv)
    paths = []
    for name, tex in tables.items():
        (out / name).write_text(tex, encoding="utf-8")
        paths.append(out / name)
    return paths
