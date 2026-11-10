"""M6: runner, cache, metadata, CLI, LaTeX export, defaults (FR-EXP-1..5)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from qmwlink import cli
from qmwlink.experiments import registry, runner
from qmwlink.io import latex
from qmwlink.io.export import paper_figs
from qmwlink.io.literature import load_literature

ROOT = Path(__file__).resolve().parents[2]
N_EXP = 9


def _e6(tmp_path: Path, **over: Any) -> Path:
    cfg = yaml.safe_load((ROOT / "configs" / "e6.yaml").read_text())
    cfg["f_ghz"] = {"start": 1, "stop": 300, "num": 8, "scale": "log"}
    cfg["literature_rar"] = str(ROOT / "configs" / "literature_rar.yaml")
    cfg.update(over)
    p = tmp_path / "e6.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_registry_and_default_configs_validate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(ROOT)
    assert list(registry.EXPERIMENTS) == [f"e{i}" for i in range(1, N_EXP + 1)]
    for spec in registry.EXPERIMENTS.values():
        got, data, h = runner.load_and_validate(spec.default_config)
        assert got is spec and data["experiment"] == spec.name and len(h) == 12
    assert registry.get("E3").name == "e3_entanglement_outage"
    assert registry.get("e6_bbr_crossover").exp_id == "e6"
    with pytest.raises(KeyError, match="unknown experiment"):
        registry.get("e42")


def test_invalid_config_fails_with_clear_message(tmp_path: Path) -> None:
    """FR-EXP-1 AC: unknown keys rejected, message names the file and the field."""
    bad = _e6(tmp_path, typo_key=1)
    with pytest.raises(runner.ConfigError, match=r"e6\.yaml.*\n\s+typo_key"):
        runner.run_config(bad, tmp_path / "out")
    with pytest.raises(runner.ConfigError, match="not found"):
        runner.run_config(tmp_path / "nope.yaml")
    p = tmp_path / "x.yaml"
    p.write_text("- 1\n")
    with pytest.raises(runner.ConfigError, match="'experiment' is required"):
        runner.run_config(p)
    p.write_text("experiment: e42\n")
    with pytest.raises(runner.ConfigError, match="unknown experiment"):
        runner.run_config(p)
    p.write_text("a: [1\n")
    with pytest.raises(runner.ConfigError, match="not valid YAML"):
        runner.run_config(p)
    p.write_text("experiment: e6_bbr_crossover\ndelta_db: nope\n")
    with pytest.raises(runner.ConfigError, match="delta_db"):
        runner.run_config(p)


def test_run_writes_outputs_metadata_and_caches(tmp_path: Path) -> None:
    """FR-EXP-3 (files, complete metadata) and FR-EXP-4 (second run is a cache hit)."""
    cfg, out = _e6(tmp_path), tmp_path / "out"
    first = runner.run_config(cfg, out)
    assert not first.cache_hit and first.experiment == "e6_bbr_crossover"
    for name in ("fig_crossover.pdf", "fig_crossover.png", "data.csv", "meta.json"):
        assert (first.out_dir / name).stat().st_size > 0
    meta = json.loads((first.out_dir / "meta.json").read_text())
    for key in ("git_commit", "config_hash", "seed", "packages", "wall_time_s", "complete"):
        assert key in meta
    assert meta["config_hash"] == first.out_dir.name and meta["complete"] is True
    assert meta["seed"] is None  # E6 is deterministic: no seed in its config
    assert {"numpy", "scipy", "qmwlink"} <= set(meta["packages"])
    mtime = (first.out_dir / "data.csv").stat().st_mtime_ns
    second = runner.run_config(cfg, out)
    assert second.cache_hit and second.out_dir == first.out_dir
    assert (first.out_dir / "data.csv").stat().st_mtime_ns == mtime
    third = runner.run_config(cfg, out, force=True)
    assert not third.cache_hit
    assert (first.out_dir / "data.csv").stat().st_mtime_ns > mtime


def test_cache_misses_when_external_input_changes(tmp_path: Path) -> None:
    lit = tmp_path / "literature_rar.yaml"
    shutil.copy(ROOT / "configs" / "literature_rar.yaml", lit)
    cfg, out = _e6(tmp_path, literature_rar=str(lit)), tmp_path / "out"
    assert not runner.run_config(cfg, out).cache_hit
    assert runner.run_config(cfg, out).cache_hit
    lit.write_text("entries: []\n")
    res = runner.run_config(cfg, out)  # FR-EXP-7 (b): empty file -> no overlay, run succeeds
    assert not res.cache_hit
    assert json.loads((res.out_dir / "meta.json").read_text())["n_literature_points"] == 0
    (res.out_dir / "meta.json").write_text("{not json")
    assert not runner.run_config(cfg, out).cache_hit  # unreadable metadata is not a hit


def test_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, out = _e6(tmp_path), tmp_path / "out"
    assert cli.main(["run", str(cfg), "--out", str(out)]) == 0
    assert "computed" in capsys.readouterr().out
    assert cli.main(["run", str(cfg), "--out", str(out)]) == 0
    assert "cache hit" in capsys.readouterr().out
    assert cli.main(["run", str(cfg), "--out", str(out), "--force"]) == 0
    assert "computed" in capsys.readouterr().out
    assert cli.main(["list"]) == 0
    listing = capsys.readouterr().out
    assert all(f"e{i} " in listing for i in range(1, N_EXP + 1))
    assert cli.main(["run", str(_e6(tmp_path, nope=1))]) == 2
    assert "nope" in capsys.readouterr().err
    monkeypatch.chdir(ROOT)
    assert cli.main(["tables", "--out", str(tmp_path / "tables")]) == 0
    assert (tmp_path / "tables" / "params.tex").is_file()
    assert cli.main(["run", "e6", "--out", str(out)]) == 0  # experiment id -> default config
    assert cli.main(["figs", "--only", "e6", "--out", str(out)]) == 0
    assert "cache hit" in capsys.readouterr().out


def test_paper_figs_exports_figures_csv_and_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FR-EXP-2 plumbing on the two fast experiments (the full set is `make paper-figs`)."""
    monkeypatch.chdir(ROOT)
    paper = paper_figs(tmp_path, ids=["e6", "e7"])
    assert (paper / "e6_bbr_crossover" / "fig_crossover.pdf").is_file()
    assert (paper / "e7_diversity" / "nreq.csv").is_file()
    assert (paper / "e7_diversity" / "meta.json").is_file()
    names = {p.name for p in (paper / "tables").iterdir()}
    assert names == {"params.tex", "thermal.tex", "rar_literature.tex", "n_req.tex"}
    assert r"20\,017\,832" in (paper / "tables" / "n_req.tex").read_text()
    assert cli.main(["paper-figs", "--only", "e6", "--out", str(tmp_path)]) == 0


# ---- LaTeX ----------------------------------------------------------------------------------
def test_latex_escape_and_numbers() -> None:
    assert latex.escape("50% & #1_a {x} ~ ^ $ \\") == (
        r"50\% \& \#1\_a \{x\} \textasciitilde{} \textasciicircum{} \$ \textbackslash{}"
    )
    assert latex.fmt_number(1.524e-6) == r"$1.524\times10^{-6}$"
    assert latex.fmt_number(300.0) == "$300$"
    assert latex.fmt_number(0.2) == "$0.2$"
    assert latex.fmt_number(0.0) == "$0$"
    assert latex.fmt_number(20017832) == r"$20\,017\,832$"
    assert latex.fmt_number(2.5e11) == r"$2.5\times10^{11}$"
    assert latex.fmt_number(float("nan")) == "--"
    with pytest.raises(TypeError):
        latex.fmt_number(True)


def test_booktabs_structure() -> None:
    tex = latex.booktabs_table(
        ("a", latex.Raw("$b$")), [(1, "x_y"), (True, [1, 2])], caption="c", label="tab:t"
    )
    for token in (r"\toprule", r"\midrule", r"\bottomrule", r"\begin{table}", r"\label{tab:t}"):
        assert tex.count(token) == 1
    assert "a & $b$ \\\\" in tex and r"$1$ & x\_y" in tex and "yes & $1$, $2$" in tex
    assert r"\begin{table*}" in latex.booktabs_table(("a",), [], caption="c", label="l", star=True)
    with pytest.raises(ValueError, match="one cell per column"):
        latex.booktabs_table(("a", "b"), [(1,)], caption="c", label="l")


def test_thermal_table_is_computed_from_code() -> None:
    tex = latex.thermal_reference_table()
    for ref in ("1249.697", "0.99920", "0.95314", "0.603", "35.748", "3.009", "2.907"):
        assert ref in tex  # TSD §11


def test_parameter_table_requires_sources(tmp_path: Path) -> None:
    tex = latex.parameter_table(ROOT / "configs" / "defaults.yaml")
    assert "ITU-R P.676-12" in tex and r"$1.524\times10^{-6}$" in tex
    bad = tmp_path / "d.yaml"
    bad.write_text("parameters:\n  - {key: x, symbol: s, description: d, value: 1, unit: u}\n")
    with pytest.raises(ValueError, match="source"):
        latex.parameter_table(bad)


def test_n_req_and_literature_tables(tmp_path: Path) -> None:
    csv_path = tmp_path / "nreq.csv"
    csv_path.write_text("band_ghz,K,eps,n_req\n5.0,0.0,0.001,20017832\n5.0,10.0,0.001,3474510\n")
    tex = latex.n_req_table(csv_path)
    assert r"$20\,017\,832$ & $3\,474\,510$" in tex
    with pytest.raises(ValueError, match="no rows"):
        latex.n_req_table(csv_path, eps=0.5)
    lit = latex.literature_table(ROOT / "configs" / "literature_rar.yaml")
    assert lit.count(r"\\") == 1 + len(load_literature(ROOT / "configs" / "literature_rar.yaml"))
    assert r"Jing et al., Nat. Phys. 16, 911 (2020)" in lit
    paths = latex.write_tables(
        tmp_path / "t", defaults_path=ROOT / "configs" / "defaults.yaml",
        literature_path=tmp_path / "missing.yaml", nreq_csv=tmp_path / "missing.csv",
    )  # fmt: skip
    assert {p.name for p in paths} == {"params.tex", "thermal.tex"}


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_tables_compile_in_ieeetran(tmp_path: Path) -> None:
    """FR-EXP-5 AC: the exported tables compile in IEEEtran."""
    cls = subprocess.run(["kpsewhich", "IEEEtran.cls"], capture_output=True, text=True)
    if not cls.stdout.strip():
        pytest.skip("IEEEtran.cls not installed")
    csv_path = tmp_path / "nreq.csv"
    csv_path.write_text("band_ghz,K,eps,n_req\n5.0,0.0,0.001,20017832\n")
    paths = latex.write_tables(
        tmp_path, defaults_path=ROOT / "configs" / "defaults.yaml",
        literature_path=ROOT / "configs" / "literature_rar.yaml", nreq_csv=csv_path,
    )  # fmt: skip
    body = "\n".join(rf"\input{{{p.name}}}" for p in paths)
    (tmp_path / "main.tex").write_text(
        "\\documentclass[journal]{IEEEtran}\n\\usepackage{booktabs}\n\\begin{document}\n"
        f"Text.\n{body}\n\\end{{document}}\n"
    )
    proc = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout[-2000:]
    assert (tmp_path / "main.pdf").stat().st_size > 0


# ---- defaults.yaml ----------------------------------------------------------------------------
def test_defaults_agree_with_experiment_configs() -> None:
    d = {
        p["key"]: p
        for p in yaml.safe_load((ROOT / "configs/defaults.yaml").read_text())["parameters"]
    }
    e3 = yaml.safe_load((ROOT / "configs/e3.yaml").read_text())
    e5 = yaml.safe_load((ROOT / "configs/e5.yaml").read_text())
    bands = d["bands_ghz"]["value"]
    assert bands == e3["bands_ghz"] and d["K"]["value"] == e3["small_scale"]["K"]
    assert d["T_env_k"]["value"] == e3["environment"]["T_env_k"]
    assert d["A_t_m2"]["value"] == e3["link_budget"]["A_t_m2"]
    assert d["A_r_m2"]["value"] == e3["link_budget"]["A_r_m2"]
    assert d["coupling_model"]["value"] == e3["link_budget"]["coupling_model"]
    ls = e3["large_scale"]
    assert d["gamma_abs_per_m"]["value"] == [ls["gamma_abs_per_m"][b] for b in bands]
    assert d["stochastic_mean_per_m"]["value"] == [ls["stochastic_per_m"][b]["mean"] for b in bands]
    assert d["stochastic_std_per_m"]["value"] == [ls["stochastic_per_m"][b]["std"] for b in bands]
    assert d["r"]["value"] == e3["source_state"]["r"]
    assert d["P_t_w"]["value"] == e5["throughput_map"]["P_t_w"]
    assert all(p["source"].strip() for p in d.values())
