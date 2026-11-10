"""E3/E4 experiment plumbing on tiny grids (the full figures are produced by configs/)."""

from __future__ import annotations

import csv
import itertools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from qmwlink.channel.large_scale import ShiftedLogGamma
from qmwlink.experiments import e3, e4
from qmwlink.experiments.__main__ import main
from qmwlink.experiments.schema import E3Config, E4Config, FrontEndClass, GridSpec, load_config

ROOT = Path(__file__).resolve().parents[2]


def _e3_cfg(tmp_path: Path, **over: object) -> Path:
    cfg = yaml.safe_load((ROOT / "configs" / "e3.yaml").read_text())
    cfg["bands_ghz"] = [5, 30]
    cfg["distance_m"] = {"start": 10, "stop": 300, "num": 4, "scale": "log"}
    cfg["small_scale"] = {"K": [0, 10]}
    cfg.update(over)
    p = tmp_path / "e3.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_shipped_configs_validate() -> None:
    c3, h3 = load_config(ROOT / "configs" / "e3.yaml", E3Config)
    c4, h4 = load_config(ROOT / "configs" / "e4.yaml", E4Config)
    assert isinstance(c3, E3Config) and isinstance(c4, E4Config)
    assert len(h3) == len(h4) == 12 and h3 != h4
    assert c3.large_scale.gamma_abs_per_m[5.0] == 1.524e-6
    assert not c3.large_scale.per_link and c3.large_scale.stochastic_per_m is not None
    assert c3.large_scale.stochastic_per_m[300.0].mean == 1.523e-3


def test_grid_spec() -> None:
    np.testing.assert_allclose(
        GridSpec(start=1, stop=100, num=3, scale="log").values(), [1, 10, 100]
    )
    np.testing.assert_allclose(GridSpec(start=0, stop=1, num=3).values(), [0, 0.5, 1])
    with pytest.raises(ValueError, match="positive"):
        GridSpec(start=0, stop=1, num=3, scale="log").values()


def _bad(tmp_path: Path, mutate: Callable[[dict[str, Any]], None]) -> Path:
    cfg = yaml.safe_load((ROOT / "configs" / "e3.yaml").read_text())
    mutate(cfg)
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_e3_schema_rejections(tmp_path: Path) -> None:
    def no_src(c: dict[str, Any]) -> None:
        c["large_scale"]["gamma_abs_source"] = "  "

    def dict_src(c: dict[str, Any]) -> None:
        c["large_scale"]["gamma_abs_source"] = {5: "x"}

    def no_stoch_src(c: dict[str, Any]) -> None:
        del c["large_scale"]["stochastic_source"]

    def missing_band(c: dict[str, Any]) -> None:
        del c["large_scale"]["stochastic_per_m"][30]

    def both(c: dict[str, Any]) -> None:
        c["large_scale"]["stochastic"] = {"mean": 1e-4, "std": 1e-4}

    def partial(c: dict[str, Any]) -> None:
        c["large_scale"]["correlation"] = "partial"

    def no_abs(c: dict[str, Any]) -> None:
        c["bands_ghz"] = [5, 60]

    for mutate, msg in (
        (no_src, "must not be empty"), (dict_src, "source missing"),
        (no_stoch_src, "stochastic_source"), (missing_band, "missing for bands"),
        (both, "exactly one"), (partial, "full"), (no_abs, "no gamma_abs"),
    ):  # fmt: skip
        with pytest.raises(ValidationError, match=msg):
            load_config(_bad(tmp_path, mutate), E3Config)


def test_per_metre_law_scaling() -> None:
    """FR-CH-2 (b): k is distance-independent and theta scales linearly in d (rtol 1e-12)."""
    cfg, _ = load_config(ROOT / "configs" / "e3.yaml", E3Config)
    assert isinstance(cfg, E3Config)
    ls = cfg.large_scale
    for band in (5.0, 30.0, 100.0, 300.0):
        laws = [ls.law(band, d, 0.1) for d in (10.0, 100.0, 1000.0)]
        assert isinstance(laws[0], ShiftedLogGamma)
        ks = [law.k for law in laws]  # type: ignore[attr-defined]
        th = [law.theta for law in laws]  # type: ignore[attr-defined]
        assert ks[1] == pytest.approx(ks[0], rel=1e-12) and ks[2] == pytest.approx(ks[0], rel=1e-12)
        assert th[1] / th[0] == pytest.approx(10.0, rel=1e-12)
        assert th[2] / th[1] == pytest.approx(10.0, rel=1e-12)
        m1, s1 = ls.stochastic_per_m[band].mean, ls.stochastic_per_m[band].std  # type: ignore[index]
        assert ks[0] == pytest.approx(m1**2 / s1**2, rel=1e-12)
    # off-band frequencies are log-log interpolated, exact at the nodes
    assert ls.moments_g(5.0, 1.0)[0] == 4.160e-7
    lo, hi = ls.moments_g(5.0, 1.0)[0], ls.moments_g(30.0, 1.0)[0]
    assert lo < ls.moments_g(12.0, 1.0)[0] < hi
    assert ls.gamma_abs_at(5.0) == 1.524e-6 and ls.gamma_abs_at(30.0) == 4.394e-6


def test_per_link_form_is_flagged(tmp_path: Path) -> None:
    """FR-CH-2 (c): the legacy per-link form is accepted but recorded as ``per_link``."""

    def legacy(c: dict[str, Any]) -> None:
        del c["large_scale"]["stochastic_per_m"], c["large_scale"]["stochastic_source"]
        c["large_scale"]["stochastic"] = {"mean": 1e-3, "std": 5e-4}
        c["bands_ghz"] = [5]
        c["distance_m"] = {"start": 10, "stop": 20, "num": 2, "scale": "lin"}
        c["small_scale"] = {"K": [10]}

    out = e3.run(_bad(tmp_path, legacy), tmp_path / "res")
    assert json.loads((out / "meta.json").read_text())["per_link"] is True


def test_e3_run_outputs_and_physics(tmp_path: Path) -> None:
    out = e3.run(_e3_cfg(tmp_path), tmp_path / "res")
    assert {p.name for p in out.iterdir()} >= {
        "data.csv", "meta.json", "fig_p_out.pdf", "fig_p_out.png", "fig_e_n.pdf", "fig_e_n.png",
    }  # fmt: skip
    rows = list(csv.DictReader((out / "data.csv").open()))
    assert len(rows) == 2 * 2 * 4 * 2  # bands x models x distances x K
    assert {r["physical"] for r in rows if r["coupling_model"] == "goubau"} == {"True"}
    assert {r["physical"] for r in rows if r["coupling_model"] == "friis_clipped"} == {"False"}
    meta = json.loads((out / "meta.json").read_text())
    assert meta["friis_clipped_physical"] is False and meta["per_link"] is False
    assert "P.676-12" in meta["gamma_abs_source"] and "KHU" in meta["stochastic_source"]
    assert meta["location_stat"] == "annual"
    # outage (goubau, 5 GHz, K = 10) is non-decreasing with distance
    sel = [
        r
        for r in rows
        if r["band_ghz"] == "5.0" and r["K"] == "10.0" and r["coupling_model"] == "goubau"
    ]
    p = [float(r["p_out"]) for r in sel]
    assert all(b >= a - 1e-12 for a, b in itertools.pairwise(p))
    assert all(0.0 <= v <= 1.0 for v in p)
    # E[E_N] is zero exactly where the outage is exactly one
    assert all(float(r["e_log_neg"]) == 0.0 for r in rows if float(r["p_out"]) == 1.0)


def test_e3_gamma_zero_std_uses_deterministic(tmp_path: Path) -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "e3.yaml").read_text())
    cfg["bands_ghz"] = [5]
    cfg["distance_m"] = {"start": 10, "stop": 20, "num": 2, "scale": "lin"}
    cfg["small_scale"] = {"K": [10]}
    del cfg["large_scale"]["stochastic_per_m"], cfg["large_scale"]["stochastic_source"]
    cfg["large_scale"]["stochastic"] = {"mean": 1e-4, "std": 0.0}
    p = tmp_path / "det.yaml"
    p.write_text(yaml.safe_dump(cfg))
    c, _ = load_config(p, E3Config)
    assert isinstance(c, E3Config)
    rows = e3.compute(c)
    assert len(rows) == 4 and all(0.0 <= r.p_out <= 1.0 for r in rows)


def test_e4_run_and_reference_points(tmp_path: Path) -> None:
    """TSD §16.5 thresholds at 5 GHz, 300 K; the window depends only on N_add_in."""
    out = e4.run(ROOT / "configs" / "e4.yaml", tmp_path / "res")
    rows = list(csv.DictReader((out / "data.csv").open()))
    assert (out / "fig.pdf").exists() and (out / "fig.png").exists()
    pts = {
        r["front_end"]: r for r in rows if r["band_ghz"] == "5.0" and r["is_class_point"] == "True"
    }
    expect = {
        "Ideal": 0.999200, "Electro-optic, low noise": 0.999328,
        "Electro-optic, high eff.": 0.999528, "Quantum-limited PI amplifier (bound)": 0.999600,
        "Piezo-optomech., low noise": 0.999664, "Rydberg mm-wave to optical": 0.999680,
        "Piezo-optomech., high eff.": 0.999952,
    }  # fmt: skip
    for label, v in expect.items():
        assert float(pts[label]["eta_star_rx"]) == pytest.approx(v, abs=1e-6), label
        assert pts[label]["has_window"] == "True"
    hemt = pts["Cryogenic HEMT 1.5 K at 5 GHz"]
    assert hemt["has_window"] == "False" and float(hemt["eta_star_rx"]) > 1.0
    # sweep: window shrinks with N_add_in and vanishes at N_add_in = 1
    sweep = [r for r in rows if r["band_ghz"] == "5.0" and r["is_class_point"] == "False"]
    by_n = {round(float(r["n_add_in"]), 6): r for r in sweep}
    assert float(by_n[0.5]["eta_star_rx"]) == pytest.approx(0.99960, abs=5e-6)
    assert by_n[1.0]["has_window"] == "False" and by_n[0.0]["has_window"] == "True"


def test_e4_schema_requires_consistent_noise() -> None:
    with pytest.raises(ValidationError, match="N_t must equal"):
        FrontEndClass(label="x", eta_t=0.5, n_t=0.0, N_t=0.3, N_add_in=0.2, source="s")
    ok = FrontEndClass(label="x", eta_t=0.5, n_t=0.0, N_t=0.1, N_add_in=0.2, source="s")
    assert ok.N_t == 0.1


def test_cli_entry_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "e4", str(ROOT / "configs" / "e4.yaml"), "--out", str(tmp_path)],
    )
    main()
    assert "e4_receiver_window" in capsys.readouterr().out
