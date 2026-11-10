"""E3/E4 experiment plumbing on tiny grids (the full figures are produced by configs/)."""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from qmwlink.experiments import e3, e4
from qmwlink.experiments.__main__ import main
from qmwlink.experiments.schema import E3Config, E4Config, GridSpec, load_config

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
    assert c3.large_scale.gamma_abs_per_m[5.0] == 1.44e-6


def test_grid_spec() -> None:
    np.testing.assert_allclose(
        GridSpec(start=1, stop=100, num=3, scale="log").values(), [1, 10, 100]
    )
    np.testing.assert_allclose(GridSpec(start=0, stop=1, num=3).values(), [0, 0.5, 1])
    with pytest.raises(ValueError, match="positive"):
        GridSpec(start=0, stop=1, num=3, scale="log").values()


def test_e3_schema_rejects_missing_absorption_source(tmp_path: Path) -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "e3.yaml").read_text())
    del cfg["large_scale"]["gamma_abs_source"][30]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValidationError, match="source missing"):
        load_config(p, E3Config)
    cfg = yaml.safe_load((ROOT / "configs" / "e3.yaml").read_text())
    cfg["bands_ghz"] = [5, 60]
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValidationError, match="no gamma_abs"):
        load_config(p, E3Config)


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
    assert meta["friis_clipped_physical"] is False and "5.0" in meta["gamma_abs_sources"]
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
    cfg["large_scale"]["stochastic"] = {"mean": 1e-4, "std": 0.0}
    p = tmp_path / "det.yaml"
    p.write_text(yaml.safe_dump(cfg))
    c, _ = load_config(p, E3Config)
    assert isinstance(c, E3Config)
    rows = e3.compute(c)
    assert len(rows) == 4 and all(0.0 <= r.p_out <= 1.0 for r in rows)


def test_e4_run_and_reference_points(tmp_path: Path) -> None:
    out = e4.run(ROOT / "configs" / "e4.yaml", tmp_path / "res")
    rows = list(csv.DictReader((out / "data.csv").open()))
    assert (out / "fig.pdf").exists() and (out / "fig.png").exists()
    ref = [r for r in rows if r["band_ghz"] == "5.0" and r["front_end"] == "additive noise"]
    by_n = {round(float(r["n_add"]), 6): r for r in ref}
    assert float(by_n[0.5]["eta_star_rx"]) == pytest.approx(0.99960, abs=5e-6)
    assert by_n[1.0]["has_window"] == "False" and by_n[0.0]["has_window"] == "True"
    t06 = [r for r in rows if r["band_ghz"] == "5.0" and r["front_end"].endswith("0.6")]
    assert (
        float(next(r for r in t06 if float(r["n_add"]) == pytest.approx(0.01))["eta_star_rx"]) < 1.0
    )


def test_cli_entry_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "e4", str(ROOT / "configs" / "e4.yaml"), "--out", str(tmp_path)],
    )
    main()
    assert "e4_receiver_window" in capsys.readouterr().out
