"""E5/E6/E8 plumbing on tiny grids, plus config validation (FR-SVC-C5 (c),(d), FR-EXP-7)."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from qmwlink.experiments import e5, e6, e8
from qmwlink.experiments.__main__ import main
from qmwlink.experiments.rx_cfg import capacity_fn, measuring_receiver
from qmwlink.experiments.schema import (
    E5Config,
    E6Config,
    E8Config,
    RxSpec,
    load_config,
    load_literature,
)

ROOT = Path(__file__).resolve().parents[2]


def _cfg(tmp_path: Path, name: str, **over: object) -> Path:
    cfg = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
    cfg.update(over)
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_shipped_configs_validate() -> None:
    for name, model in (("e5", E5Config), ("e6", E6Config), ("e8", E8Config)):
        cfg, h = load_config(ROOT / "configs" / f"{name}.yaml", model)
        assert isinstance(cfg, model) and len(h) == 12


def test_missing_source_fails_validation(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "e5.yaml").read_text())
    del raw["rar_bands"][0]["source"]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError):
        load_config(p, E5Config)
    with pytest.raises(ValidationError):
        RxSpec(label="x", kind="rydberg", s_e=1e-7, field_convention="rms")  # no source
    with pytest.raises(ValidationError):
        RxSpec(label="x", kind="electronic", t_n_k=50.0, source="s")  # no tn_convention


def test_rx_cfg_builders() -> None:
    het = RxSpec(label="h", kind="het")
    assert measuring_receiver(het, 5e9) is not None
    assert measuring_receiver(RxSpec(label="v", kind="holevo"), 5e9) is None
    fn = capacity_fn(RxSpec(label="h", kind="hom"), 5e9, 10.0, 1000.0)
    assert float(fn(np.float64(0.5))) > 0


def test_literature_loader(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    assert load_literature(tmp_path / "missing.yaml") == []
    assert any("missing" in r.message for r in caplog.records)
    (tmp_path / "empty.yaml").write_text("entries: []\n")
    assert load_literature(tmp_path / "empty.yaml") == []
    ok = {"entries": [{"label": "a", "f_ghz": 5, "s_e": 2.0, "field_convention": "peak",
                       "sideband": "dsb", "source": "s"}]}  # fmt: skip
    (tmp_path / "ok.yaml").write_text(yaml.safe_dump(ok))
    (e,) = load_literature(tmp_path / "ok.yaml")
    assert e.s_e_rms() == pytest.approx(2.0 / math.sqrt(2.0))
    del ok["entries"][0]["field_convention"]
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump(ok))
    with pytest.raises(ValidationError):
        load_literature(tmp_path / "bad.yaml")


def test_e6_run_and_slope(tmp_path: Path) -> None:
    out = e6.run(_cfg(tmp_path, "e6", f_ghz={"start": 1, "stop": 300, "num": 12, "scale": "log"}),
                 tmp_path / "r")  # fmt: skip
    meta = json.loads((out / "meta.json").read_text())
    assert meta["bbr_slope_rj"] == pytest.approx(1.0, abs=5e-3)
    rows = list(csv.DictReader((out / "data.csv").open()))
    dsb3 = [r for r in rows if r["sideband"] == "dsb" and float(r["delta_db"]) == 3.0]
    assert any(r["reason"] and "image floor" in r["reason"] for r in dsb3)
    assert (out / "fig_crossover.pdf").exists()


def test_e8_run(tmp_path: Path) -> None:
    out = e8.run(
        _cfg(tmp_path, "e8", distance_m=[100.0], small_scale={"K": [10]},
             eps={"start": 1e-3, "stop": 1e-1, "num": 3, "scale": "log"}),
        tmp_path / "r",
    )  # fmt: skip
    rows = list(csv.DictReader((out / "data.csv").open()))
    by = {(r["receiver"], float(r["eps"])): float(r["c_eps"]) for r in rows}
    eps = sorted({e for _, e in by})
    for rcv in ("Holevo", "ideal het", "LNA", "RAR"):
        seq = [by[(rcv, e)] for e in eps]
        assert seq == sorted(seq)  # C_eps non-decreasing in eps
    for e in eps:
        assert by[("Holevo", e)] >= by[("ideal het", e)] - 1e-12
        assert by[("ideal het", e)] >= by[("LNA", e)] - 1e-12


def test_e5_run_masks_outside_band_range(tmp_path: Path) -> None:
    out = e5.run(
        _cfg(tmp_path, "e5", bands_ghz=[5, 30], n_s=[1e3],
             f_ghz={"start": 3, "stop": 60, "num": 4, "scale": "log"},
             distance_m={"start": 50, "stop": 500, "num": 2, "scale": "log"},
             small_scale={"K": [10]}),
        tmp_path / "r",
    )  # fmt: skip
    rows = list(csv.DictReader((out / "map.csv").open()))
    for r in rows:
        f = float(r["f_ghz"])
        masked = math.isnan(float(r["rho_db"]))
        assert masked == (f < 5 - 1e-9 or f > 300 + 1e-9)
    assert any(math.isnan(float(r["rho_db"])) for r in rows)  # 3 GHz < lowest band
    assert any(not math.isnan(float(r["rho_db"])) for r in rows)
    assert (out / "capacity.csv").exists() and (out / "fig_rho_map.png").exists()


def test_cli_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _cfg(tmp_path, "e6", f_ghz={"start": 1, "stop": 10, "num": 3, "scale": "log"})
    monkeypatch.setattr("sys.argv", ["x", "e6", str(p), "--out", str(tmp_path / "r")])
    main()
    assert list((tmp_path / "r").glob("e6_*/*/data.csv"))
