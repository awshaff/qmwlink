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

from qmwlink.constants import K_B, H
from qmwlink.experiments import e5, e6, e8
from qmwlink.experiments.__main__ import main
from qmwlink.experiments.e5 import _tables
from qmwlink.experiments.rx_cfg import capacity_fn, measuring_receiver
from qmwlink.experiments.schema import (
    E5Config,
    E6Config,
    E8Config,
    RxSpec,
    load_config,
    load_literature,
)
from qmwlink.receivers.rydberg import RydbergReceiver

ROOT = Path(__file__).resolve().parents[2]


_E5_TM = yaml.safe_load((ROOT / "configs" / "e5.yaml").read_text())["throughput_map"]


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
    del raw["throughput_map"]["lna"]["bands"][0]["source"]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError):
        load_config(p, E5Config)
    raw = yaml.safe_load((ROOT / "configs" / "e5.yaml").read_text())
    raw["receivers"].append({"label": "x", "kind": "electronic", "t_n_k": 50.0,
                             "tn_convention": "rj", "source": "s"})  # fmt: skip
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError, match="ideal kinds only"):
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
    ok = {"entries": [{"label": "a", "f_hz": 5e9, "s_e": 2.0, "field_convention": "peak",
                       "sideband": "dsb", "source": "s", "b_hz": 1e6}]}  # fmt: skip
    (tmp_path / "ok.yaml").write_text(yaml.safe_dump(ok))
    (e,) = load_literature(tmp_path / "ok.yaml")
    assert e.s_e_rms() == pytest.approx(2.0 / math.sqrt(2.0)) and e.b_hz == 1e6
    del ok["entries"][0]["field_convention"]
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump(ok))
    with pytest.raises(ValidationError):
        load_literature(tmp_path / "bad.yaml")


def test_shipped_literature_reproduces_tsd_table() -> None:
    """TSD §16.3: N_amp and T_equiv of the published points (peak read for unstated ones)."""
    ent = {e.label.split()[0]: e for e in load_literature(ROOT / "configs" / "literature_rar.yaml")}
    assert set(ent) == {"Jing", "Cai", "Borowka", "Yang", "Legaie", "She"}
    expect = {  # N_amp, T_equiv [K] from TSD §16.3
        "Jing": (1.9e6, 6.5e5), "Cai": (2.8e4, 1.4e4), "Borowka": (2.1e3, 1.4e3),
        "Yang": (2.0e5, 1.5e5), "Legaie": (1.5e3, 7.0e3), "She": (4.6e4, 7.5e5),
    }  # fmt: skip
    for name, (n_ref, t_ref) in expect.items():
        e = ent[name]
        r = RydbergReceiver.from_sensitivity(
            e.s_e, e.f_hz, field_convention=e.field_convention, sideband=e.sideband,
            includes_bbr=e.includes_bbr,
        )  # fmt: skip
        assert r.n_amp == pytest.approx(n_ref, rel=0.06), name
        assert r.n_amp * H * e.f_hz / K_B == pytest.approx(t_ref, rel=0.06), name
    assert ent["Borowka"].sideband == "ssb" and ent["Borowka"].includes_bbr
    assert {k for k, e in ent.items() if e.b_hz is not None} == {"Borowka", "Yang", "Legaie"}
    # BBR removal is a power subtraction
    b = ent["Borowka"]
    assert b.s_e_intrinsic_rms() < b.s_e_rms()


def test_rar_lna_tables_mask_and_interpolate() -> None:
    cfg, _ = load_config(ROOT / "configs" / "e5.yaml", E5Config)
    assert isinstance(cfg, E5Config)
    rar_t, lna_t = _tables(cfg)
    assert rar_t.f_range_hz == (13.9e9, 95.99e9)  # only entries with b_hz
    assert rar_t.receiver_at(10e9) is None and rar_t.receiver_at(200e9) is None
    assert lna_t.receiver_at(300e9) is None and lna_t.receiver_at(2e9) is None
    r = rar_t.receiver_at(13.9e9)
    assert r is not None and r.bandwidth_hz == pytest.approx(16e6) and r.sideband == "ssb"
    r2 = rar_t.receiver_at(50e9)
    assert r2 is not None and r2.sideband == "dsb"  # nearest entry in log f (Legaie)
    assert 127e3 < float(r2.bandwidth_hz or 0) < 10.2e6
    lna = lna_t.receiver_at(5e9)
    assert lna is not None and lna.sideband == "ssb" and lna.bandwidth_hz == pytest.approx(160e6)
    assert lna.n_amp == pytest.approx(238.0, rel=5e-3)  # TSD §16.4: 57 K at 5 GHz
    eq = lna_t.receiver_at(30e9, 16e6)
    assert eq is not None and eq.bandwidth_hz == 16e6


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
        _cfg(tmp_path, "e8", distance_m=[40.0], small_scale={"K": [10]},
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
    assert json.loads((out / "meta.json").read_text())["per_link"] is False


def test_e5_run_masks_outside_band_range(tmp_path: Path) -> None:
    out = e5.run(
        _cfg(tmp_path, "e5", n_s=[1e3], bands_ghz=[15, 60],
             throughput_map={**_E5_TM,
                             "f_ghz": {"start": 8, "stop": 150, "num": 4, "scale": "log"},
                             "distance_m": {"start": 20, "stop": 200, "num": 2, "scale": "log"},
                             "K": [10]}),
        tmp_path / "r",
    )  # fmt: skip
    rows = list(csv.DictReader((out / "map.csv").open()))
    for r in rows:
        f = float(r["f_ghz"])
        masked = math.isnan(float(r["rho_db"]))
        assert masked == (f < 13.9 - 1e-9 or f > 95.99 + 1e-9)
        assert masked == math.isnan(float(r["rho_equal_b_db"]))
    assert any(math.isnan(float(r["rho_db"])) for r in rows)
    live = [r for r in rows if not math.isnan(float(r["rho_db"]))]
    assert live
    # equal-bandwidth variant: B_LNA == B_RAR
    assert all(float(r["b_rar_hz"]) > 0 for r in live)
    cap = list(csv.DictReader((out / "capacity.csv").open()))
    assert {"Holevo", "ideal het", "ideal hom", "LNA", "RAR"} <= {r["receiver"] for r in cap}
    assert (out / "fig_rho_map_equal_b.png").exists()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["rar_f_range_hz"] == [13.9e9, 95.99e9] and meta["P_t_w"] == 0.2


def test_cli_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _cfg(tmp_path, "e6", f_ghz={"start": 1, "stop": 10, "num": 3, "scale": "log"})
    monkeypatch.setattr("sys.argv", ["x", "e6", str(p), "--out", str(tmp_path / "r")])
    main()
    assert list((tmp_path / "r").glob("e6_*/*/data.csv"))
