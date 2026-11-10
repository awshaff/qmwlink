"""M7: E9 on a small grid, the lindblad config schema and the engine benchmark."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from qmwlink import cli
from qmwlink.experiments import e9
from qmwlink.experiments.schema import E9Config
from qmwlink.receivers.rydberg_physics import bench
from qmwlink.receivers.rydberg_physics.model import GAMMA2_RB87_PER_S

ROOT = Path(__file__).resolve().parents[2]
TP = 2.0 * math.pi


def _data(**lind: Any) -> dict[str, Any]:
    cfg = yaml.safe_load((ROOT / "configs" / "e9.yaml").read_text())
    cfg["lindblad"]["doppler"] = {"n_sigma": 5, "n_v_start": 513, "rtol": 1e-3, "n_v_max": 65537}
    cfg["lindblad"]["sweep"] = {"omega_c_over_2pi_mhz": [5, 10], "e_lo_pk_v_per_m": [0.66, 1.3252]}
    cfg["lindblad"].update(lind)
    cfg["link"]["distance_m"] = {"start": 10, "stop": 1e5, "num": 9, "scale": "log"}
    cfg["crossover"]["f_ghz"] = {"start": 1, "stop": 300, "num": 8, "scale": "log"}
    cfg["crossover"]["literature_rar"] = str(ROOT / "configs" / "literature_rar.yaml")
    return cfg


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def e9_out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("e9")
    p = tmp / "e9.yaml"
    p.write_text(yaml.safe_dump(_data()))
    return e9.run(p, tmp / "out")


def test_e9_outputs_and_metadata(e9_out: Path) -> None:
    """FR-RX-6 AC: method used and max weak-probe/full discrepancy are in the run metadata."""
    for name in ("fig_linear_range", "fig_sweep", "fig_crossover_overlay"):
        assert (e9_out / f"{name}.pdf").stat().st_size > 0
        assert (e9_out / f"{name}.png").stat().st_size > 0
    meta = json.loads((e9_out / "meta.json").read_text())
    assert meta["method"] == "auto" and meta["engine_used"] == "weak_probe"
    assert 0.0 < meta["max_weak_probe_full_discrepancy"] < meta["weak_probe_tol"] == 0.01
    assert meta["recomputed_with_full"] is False and len(meta["checkpoints"]) == 4
    assert meta["sideband"] == "dsb" and meta["field_convention"] == "rms"
    assert meta["n_literature_points"] == 6
    assert meta["placeholders"] == ["coupling", "detector", "mw", "probe", "transit", "vapor"]
    base = meta["baseline"]
    assert base["e_1db_case"] == "compression"
    assert 0 < base["e_1db_pk_v_per_m"] < 1.3252
    assert base["omega_p_over_2pi_mhz"] == pytest.approx(0.0838, rel=1e-2)
    assert base["s_e_over_bbr"] > 1.0 and base["n_amp"] > 0.5


def test_e9_sweep_and_linear_range_rows(e9_out: Path) -> None:
    sweep = _rows(e9_out / "data.csv")
    assert len(sweep) == 4 and {r["engine"] for r in sweep} == {"weak_probe"}
    for r in sweep:
        assert float(r["s_e_rms"]) > 0 and float(r["n_amp"]) >= 0.5 and float(r["b_3db_hz"]) > 0
        # Omega_LO = d34 E_LO / hbar with the placeholder dipole: 20 MHz at 1.3252 V/m
        assert float(r["omega_lo_over_2pi_mhz"]) == pytest.approx(
            20.0 * float(r["e_lo_pk_v_per_m"]) / 1.3252, rel=1e-3
        )
    lin = _rows(e9_out / "linear_range.csv")
    fields = [float(r["e_rx_pk_v_per_m"]) for r in lin]
    assert fields == sorted(fields, reverse=True)  # received field falls with distance
    e_1db = float(lin[0]["e_1db_pk_v_per_m"])
    for r in lin:
        assert (r["linear_regime"] == "True") == (float(r["e_rx_pk_v_per_m"]) <= e_1db)
        assert float(r["p_rx_w"]) == pytest.approx(0.2 * float(r["eta"]))
    # 0.2 W into a 2 m^2 aperture at 10 m is 8.7 V/m peak: beyond the linear range
    assert lin[0]["linear_regime"] == "False" and lin[-1]["linear_regime"] == "True"
    assert fields[0] == pytest.approx(math.sqrt(2 * 376.730313 * 0.2 * float(lin[0]["eta"]) / 2.0))
    assert len(_rows(e9_out / "crossover.csv")) == 8 * 2 * 2


def test_e9_build_params_and_schema() -> None:
    cfg = E9Config.model_validate(_data())
    p = e9.build_params(cfg.lindblad)
    assert p.gamma2_per_s == pytest.approx(GAMMA2_RB87_PER_S)  # 2 -> 1 defaulted (Steck)
    assert p.gamma3_per_s == 6283.0 and p.gamma4_per_s == 6283.0
    assert p.decays_per_s == ((2, 1, GAMMA2_RB87_PER_S), (3, 2, 6283.0), (4, 1, 6283.0))
    assert p.omega_c_rad_s == pytest.approx(TP * 5e6) and p.eta_d == 0.9
    assert p.d12_cm == pytest.approx(2.5345e-29, rel=1e-4)  # cycling-transition dipole
    assert e9.doppler_settings(cfg.lindblad).n_v_start == 513
    explicit = _data(probe={"wavelength_m": 780.241e-9, "power_w": 1e-8, "waist_m": 1e-3,
                            "d12_cm": 1.0e-29})  # fmt: skip
    assert e9.build_params(E9Config.model_validate(explicit).lindblad).d12_cm == 1.0e-29

    def bad(mutate: Any, match: str) -> None:
        data = _data()
        mutate(data)
        with pytest.raises(ValidationError, match=match):
            E9Config.model_validate(data)

    bad(lambda d: d["lindblad"]["sources"].pop("mw"), "sources missing")
    bad(lambda d: d["lindblad"]["probe"].pop("power_w"), "probe needs power_w")
    bad(lambda d: d["lindblad"]["coupling"].pop("omega_c_over_2pi_mhz"), "coupling needs")
    bad(lambda d: d["lindblad"]["decays"][0].pop("source"), "source")
    bad(lambda d: d["lindblad"]["decays"][0].update(source="  "), "non-empty source")
    bad(lambda d: d["lindblad"].update(n_levels=3), "n_levels")
    bad(lambda d: d["lindblad"].update(method="fast"), "method")
    bad(lambda d: d["lindblad"].update(typo=1), "typo")
    bad(lambda d: d["lindblad"]["mw"].pop("f_hz"), "f_hz")
    beam = _data(coupling={"wavelength_m": 480e-9, "power_w": 0.02, "waist_m": 2e-3,
                           "d23_cm": 1e-31})  # fmt: skip
    assert e9.build_params(E9Config.model_validate(beam).lindblad).omega_c_rad_s is None


def test_e9_runs_without_literature_file_and_without_1db_point(tmp_path: Path) -> None:
    """FR-EXP-7 (b): a missing literature file only drops the overlay."""
    data = _data(method="weak_probe")
    data["lindblad"]["sweep"] = {"omega_c_over_2pi_mhz": [5], "e_lo_pk_v_per_m": [1.3252]}
    data["crossover"]["literature_rar"] = str(tmp_path / "missing.yaml")
    cfg = E9Config.model_validate(data)
    _, lit = e9.crossover_rows(cfg)
    assert lit == []
    _, sweep = e9.compute_sweep(cfg)
    assert sweep.method == "weak_probe" and math.isnan(sweep.max_discrepancy)
    from dataclasses import replace

    base = replace(sweep.results[0], e_1db_pk_v_per_m=float("nan"), e_1db_case="none")
    lin, rx = e9.compute_linear(cfg, base)
    assert rx.e_1db_pk_v_per_m is None and all(r.linear_regime for r in lin)
    assert e9._fig_linear(lin) is not None


# ---- benchmark (FR-LIN-2, FR-LIN-3: recorded, not gated) -----------------------------------------
def test_bench_writes_and_replaces_its_section(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    md = tmp_path / "RESULTS.md"
    md.write_text("# RESULTS\n\nintro\n")
    text = bench.run_bench(md)
    doc = md.read_text()
    assert doc.startswith("# RESULTS\n\nintro\n") and text in doc
    assert doc.count(bench.START) == 1 and "Weak-probe closed form" in doc and "ns |" in doc
    md.write_text(doc + "\ntrailing section\n")
    bench.run_bench(md)
    again = md.read_text()
    assert again.count(bench.START) == 1 and again.endswith("trailing section\n")
    assert bench.time_weak_probe(n_v=1000, repeats=2) > 0
    assert bench.time_full(n_v=100, repeats=1) > 0
    assert cli.main(["bench", "--no-write"]) == 0
    assert bench.START in capsys.readouterr().out
    fresh = tmp_path / "new.md"
    bench.run_bench(fresh)
    assert "Receiver engine benchmarks" in fresh.read_text()
