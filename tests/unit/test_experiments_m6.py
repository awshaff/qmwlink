"""M6: E1 (RQ1), E2 (RQ2) and the obstacle generator (FR-CH-4) on small grids."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from pydantic import ValidationError
from scipy import stats

import tolerances as tol
from qmwlink.channel.large_scale import LogNormalV0, ShiftedLogGamma, matched_laws
from qmwlink.channel.obstacles import ObstacleModel, diagnose
from qmwlink.channel.small_scale import SmallScale
from qmwlink.experiments import e1, e2
from qmwlink.experiments.schema import E1Config, E2Config, LawComparison
from qmwlink.numerics.integrate import checked_quad
from qmwlink.numerics.mcagree import check_moment

ROOT = Path(__file__).resolve().parents[2]


def _cfg(name: str, tmp_path: Path, **over: Any) -> Path:
    cfg = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
    for key, val in over.items():
        node = cfg
        *parents, leaf = key.split("__")
        for p in parents:
            node = node[p]
        node[leaf] = val
    if "rar" in cfg:
        cfg["rar"]["literature"] = str(ROOT / "configs" / "literature_rar.yaml")
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


# ---- E1 ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def e1_out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("e1")
    cfg = _cfg(
        "e1", tmp, K=[0, 10],
        eta_s_bar={"start": 0.9, "stop": 1.0, "num": 3, "scale": "lin"},
        pdf_eta={"start": 0.0, "stop": 1.2, "num": 25, "scale": "lin"},
        compound__bands_ghz=[100],
        compound__distance_m={"start": 30, "stop": 300, "num": 2, "scale": "log"},
    )  # fmt: skip
    return e1.run(cfg, tmp / "out")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def test_e1_outputs(e1_out: Path) -> None:
    for name in ("fig_pdf", "fig_removed_mass", "fig_compound"):
        assert (e1_out / f"{name}.pdf").stat().st_size > 0
        assert (e1_out / f"{name}.png").stat().st_size > 0
    meta = json.loads((e1_out / "meta.json").read_text())
    assert meta["surrogate_physical"] is False and meta["per_link"] is False
    assert len(_rows(e1_out / "data.csv")) == 6 and len(_rows(e1_out / "compound.csv")) == 4


def test_e1_deterministic_identity_and_reference(e1_out: Path) -> None:
    """FR-SVC-E2 (e): P_out/P_out^surr = 1/Z for eta_l = 1; removed mass = 1 - Z."""
    for r in _rows(e1_out / "data.csv"):
        z = float(r["z"])
        assert float(r["ratio"]) == pytest.approx(1.0 / z, rel=tol.RTOL_RQ1_IDENTITY)
        assert float(r["removed_mass"]) == pytest.approx(1.0 - z, abs=1e-15)
        assert float(r["delta_p_out"]) == pytest.approx(
            float(r["p_out"]) - float(r["p_out_surr"]), abs=1e-15
        )
        assert 0.0 < float(r["p_out_surr"]) < float(r["p_out"]) <= 1.0


def test_e1_pdf_rows_truncation(e1_out: Path) -> None:
    """Truncated pdf is the surrogate / Z inside [0, 1] and 0 beyond; surrogate keeps its tail."""
    rows = _rows(e1_out / "pdf.csv")
    for k in (0.0, 10.0):
        z = SmallScale(k, 0.999).z
        sel = [r for r in rows if float(r["K"]) == k]
        for r in sel:
            eta, t, s = float(r["eta"]), float(r["pdf_truncated"]), float(r["pdf_surrogate"])
            if eta <= 1.0:
                assert t == pytest.approx(s / z, rel=1e-12)
            else:
                assert t == 0.0 and s > 0.0


def test_e1_compound_bounds(e1_out: Path) -> None:
    for r in _rows(e1_out / "compound.csv"):
        assert 0.0 <= float(r["p_out_surr"]) <= float(r["p_out"]) <= 1.0
        assert float(r["ratio"]) >= 1.0 - 1e-12
        assert float(r["ratio"]) <= 1.0 / (1.0 - float(r["removed_mass"])) + 1e-9


def test_e1_schema(tmp_path: Path) -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "e1.yaml").read_text())
    E1Config.model_validate(cfg)
    cfg["compound"]["bands_ghz"] = [7]
    with pytest.raises(ValidationError, match="no gamma_abs entry"):
        E1Config.model_validate(cfg)


# ---- E2 ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def e2_cfg(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("e2")
    return _cfg(
        "e2", tmp, law_comparison__sigma_ratio=[1e-3, 1.0],
        pdf_eta={"start": 0.99, "stop": 1.004, "num": 15, "scale": "lin"},
    )  # fmt: skip


@pytest.fixture(scope="module")
def e2_out(e2_cfg: Path) -> Path:
    return e2.run(e2_cfg, e2_cfg.parent / "out")


def test_e2_outputs_and_flags(e2_out: Path) -> None:
    """FR-CH-3 (a),(c): mass above 1 = Phi(-1); V0 rows carry physical=False and clipped mass."""
    rows = _rows(e2_out / "data.csv")
    assert len(rows) == 6
    for name in ("fig_law_comparison", "fig_pdf"):
        assert (e2_out / f"{name}.pdf").stat().st_size > 0
    v0 = next(r for r in rows if r["law"] == "lognormal_v0" and float(r["sigma_ratio"]) == 1.0)
    assert v0["physical"] == "False"
    assert float(v0["mass_eta_l_above_one"]) == pytest.approx(
        stats.norm.cdf(-1.0), abs=tol.ATOL_MASS_ABOVE_ONE
    )
    assert 0.0 < float(v0["clipped_mass"]) < float(v0["mass_eta_l_above_one"])
    for r in rows:
        if r["law"] != "lognormal_v0":
            assert r["physical"] == "True"
            assert float(r["clipped_mass"]) == 0.0 and float(r["mass_eta_l_above_one"]) == 0.0
        if r["law"] == "loggamma":
            assert float(r["delta_p_out"]) == 0.0 and float(r["rel_delta_c_het"]) == 0.0
        assert float(r["c_holevo"]) >= float(r["c_het"]) >= float(r["c_rx"]) > 0.0
    meta = json.loads((e2_out / "meta.json").read_text())
    assert meta["lognormal_v0_physical"] is False and meta["eta_gt1"] == "clip"
    assert meta["max_clipped_mass"] == pytest.approx(float(v0["clipped_mass"]))
    assert meta["sourced_sigma_ratio"] == pytest.approx(6.760e-5 / (4.394e-6 + 2.839e-5))


def test_e2_laws_agree_as_sigma_to_zero(e2_out: Path) -> None:
    """FR-CH-3 (b): sigma_Psi -> 0 gives the same P_out and ergodic capacity (rtol 1e-6)."""
    rows = [r for r in _rows(e2_out / "data.csv") if float(r["sigma_ratio"]) == 1e-3]
    ref = next(r for r in rows if r["law"] == "loggamma")
    for r in rows:
        for col in ("p_out", "c_holevo", "c_het", "c_rx"):
            assert float(r[col]) == pytest.approx(float(ref[col]), rel=tol.RTOL_LAW_LIMIT)


def test_e2_clipped_mass_matches_mc_and_is_zero_for_physical_laws() -> None:
    small = SmallScale(10.0, 0.999)
    lg, v0, cond = matched_laws(1e-4, 2e-3, 2.1e-3)
    assert isinstance(lg, ShiftedLogGamma) and isinstance(v0, LogNormalV0)
    assert e2.clipped_mass(small, lg) == 0.0 and e2.clipped_mass(small, cond) == 0.0
    got = e2.clipped_mass(small, v0)
    rng = np.random.default_rng(tol.MC_SEED)
    eta = v0.sample_eta(rng, tol.MC_N) * small.sample_sum(rng, tol.MC_N)
    assert check_moment((eta > 1.0).astype(float), got).ok
    assert 0.0 < got < v0.mass_above_one()


def test_e2_clip_only_touches_eta_above_one() -> None:
    cap = e2._clip(lambda e: 2.0 * e)
    np.testing.assert_array_equal(cap(np.array([0.2, 1.0, 1.7])), [0.4, 2.0, 2.0])


def test_e2_schema_and_errors(e2_cfg: Path, tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="sigma_ratio"):
        LawComparison(sigma_ratio=[0.1, 0.0])
    with pytest.raises(ValidationError):
        LawComparison(sigma_ratio=[0.1], eta_gt1="drop")  # type: ignore[arg-type]
    data = yaml.safe_load(e2_cfg.read_text())
    data["band_ghz"] = 5  # below the RAR literature range with b_hz
    with pytest.raises(ValueError, match="no RAR input"):
        e2.compute(E2Config.model_validate(data))
    data = yaml.safe_load(e2_cfg.read_text())
    data["large_scale"]["stochastic_per_m"][30]["mean"] = 0.0
    with pytest.raises(ValueError, match="positive stochastic mean"):
        e2.compute(E2Config.model_validate(data))


# ---- FR-CH-4 obstacle generator -----------------------------------------------------------------
def test_obstacle_moment_match_is_exact() -> None:
    m = ObstacleModel(rate_per_m=0.02, k_o=2.5, theta_o=0.03, d_m=150.0)
    assert m.mean_count == pytest.approx(3.0)
    assert m.mu_g == pytest.approx(3.0 * 2.5 * 0.03, rel=1e-15)
    assert m.var_g == pytest.approx(3.0 * 2.5 * 3.5 * 0.03**2, rel=1e-15)
    law = m.matched_loggamma(psi0=1e-3)
    assert law.psi0 == 1e-3
    assert law.mu_g == pytest.approx(m.mu_g, rel=1e-14)
    assert law.sigma_g**2 == pytest.approx(m.var_g, rel=1e-14)
    assert m.atom_at_zero == pytest.approx(np.exp(-3.0))


def test_obstacle_sampler_moments_and_atom() -> None:
    m = ObstacleModel(rate_per_m=0.02, k_o=2.5, theta_o=0.03, d_m=150.0)
    g = m.sample_g(np.random.default_rng(tol.MC_SEED), tol.MC_N)
    assert np.all(g >= 0.0)
    assert check_moment(g, m.mu_g).ok
    assert check_moment((g - m.mu_g) ** 2, m.var_g).ok
    assert check_moment((g == 0.0).astype(float), m.atom_at_zero).ok
    assert m.sample_g(np.random.default_rng(0), 0).shape == (0,)
    with pytest.raises(ValueError, match="size"):
        m.sample_g(np.random.default_rng(0), -1)


def test_obstacle_diagnostic_reports_without_threshold() -> None:
    """FR-CH-4 AC: histogram vs fitted Gamma is reported (diagnostic only)."""
    m = ObstacleModel(rate_per_m=0.05, k_o=2.0, theta_o=0.02, d_m=100.0)
    d = diagnose(m, np.random.default_rng(tol.MC_SEED), size=200_000, bins=40)
    assert d.bin_centers.shape == d.hist_density.shape == d.gamma_density.shape == (40,)
    assert np.all(np.isfinite(d.gamma_density)) and np.all(d.hist_density >= 0)
    assert d.sample_fraction_zero == pytest.approx(d.atom_at_zero, abs=5e-3)
    assert 0.0 < d.max_abs_cdf_diff < 1.0
    area = checked_quad(lambda x: float(m.matched_loggamma().psi_pdf(x)), 0.0, 5.0)[0]
    assert area == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("field", ["rate_per_m", "k_o", "theta_o", "d_m"])
def test_obstacle_validation(field: str) -> None:
    kw = {"rate_per_m": 0.1, "k_o": 1.0, "theta_o": 0.1, "d_m": 10.0}
    kw[field] = 0.0
    with pytest.raises(ValueError, match=field):
        ObstacleModel(**kw)
