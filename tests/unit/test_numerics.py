"""checked_quad, rejection_sample, MC-agree rule. (TSD §6.4, §10.1, §13)"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from scipy import stats

from qmwlink.numerics.integrate import checked_quad
from qmwlink.numerics.mcagree import (
    bonferroni_z,
    check_moment,
    check_probabilities,
    ecdf_estimate,
    ks_agree,
)
from qmwlink.numerics.sampling import rejection_sample


def test_checked_quad_value_and_breakpoints() -> None:
    val, err = checked_quad(lambda x: np.exp(-x), 0.0, 5.0, points=[1.0, 2.0, 99.0, np.nan])
    assert val == pytest.approx(1 - np.exp(-5.0), rel=1e-12)
    assert err < 1e-8
    assert checked_quad(lambda x: 1.0, 2.0, 2.0) == (0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        checked_quad(lambda x: 1.0, 0.0, np.inf)


def test_checked_quad_falls_back_to_mpmath_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    # limit=1 (a single Gauss-Kronrod panel) cannot reach 1e-8 on this integrand -> fallback
    with caplog.at_level(logging.WARNING):
        val, _ = checked_quad(lambda x: float(np.cos(40.0 * x)), 0.0, 3.0, limit=1)
    assert "falling back to mpmath" in caplog.text
    assert val == pytest.approx(np.sin(120.0) / 40.0, abs=1e-9)


def test_bonferroni_z_values() -> None:
    assert bonferroni_z(100) == pytest.approx(4.417, abs=2e-3)  # spec: ~4.4
    assert bonferroni_z(1) == pytest.approx(stats.norm.isf(5e-4))
    with pytest.raises(ValueError, match="m must"):
        bonferroni_z(0)


def test_ecdf_estimate_and_probability_check() -> None:
    rng = np.random.default_rng(0)
    s = rng.uniform(size=200_000)
    xs = np.linspace(0.05, 0.95, 50)
    p_hat, se = ecdf_estimate(s, xs)
    assert np.all(se > 0)
    assert check_probabilities(p_hat, se, xs).ok
    bad = check_probabilities(p_hat, se, xs + 0.01)
    assert not bad.ok and bad.max_z > bad.z_crit
    # se stays positive at the extremes
    _, se_edge = ecdf_estimate(s, [-1.0, 2.0])
    assert np.all(se_edge > 0)


def test_probability_check_errors() -> None:
    with pytest.raises(ValueError, match="same shape"):
        check_probabilities([0.5], [0.1, 0.1], [0.5])
    with pytest.raises(ValueError, match="> 0"):
        check_probabilities([0.5], [0.0], [0.5])


def test_check_moment_and_ks() -> None:
    rng = np.random.default_rng(1)
    s = rng.normal(2.0, 1.0, 500_000)
    assert check_moment(s, 2.0).ok
    assert not check_moment(s, 2.01).ok
    with pytest.raises(ValueError, match="zero variance"):
        check_moment(np.ones(10), 1.0)
    ok, d, p = ks_agree(s, stats.norm(2.0, 1.0).cdf)
    assert ok and d < 5e-3 and p >= 1e-3
    assert not ks_agree(s, stats.norm(2.1, 1.0).cdf)[0]


def test_rejection_sample_exact_size_and_acceptance() -> None:
    rng = np.random.default_rng(2)
    out = rejection_sample(lambda n: rng.uniform(size=n), lambda x: x < 0.25, 1000, 0.25)
    assert out.shape == (1000,) and np.all(out < 0.25)
    rows = rejection_sample(lambda n: rng.uniform(size=(n, 3)), lambda x: x.sum(1) < 1, 500, 0.17)
    assert rows.shape == (500, 3)
    assert rejection_sample(lambda n: rng.uniform(size=n), lambda x: x < 1, 0, 1.0).shape == (0,)
    with pytest.raises(ValueError, match="size"):
        rejection_sample(lambda n: rng.uniform(size=n), lambda x: x < 1, -1, 1.0)


def test_rejection_sample_hopeless_acceptance_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import qmwlink.numerics.sampling as smp

    monkeypatch.setattr(smp, "MAX_ROUNDS", 3)
    rng = np.random.default_rng(3)
    with pytest.raises(RuntimeError, match="acceptance rate"):
        smp.rejection_sample(lambda n: rng.uniform(size=n), lambda x: x > 2.0, 10, 0.5)


def test_checked_quad_nonfinite_triggers_fallback(caplog: pytest.LogCaptureFixture) -> None:
    calls = {"n": 0}

    def f(x: float) -> float:
        calls["n"] += 1
        return float("nan") if calls["n"] <= 21 else 1.0  # the scipy pass sees nan, mpmath not

    with caplog.at_level(logging.WARNING):
        val, _ = checked_quad(f, 0.0, 1.0)
    assert "falling back" in caplog.text and np.isfinite(val)
