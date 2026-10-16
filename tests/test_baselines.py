import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from econocausal.baselines import SPECS, demean, fit_baseline, run_baselines
from econocausal.simulate import TRUE_ELASTICITY, simulate_panel


@pytest.fixture(scope="module")
def unbalanced_panel():
    """Small synthetic panel with ~30% of rows dropped, so it is unbalanced like the real data."""
    df = simulate_panel(n_products=40, n_weeks=30, seed=3)
    keep = np.random.default_rng(0).random(len(df)) > 0.3
    return df[keep].reset_index(drop=True)


def test_demean_one_group_removes_group_means():
    df = pd.DataFrame({"g": ["a", "a", "b", "b"], "x": [1.0, 3.0, 10.0, 14.0]})
    out = demean(df, ["x"], ["g"])
    assert out["x"].tolist() == [-1.0, 1.0, -2.0, 2.0]


def test_two_way_demeaning_matches_dummy_regression(unbalanced_panel):
    df = unbalanced_panel
    dummies = smf.ols(
        "log_quantity ~ log_price + C(stock_code) + C(week_start)", data=df
    ).fit().params["log_price"]
    result = fit_baseline(df, ["stock_code", "week_start"], [])
    assert result["estimate"] == pytest.approx(dummies, abs=1e-8)


def test_controls_match_dummy_regression(unbalanced_panel):
    df = unbalanced_panel
    dummies = smf.ols(
        "log_quantity ~ log_price + guest_share + uk_share + C(stock_code) + C(week_start)", data=df
    ).fit().params["log_price"]
    result = fit_baseline(df, ["stock_code", "week_start"], ["guest_share", "uk_share"])
    assert result["estimate"] == pytest.approx(dummies, abs=1e-8)


def test_pooled_matches_simple_regression(unbalanced_panel):
    df = unbalanced_panel
    slope = np.polyfit(df["log_price"], df["log_quantity"], 1)[0]
    assert fit_baseline(df, [], [])["estimate"] == pytest.approx(slope)


def test_confidence_interval_contains_estimate(unbalanced_panel):
    r = fit_baseline(unbalanced_panel, ["stock_code"], [])
    assert r["ci_low"] < r["estimate"] < r["ci_high"]
    assert r["std_error"] > 0


def test_run_baselines_output():
    df = simulate_panel(n_products=60, n_weeks=52)
    results = run_baselines(df, "synthetic", true_value=TRUE_ELASTICITY)
    assert results["method"].tolist() == list(SPECS)
    assert (results["dataset"] == "synthetic").all()
    assert np.allclose(results["error"], results["estimate"] - TRUE_ELASTICITY)


def test_baselines_are_biased_on_synthetic_data():
    """Every naive baseline misses the true elasticity, which is what DML has to fix."""
    results = run_baselines(simulate_panel(), "synthetic", true_value=TRUE_ELASTICITY)
    assert (results["error"].abs() > 0.1).all()
