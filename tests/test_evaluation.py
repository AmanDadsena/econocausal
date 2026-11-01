import pandas as pd
import pytest

from econocausal.evaluation import run_one, summarise
from econocausal.simulate import TRUE_ELASTICITY


def test_summarise_known_values():
    runs = pd.DataFrame({
        "method": ["m"] * 4,
        "estimate": [-1.7, -1.9, -1.8, -1.8],
        "std_error": [0.1, 0.1, 0.1, 0.1],
        "covers": [True, True, True, False],
    })
    s = summarise(runs).iloc[0]
    assert s["coverage"] == 0.75
    assert s["bias"] == pytest.approx(0.0)
    assert s["rmse"] == pytest.approx((0.02 / 4) ** 0.5)  # errors +0.1, -0.1, 0, 0
    assert s["coverage_ci_low"] < 0.75 < s["coverage_ci_high"]
    assert s["sd_estimates"] == pytest.approx(pd.Series([-1.7, -1.9, -1.8, -1.8]).std())


def test_run_one_fits_both_methods():
    rows = run_one(seed=0)
    assert [r["method"] for r in rows] == ["dml_manual", "ols_product_week_fe_controls"]
    for r in rows:
        assert r["ci_low"] < r["estimate"] < r["ci_high"]
        assert r["covers"] == (r["ci_low"] <= TRUE_ELASTICITY <= r["ci_high"])
