import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from lightgbm import LGBMRegressor

from econocausal.baselines import run_baselines
from econocausal.dml import build_features, estimate_by_segment, estimate_elasticity, final_stage
from econocausal.simulate import TRUE_ELASTICITY, simulate_panel


def fast_model():
    """Lighter LightGBM settings so the tests run quickly."""
    return LGBMRegressor(
        n_estimators=150, learning_rate=0.1, num_leaves=15, min_child_samples=40, verbose=-1
    )


def fit(df, **kwargs):
    return estimate_elasticity(df, model_y=fast_model(), model_t=fast_model(), **kwargs)


@pytest.fixture(scope="module")
def synthetic():
    return simulate_panel(n_products=150, n_weeks=104, seed=1)


@pytest.fixture(scope="module")
def dml_result(synthetic):
    return fit(synthetic)


# --- final stage -------------------------------------------------------------

def test_final_stage_exact_on_noise_free_data():
    t = np.array([1.0, -2.0, 0.5, 3.0, -1.5, 0.2])
    theta, se, low, high = final_stage(-1.8 * t, t, np.array([0, 0, 1, 1, 2, 2]))
    assert theta == pytest.approx(-1.8)
    assert se == pytest.approx(0.0, abs=1e-12)


def test_final_stage_standard_error_matches_statsmodels():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(200), 20)
    t = rng.normal(size=len(groups))
    y = -1.5 * t + rng.normal(size=len(groups)) + np.repeat(rng.normal(size=200), 20) * t
    theta, se, _, _ = final_stage(y, t, groups)

    ols = sm.OLS(y, t).fit(cov_type="cluster", cov_kwds={"groups": groups})
    assert theta == pytest.approx(ols.params[0])
    # statsmodels also scales by (N-1)/(N-K), which is ~1 here.
    assert se == pytest.approx(ols.bse[0], rel=0.01)


# --- features ----------------------------------------------------------------

def test_build_features_adds_product_means_and_skips_missing_columns(synthetic):
    X = build_features(synthetic)
    assert "has_bank_holiday" not in X  # not in the synthetic data
    assert {"guest_share_product_mean", "uk_share_product_mean"} <= set(X.columns)
    assert str(X["category"].dtype) == "category"
    assert "n_invoices" not in X and "log_price" not in X


# --- estimator on synthetic data -------------------------------------------------

def test_dml_recovers_true_elasticity(dml_result):
    assert dml_result.estimate == pytest.approx(TRUE_ELASTICITY, abs=0.1)
    assert dml_result.ci_low < TRUE_ELASTICITY < dml_result.ci_high


def test_dml_beats_every_naive_baseline(synthetic, dml_result):
    baselines = run_baselines(synthetic, "synthetic", true_value=TRUE_ELASTICITY)
    dml_error = abs(dml_result.estimate - TRUE_ELASTICITY)
    assert (baselines["error"].abs() > dml_error).all()


def test_dml_result_fields(synthetic, dml_result):
    assert dml_result.n_obs == len(synthetic)
    assert dml_result.n_groups == synthetic["stock_code"].nunique()
    assert dml_result.std_error > 0
    assert 0 < dml_result.outcome_r2 < 1
    assert 0 < dml_result.treatment_r2 < 1


def test_dml_is_reproducible_with_same_seed(synthetic):
    small = synthetic[synthetic["stock_code"].isin(synthetic["stock_code"].unique()[:40])]
    assert fit(small, seed=3).estimate == fit(small, seed=3).estimate


def test_dml_cannot_fix_hidden_confounding():
    df = simulate_panel(n_products=150, n_weeks=104, seed=1, hidden_confounding=1.0)
    result = fit(df)
    assert not (result.ci_low < TRUE_ELASTICITY < result.ci_high)
    assert abs(result.estimate - TRUE_ELASTICITY) > 0.5


def test_estimate_by_segment_recovers_category_elasticities():
    df = simulate_panel(n_products=300, n_weeks=104, seed=2, heterogeneous=True)
    by_category = estimate_by_segment(
        df, "category", model_y=fast_model(), model_t=fast_model()
    ).set_index("value")
    truth = df.groupby("category")["true_elasticity"].first()

    assert set(by_category.index) == set(truth.index)
    truth = truth.reindex(by_category.index)
    assert (by_category["estimate"] - truth).abs().max() < 0.2
    # Most intervals should contain the truth (with 95% CIs, 6 categories).
    covered = (by_category["ci_low"] < truth) & (truth < by_category["ci_high"])
    assert covered.sum() >= 5
    # The ordering of the most and least elastic categories is recovered.
    assert by_category["estimate"].idxmin() == "christmas"
    assert by_category["estimate"].idxmax() == "stationery_cards"
    assert by_category["n_obs"].sum() == len(df)


def test_works_on_unbalanced_panel(synthetic):
    keep = np.random.default_rng(0).random(len(synthetic)) > 0.3
    result = fit(synthetic[keep])
    assert result.estimate == pytest.approx(TRUE_ELASTICITY, abs=0.15)
