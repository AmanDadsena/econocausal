import numpy as np
import pytest
from econml.grf import CausalForest

from econocausal.forest import effect_modifiers, product_elasticities, summarise_by
from econocausal.simulate import simulate_panel
from test_dml import fast_model


def small_forest():
    return CausalForest(n_estimators=100, min_samples_leaf=50, random_state=0, n_jobs=2)


@pytest.fixture(scope="module")
def heterogeneous():
    return simulate_panel(n_products=240, n_weeks=104, seed=4, heterogeneous=True)


@pytest.fixture(scope="module")
def forest_result(heterogeneous):
    return product_elasticities(
        heterogeneous, forest=small_forest(), model_y=fast_model(), model_t=fast_model()
    )


def test_effect_modifiers_are_constant_within_each_product(heterogeneous):
    X = effect_modifiers(heterogeneous)
    assert X.groupby(heterogeneous["stock_code"]).nunique().max().max() == 1
    assert {"log_price_level", "guest_share_mean", "uk_share_mean"} <= set(X.columns)


def test_one_row_per_product_with_valid_intervals(heterogeneous, forest_result):
    products, importances = forest_result
    assert len(products) == heterogeneous["stock_code"].nunique()
    assert (products["ci_low"] < products["estimate"]).all()
    assert (products["estimate"] < products["ci_high"]).all()
    assert products["n_obs"].sum() == len(heterogeneous)
    assert importances.sum() == pytest.approx(1.0)


def test_recovers_category_elasticities(heterogeneous, forest_result):
    products, _ = forest_result
    truth = heterogeneous.groupby("stock_code")["true_elasticity"].first()
    products = products.assign(truth=products["stock_code"].map(truth))

    # With this small, fast forest the worst category error over seeds 1-6 was 0.10-0.24
    # and the correlation with the truth was always above 0.91.
    by_category = products.groupby("category")[["estimate", "truth"]].mean()
    assert (by_category["estimate"] - by_category["truth"]).abs().max() < 0.25
    assert np.corrcoef(products["estimate"], products["truth"])[0, 1] > 0.9


def test_category_is_the_most_important_modifier_on_synthetic_data(forest_result):
    # In the synthetic data only the category changes the elasticity.
    _, importances = forest_result
    assert importances.index[0].startswith("category_")


def test_summarise_by_category(forest_result):
    products, _ = forest_result
    summary = summarise_by(products, "category")
    assert summary["n_products"].sum() == len(products)
    assert summary["value"].iloc[0] == "christmas"  # most elastic first
    assert summary["estimate"].is_monotonic_increasing
