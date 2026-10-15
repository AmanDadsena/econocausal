import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from econocausal.simulate import (
    CATEGORY_ELASTICITY_SHIFT,
    TRUE_ELASTICITY,
    guest_demand_effect,
    guest_price_premium,
    simulate_panel,
)

SMALL = dict(n_products=120, n_weeks=104)


@pytest.fixture(scope="module")
def panel():
    return simulate_panel(**SMALL)


def within(df, col):
    return df[col] - df.groupby("stock_code")[col].transform("mean")


def oracle_slope(df):
    """Within-product regression with week effects and the exact guest functions.
    This is the 'correct' adjustment, so it should recover the true elasticity."""
    d = pd.DataFrame(
        {
            "q": within(df, "log_quantity"),
            "p": within(df, "log_price"),
            "gp": guest_price_premium(df["guest_share"]),
            "gq": guest_demand_effect(df["guest_share"]),
            "uk": df["uk_share"],
            "week": df["week_start"].astype(str),
            "stock_code": df["stock_code"],
        }
    )
    d["gp"] = within(d, "gp")
    d["gq"] = within(d, "gq")
    return smf.ols("q ~ p + C(week) + gp + gq + uk", data=d).fit().params["p"]


def test_same_seed_gives_same_data():
    pd.testing.assert_frame_equal(simulate_panel(seed=7, **SMALL), simulate_panel(seed=7, **SMALL))


def test_different_seed_gives_different_data():
    a = simulate_panel(seed=1, **SMALL)
    b = simulate_panel(seed=2, **SMALL)
    assert not np.allclose(a["log_price"], b["log_price"])


def test_shape_and_columns(panel):
    assert len(panel) == SMALL["n_products"] * SMALL["n_weeks"]
    assert panel["stock_code"].nunique() == SMALL["n_products"]
    expected = {
        "stock_code", "week_start", "category", "price", "quantity", "log_price",
        "log_quantity", "year", "month", "week_of_year", "weeks_to_christmas",
        "is_peak_season", "guest_share", "uk_share", "true_elasticity",
    }
    assert expected <= set(panel.columns)
    assert not panel.isna().any().any()


def test_values_are_consistent(panel):
    assert np.allclose(np.log(panel["price"]), panel["log_price"])
    assert np.allclose(np.log(panel["quantity"]), panel["log_quantity"])
    assert panel["guest_share"].between(0, 1).all()
    assert set(panel["category"]) <= set(CATEGORY_ELASTICITY_SHIFT)


def test_customer_mix_resembles_real_data(panel):
    # Real data: 44% of product-weeks have no guests, 13% only guests.
    assert 0.35 < (panel["guest_share"] == 0).mean() < 0.55
    assert 0.08 < (panel["guest_share"] == 1).mean() < 0.18


def test_constant_elasticity_by_default(panel):
    assert np.allclose(panel["true_elasticity"], TRUE_ELASTICITY)
    assert panel.attrs["true_elasticity"] == pytest.approx(TRUE_ELASTICITY)


def test_heterogeneous_elasticity_keeps_the_average():
    df = simulate_panel(heterogeneous=True, **SMALL)
    per_category = df.groupby("category")["true_elasticity"].first()
    assert per_category.nunique() == len(CATEGORY_ELASTICITY_SHIFT)
    # Christmas items are the most price sensitive, stationery the least.
    assert per_category.idxmin() == "christmas"
    assert per_category.idxmax() == "stationery_cards"
    product_mean = df.groupby("stock_code")["true_elasticity"].first().mean()
    assert product_mean == pytest.approx(TRUE_ELASTICITY)


def test_naive_regression_is_biased(panel):
    pooled = smf.ols("log_quantity ~ log_price", data=panel).fit().params["log_price"]
    assert abs(pooled - TRUE_ELASTICITY) > 0.4


def test_correct_adjustment_recovers_true_elasticity(panel):
    # Over seeds 0-9 this panel size gives estimates between -1.74 and -1.92,
    # so the tolerance allows for sampling noise.
    assert oracle_slope(panel) == pytest.approx(TRUE_ELASTICITY, abs=0.15)


def test_hidden_confounding_breaks_recovery():
    df = simulate_panel(hidden_confounding=1.0, **SMALL)
    assert abs(oracle_slope(df) - TRUE_ELASTICITY) > 0.5
