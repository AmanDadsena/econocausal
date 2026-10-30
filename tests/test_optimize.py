import math

import numpy as np
import pandas as pd
import pytest

from econocausal.optimize import (
    PricingRules,
    best_price,
    demand,
    grid_optimum,
    optimize_products,
    pricing_inputs,
    profit,
    recommend_price,
    summarise,
    unconstrained_optimum,
)


def test_demand_equals_base_quantity_at_base_price():
    assert demand(2.0, 2.0, 100, -1.5) == pytest.approx(100)


def test_demand_responds_with_the_elasticity():
    # A 1% price rise changes units by about θ%.
    assert demand(2.0 * 1.01, 2.0, 100, -1.5) == pytest.approx(100 * 1.01**-1.5)
    assert demand(4.0, 2.0, 100, -2.0) == pytest.approx(25)


def test_profit_is_margin_times_units():
    assert profit(3.0, 2.0, 100, -1.0, 1.0) == pytest.approx((3 - 1) * 100 * 1.5**-1)


@pytest.mark.parametrize("elasticity", [-1.2, -1.5, -2.0, -3.5])
@pytest.mark.parametrize("unit_cost", [0.5, 1.2])
def test_lerner_price_matches_brute_force(elasticity, unit_cost):
    expected = unconstrained_optimum(elasticity, unit_cost)
    found = grid_optimum(2.0, 100, elasticity, unit_cost, low=unit_cost * 1.001, high=expected * 3)
    assert found == pytest.approx(expected, rel=1e-3)


def test_lerner_formula_values():
    assert unconstrained_optimum(-2.0, 1.0) == pytest.approx(2.0)   # 100% markup on cost
    assert unconstrained_optimum(-3.0, 1.0) == pytest.approx(1.5)   # 50% markup


def test_inelastic_demand_has_no_finite_optimum():
    assert math.isinf(unconstrained_optimum(-0.8, 1.0))
    assert math.isinf(unconstrained_optimum(-1.0, 1.0))
    # Profit keeps rising with price.
    prices = np.array([2.0, 4.0, 8.0])
    assert np.all(np.diff(profit(prices, 2.0, 100, -0.8, 1.0)) > 0)


# --- business rules ----------------------------------------------------------------

def test_rules_validate_their_inputs():
    with pytest.raises(ValueError):
        PricingRules(max_change=0)
    with pytest.raises(ValueError):
        PricingRules(min_margin=1.2)
    with pytest.raises(ValueError):
        PricingRules(objective="sales")


@pytest.mark.parametrize("elasticity", [-0.6, -1.1, -1.3, -2.0, -2.5, -4.0, -8.0])
@pytest.mark.parametrize("cost_ratio", [0.3, 0.6, 0.75])
def test_best_price_matches_brute_force_inside_the_allowed_range(elasticity, cost_ratio):
    rules = PricingRules(cost_ratio=cost_ratio)
    p0, cost = 2.0, cost_ratio * 2.0
    price, _ = best_price(p0, elasticity, cost, rules)
    low = max(p0 * 0.8, cost / (1 - rules.min_margin))
    expected = grid_optimum(p0, 100, elasticity, cost, low=low, high=p0 * 1.2)
    assert price == pytest.approx(expected, abs=1e-4)


def test_inelastic_demand_hits_the_upper_limit():
    price, reason = best_price(2.0, -0.8, 1.2, PricingRules())
    assert price == pytest.approx(2.4)
    assert reason == "upper price limit"


def test_very_elastic_demand_hits_the_lower_limit():
    # c·θ/(1+θ) = 0.6·(−8)/(−7) ≈ 0.686, below the 20% cut floor of 0.8.
    price, reason = best_price(1.0, -8.0, 0.6, PricingRules())
    assert price == pytest.approx(0.8)
    assert reason == "lower price limit"


def test_optimum_inside_the_limits_is_kept():
    # c·θ/(1+θ) = 0.6·(−3)/(−2) = 0.9, inside [0.8, 1.2].
    price, reason = best_price(1.0, -3.0, 0.6, PricingRules())
    assert price == pytest.approx(0.9)
    assert reason == "optimal within limits"


def test_minimum_margin_can_set_the_floor():
    # Cost 0.7 with a 20% minimum margin needs a price of at least 0.875 (> 0.8).
    price, reason = best_price(1.0, -8.0, 0.7, PricingRules(cost_ratio=0.7))
    assert price == pytest.approx(0.875)
    assert reason == "minimum margin"
    assert (price - 0.7) / price == pytest.approx(0.2)


def test_infeasible_when_margin_needs_a_price_above_the_range():
    rules = PricingRules(cost_ratio=0.98, min_margin=0.3)
    rec = recommend_price(1.0, 50, -2.0, rules)
    assert rec["recommended_price"] is None
    assert rec["reason"].startswith("infeasible")


def test_revenue_objective_cuts_price_for_elastic_demand():
    price, _ = best_price(1.0, -2.0, 0.6, PricingRules(objective="revenue"))
    assert price == pytest.approx(0.8)


def test_recommendation_numbers_are_consistent():
    rec = recommend_price(2.0, 100, -1.3, PricingRules(), elasticity_low=-1.6, elasticity_high=-1.0)
    assert rec["recommended_price"] == pytest.approx(2.4)
    assert rec["price_change_pct"] == pytest.approx(20)
    assert rec["expected_units"] == pytest.approx(100 * 1.2**-1.3)
    assert rec["expected_revenue"] == pytest.approx(2.4 * rec["expected_units"])
    assert rec["expected_profit"] == pytest.approx((2.4 - 1.2) * rec["expected_units"])
    # Higher price, fewer units, lower revenue, higher profit.
    assert rec["units_change_pct"] < 0 and rec["revenue_change_pct"] < 0 < rec["profit_change_pct"]
    assert rec["price_at_ci_low"] == pytest.approx(2.4)
    assert rec["price_at_ci_high"] == pytest.approx(2.4)


def test_negative_inputs_are_rejected():
    with pytest.raises(ValueError):
        recommend_price(0, 10, -1.5)


# --- whole catalogue --------------------------------------------------------------

def make_panel():
    weeks = pd.date_range("2011-01-03", periods=20, freq="7D")
    rows = [{"stock_code": "A", "week_start": w, "price": 2.0, "quantity": 10} for w in weeks]
    rows += [{"stock_code": "B", "week_start": w, "price": 5.0, "quantity": 4} for w in weeks[::2]]
    rows += [{"stock_code": "OLD", "week_start": weeks[0], "price": 1.0, "quantity": 99}]
    return pd.DataFrame(rows)


def test_pricing_inputs_use_the_recent_window():
    inputs = pricing_inputs(make_panel(), recent_weeks=13).set_index("stock_code")
    assert "OLD" not in inputs.index  # no sales in the last 13 weeks
    assert inputs.loc["A", "current_price"] == 2.0
    assert inputs.loc["A", "current_units"] == pytest.approx(10)
    # B sold every other week: zero weeks count, so the weekly average is lower.
    assert inputs.loc["B", "current_units"] == pytest.approx(4 * inputs.loc["B", "weeks_sold"] / 13)


def test_pricing_inputs_for_one_product_use_the_shared_window():
    panel = make_panel()
    last_week = panel["week_start"].max()
    alone = pricing_inputs(panel[panel["stock_code"] == "B"], last_week=last_week)
    together = pricing_inputs(panel).set_index("stock_code").loc["B"]
    assert alone["current_units"].iloc[0] == pytest.approx(together["current_units"])


def test_optimize_products_and_summary():
    elasticities = pd.DataFrame({
        "stock_code": ["A", "B"], "description": ["MUG", "BAG"], "category": ["k", "b"],
        "estimate": [-1.3, -3.0], "ci_low": [-1.6, -3.5], "ci_high": [-1.0, -2.5],
    })
    recs = optimize_products(pricing_inputs(make_panel()), elasticities).set_index("stock_code")
    assert recs.loc["A", "recommended_price"] == pytest.approx(2.4)
    assert recs.loc["B", "recommended_price"] == pytest.approx(5.0 * 0.6 * 1.5)  # c·θ/(1+θ)
    summary = summarise(recs.reset_index())
    assert summary["products_priced"] == 2
    assert summary["price_up"] == 1 and summary["price_down"] == 1
    assert summary["expected_weekly_profit"] > summary["current_weekly_profit"]


@pytest.mark.parametrize("bad", [
    {"base_price": float("nan")}, {"base_quantity": float("inf")}, {"elasticity": float("nan")},
])
def test_non_finite_inputs_are_rejected(bad):
    args = {"base_price": 2.0, "base_quantity": 10.0, "elasticity": -1.5, **bad}
    with pytest.raises(ValueError, match="must be finite"):
        recommend_price(**args)
