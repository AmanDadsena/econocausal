import math

import numpy as np
import pytest

from econocausal.optimize import demand, grid_optimum, profit, unconstrained_optimum


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
