"""Profit-maximising prices from estimated elasticities.

Demand model (constant elasticity around the current price):

    q(p) = q0 · (p / p0)^θ

where p0 and q0 are the product's current price and weekly units, and θ is its estimated
price elasticity. With unit cost c, weekly profit is

    π(p) = (p − c) · q(p)

For θ < −1 the profit has a single maximum at the "inverse elasticity" (Lerner) price

    p* = c · θ / (1 + θ)

For θ ≥ −1 demand is inelastic: raising the price always raises profit, so the model alone
would recommend an unbounded price. Business limits (`constraints.py`) are therefore always
applied before a price is recommended.

The constant-elasticity curve is only trusted close to prices the product has actually been
sold at, which is another reason to limit how far a price may move.
"""

import numpy as np


def demand(price, base_price, base_quantity, elasticity):
    """Expected weekly units at `price` under the constant-elasticity demand curve."""
    return base_quantity * (np.asarray(price, dtype=float) / base_price) ** elasticity


def profit(price, base_price, base_quantity, elasticity, unit_cost):
    """Expected weekly profit at `price`."""
    price = np.asarray(price, dtype=float)
    return (price - unit_cost) * demand(price, base_price, base_quantity, elasticity)


def unconstrained_optimum(elasticity: float, unit_cost: float) -> float:
    """Profit-maximising price with no limits: c·θ/(1+θ) for θ < −1, otherwise infinite."""
    if elasticity >= -1:
        return float("inf")
    return unit_cost * elasticity / (1 + elasticity)


def grid_optimum(base_price, base_quantity, elasticity, unit_cost, low, high, n=20_001) -> float:
    """Brute-force check: the best price on a fine grid between `low` and `high`."""
    grid = np.linspace(low, high, n)
    return float(grid[np.argmax(profit(grid, base_price, base_quantity, elasticity, unit_cost))])
