"""Profit-maximising prices from estimated elasticities.

Demand model (constant elasticity around the current price):

    q(p) = q0 · (p / p0)^θ

where p0 and q0 are the product's current price and weekly units, and θ is its estimated
price elasticity. With unit cost c, weekly profit is

    π(p) = (p − c) · q(p)

For θ < −1 the profit has a single maximum at the "inverse elasticity" (Lerner) price

    p* = c · θ / (1 + θ)

For θ ≥ −1 demand is inelastic: raising the price always raises profit, so the model alone
would recommend an unbounded price. Business limits (`PricingRules`) are therefore always
applied before a price is recommended.

The constant-elasticity curve is only trusted close to prices the product has actually been
sold at, which is another reason to limit how far a price may move.

Real cost data is not available, so unit cost is an **assumption**: a fixed share of the
current price (`cost_ratio`). Recommendations depend on it and should be read with that in mind.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


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


@dataclass(frozen=True)
class PricingRules:
    """Business limits for a price recommendation.

    max_change : largest allowed move from the current price, as a share (0.2 = ±20%).
    min_margin : smallest allowed margin (price − cost) / price.
    cost_ratio : assumed unit cost as a share of the current price.
    objective  : "profit" or "revenue".
    """

    max_change: float = 0.20
    min_margin: float = 0.20
    cost_ratio: float = 0.60
    objective: str = "profit"

    def __post_init__(self):
        if not 0 < self.max_change < 1:
            raise ValueError("max_change must be between 0 and 1")
        if not 0 <= self.min_margin < 1:
            raise ValueError("min_margin must be between 0 and 1")
        if not 0 <= self.cost_ratio < 1:
            raise ValueError("cost_ratio must be between 0 and 1")
        if self.objective not in {"profit", "revenue"}:
            raise ValueError("objective must be 'profit' or 'revenue'")


DEFAULT_RULES = PricingRules()


def best_price(base_price: float, elasticity: float, unit_cost: float, rules: PricingRules):
    """Best price within the rules, and which limit (if any) decided it.

    Profit (and revenue, which is profit with zero cost) is single-peaked in price for
    θ < −1 and always increasing for θ ≥ −1, so the best price inside an interval is the
    unconstrained optimum clipped to that interval.
    Returns (price, reason); price is None when no price satisfies every rule.
    """
    low = base_price * (1 - rules.max_change)
    high = base_price * (1 + rules.max_change)
    margin_floor = unit_cost / (1 - rules.min_margin)
    if margin_floor > high:
        return None, "infeasible: minimum margin needs a price above the allowed range"

    floor_is_margin = margin_floor > low
    low = max(low, margin_floor)
    target = unconstrained_optimum(elasticity, unit_cost if rules.objective == "profit" else 0.0)

    if target >= high:
        return high, "upper price limit"
    if target <= low:
        return low, "minimum margin" if floor_is_margin else "lower price limit"
    return target, "optimal within limits"


def recommend_price(
    base_price: float,
    base_quantity: float,
    elasticity: float,
    rules: PricingRules = DEFAULT_RULES,
    elasticity_low: float | None = None,
    elasticity_high: float | None = None,
) -> dict:
    """Recommended price for one product with the expected effect on units, revenue and profit.

    `elasticity_low` / `elasticity_high` (the ends of the elasticity's confidence interval)
    are optional; when given, the recommended price is also computed at both ends to show
    how sensitive the decision is to estimation uncertainty.
    """
    if base_price <= 0 or base_quantity < 0:
        raise ValueError("base_price must be positive and base_quantity non-negative")

    unit_cost = rules.cost_ratio * base_price
    price, reason = best_price(base_price, elasticity, unit_cost, rules)
    out = {
        "current_price": base_price,
        "unit_cost": unit_cost,
        "elasticity": elasticity,
        "recommended_price": price,
        "reason": reason,
    }
    if price is None:
        return out

    q0 = base_quantity
    q1 = float(demand(price, base_price, base_quantity, elasticity))
    rev0, rev1 = base_price * q0, price * q1
    pro0, pro1 = (base_price - unit_cost) * q0, (price - unit_cost) * q1
    out.update({
        "price_change_pct": 100 * (price / base_price - 1),
        "current_units": q0,
        "expected_units": q1,
        "units_change_pct": 100 * (q1 / q0 - 1) if q0 else 0.0,
        "current_revenue": rev0,
        "expected_revenue": rev1,
        "revenue_change_pct": 100 * (rev1 / rev0 - 1) if rev0 else 0.0,
        "current_profit": pro0,
        "expected_profit": pro1,
        "profit_change_pct": 100 * (pro1 / pro0 - 1) if pro0 else 0.0,
    })
    for name, e in [("price_at_ci_low", elasticity_low), ("price_at_ci_high", elasticity_high)]:
        if e is not None:
            out[name] = best_price(base_price, e, unit_cost, rules)[0]
    return out


def pricing_inputs(
    panel: pd.DataFrame, recent_weeks: int = 13, last_week: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Current price and weekly units for every product, from the last `recent_weeks` weeks.

    current price  median weekly price over the product's sale weeks in that window
    current units  total units in the window / recent_weeks (weeks with no sales count as 0)
    Products with no sales in the window are left out.

    `last_week` is the final week of the window; by default the last week in `panel`. Pass
    the data's last week when `panel` holds only some products, so every product is priced
    on the same window.
    """
    last_week = panel["week_start"].max() if last_week is None else pd.Timestamp(last_week)
    start = last_week - pd.Timedelta(weeks=recent_weeks - 1)
    recent = panel[panel["week_start"] >= start]
    return (
        recent.groupby("stock_code")
        .agg(current_price=("price", "median"), units_in_window=("quantity", "sum"),
             weeks_sold=("week_start", "nunique"))
        .assign(current_units=lambda d: d["units_in_window"] / recent_weeks)
        .drop(columns="units_in_window")
        .reset_index()
    )


def optimize_products(
    inputs: pd.DataFrame, elasticities: pd.DataFrame, rules: PricingRules = DEFAULT_RULES
) -> pd.DataFrame:
    """Recommend a price for every product that has both pricing inputs and an elasticity.

    `elasticities` needs columns stock_code, estimate, ci_low, ci_high
    (e.g. the causal forest's `product_elasticity` table)."""
    merged = inputs.merge(
        elasticities[["stock_code", "description", "category", "estimate", "ci_low", "ci_high"]],
        on="stock_code",
    )
    rows = []
    for r in merged.itertuples(index=False):
        rec = recommend_price(r.current_price, r.current_units, r.estimate, rules,
                              elasticity_low=r.ci_low, elasticity_high=r.ci_high)
        rows.append({"stock_code": r.stock_code, "description": r.description,
                     "category": r.category, "weeks_sold": r.weeks_sold, **rec})
    return pd.DataFrame(rows)


def summarise(recommendations: pd.DataFrame) -> dict:
    """Portfolio totals: current vs expected weekly revenue and profit."""
    ok = recommendations.dropna(subset=["recommended_price"])
    total = ok[["current_revenue", "expected_revenue", "current_profit", "expected_profit"]].sum()
    return {
        "products": len(recommendations),
        "products_priced": len(ok),
        "price_up": int((ok["price_change_pct"] > 0.5).sum()),
        "price_down": int((ok["price_change_pct"] < -0.5).sum()),
        "current_weekly_revenue": float(total["current_revenue"]),
        "expected_weekly_revenue": float(total["expected_revenue"]),
        "revenue_change_pct": float(100 * (total["expected_revenue"] / total["current_revenue"] - 1)),
        "current_weekly_profit": float(total["current_profit"]),
        "expected_weekly_profit": float(total["expected_profit"]),
        "profit_change_pct": float(100 * (total["expected_profit"] / total["current_profit"] - 1)),
        "reasons": ok["reason"].value_counts().to_dict(),
    }


def main() -> None:
    """Recommend prices for every product on the real data and store them.

    Usage:
        python -m econocausal.optimize
    """
    from econocausal.data.db import get_engine, load_panel, read_table, save_table
    from econocausal.paths import PROJECT_ROOT

    engine = get_engine()
    elasticities = read_table("product_elasticity", engine).query(
        "dataset == 'real' and method == 'causal_forest'"
    )
    inputs = pricing_inputs(load_panel())
    rules = PricingRules()
    recs = optimize_products(inputs, elasticities, rules)

    save_table(recs, "price_recommendations", engine)
    recs.round(4).to_csv(PROJECT_ROOT / "reports" / "price_recommendations.csv", index=False)

    summary = summarise(recs)
    print(f"Rules: +/-{rules.max_change:.0%} price change, >={rules.min_margin:.0%} margin, "
          f"cost assumed {rules.cost_ratio:.0%} of current price, objective {rules.objective}")
    print(f"Products priced: {summary['products_priced']:,} of {summary['products']:,} "
          f"(up {summary['price_up']}, down {summary['price_down']})")
    print(f"Why: {summary['reasons']}")
    print(f"Weekly revenue GBP {summary['current_weekly_revenue']:,.0f} to GBP {summary['expected_weekly_revenue']:,.0f} "
          f"({summary['revenue_change_pct']:+.1f}%)")
    print(f"Weekly profit  GBP {summary['current_weekly_profit']:,.0f} to GBP {summary['expected_weekly_profit']:,.0f} "
          f"({summary['profit_change_pct']:+.1f}%)")

    print("\nSensitivity to the assumed cost:")
    rows = []
    for ratio in [0.4, 0.5, 0.6, 0.7, 0.75]:
        s = summarise(optimize_products(inputs, elasticities, PricingRules(cost_ratio=ratio)))
        rows.append({"cost_ratio": ratio, "price_up": s["price_up"], "price_down": s["price_down"],
                     "revenue_change_pct": s["revenue_change_pct"], "profit_change_pct": s["profit_change_pct"]})
    sensitivity = pd.DataFrame(rows)
    sensitivity.round(2).to_csv(PROJECT_ROOT / "reports" / "pricing_cost_sensitivity.csv", index=False)
    print(sensitivity.round({"revenue_change_pct": 1, "profit_change_pct": 1}).to_string(index=False))
    print("\nSaved to reports/ and the `price_recommendations` table")


if __name__ == "__main__":
    main()
