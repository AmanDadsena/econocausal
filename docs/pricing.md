# Price Optimizer

Code: [`src/econocausal/optimize.py`](../src/econocausal/optimize.py) ·
Tests: [`tests/test_optimize.py`](../tests/test_optimize.py) ·
Run: `python -m econocausal.optimize` ·
Results: [`reports/price_recommendations.csv`](../reports/price_recommendations.csv),
[`reports/pricing_cost_sensitivity.csv`](../reports/pricing_cost_sensitivity.csv)

## Model

Around a product's current price `p0` and weekly units `q0`, demand follows its estimated
elasticity θ (from the causal forest):

```
q(p) = q0 · (p / p0)^θ            weekly units at price p
π(p) = (p − c) · q(p)             weekly profit, unit cost c
```

- For θ < −1 profit peaks at the **Lerner / inverse-elasticity price** `p* = c · θ / (1 + θ)`.
  Example: c = £1.20, θ = −3 → p* = £1.80.
- For θ ≥ −1 (inelastic) profit rises with every price increase, so the model alone has no finite answer.

Both cases are handled the same way: the best price inside the allowed range is the unconstrained optimum
clipped to that range (profit is single-peaked for θ < −1 and increasing for θ ≥ −1). Tests check this
against a brute-force search over 20,001 prices for 7 elasticities x 3 cost levels.

## Business rules (`PricingRules`)

| Rule | Default | Why |
|------|---------|-----|
| Maximum price change | ±20% of the current price | The constant-elasticity curve is only trusted near prices the product has actually been sold at (the median product's prices vary by about ±28%). |
| Minimum margin | 20% | Never recommend a price that leaves less than 20% margin. |
| Unit cost | 60% of the current price | **Assumption**: the data has no cost information. See the sensitivity table below. |
| Objective | profit | `revenue` is also supported. |

Each recommendation records which rule decided it (`reason`): *upper price limit*, *lower price limit*,
*minimum margin*, *optimal within limits*, or *infeasible* (when the margin rule needs a price above the
allowed range; the current price is then kept).

**Inputs per product:** the current price is the median weekly price over the last 13 weeks of data
(weeks starting 5 Sep – 28 Nov 2011, i.e. 5 Sep – 4 Dec); current weekly units are total units in those 13 weeks divided by 13, so weeks
with no sales count as zero. Products with no sales in that window (240 of 1,877) are not priced.

**Uncertainty:** the price is also recomputed at both ends of the product's 95% elasticity interval
(`price_at_ci_low`, `price_at_ci_high`). If they differ from the recommendation, the decision depends on
where in its interval the true elasticity lies.

## Results on the real data (cost = 60% of price)

| | Value |
|---|---|
| Products priced | 1,637 |
| Recommendation | **raise the price for all 1,637** |
| Decided by the +20% limit | 1,630 |
| Optimal inside the limits | 7 (elasticity about −2.0 to −2.3, e.g. Kensington Coffee Set +6.3%) |
| Weekly revenue | £120,097 → £114,635 (**−4.5%**) |
| Weekly profit | £48,039 → £57,300 (**+19.3%**) |

**Why almost every price goes up.** Most estimated elasticities are around −1.3. At that elasticity the
profit-maximising markup is θ/(1+θ) = 4.3 times cost, far above today's prices if cost is 60% of the
price. So the optimum lies well above the current price and the +20% limit decides. Units and revenue
fall a little, but each unit earns much more margin, so profit rises. 12% of products have an elasticity
above −1 (inelastic), where raising the price increases revenue as well.

**Largest expected profit gains** are seasonal best sellers such as *Paper Chain Kit 50's Christmas*
(£2.95 → £3.54) and *Small Popcorn Holder* (£0.85 → £1.02). The window is the pre-Christmas peak, so
their weekly volumes, and the £ figures above, are peak-season numbers.

## Sensitivity to the cost assumption

| Cost as % of price | Price up | Price down | Revenue change | Profit change |
|-------------------:|---------:|-----------:|---------------:|--------------:|
| 40% | 1,493 | 137 | −2.1% | +6.5% |
| 50% | 1,628 | 7 | −4.1% | +11.4% |
| **60% (default)** | 1,637 | 0 | −4.5% | +19.3% |
| 70% | 1,637 | 0 | −4.6% | +32.5% |
| 75% | 1,637 | 0 | −4.6% | +43.1% |

The **direction** (raise prices) holds for any cost share of 50% or more; only at low costs do some
products get a price cut. The **size** of the profit gain depends strongly on the cost assumption, so the
percentage should not be quoted without it.

## Limitations

- Real unit costs are unknown; the cost share is an assumption.
- The constant-elasticity curve is a local approximation; a +20% move is at the edge of the observed price
  range for many products.
- Customers' reactions to a deliberate, permanent price rise (and competitors' responses) may differ from
  their reactions to the week-to-week price changes in the data.
- Recommended next step for a real business: test the price change on a small set of products first
  (an A/B or staggered rollout) and re-estimate the elasticity from the results.
