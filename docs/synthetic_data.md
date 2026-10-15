# Synthetic Data Design

Code: [`src/econocausal/simulate.py`](../src/econocausal/simulate.py) ·
Tests: [`tests/test_simulate.py`](../tests/test_simulate.py)

## Why synthetic data

On the real Online Retail data the true price elasticity is unknown, so there is no way to
tell whether an estimate of −1.5 is right or wrong. A synthetic dataset fixes this: the data is
generated from a model where **the true elasticity is set by us (−1.8)**, with the same kinds of
confounding found in the EDA. Every method can then be scored against the known answer.

```python
from econocausal.simulate import simulate_panel

df = simulate_panel()                         # 300 products x 104 weeks, true elasticity -1.8
df = simulate_panel(heterogeneous=True)       # elasticity differs by category
df = simulate_panel(hidden_confounding=1.0)   # an unobserved confounder (assumption broken)
```

The output has the same main columns as the real `weekly_sales` table (`stock_code`, `week_start`,
`category`, `price`, `quantity`, `log_price`, `log_quantity`, calendar features, `guest_share`,
`uk_share`), plus `true_elasticity` for each row.

## Model

For product *i* in week *t*:

**Price (treatment)**

```
log P_it = b_i + 0.35 · S(w_t) + G_p(g_it) + h · 0.15 · U_it + v_it,      v_it ~ N(0, 0.12²)
```

**Quantity (outcome)**

```
log Q_it = a_i + θ_i · (log P_it − b_i) + S(w_t) + G_q(g_it) + 0.3 · (uk_it − 0.92) + h · 0.5 · U_it + e_it,
e_it ~ N(0, 0.5²)
```

| Symbol | Meaning | How it is generated |
|--------|---------|---------------------|
| `b_i` | product's base log price | N(0.75, 0.8²) |
| `a_i` | product's base log demand | 3.0 − 0.9 · b_i + N(0, 0.8²): **cheaper products are more popular** |
| `θ_i` | true elasticity | −1.8, or −1.8 + category shift when `heterogeneous=True` |
| `S(w)` | seasonal demand by week of year | 0.45 · exp(−½((w − 45)/4)²) + 0.12 · cos(2π(w − 1)/52): peak around early November |
| `g_it` | guest share (lines with no customer ID) | product rate ~ Beta(1, 2.5) + N(0, 0.25²), clipped to [0, 1]; then 35% of weeks set to 0 and 12% set to 1 |
| `G_p(g)` | price premium from guests | 0.1 · g + 0.6 · 1[g > 0.5]: **jump to the retail tier** once guests are the majority |
| `G_q(g)` | demand effect of guests | −1.2 · g²: weeks bought mostly by guests sell less |
| `uk_it` | UK share of lines | N(0.92, 0.1²), clipped to [0, 1] |
| `U_it` | hidden demand shock | N(0, 1), **not** returned in the data |
| `h` | `hidden_confounding` parameter | 0 by default |

## How each confounder biases naive estimates

| Confounder | Effect on price | Effect on quantity | Mirrors this EDA finding |
|------------|-----------------|--------------------|--------------------------|
| Product popularity (`a_i` vs `b_i`) | cheap products | sell more | pooled slope much flatter than within-product slope |
| Season `S(w)` | higher when demand is high | peak in Oct–Nov | price moves with the season |
| Guest share `g` | step up to the retail tier | lower when only guests buy | guest lines pay about 2x; all-guest weeks sell less |
| Hidden shock `U` (optional) | higher | higher | stands in for unmeasured things like competitor prices or marketing |

The seasonal and guest effects are **nonlinear** (a bell curve and a step), so adding them as
straight-line controls in a regression does not remove the bias completely. A flexible model can
learn these shapes. This is the gap Double ML is designed to close.

## Calibration against the real data

| Property | Real panel | Synthetic (default) |
|----------|-----------:|--------------------:|
| Product-weeks with no guest lines | 44% | 44% |
| Product-weeks with only guest lines | 13% | 13% |
| Within-product SD of log price (median over products) | 0.28 | 0.31 |
| Share of log-price variance within products | 13.3% | 13.5% |
| Median weekly price | £2.10 | £2.58 |

The synthetic panel is **balanced** (every product appears every week). The real panel only has weeks
with sales.

## Validation: what the generator produces

Default settings (`seed=42`, 300 products x 104 weeks = 31,200 rows), true elasticity **−1.8**:

| Estimate | Slope | Error |
|----------|------:|------:|
| Pooled OLS, no controls | −1.15 | +0.65 |
| Within product | −2.74 | −0.94 |
| Within product + week effects | −2.91 | −1.11 |
| Within product + week effects + guest share (linear) | −1.93 | −0.13 |
| Within product + week effects + exact guest functions ("oracle") | **−1.82** | −0.02 |

- Naive estimates are wrong in **both directions**, depending on which confounders they ignore.
- Even with all the right variables, linear controls leave a bias of about 0.13 because the
  true effects are nonlinear.
- With the correct functional form (which a real analyst does not know), the true value is recovered.
  Across seeds 0–9 (120 products), the oracle estimate stays between −1.74 and −1.92, while pooled OLS
  stays between −1.02 and −1.26.
- With `hidden_confounding=1.0`, even the oracle breaks (estimate about +0.2). No method can fix a
  confounder that is not in the data, which is why "no unobserved confounding" is listed as a key
  assumption in the problem statement.

## Heterogeneous mode

With `heterogeneous=True`, the elasticity depends on the category (shifts are centred so the average
over products is still −1.8). With the default seed:

| Category | True elasticity |
|----------|----------------:|
| christmas | −2.49 |
| jewellery | −2.09 |
| home_decor | −1.89 |
| kitchen_dining | −1.69 |
| bags | −1.39 |
| stationery_cards | −1.19 |

This is used later to check whether `CausalForestDML` recovers elasticities per segment.

## What the tests check

`tests/test_simulate.py`:

- the same seed gives identical data, and a different seed gives different data
- shape, columns, no missing values; price and quantity match their logs
- the customer mix is close to the real data
- the default elasticity is constant at −1.8
- heterogeneous mode: christmas is most elastic, stationery least, and the average stays −1.8
- pooled OLS is biased by more than 0.4
- correct adjustment recovers −1.8 within ±0.15
- hidden confounding breaks recovery
