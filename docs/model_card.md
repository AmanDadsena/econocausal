# Model Card: EconoCausal price elasticity model

Version 1.0.0 · Author: Aman Singh Dadsena · Code: `src/econocausal/dml.py`, `forest.py`, `optimize.py`

## What it does

Estimates the **causal price elasticity of demand**, the % change in weekly units sold for a 1%
change in price, for products of an online gift retailer, and uses it to recommend prices.

| Output | Level | Method |
|--------|-------|--------|
| Average elasticity | all products | Double Machine Learning (own implementation), EconML LinearDML as a check |
| Elasticity per category | 11 categories | same DML residuals, separate final stage per category |
| Elasticity per product | 1,877 products | causal forest (econml generalized random forest) on the DML residuals |
| Recommended price | 1,637 recently sold products | profit-maximising price from the elasticity, within business rules |

## Intended use

- Understanding how price-sensitive products and segments are, and comparing them.
- Exploring "what if" prices for a product (dashboard simulator, `POST /simulate`).
- Shortlisting products whose prices are worth testing.

**Not intended** for setting prices automatically without a test: recommendations should be tried on a few
products first (A/B or staggered rollout) and the elasticity re-estimated. Not intended for other retailers,
time periods or sales channels without re-fitting.

## Data

- **UCI Online Retail II** (Chen, 2019): all transactions of a UK online giftware retailer, Dec 2009 – Dec 2011,
  1,067,371 rows. Many customers are wholesalers.
- Cleaned to 1,002,231 product sales (cancellations, returns, non-product codes, duplicates and outliers
  removed; see [data_dictionary.md](data_dictionary.md)).
- Modelling unit: **product x week**, 126,672 rows, 1,877 products with at least 30 weeks of sales and 5
  different weekly prices. Weekly price = median order-line price (not revenue / units, which bulk discounts
  would bias).
- **Synthetic data** with a known elasticity of −1.8 and the same kinds of confounding, for validation
  ([synthetic_data.md](synthetic_data.md)).

## Model

`log Q = θ · log P + g(X) + ε`, `log P = m(X) + v`

| Part | Choice |
|------|--------|
| Treatment / outcome | log weekly median price / log weekly units |
| Product effects | removed by demeaning (before and after the ML step) |
| Confounders X | week of year, weeks to Christmas, month, year, peak season, bank holiday, guest share, UK share, category, product averages of guest and UK share |
| Not used as controls | number of orders and customers (partly caused by price) |
| Nuisance models g, m | LightGBM, 400 trees, learning rate 0.05, 31 leaves, ≥ 50 rows per leaf |
| Cross-fitting | 5 folds, split by product |
| Inference | standard errors clustered by product; 95% intervals |
| Causal forest | 400 honest trees, ≥ 50 rows per leaf; effect modifiers: category, typical price, average guest and UK share |
| Seeds | fixed; a full rebuild reproduces all results exactly |

## Performance

### Synthetic data (true elasticity −1.8)

| Method | Estimate | 95% CI contains −1.8 |
|--------|---------:|:--------------------:|
| Best naive OLS (product + week FE + controls) | −1.93 | no |
| **Own DML** | **−1.806** | yes |
| EconML LinearDML | −1.816 | yes |

**Coverage study** (100 synthetic datasets, [evaluation.md](evaluation.md)):

| | Own DML | Best naive OLS |
|---|---:|---:|
| 95% interval contains the truth | **94%** (consistent with 95%; Wilson range 87.5%–97.2%) | 0% |
| Bias | +0.011 | −0.123 |
| RMSE | 0.033 | 0.126 |
| Mean reported SE vs actual spread | 0.033 vs 0.031 | 0.027 vs 0.027 |

DML's intervals are honest and its standard errors are calibrated; OLS is confidently wrong. DML has a
small bias towards zero (about a third of a standard error).

### Real data (126,672 product-weeks)

| Quantity | Value |
|----------|-------|
| Average elasticity (own DML) | **−1.28** (95% CI −1.32 to −1.23) |
| EconML LinearDML | −1.37 (95% CI −1.39 to −1.35); difference explained in [models.py](../src/econocausal/models.py) |
| Nuisance model fit (out-of-fold R²) | quantity 0.32, price 0.47 |
| Category range | −1.57 (garden) to −0.90 (Easter) |
| Product elasticities | median −1.31; 99.9% of 95% intervals below zero; median interval width 0.56 |
| Main drivers of differences | share of guest buyers (importance 0.61), price level (0.26) |

### Robustness (real data)

| Check | Result | Expected |
|-------|-------:|----------|
| Placebo: prices shuffled within products | −0.001 | about 0 |
| Random noise added as a confounder | −1.276 | about −1.275 |
| Random halves of the products (3 runs) | −1.23 to −1.27 | about −1.275 |

## Pricing recommendations

Profit-maximising price within ±20% of the current price and a 20% minimum margin; unit cost **assumed** to
be 60% of the current price (no cost data). Result: raise prices for all 1,637 products (1,630 at the +20%
limit); expected weekly profit +19.3%, revenue −4.5%. The direction holds for any cost share of 50% or more;
the size depends strongly on it (+11% at 50%, +33% at 70%). Details: [pricing.md](pricing.md).

## Limitations and risks

| Limitation | Effect | Mitigation |
|------------|--------|-----------|
| **Unobserved confounders** (competitor prices, marketing, stock-outs) | biased estimates; on synthetic data a hidden confounder moves the estimate from −1.8 to +0.22 | stated as the main assumption; robustness checks cannot rule it out |
| Weeks with no sales are missing | may make demand look slightly less elastic | documented |
| Constant elasticity within a product | a local approximation | price changes limited to ±20%; `/simulate` rejects prices beyond 20%–500% of current |
| No cost data | profit figures depend on an assumed cost share | sensitivity table; direction reported, size qualified |
| One UK retailer, 2009–2011, mostly wholesale | may not transfer to other shops or today | re-fit before use elsewhere |
| Product-level intervals are wide (median 0.56) | individual product values are uncertain | use for patterns and shortlisting; decisions use the interval ends too |
| Recommendations come from a pre-Christmas window | peak-season volumes | labelled in the dashboard and docs |

## Ethical considerations

- The data contains customer IDs but no names or contact details; the models and the API use only
  product-level aggregates, and no customer-level data is served or stored outside the local database.
- Price recommendations affect all customers of a product equally; the model does not personalise prices.
- Raising prices on price-insensitive products shifts value from customers to the retailer; a business
  should weigh this against customer trust before acting.

## How to reproduce

```bash
python -m econocausal.pipeline      # data, models, pricing (about 11 minutes)
python -m econocausal.evaluation    # coverage study
pytest                              # tests
```
