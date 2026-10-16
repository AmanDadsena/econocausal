# Week 1 Progress Notes (10 – 16 Oct 2026)

**Focus:** setup, data pipeline, EDA, synthetic data, naive baselines
**Status:** all Week 1 items in the [roadmap](../roadmap.md) are done.
21 commits on 7 days, 37 automated tests passing.

## Work done

| Day | Work | Output |
|-----|------|--------|
| 10 Oct | Repository structure, README, requirements | project skeleton |
| 11 Oct | Problem statement with causal diagram, roadmap, references | [problem_statement.md](../problem_statement.md), [roadmap.md](../roadmap.md), [references.md](../references.md) |
| 12 Oct | Download script and Excel-to-parquet loader for Online Retail II; data profiling | [data_dictionary.md](../data_dictionary.md) |
| 13 Oct | Cleaning pipeline, product x week panel, SQLite storage | `weekly_sales`, `products`, `cleaning_log` tables |
| 14 Oct | EDA notebook (7 figures) and written findings; new `guest_share` feature | [eda_findings.md](../eda_findings.md), [01_eda.ipynb](../../notebooks/01_eda.ipynb) |
| 15 Oct | Synthetic data generator with a known elasticity of −1.8 | [synthetic_data.md](../synthetic_data.md) |
| 16 Oct | Naive OLS baselines on both datasets, `results` table | [baseline_results.csv](../../reports/baseline_results.csv) |

## Key numbers

- Raw data: 1,067,371 transactions → **1,002,231** clean product sales.
- Modelling panel: **126,672 product-weeks, 1,877 products**, 30 Nov 2009 – 28 Nov 2011.
- Prices change within products in **37%** of weeks (median within-product SD of log price 0.28).
- One-off buyers without a customer ID pay **1.98x** the product's usual price.

## Baseline results

Log-log OLS, standard errors clustered by product. On synthetic data the true elasticity is **−1.8**.

| Method | Synthetic | Error vs truth | Real data |
|--------|----------:|---------------:|----------:|
| Pooled, no controls | −1.15 | +0.65 | −0.67 |
| Product fixed effects | −2.74 | −0.94 | −1.54 |
| Product + week fixed effects | −2.91 | −1.11 | −1.60 |
| Product + week FE + guest/UK share | −1.93 | −0.13 | −1.53 |

On synthetic data, **none of the baselines recovers −1.8**. Their 95% confidence intervals are narrow
(e.g. −1.96 to −1.89 for the best one) and all exclude the true value. The problem is bias, not noise:
more data would not fix it. This is the benchmark the DML estimator has to beat next week.

On real data the estimates move from −0.67 to about −1.5 depending on the controls, in line with the EDA.

## Decisions and reasons

| Decision | Reason |
|----------|--------|
| Weekly price = **median** of order-line prices | Bulk orders get lower unit prices; a revenue-weighted price would partly be caused by quantity (reverse causality). |
| Keep products with ≥ 30 weeks and ≥ 5 price levels | No price variation means no information about elasticity. |
| Drop the last, incomplete week | Data ends on a Friday, so that week's quantities would be too low. |
| Add `guest_share` as a control | Found in EDA: guest lines pay the retail price tier, so customer mix drives weekly price and quantity. |
| Do **not** control for `n_invoices` / `n_customers` | They are partly caused by price, so controlling for them would hide part of the effect. |
| Exact two-way fixed effects via alternating demeaning | The real panel is unbalanced; a single demeaning pass is only an approximation. A test checks it against the full dummy regression (match to 1e-8). |
| Synthetic data with nonlinear confounding | Linear controls should not be enough on their own, as in the real data, so the synthetic test is a fair challenge for DML. |

## Problems and fixes

- **Reading the Excel file took about 4 minutes** → converted once to parquet; it now loads in under a second.
- **The panel build took about 3 minutes** → replaced per-group Python functions with vectorised pandas operations. It now takes 15 s, with output checked to be identical.
- **A bank holiday test failed** → the test was wrong: Christmas 2010 was a Saturday, so the holidays moved to 27–28 Dec. Fixed the test, not the data.
- **Synthetic recovery test was too strict** → checked 10 seeds (range −1.74 to −1.92) and set the tolerance to ±0.15 to match.
- **An EDA chart title overstated a pattern** (guest share vs quantity) → rewrote it to describe what the data actually shows.

## Next week (17 – 23 Oct)

- Cross-fitting helper and my own DML estimator with standard errors
- Check it recovers −1.8 on synthetic data
- EconML `LinearDML` and comparison against the baselines above
- Category-level elasticities on real data
- FastAPI endpoints and first Streamlit page
- Mid Review preparation (review window 20 – 27 Oct)

## Open questions for mentor

- Exact Mid Review date and time.
- Required platform and format for the Mid Review social media video.
