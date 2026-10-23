# Week 2 Progress Notes (17 – 23 Oct 2026)

**Focus:** Double ML estimator, EconML comparison, first real-data results, API and dashboard
**Status:** all Week 2 items in the [roadmap](../roadmap.md) are done; three Week 3 items
(causal forest, robustness checks, price optimizer) were started early. 129 automated tests passing.

## Work done

| Day | Work | Output |
|-----|------|--------|
| 17 Oct | Cross-fitting helper (folds split by product); own DML estimator with clustered standard errors; tests that it recovers the true elasticity | `crossfit.py`, `dml.py`, `tests/test_dml.py` |
| 18 Oct | EconML LinearDML wrapper; comparison of OLS, own DML and EconML on synthetic data; elasticity by category on real data | `models.py`, [02_dml_synthetic.ipynb](../../notebooks/02_dml_synthetic.ipynb), `run_dml.py` |
| 19 Oct | FastAPI REST API (8 endpoints) on SQLite; first Streamlit dashboard; run instructions | `api/main.py`, `app/streamlit_app.py` |
| 20 Oct | Viva notes, README status and architecture diagram, Mid Review demo checklist | [viva_notes.md](../viva_notes.md), [mid_review.md](../mid_review.md) |
| 21 Oct | No commit (missed day) | – |
| 22 Oct | Causal forest for product-level elasticities; robustness checks (placebo, random common cause, data subsets) | `forest.py`, `robustness.py`, [03_heterogeneity.ipynb](../../notebooks/03_heterogeneity.ipynb), [robustness.md](../robustness.md) |
| 23 Oct | Profit-maximising price optimizer with price limits, margin floor and cost sensitivity | `optimize.py`, [pricing.md](../pricing.md) |

## Key results

**Synthetic data (true elasticity −1.8)**

| Method | Estimate | 95% CI contains −1.8? |
|--------|---------:|:---------------------:|
| Best naive OLS (product + week FE + controls) | −1.93 | no |
| DML, own implementation | **−1.806** | yes |
| DML, EconML LinearDML | −1.816 | yes |

The DML result held across 6 different random seeds.

**Real data (126,672 product-weeks, 1,877 products)**

| Method | Estimate | 95% CI |
|--------|---------:|--------|
| Best naive OLS | −1.53 | −1.58 to −1.48 |
| DML, own implementation | **−1.28** | −1.32 to −1.23 |
| DML, EconML LinearDML | −1.37 | −1.40 to −1.35 |

A 1% price increase lowers weekly units sold by about 1.3%.

- **By category:** garden (−1.57) and candles/lighting (−1.53) are the most price sensitive;
  jewellery (−0.96) and Easter (−0.90) the least.
- **By product (causal forest):** median −1.31; 99.9% of products have an interval below zero.
  Products bought mostly by registered (wholesale) customers are the most price sensitive (about −1.54),
  and more expensive products are more price sensitive (−1.42 vs −1.17 for the cheapest).
- **Robustness:** placebo with shuffled prices −0.001; random fake confounder −1.276; random halves of
  the products −1.23 to −1.27. All checks pass.
- **Pricing (cost assumed 60% of price, ±20% limit):** raise prices on all 1,637 recently sold products,
  1,630 of them up to the +20% limit; weekly revenue −4.5%, weekly profit +19.3%. The direction holds for
  any cost share of 50% or more; the size of the gain depends on the cost assumption.

## Decisions and reasons

| Decision | Reason |
|----------|--------|
| Cross-fitting folds split by product | Weeks of the same product are correlated; random folds would let a model learn a product's own noise. |
| LightGBM for the nuisance models | The confounder effects are nonlinear (a price step for guest buyers, a seasonal curve); a linear control left a 0.13 bias on synthetic data. |
| Standard errors clustered by product | Weeks of the same product are not independent. Checked against statsmodels (within 1%). |
| Demean residuals by product a second time | Models trained on other products cannot know a product's average error; on the real data those errors are correlated (−0.65) and would bring back between-product bias. Without this step the estimate is −1.368, the same as EconML. |
| Causal forest fitted on the same DML residuals | Keeps the product-level estimates consistent with the main DML estimate. |
| API returns the product-level estimate, falling back to category level | The API never claims more precision than the model has. |
| LightGBM limited to 4 threads | Measured: using all 16 cores was about twice as slow on this data size. |
| Price changes limited to ±20% | The constant-elasticity curve is only trusted near observed prices; without a limit, inelastic products would get an unbounded price. |
| Cost assumed at 60% of price, with a sensitivity table | No cost data is available; the table shows which conclusions depend on it. |

## Problems and fixes

- **Own DML and EconML differed by 0.1 on real data** → traced it to the second demeaning step and proved
  it by switching the step off (−1.368 vs EconML −1.371).
- **Floating-point noise** (true value stored as −1.8000000000000005) → rounded at the source and
  regenerated the results.
- **A top-selling product (White Hanging Heart) is not in the API** → not a bug: its weekly price was
  always £2.95, so its elasticity cannot be estimated and the filter excludes it.
- **Dashboard chart label cut off in dark mode** → repositioned and checked the layout in the browser.
- **Causal forest crashed when passed in as an argument** → an unfitted econml forest raises an error
  when tested as true/false; replaced with an explicit `is None` check.
- **Optimizer crashed when printing on Windows** → the console could not print symbols like "≥" and "£";
  console messages now use plain ASCII.
- **Missed commit on 21 Oct** → work done on 22 Oct; committing daily from here on.

## Next week (24 – 30 Oct)

- Optimizer and price-simulator API endpoints with tests
- Full dashboard: simulator and optimizer pages
- Docker, CI pipeline and deployment

## Open questions for mentor

- Exact Mid Review date and time.
- Platform and format for the Mid Review social media video.
