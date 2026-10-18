# EconoCausal

Dynamic pricing using Double Machine Learning (DML).

## Problem

A retailer wants to know how demand for a product changes when its price changes,
i.e. the **price elasticity of demand**. Once we know the elasticity, we can pick
prices that improve revenue or profit.

The catch is that prices in historical data are not set randomly. Stores raise prices
when demand is already high (holidays, trending products) and discount when it is low.
If we just regress sales on price, the model mixes up "price affects demand" with
"demand affects price", and the elasticity comes out biased (usually it looks like
customers are less price sensitive than they really are).

This project uses **Double Machine Learning** to estimate the causal effect of price on
demand while controlling for these confounders, and then uses the estimates to
recommend prices.

- Problem statement, causal diagram and assumptions: [docs/problem_statement.md](docs/problem_statement.md)
- Week-wise plan and milestones: [docs/roadmap.md](docs/roadmap.md)
- Papers, libraries and data sources: [docs/references.md](docs/references.md)

## Approach (planned)

1. Clean transaction data and aggregate it to product x week level.
2. Explore price variation, seasonality and demand patterns (EDA).
3. Build a synthetic dataset where the true elasticity is known, to check whether
   each method recovers it.
4. Baselines: naive OLS on log(price) vs log(quantity).
5. DML:
   - my own implementation with cross-fitting
   - EconML `LinearDML` for comparison
   - `CausalForestDML` for elasticity by segment (category, season, country)
6. Robustness checks (placebo treatment, random common cause, data subsets).
7. Price optimizer that maximises expected profit under business constraints.
8. FastAPI backend + SQLite database + Streamlit dashboard, deployed online.

## Data

- [Online Retail II (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/502/online+retail+ii):
  around 1M transactions from a UK online retailer, 2009-2011.
- A synthetic dataset generated in this repo with a known true elasticity (−1.8) and the same kinds of
  confounding as the real data, used to check which methods recover the true value.
  Design and validation: [docs/synthetic_data.md](docs/synthetic_data.md)

Raw data is not committed to the repo. To get it:

```bash
python -m econocausal.data.download   # download and extract the Excel file
python -m econocausal.data.load       # convert it to parquet for fast loading
python -m econocausal.data.build      # clean, build the weekly panel, save to SQLite
```

The build step creates `data/econocausal.db` with three tables: `weekly_sales`
(product x week panel used for modelling), `products` (one row per product) and
`cleaning_log` (rows removed by each cleaning step).

Column descriptions and data quality notes: [docs/data_dictionary.md](docs/data_dictionary.md)

## Early findings (EDA)

- Prices change within products in about a third of weeks (typical swing ±28%), enough to estimate elasticities.
- Demand peaks in Oct–Nov while prices dip, and prices rise in December: season affects both.
- One-off buyers without a customer ID pay about 2x the usual price, so customer mix drives weekly prices.
- The naive price slope moves from −0.67 (pooled) to about −1.5 (within product) depending on the
  controls, which is why Double ML is needed.

Details: [docs/eda_findings.md](docs/eda_findings.md) · notebook: [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)

## Baselines

Naive log-log OLS (`python -m econocausal.baselines`). On the synthetic data the true elasticity is −1.8.

| Method | Synthetic | Real |
|--------|----------:|-----:|
| Pooled, no controls | −1.15 | −0.67 |
| Product fixed effects | −2.74 | −1.54 |
| Product + week fixed effects | −2.91 | −1.60 |
| Product + week FE + guest/UK share | −1.93 | −1.53 |

None of them recovers the true value on synthetic data, and their confidence intervals exclude it.
Full table: [reports/baseline_results.csv](reports/baseline_results.csv)

## Double ML (own implementation)

[`src/econocausal/dml.py`](src/econocausal/dml.py): product fixed effects removed by demeaning,
LightGBM nuisance models cross-fitted over folds split by product, residual-on-residual final stage,
standard errors clustered by product.

| Synthetic setting (true value −1.8) | DML estimate | 95% CI |
|-------------------------------------|-------------:|--------|
| Default | **−1.806** | −1.851 to −1.760 |
| Elasticity varies by category (average −1.8) | **−1.806** | −1.871 to −1.741 |
| Hidden confounder (assumption broken) | +0.22 | fails, as expected |

The best naive baseline on the same data gives −1.93 (CI −1.96 to −1.89), which excludes the true value.
EconML's `LinearDML` with the same inputs gives −1.816 (−1.861 to −1.771).
Comparison notebook: [notebooks/02_dml_synthetic.ipynb](notebooks/02_dml_synthetic.ipynb)

![Method comparison on synthetic data](reports/figures/08_method_comparison_synthetic.png)

## First results on real data

`python -m econocausal.run_dml`

| Method | Elasticity | 95% CI |
|--------|-----------:|--------|
| Best naive baseline (product + week FE + controls) | −1.53 | −1.58 to −1.48 |
| **DML (own implementation)** | **−1.28** | −1.32 to −1.23 |
| DML (EconML LinearDML) | −1.37 | −1.40 to −1.35 |

A 1% price increase lowers weekly units by about 1.3% on average. As on the synthetic data, the
linear baseline overstates price sensitivity, because the guest-buyer price step is not linear.

The two DML versions differ because the own implementation removes each product's average residual a
second time after the ML step and EconML does not. Without that step the own implementation gives
−1.368, matching EconML; the step is needed for a true within-product estimate on this unbalanced panel
(see the note in [`models.py`](src/econocausal/models.py)).

**By category** (own DML; one set of nuisance models, separate final stage per category):

| Category | Elasticity | 95% CI | Products |
|----------|-----------:|--------|---------:|
| garden | −1.57 | −1.78 to −1.36 | 38 |
| candles_lighting | −1.53 | −1.66 to −1.39 | 157 |
| bags | −1.37 | −1.55 to −1.18 | 94 |
| other | −1.29 | −1.36 to −1.22 | 699 |
| stationery_cards | −1.28 | −1.44 to −1.12 | 168 |
| home_decor | −1.26 | −1.37 to −1.15 | 249 |
| kitchen_dining | −1.22 | −1.35 to −1.09 | 258 |
| toys_kids | −1.18 | −1.48 to −0.88 | 59 |
| christmas | −1.11 | −1.31 to −0.91 | 60 |
| jewellery | −0.96 | −1.20 to −0.72 | 71 |
| easter | −0.90 | −1.25 to −0.55 | 24 |

Every category is price sensitive. Jewellery and Easter items are the least elastic, though their
intervals are wide because they have few products. Full results:
[reports/category_elasticities_real.csv](reports/category_elasticities_real.csv)

## Tech stack

Python, pandas, scikit-learn, LightGBM, EconML, FastAPI, SQLite (SQLAlchemy), Streamlit, pytest.

## Project structure

```
econocausal/
├── api/                  FastAPI service
├── app/                  Streamlit dashboard
├── data/
│   ├── raw/              original downloaded data (not tracked)
│   └── processed/        cleaned / aggregated data (not tracked)
├── docs/                 problem statement, notes, report
├── notebooks/            EDA and experiments
├── reports/figures/      saved plots
├── src/econocausal/      main package (data pipeline, models, optimizer)
└── tests/
```

## Setup

Requires Python 3.12 or newer (developed on Python 3.14).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Progress

| Week | Focus | Status |
|------|-------|--------|
| 1 | Setup, data pipeline, EDA, synthetic data, naive baselines | Done ([notes](docs/progress/week1.md)) |
| 2 | DML (manual + EconML), API, first dashboard | Next |
| 3 | Heterogeneous effects, robustness checks, price optimizer | Not started |
| 4 | Full dashboard, Docker, CI, deployment | Not started |

## Author

Aman Singh Dadsena, Data Science & ML Intern, Infotact Solutions
