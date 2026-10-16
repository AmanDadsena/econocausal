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
