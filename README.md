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
- A synthetic dataset generated in this repo with a known true elasticity.

Raw data is not committed to the repo. Download steps will be added under `src/econocausal/data/`.

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

Python 3.11 is recommended (some causal inference libraries don't support the newest Python versions yet).

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
| 1 | Setup, data pipeline, EDA, synthetic data | In progress |
| 2 | Baselines, DML (manual + EconML), API, first dashboard | Not started |
| 3 | Heterogeneous effects, robustness checks, price optimizer | Not started |
| 4 | Full dashboard, Docker, CI, deployment | Not started |

## Author

Aman Singh Dadsena, Data Science & ML Intern, Infotact Solutions
