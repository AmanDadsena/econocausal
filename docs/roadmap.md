# Roadmap

Project window: **10 Oct 2026 – 10 Nov 2026**

| Review | Window | What is checked |
|--------|--------|-----------------|
| Mid Review | 20 – 27 Oct | Week 1 + Week 2 work |
| Final Review | 5 – 10 Nov | Week 3 + Week 4 work, deployment, documentation |

## Week 1 (10 – 16 Oct): setup, data and EDA

- [x] Repository structure, README, requirements
- [x] Problem statement and causal diagram
- [ ] Download script and data dictionary for Online Retail II
- [ ] Cleaning: cancellations, returns, invalid prices, outliers
- [ ] Product x week panel stored in SQLite
- [ ] EDA notebook and written findings
- [ ] Synthetic dataset with a known elasticity
- [ ] Naive OLS baselines

**Milestone:** clean weekly data in the database, and a clear picture of how much
prices actually vary.

## Week 2 (17 – 23 Oct): causal core

- [ ] Cross-fitting helper for nuisance models
- [ ] Manual DML estimator with standard errors and confidence intervals
- [ ] EconML `LinearDML` and comparison with the manual version
- [ ] Category-level elasticities on real data
- [ ] FastAPI service: health, products, elasticity endpoints
- [ ] First Streamlit page

**Milestone (Mid Review):** on synthetic data, DML recovers the true elasticity while
OLS does not; results available through the API and dashboard.

## Week 3 (24 – 30 Oct): advanced features and product

- [ ] `CausalForestDML` for segment-level elasticities
- [ ] Robustness checks: placebo, random common cause, data subsets
- [ ] Price optimizer with price bounds and margin constraints
- [ ] Optimizer and simulator endpoints with tests
- [ ] Full dashboard: elasticity, segments, simulator, optimizer
- [ ] Docker, CI, deployment

**Milestone:** working deployed app with price recommendations.

## Week 4 (31 Oct – 10 Nov): evaluation, documentation, final review

- [ ] Simulation study for confidence interval coverage
- [ ] Evaluation results and model card
- [ ] Architecture document
- [ ] Project report and presentation
- [ ] README with screenshots and live links
- [ ] Final clean-up and release

**Milestone (Final Review):** complete, documented, deployed project.

## Working rules

- Commit and push every day until the Final Review is done.
- Short progress update to the mentor every Friday.
- Keep raw data out of git; the pipeline must rebuild everything from the download.
