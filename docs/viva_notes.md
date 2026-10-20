# Viva Notes

Short answers to the questions most likely to come up, using this project's own numbers.
Explain in plain words first, then use the technical term.

## 1. One-minute summary

> Retailers want to know how much demand changes when they change a price: the **price elasticity**.
> The problem is that historical prices were not set at random. They move with season, with product
> popularity and, in this shop, with *who* is buying. A plain regression mixes those effects up with
> the effect of price. I used **Double Machine Learning**: two ML models remove the effect of these
> confounders from both price and quantity, and the elasticity is estimated from what is left.
> I proved it works on synthetic data with a known answer (true −1.8, DML −1.806, every OLS method
> misses), then applied it to 1 million real transactions: the average elasticity is **−1.28**,
> meaning a 1% price rise lowers weekly units by about 1.3%. The results are served through a FastAPI
> backend with a SQLite database and shown in a Streamlit dashboard.

## 2. Core concepts

**What is price elasticity?**
The % change in quantity for a 1% change in price. With `log Q = θ · log P + ...`, θ is the elasticity.
θ = −1.28 means a 1% price increase lowers units sold by about 1.28%. |θ| > 1 is "elastic": revenue
falls when the price goes up.

**What is a confounder?**
Something that affects both the price and the quantity. Here: season (Oct–Nov demand peak, December
price jump), product popularity (cheap products sell more) and customer mix (guest buyers pay a higher
price and buy less).

**Why is a simple regression biased?**
Because it credits price with effects that really come from the confounders. On the synthetic data the
pooled OLS slope is −1.15 against a true −1.8; within-product OLS is −2.74. The bias can go either way
depending on which confounder is ignored.

**What does DML do, step by step?** (`src/econocausal/dml.py`)
1. Remove each product's own average from log quantity and log price (product fixed effects).
2. ML model 1 predicts log quantity from the confounders.
3. ML model 2 predicts log price from the confounders.
4. Keep the residuals: the parts the confounders cannot explain.
5. Regress the quantity residual on the price residual. The slope is the elasticity.
6. Standard errors clustered by product.

**Why does that work?** (Frisch–Waugh–Lovell)
In a linear regression, the coefficient on price equals the slope you get after removing the controls
from both price and quantity. DML does the same, but with ML models, so the controls can have
nonlinear effects.

**What is cross-fitting and why is it needed?** (`src/econocausal/crossfit.py`)
Each row is predicted by a model that was trained without it (5 folds). If a model predicted its own
training rows, it would partly fit the noise, the residuals would be too small and the estimate biased.
My folds are split **by product**, so all weeks of a product are in the same fold. Otherwise a model
could learn a product's own noise from its other weeks.

**What is Neyman orthogonality?**
The final estimate uses residuals from *both* models, so small errors in either model only affect it at
second order (error × error). That is why imperfect ML models still give a valid estimate and valid
confidence intervals.

**Why LightGBM?**
The confounder effects are nonlinear: the guest-buyer price effect is a *step* (jump to the retail tier
when guests are over half the order lines) and seasonality is a *bell curve*. Trees fit steps and curves
naturally. A linear control on the synthetic data still leaves a 0.13 bias (−1.93 vs −1.8).

**What do the nuisance R² values mean?**
Out-of-sample R² of the two ML models. On real data: quantity 0.32, price 0.47. So about half of the
within-product price variation is explained by season and customer mix; the elasticity comes from the
other half, the "clean" variation.

## 3. Data decisions

| Question | Answer |
|----------|--------|
| What data? | UCI Online Retail II: 1,067,371 transactions, UK gift retailer, Dec 2009 – Dec 2011 |
| What was removed? | Cancellations (19,500), zero/negative quantity or price (6,202), non-product codes like postage and fees (4,792), exact duplicates (33,663), price outliers more than 10x from the product median (882), extreme bulk lines (101). 1,002,231 rows remain. |
| Unit of analysis? | One product in one week. 126,672 product-weeks, 1,877 products. |
| Why the **median** weekly price? | Bulk orders get lower unit prices. A revenue-weighted price would partly be *caused by* the quantity, which is reverse causality. The median is not pulled down by one big order. |
| Why only products with ≥ 30 weeks and ≥ 5 price levels? | Without price changes there is nothing to estimate. Example: White Hanging Heart (85123A) sold every week but always at £2.95, so it is excluded. |
| Why drop the last week? | The data ends on a Friday, so that week is incomplete. |
| What is `guest_share`? | Share of a week's order lines with no customer ID. Those buyers pay about 1.98x the usual price (retail tier). It correlates 0.51 with price within a product. I found this in the EDA and added it as a control. |
| Why **not** control for number of orders or customers? | They are partly *caused by* the price (a cheaper product gets more orders). Controlling for them would hide part of the effect. |

## 4. Results

**Synthetic data (true elasticity −1.8)**

| Method | Estimate | 95% CI contains −1.8? |
|--------|---------:|:---------------------:|
| OLS pooled | −1.15 | no |
| OLS product FE | −2.74 | no |
| OLS product + week FE | −2.91 | no |
| OLS product + week FE + controls | −1.93 | no |
| DML (own) | **−1.806** | yes |
| DML (EconML) | −1.816 | yes |

The best OLS interval (−1.96 to −1.89) is narrow and in the wrong place: the problem is **bias, not
noise**, and more data would not fix it. The DML result held across 6 different random seeds.

**Real data**

| Method | Estimate | 95% CI |
|--------|---------:|--------|
| Best naive OLS | −1.53 | −1.58 to −1.48 |
| DML (own) | **−1.28** | −1.32 to −1.23 |
| DML (EconML) | −1.37 | −1.40 to −1.35 |

By category: garden (−1.57) and candles/lighting (−1.53) are the most price sensitive; jewellery
(−0.96) and Easter (−0.90) the least (wide intervals, few products).

**Why do my DML and EconML differ on real data?**
My version removes each product's average residual a second time after the ML step. The models are
trained on other products, so they cannot know a product's typical error, and those errors are
correlated between the two models (−0.65). Leaving them in brings between-product differences back
into a within-product estimate. I checked it: without that step my code gives −1.368, the same as
EconML's −1.371. On the balanced synthetic data the two agree.

## 5. Limitations (say these before being asked)

- **Unobserved confounders** (competitor prices, marketing, stock-outs) are not in the data. DML cannot
  fix them; the synthetic `hidden_confounding` test shows the estimate breaks (+0.22 instead of −1.8).
- **Weeks with zero sales** are missing (price unobserved), which may make demand look slightly less
  elastic.
- **Constant elasticity** within a category is an assumption; segment-level estimates relax it partly.
- **One retailer, 2009–2011, mostly UK**: results may not transfer.
- **No real cost data**: the price optimizer will need an assumed cost.

## 6. Questions about the code

| Question | Where |
|----------|-------|
| How is the data cleaned? | `src/econocausal/data/clean.py`, one function per rule, log of rows removed |
| How is the weekly panel built? | `src/econocausal/data/panel.py` |
| How are fixed effects removed exactly on an unbalanced panel? | `demean` in `src/econocausal/baselines.py`: alternates between product and week demeaning until it converges; a test matches the full dummy regression to 1e-8 |
| How is the standard error computed? | `final_stage` in `src/econocausal/dml.py`: DML score summed per product (clustered); a test matches statsmodels within 1% |
| How do you know DML works? | `tests/test_dml.py`: recovers −1.8, beats every baseline, fails on hidden confounding as it should |
| How is the API protected from bad input? | Parameterised SQL only, input limits checked (422), unknown products 404; a test sends a SQL-injection string |
| How many tests? | 72, run with `pytest` |

## 7. Tricky questions

**"Why not just use fixed effects?"** Fixed effects remove product differences, but not season or
customer mix. On synthetic data product FE gives −2.74 (true −1.8).

**"Why not just add the controls linearly?"** The effects are nonlinear (a price step, a seasonal
curve). Linear controls leave a 0.13 bias on synthetic data; DML removes it.

**"Is −1.28 causal?"** Only under the assumption that all important confounders are observed. I have
season, calendar, customer mix, category and product effects, but not competitor prices or marketing.

**"What would you do with more time?"** Product-level elasticities with a causal forest, robustness
checks (placebo, random common cause), and a price optimizer. These are the next steps in the roadmap.
