# Robustness Checks

Code: [`src/econocausal/robustness.py`](../src/econocausal/robustness.py) ·
Tests: [`tests/test_robustness.py`](../tests/test_robustness.py) ·
Run: `python -m econocausal.robustness` · Results: [`reports/robustness_real.csv`](../reports/robustness_real.csv)

## Why

A good estimate should behave predictably when the data is changed in ways where the right
answer is known. These "refutation tests" (the same idea as DoWhy's refuters) try to break the
estimate. Passing them does not prove the result is causal (no test can rule out an
unmeasured confounder), but failing one would point to a problem with the method or the data.

## The checks

| Check | What changes | A trustworthy estimate should... |
|-------|--------------|----------------------------------|
| **Placebo treatment** | Each product's weekly prices are randomly reshuffled among its own weeks. Price level and spread stay the same; only the link between a week's price and that week's sales is broken. | drop to about **0** (confidence interval contains 0) |
| **Random common cause** | A column of pure random noise is added as an extra confounder. | stay within **0.1** of the original |
| **Data subsets** | Re-estimated three times, each on a random half of the products. | stay within **0.1** of the original |

## Results on the real data

Own DML, same settings as the main estimate (126,672 product-weeks, 1,877 products).

| Check | Estimate | 95% CI | Expected | Passed |
|-------|---------:|--------|----------|:------:|
| Original | −1.275 | −1.320 to −1.230 | – | – |
| Placebo treatment | **−0.001** | −0.019 to 0.017 | about 0 | yes |
| Random common cause | −1.276 | −1.321 to −1.231 | within 0.1 of −1.275 | yes |
| Subset 1 (half the products) | −1.257 | −1.321 to −1.193 | within 0.1 | yes |
| Subset 2 | −1.231 | −1.285 to −1.177 | within 0.1 | yes |
| Subset 3 | −1.270 | −1.327 to −1.213 | within 0.1 | yes |

**All checks pass.**

- **Placebo:** with shuffled prices the estimate is essentially zero (−0.001, interval −0.019 to 0.017).
  The −1.28 therefore comes from the timing of real price changes, not from something built into the
  method, the product averages or the confounder models.
- **Random common cause:** adding a meaningless variable changes the estimate by 0.001. The ML models do
  not pick up noise and shift the answer.
- **Subsets:** halves of the data give −1.23 to −1.27. The result does not depend on a few products.

## The same checks on synthetic data

`tests/test_robustness.py` runs the checks on synthetic data (true elasticity −1.8) with smaller
models: the placebo estimate is within 0.1 of zero, the random confounder moves the estimate by less
than 0.05, and every half-sample estimate is within 0.2 of −1.8.

## What these checks cannot show

- An **unmeasured confounder** (competitor prices, marketing, stock-outs) would bias every row above in
  the same way. The synthetic `hidden_confounding` test shows how large that bias can be (+0.22 instead
  of −1.8), which is why this stays listed as the main limitation.
- The checks test this estimator on this panel; they do not test the panel itself (for example, the
  missing zero-sales weeks).
