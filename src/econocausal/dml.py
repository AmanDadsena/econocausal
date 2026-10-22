"""Double Machine Learning estimator of the price elasticity (own implementation).

Model (partially linear, in logs):

    log Q = θ · log P + g(X) + ε          outcome equation
    log P =          m(X) + v            treatment equation

Steps:
1. Remove product fixed effects: subtract each product's own average from log Q and
   log P, so differences between products (popular vs niche, cheap vs expensive)
   cannot bias the estimate.
2. Cross-fit two ML models (LightGBM): one predicts log Q from the confounders X, the
   other predicts log P from X. Each row is predicted by a model that never saw it.
3. Take the residuals Ỹ = log Q − ĝ(X) and T̃ = log P − m̂(X). T̃ is the part of the price
   that the confounders cannot explain.
4. θ̂ = Σ T̃·Ỹ / Σ T̃² (regression of Ỹ on T̃ through the origin).
5. Standard error from the DML score, clustered by product.

Steps 3-4 are the Frisch-Waugh-Lovell idea with ML models instead of linear regressions.
Because the final estimate depends on residuals from *both* models, small errors in either
model only affect it at second order (Neyman orthogonality). That is what makes valid
confidence intervals possible even though the ML models themselves are imperfect.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats

from econocausal.baselines import demean
from econocausal.crossfit import cross_fit_predict, make_group_folds

OUTCOME = "log_quantity"
TREATMENT = "log_price"
GROUP = "stock_code"

# Confounders: calendar, customer mix and product category. n_invoices / n_customers are
# left out on purpose: they are partly caused by the price (see docs/eda_findings.md).
CANDIDATE_FEATURES = [
    "week_of_year",
    "weeks_to_christmas",
    "month",
    "year",
    "is_peak_season",
    "has_bank_holiday",
    "guest_share",
    "uk_share",
    "category",
]
# Product averages of time-varying confounders (Mundlak terms). After demeaning, a row's
# expected value depends on the product's usual customer mix as well as this week's.
PRODUCT_MEAN_FEATURES = ["guest_share", "uk_share"]


def default_model(seed: int = 0) -> LGBMRegressor:
    """LightGBM settings for the nuisance models: shallow-ish trees, many rows per leaf
    and moderate shrinkage, which suits smooth seasonal and customer-mix effects.

    n_jobs=4: on these data sizes using every core is slower because of thread overhead
    (measured on the real panel: 45 s with 4 threads vs 54 s with 1; all 16 cores was
    about twice as slow as 1 on the synthetic panel). Results do not depend on it."""
    return LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )


@dataclass
class DMLResult:
    estimate: float
    std_error: float
    ci_low: float
    ci_high: float
    n_obs: int
    n_groups: int
    outcome_r2: float
    treatment_r2: float

    def to_dict(self) -> dict:
        return asdict(self)


def build_features(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Confounder matrix X: available candidate features plus product-mean terms."""
    features = features or [f for f in CANDIDATE_FEATURES if f in df.columns]
    X = df[features].copy()
    for col in PRODUCT_MEAN_FEATURES:
        if col in features:
            X[f"{col}_product_mean"] = df.groupby(GROUP)[col].transform("mean")
    if "category" in X:
        X["category"] = X["category"].astype("category")
    return X


def final_stage(y_res: np.ndarray, t_res: np.ndarray, groups: np.ndarray, level: float = 0.95):
    """θ̂ from residuals, with a standard error clustered by group.

    The DML score for one row is ψ = (Ỹ − θ·T̃)·T̃. Summing ψ within each product and
    using the spread of those sums across products gives a standard error that allows
    weeks of the same product to be correlated.
    """
    denom = np.sum(t_res**2)
    theta = float(np.sum(t_res * y_res) / denom)

    score = (y_res - theta * t_res) * t_res
    cluster_sums = pd.Series(score).groupby(groups).sum().to_numpy()
    n_clusters = len(cluster_sums)
    variance = n_clusters / (n_clusters - 1) * np.sum(cluster_sums**2) / denom**2
    se = float(np.sqrt(variance))

    z = float(stats.norm.ppf(0.5 + level / 2))
    return theta, se, theta - z * se, theta + z * se


def residualize(
    df: pd.DataFrame,
    features: list[str] | None = None,
    model_y=None,
    model_t=None,
    n_folds: int = 5,
    product_fixed_effects: bool = True,
    seed: int = 0,
) -> tuple[pd.DataFrame, float, float]:
    """Steps 1-3: cross-fitted residuals of log quantity (`y`) and log price (`t`).

    Returns the residuals (one row per input row, with `stock_code`) and the out-of-fold
    R² of the outcome and treatment models.

    Parameters
    ----------
    df : panel with `log_quantity`, `log_price`, `stock_code` and confounder columns.
    features : confounder columns; default is every column in CANDIDATE_FEATURES present in df.
    model_y, model_t : sklearn-style regressors for the outcome and treatment; LightGBM by default.
    n_folds : number of cross-fitting folds (split by product).
    product_fixed_effects : remove product averages before and after the ML step.
    seed : controls the fold split and the LightGBM randomness.
    """
    df = df.reset_index(drop=True)
    groups = df[GROUP].to_numpy()
    X = build_features(df, features)

    if product_fixed_effects:
        target = demean(df, [OUTCOME, TREATMENT], [GROUP])
    else:
        target = df[[OUTCOME, TREATMENT]].astype("float64")
    y, t = target[OUTCOME].to_numpy(), target[TREATMENT].to_numpy()

    folds = make_group_folds(df[GROUP], n_folds=n_folds, seed=seed)
    y_hat, outcome_r2 = cross_fit_predict(model_y or default_model(seed), X, y, folds)
    t_hat, treatment_r2 = cross_fit_predict(model_t or default_model(seed), X, t, folds)

    residuals = pd.DataFrame({"y": y - y_hat, "t": t - t_hat, GROUP: groups})
    if product_fixed_effects:
        # Models trained on other products cannot know this product's average error, so
        # remove it, exactly as the fixed effects were removed from the inputs.
        residuals[["y", "t"]] = demean(residuals, ["y", "t"], [GROUP])
    return residuals, outcome_r2, treatment_r2


def estimate_elasticity(
    df: pd.DataFrame, residualized: tuple | None = None, **kwargs
) -> DMLResult:
    """Estimate the average price elasticity with DML.

    `residualized` is the output of `residualize(df)`; pass it to reuse residuals that
    were already computed. Otherwise keyword arguments go to `residualize`
    (features, models, folds, seed).
    """
    residuals, outcome_r2, treatment_r2 = residualized or residualize(df, **kwargs)
    groups = residuals[GROUP].to_numpy()
    theta, se, low, high = final_stage(residuals["y"].to_numpy(), residuals["t"].to_numpy(), groups)
    return DMLResult(
        estimate=theta,
        std_error=se,
        ci_low=low,
        ci_high=high,
        n_obs=len(residuals),
        n_groups=len(np.unique(groups)),
        outcome_r2=outcome_r2,
        treatment_r2=treatment_r2,
    )


def estimate_by_segment(
    df: pd.DataFrame, segment: str = "category", residualized: tuple | None = None, **kwargs
) -> pd.DataFrame:
    """Elasticity for each value of `segment` (e.g. each product category).

    The nuisance models are fitted once on all the data; the final stage is then run
    separately on each segment's residuals. This is the same as a DML model where the
    elasticity is allowed to differ by segment (a separate θ per category), and it uses
    all products to learn the confounder effects, so small categories still get good
    nuisance models. `residualized` works as in `estimate_elasticity`.
    """
    df = df.reset_index(drop=True)
    residuals, _, _ = residualized or residualize(df, **kwargs)
    rows = []
    for value, part in residuals.groupby(df[segment], observed=True):
        groups = part[GROUP].to_numpy()
        theta, se, low, high = final_stage(part["y"].to_numpy(), part["t"].to_numpy(), groups)
        rows.append({
            "segment": segment,
            "value": value,
            "estimate": theta,
            "std_error": se,
            "ci_low": low,
            "ci_high": high,
            "n_obs": len(part),
            "n_groups": len(np.unique(groups)),
        })
    return pd.DataFrame(rows).sort_values("estimate").reset_index(drop=True)
