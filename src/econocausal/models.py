"""EconML LinearDML, used as an independent check on the manual DML in dml.py.

Both use the same inputs so that any difference comes from the implementation, not the setup:
the same confounders, the same product demeaning, the same LightGBM settings and the same
product-based folds (passed to EconML as explicit train/test splits).

Two known differences:
* EconML's default inference uses heteroskedasticity-robust (HC1) standard errors, not
  standard errors clustered by product, so its intervals are usually somewhat narrower.
* The manual version demeans the residuals by product a second time; EconML does not.
"""

import numpy as np
import pandas as pd
from econml.dml import LinearDML

from econocausal.baselines import demean
from econocausal.crossfit import make_group_folds
from econocausal.dml import GROUP, OUTCOME, TREATMENT, DMLResult, build_features, default_model


def estimate_elasticity_econml(
    df: pd.DataFrame,
    features: list[str] | None = None,
    model_y=None,
    model_t=None,
    n_folds: int = 5,
    seed: int = 0,
) -> DMLResult:
    """Average price elasticity with EconML's LinearDML (product fixed effects removed first)."""
    df = df.reset_index(drop=True)
    # EconML needs a numeric matrix, so the category column is one-hot encoded.
    W = pd.get_dummies(build_features(df, features), columns=["category"], dtype=float)
    target = demean(df, [OUTCOME, TREATMENT], [GROUP])

    folds = make_group_folds(df[GROUP], n_folds=n_folds, seed=seed)
    splits = [(np.flatnonzero(folds != k), np.flatnonzero(folds == k)) for k in range(n_folds)]

    est = LinearDML(
        model_y=model_y or default_model(seed),
        model_t=model_t or default_model(seed),
        cv=splits,
        random_state=seed,
    )
    est.fit(target[OUTCOME].to_numpy(), target[TREATMENT].to_numpy(), W=W.to_numpy())

    low, high = est.intercept__interval(alpha=0.05)
    return DMLResult(
        estimate=float(est.intercept_),
        std_error=float(est.intercept__inference().stderr),
        ci_low=float(low),
        ci_high=float(high),
        n_obs=len(df),
        n_groups=df[GROUP].nunique(),
        # Mean out-of-fold R² across folds.
        outcome_r2=float(np.mean(est.nuisance_scores_y)),
        treatment_r2=float(np.mean(est.nuisance_scores_t)),
    )
