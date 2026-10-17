"""Cross-fitting: out-of-fold predictions for the DML nuisance models.

In DML the outcome and treatment models must not predict rows they were trained on.
An overfitted model would "explain" part of the noise in its own training rows, the
residuals would be too small, and the final elasticity would be biased. Cross-fitting
avoids this: the data is split into K folds, a model is trained on K-1 folds and
predicts the held-out fold, and this rotates until every row has a prediction from a
model that never saw it.

Folds are formed by **product**, so all weeks of a product fall in the same fold. Weeks
of the same product are correlated, and splitting them across folds would let a model
learn a product's own noise.
"""

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone


def make_group_folds(groups: pd.Series, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Assign each row a fold number (0..n_folds-1) so that each group sits in one fold.

    Groups are shuffled with `seed` and then dealt out round-robin, which gives folds
    with an equal number of groups.
    """
    unique = pd.unique(groups)
    if len(unique) < n_folds:
        raise ValueError(f"Need at least {n_folds} groups for {n_folds} folds, got {len(unique)}")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    fold_of_group = pd.Series(np.arange(len(shuffled)) % n_folds, index=shuffled)
    return fold_of_group.loc[groups].to_numpy()


def cross_fit_predict(
    model: RegressorMixin, X: pd.DataFrame, y: pd.Series, folds: np.ndarray
) -> tuple[np.ndarray, float]:
    """Out-of-fold predictions of `y` from `X`.

    Returns the predictions and the out-of-fold R², which shows how much of `y` the
    confounders explain.
    """
    y = np.asarray(y, dtype=float)
    predictions = np.empty(len(y))
    for fold in np.unique(folds):
        test = folds == fold
        fitted = clone(model).fit(X[~test], y[~test])
        predictions[test] = fitted.predict(X[test])

    r2 = 1 - np.sum((y - predictions) ** 2) / np.sum((y - y.mean()) ** 2)
    return predictions, float(r2)
