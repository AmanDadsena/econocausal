import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

from econocausal.crossfit import cross_fit_predict, make_group_folds


class MemorisingModel(BaseEstimator, RegressorMixin):
    """Predicts the training target for rows it has seen, and -1 for unseen rows.
    Used to prove that no row is predicted by a model trained on it."""

    def fit(self, X, y):
        self.seen_ = dict(zip(X["row_id"], y))
        return self

    def predict(self, X):
        return np.array([self.seen_.get(r, -1.0) for r in X["row_id"]])


def test_each_group_is_in_exactly_one_fold():
    groups = pd.Series(np.repeat(["a", "b", "c", "d", "e", "f", "g"], 4))
    folds = make_group_folds(groups, n_folds=3)
    assert pd.DataFrame({"g": groups, "f": folds}).groupby("g")["f"].nunique().eq(1).all()
    assert set(folds) == {0, 1, 2}


def test_folds_are_balanced_and_reproducible():
    groups = pd.Series(np.repeat(np.arange(50), 3))
    folds = make_group_folds(groups, n_folds=5, seed=1)
    groups_per_fold = pd.Series(folds).groupby(groups).first().value_counts()
    assert groups_per_fold.tolist() == [10] * 5
    assert np.array_equal(folds, make_group_folds(groups, n_folds=5, seed=1))
    assert not np.array_equal(folds, make_group_folds(groups, n_folds=5, seed=2))


def test_too_few_groups_raises():
    with pytest.raises(ValueError, match="at least 5 groups"):
        make_group_folds(pd.Series(["a", "b", "c"]), n_folds=5)


def test_no_row_is_predicted_by_a_model_that_saw_it():
    n = 60
    X = pd.DataFrame({"row_id": np.arange(n)})
    y = np.arange(n, dtype=float) + 100
    folds = make_group_folds(pd.Series(np.repeat(np.arange(12), 5)), n_folds=4)
    predictions, _ = cross_fit_predict(MemorisingModel(), X, y, folds)
    assert (predictions == -1.0).all()


def test_out_of_fold_r2_on_a_learnable_target():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=500)})
    y = 2 * X["x"] + rng.normal(scale=0.5, size=500)
    folds = make_group_folds(pd.Series(np.arange(500) // 5), n_folds=5)
    predictions, r2 = cross_fit_predict(LinearRegression(), X, y, folds)
    assert 0.9 < r2 < 0.97  # true R² is 4 / (4 + 0.25) = 0.94
    assert predictions.shape == (500,)
