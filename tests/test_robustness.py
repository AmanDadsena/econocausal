import pytest

from econocausal.dml import estimate_elasticity
from econocausal.robustness import data_subsets, placebo_treatment, random_common_cause, run_all
from econocausal.simulate import TRUE_ELASTICITY, simulate_panel
from test_dml import fast_model

FAST = dict(model_y=fast_model(), model_t=fast_model())


@pytest.fixture(scope="module")
def synthetic():
    return simulate_panel(n_products=150, n_weeks=104, seed=1)


@pytest.fixture(scope="module")
def original(synthetic):
    return estimate_elasticity(synthetic, **FAST)


def test_placebo_estimate_is_close_to_zero(synthetic):
    row = placebo_treatment(synthetic, **FAST)
    assert row["passed"]
    assert abs(row["estimate"]) < 0.1


def test_placebo_keeps_each_products_prices(synthetic):
    # The shuffle must only reorder prices within a product.
    import numpy as np

    rng = np.random.default_rng(0)
    shuffled = synthetic.groupby("stock_code")["log_price"].transform(lambda s: rng.permutation(s.to_numpy()))
    before = synthetic.groupby("stock_code")["log_price"].apply(sorted)
    after = shuffled.groupby(synthetic["stock_code"]).apply(sorted)
    assert before.equals(after)


def test_random_common_cause_barely_moves_the_estimate(synthetic, original):
    row = random_common_cause(synthetic, original, **FAST)
    assert row["passed"]
    assert abs(row["change"]) < 0.05


def test_subsets_stay_close_to_truth(synthetic, original):
    rows = data_subsets(synthetic, original, n_subsets=2, **FAST)
    assert len(rows) == 2
    for row in rows:
        assert row["n_obs"] == pytest.approx(len(synthetic) / 2, rel=0.01)
        assert row["estimate"] == pytest.approx(TRUE_ELASTICITY, abs=0.2)


def test_run_all_lists_every_check(synthetic, original):
    table = run_all(synthetic, original=original, **FAST)
    assert table["check"].tolist() == [
        "original", "placebo_treatment", "random_common_cause", "subset_1", "subset_2", "subset_3",
    ]
    assert table["passed"].all()
