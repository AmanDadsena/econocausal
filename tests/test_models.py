import pytest

from econocausal.dml import estimate_elasticity
from econocausal.models import estimate_elasticity_econml
from econocausal.simulate import TRUE_ELASTICITY, simulate_panel

from test_dml import fast_model


@pytest.fixture(scope="module")
def synthetic():
    return simulate_panel(n_products=150, n_weeks=104, seed=1)


@pytest.fixture(scope="module")
def econml_result(synthetic):
    return estimate_elasticity_econml(synthetic, model_y=fast_model(), model_t=fast_model())


def test_econml_recovers_true_elasticity(econml_result):
    assert econml_result.estimate == pytest.approx(TRUE_ELASTICITY, abs=0.1)
    assert econml_result.ci_low < econml_result.estimate < econml_result.ci_high


def test_econml_agrees_with_manual_dml(synthetic, econml_result):
    manual = estimate_elasticity(synthetic, model_y=fast_model(), model_t=fast_model())
    assert econml_result.estimate == pytest.approx(manual.estimate, abs=0.05)


def test_econml_result_fields(synthetic, econml_result):
    assert econml_result.n_obs == len(synthetic)
    assert econml_result.n_groups == synthetic["stock_code"].nunique()
    assert 0 < econml_result.outcome_r2 < 1
    assert 0 < econml_result.treatment_r2 < 1
