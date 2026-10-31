"""Refutation tests for the DML elasticity.

Each check changes the data in a way where we know what a trustworthy estimator should do:

    placebo_treatment     Replace each product's prices with a random reshuffle of its own
                          prices. The shuffled "price" cannot cause anything, so the estimate
                          should be close to 0.
    random_common_cause   Add a column of pure noise as an extra confounder. Nothing real
                          changes, so the estimate should barely move.
    data_subsets          Re-estimate on random halves of the products. The estimate should
                          stay close to the full-data value.

Passing these does not prove the estimate is causal (nothing can rule out an unmeasured
confounder), but failing one would show a problem with the method or the data.

Usage:
    python -m econocausal.robustness
"""

import numpy as np
import pandas as pd

from econocausal.dml import CANDIDATE_FEATURES, GROUP, TREATMENT, DMLResult, estimate_elasticity

# How close a re-estimate must be to count as "stable", in elasticity units.
STABILITY_TOLERANCE = 0.1


def _row(check: str, result: DMLResult, expected: str, passed: bool, **extra) -> dict:
    return {
        "check": check,
        "estimate": result.estimate,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "n_obs": result.n_obs,
        "expected": expected,
        "passed": bool(passed),
        **extra,
    }


def placebo_treatment(df: pd.DataFrame, seed: int = 0, **kwargs) -> dict:
    """Shuffle prices within each product and re-estimate. Expected: about 0.

    Shuffling within a product keeps each product's price level and spread, so only the
    link between a week's price and that week's sales is broken."""
    rng = np.random.default_rng(seed)
    shuffled = df.copy()
    shuffled[TREATMENT] = shuffled.groupby(GROUP)[TREATMENT].transform(
        lambda s: rng.permutation(s.to_numpy())
    )
    result = estimate_elasticity(shuffled, seed=seed, **kwargs)
    return _row("placebo_treatment", result, "close to 0 (CI contains 0)",
                result.ci_low <= 0 <= result.ci_high)


def random_common_cause(df: pd.DataFrame, original: DMLResult, seed: int = 0, **kwargs) -> dict:
    """Add a pure-noise confounder and re-estimate. Expected: about the original value."""
    rng = np.random.default_rng(seed)
    noisy = df.assign(random_noise=rng.normal(size=len(df)))
    features = [f for f in CANDIDATE_FEATURES if f in df.columns] + ["random_noise"]
    result = estimate_elasticity(noisy, features=features, seed=seed, **kwargs)
    change = result.estimate - original.estimate
    return _row("random_common_cause", result, f"within {STABILITY_TOLERANCE} of {original.estimate:.3f}",
                abs(change) <= STABILITY_TOLERANCE, change=change)


def data_subsets(
    df: pd.DataFrame, original: DMLResult, n_subsets: int = 3, fraction: float = 0.5,
    seed: int = 0, **kwargs,
) -> list[dict]:
    """Re-estimate on random subsets of products. Expected: about the original value."""
    rng = np.random.default_rng(seed)
    products = df[GROUP].unique()
    rows = []
    for i in range(n_subsets):
        keep = rng.choice(products, size=int(len(products) * fraction), replace=False)
        result = estimate_elasticity(df[df[GROUP].isin(keep)], seed=seed, **kwargs)
        change = result.estimate - original.estimate
        rows.append(_row(f"subset_{i + 1}", result,
                         f"within {STABILITY_TOLERANCE} of {original.estimate:.3f}",
                         abs(change) <= STABILITY_TOLERANCE, change=change))
    return rows


def run_all(df: pd.DataFrame, original: DMLResult | None = None, seed: int = 0, **kwargs) -> pd.DataFrame:
    """Run every check and return one row per check (the original estimate first)."""
    original = original or estimate_elasticity(df, seed=seed, **kwargs)
    rows = [
        _row("original", original, "-", True),
        placebo_treatment(df, seed=seed, **kwargs),
        random_common_cause(df, original, seed=seed, **kwargs),
        *data_subsets(df, original, seed=seed, **kwargs),
    ]
    return pd.DataFrame(rows)


def main() -> None:
    """Run every check on the real panel; save to the `robustness` table and
    reports/robustness_real.csv."""
    import time
    import warnings

    from econocausal.data.db import get_engine, load_panel, upsert_table
    from econocausal.paths import PROJECT_ROOT

    warnings.filterwarnings("ignore", category=UserWarning)
    panel = load_panel()
    start = time.time()
    results = run_all(panel).assign(dataset="real")
    print(f"Robustness checks done in {time.time() - start:.0f}s\n")

    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        print(results[["check", "estimate", "ci_low", "ci_high", "expected", "passed"]].to_string(index=False))

    results.round(4).to_csv(PROJECT_ROOT / "reports" / "robustness_real.csv", index=False)
    upsert_table(results, "robustness", ["dataset", "check"], get_engine())
    print("\nSaved to reports/robustness_real.csv and the `robustness` table")


if __name__ == "__main__":
    main()
