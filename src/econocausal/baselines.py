"""Naive OLS baselines for the price elasticity.

These are the estimates a DML model has to beat. Each one regresses log quantity on
log price with a different set of controls:

    pooled                        no controls
    product_fe                    product fixed effects (each product vs its own average)
    product_week_fe               product + week fixed effects
    product_week_fe_controls      product + week fixed effects + guest_share, uk_share (linear)

Fixed effects are removed by demeaning (Frisch-Waugh-Lovell): the slope from regressing
the demeaned outcome on the demeaned price equals the slope from the full regression with
dummy variables. For two sets of fixed effects on an unbalanced panel, demeaning by product
and then by week once is not exact, so `demean` alternates between the two until the
change is negligible.

Standard errors are clustered by product. They do not correct the degrees of freedom for
the absorbed fixed effects, which makes them very slightly too small; with ~100 weeks per
product the difference is negligible.

Usage:
    python -m econocausal.baselines
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

OUTCOME = "log_quantity"
TREATMENT = "log_price"
LINEAR_CONTROLS = ["guest_share", "uk_share"]

SPECS = {
    "pooled": {"fixed_effects": [], "controls": []},
    "product_fe": {"fixed_effects": ["stock_code"], "controls": []},
    "product_week_fe": {"fixed_effects": ["stock_code", "week_start"], "controls": []},
    "product_week_fe_controls": {
        "fixed_effects": ["stock_code", "week_start"],
        "controls": LINEAR_CONTROLS,
    },
}


def demean(
    df: pd.DataFrame, columns: list[str], groups: list[str], tol: float = 1e-10, max_iter: int = 1000
) -> pd.DataFrame:
    """Remove group means for one or more grouping variables.

    With one group this is a single pass. With several groups it alternates between them
    (method of alternating projections) until the largest change is below `tol`.
    """
    out = df[columns].astype("float64").copy()
    if not groups:
        return out - out.mean()

    keys = [df[g] for g in groups]
    for _ in range(max_iter):
        largest_change = 0.0
        for key in keys:
            means = out.groupby(key).transform("mean")
            out -= means
            largest_change = max(largest_change, float(np.abs(means.to_numpy()).max()))
        if len(groups) == 1 or largest_change < tol:
            break
    return out


def fit_baseline(df: pd.DataFrame, fixed_effects: list[str], controls: list[str]) -> dict:
    """Fit one OLS specification and return the price coefficient with its uncertainty."""
    columns = [OUTCOME, TREATMENT, *controls]
    resid = demean(df, columns, fixed_effects)

    X = resid[[TREATMENT, *controls]]
    if not fixed_effects:
        X = sm.add_constant(df[[TREATMENT, *controls]].astype("float64"))
        y = df[OUTCOME].astype("float64")
    else:
        y = resid[OUTCOME]

    fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df["stock_code"])[0]})
    low, high = fit.conf_int().loc[TREATMENT]
    return {
        "estimate": float(fit.params[TREATMENT]),
        "std_error": float(fit.bse[TREATMENT]),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_obs": int(fit.nobs),
    }


def run_baselines(df: pd.DataFrame, dataset: str, true_value: float | None = None) -> pd.DataFrame:
    """Fit every baseline specification on one dataset."""
    rows = []
    for method, spec in SPECS.items():
        result = fit_baseline(df, spec["fixed_effects"], spec["controls"])
        rows.append({"dataset": dataset, "method": method, **result})
    results = pd.DataFrame(rows)
    results["true_value"] = np.nan if true_value is None else float(true_value)
    results["error"] = results["estimate"] - results["true_value"]
    return results


def main() -> None:
    """Fit every baseline on the synthetic and the real panel; save to the `results`
    table and reports/baseline_results.csv."""
    from econocausal.data.db import get_engine, load_panel, save_results
    from econocausal.paths import PROJECT_ROOT
    from econocausal.simulate import simulate_panel

    synthetic = simulate_panel()
    results = pd.concat(
        [
            run_baselines(synthetic, "synthetic", true_value=synthetic.attrs["true_elasticity"]),
            run_baselines(load_panel(), "real"),
        ],
        ignore_index=True,
    )

    out = PROJECT_ROOT / "reports" / "baseline_results.csv"
    results.round(4).to_csv(out, index=False)
    save_results(results, get_engine())

    with pd.option_context("display.width", 120, "display.float_format", "{:.3f}".format):
        print(results.drop(columns=["true_value"]).to_string(index=False))
    print(f"\nSaved to {out} and the `results` table")


if __name__ == "__main__":
    main()
