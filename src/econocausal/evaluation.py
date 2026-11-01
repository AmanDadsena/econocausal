"""Simulation study: are the 95% confidence intervals honest?

A 95% interval should contain the true value in about 95% of repeated samples. On real data
this cannot be checked, because the true elasticity is unknown. Here the synthetic generator
is run many times with different seeds (true elasticity -1.8 every time); for each dataset
the own DML estimator and the best naive OLS baseline are fitted, and we record whether each
interval contains -1.8.

Reported per method:
    coverage      share of intervals that contain the true value (target 0.95)
    bias          mean(estimate) - true value
    rmse          root mean squared error of the estimates
    mean_se       average reported standard error
    sd_estimates  standard deviation of the estimates across datasets
                  (if mean_se is close to sd_estimates, the standard errors are well calibrated)

Usage:
    python -m econocausal.evaluation              # 100 datasets
    python -m econocausal.evaluation --reps 20    # quicker
    python -m econocausal.evaluation --plot-only  # redraw the figure from saved results
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats

from econocausal.baselines import fit_baseline
from econocausal.dml import estimate_elasticity
from econocausal.paths import PROJECT_ROOT
from econocausal.simulate import TRUE_ELASTICITY, simulate_panel

REPORTS = PROJECT_ROOT / "reports"

# A smaller panel and lighter LightGBM settings than the main analysis keep 100 repetitions
# to a few minutes; each dataset still has 15,600 product-weeks.
STUDY_PANEL = {"n_products": 150, "n_weeks": 104}


def study_model(seed: int) -> LGBMRegressor:
    """LightGBM settings for the study (lighter than the main analysis)."""
    return LGBMRegressor(n_estimators=200, learning_rate=0.08, num_leaves=15,
                         min_child_samples=40, random_state=seed, n_jobs=4, verbose=-1)


def run_one(seed: int) -> list[dict]:
    """Fit own DML and the best OLS baseline on one synthetic dataset."""
    df = simulate_panel(seed=seed, **STUDY_PANEL)
    dml = estimate_elasticity(df, model_y=study_model(seed), model_t=study_model(seed), seed=seed)
    ols = fit_baseline(df, ["stock_code", "week_start"], ["guest_share", "uk_share"])
    rows = [
        {"seed": seed, "method": "dml_manual", "estimate": dml.estimate, "std_error": dml.std_error,
         "ci_low": dml.ci_low, "ci_high": dml.ci_high},
        {"seed": seed, "method": "ols_product_week_fe_controls", "estimate": ols["estimate"],
         "std_error": ols["std_error"], "ci_low": ols["ci_low"], "ci_high": ols["ci_high"]},
    ]
    for row in rows:
        row["covers"] = row["ci_low"] <= TRUE_ELASTICITY <= row["ci_high"]
    return rows


def summarise(runs: pd.DataFrame, truth: float = TRUE_ELASTICITY) -> pd.DataFrame:
    """Coverage, bias, RMSE and standard-error calibration per method.

    `coverage_ci_low/high` is a 95% Wilson interval for the coverage itself, so with a
    finite number of repetitions we can say whether coverage is consistent with 95%."""
    rows = []
    z = stats.norm.ppf(0.975)
    for method, part in runs.groupby("method", sort=False):
        n = len(part)
        p = part["covers"].mean()
        centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
        errors = part["estimate"] - truth
        rows.append({
            "method": method,
            "datasets": n,
            "coverage": p,
            "coverage_ci_low": centre - half,
            "coverage_ci_high": centre + half,
            "mean_estimate": part["estimate"].mean(),
            "bias": errors.mean(),
            "rmse": float(np.sqrt((errors**2).mean())),
            "mean_se": part["std_error"].mean(),
            "sd_estimates": part["estimate"].std(ddof=1),
        })
    return pd.DataFrame(rows)


def plot_study(runs: pd.DataFrame, truth: float = TRUE_ELASTICITY) -> None:
    """Save reports/figures/12_coverage_study.png: the spread of estimates for each method,
    and every DML interval sorted by its estimate, with intervals that miss the truth marked."""
    import matplotlib.pyplot as plt

    from econocausal.viz import BLUE, GREY, ORANGE, TEXT_MUTED, save_figure, set_style

    set_style()
    dml = runs[runs["method"] == "dml_manual"].sort_values("estimate").reset_index(drop=True)
    ols = runs[runs["method"] != "dml_manual"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))
    bins = np.linspace(min(runs["ci_low"].min(), truth) - 0.02, max(runs["ci_high"].max(), truth) + 0.02, 40)
    left.hist(ols["estimate"], bins=bins, color=GREY, label="Best naive OLS", alpha=0.9)
    left.hist(dml["estimate"], bins=bins, color=BLUE, label="Own DML", alpha=0.9)
    left.axvline(truth, color=TEXT_MUTED, linestyle="--", linewidth=1)
    left.text(truth, left.get_ylim()[1] * 0.95, f"  true {truth:g}", color=TEXT_MUTED, va="top")
    left.set_xlabel("Estimated elasticity")
    left.set_ylabel("Datasets")
    left.set_title(f"Estimates on {len(dml)} synthetic datasets")
    left.legend(loc="upper left")

    misses = ~dml["covers"]
    colors = np.where(misses, ORANGE, BLUE)
    y = np.arange(len(dml))
    right.hlines(y, dml["ci_low"], dml["ci_high"], colors=colors, linewidth=1.2)
    right.scatter(dml["estimate"], y, color=colors, s=6, zorder=3)
    right.axvline(truth, color=TEXT_MUTED, linestyle="--", linewidth=1)
    right.set_yticks([])
    right.set_xlabel("Own DML estimate with 95% interval")
    right.set_title(f"{(~misses).sum()} of {len(dml)} intervals contain the true value")
    right.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, "12_coverage_study")


def main(argv: list[str] | None = None) -> None:
    """Run the study, print the summary and save both tables to reports/."""
    parser = argparse.ArgumentParser(description="Confidence-interval coverage study.")
    parser.add_argument("--reps", type=int, default=100, help="number of synthetic datasets")
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw the figure from reports/coverage_study_runs.csv")
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore", category=UserWarning)
    if args.plot_only:
        plot_study(pd.read_csv(REPORTS / "coverage_study_runs.csv"))
        print("Saved reports/figures/12_coverage_study.png")
        return

    start = time.time()
    runs = []
    for i, seed in enumerate(range(args.reps), start=1):
        runs += run_one(seed)
        if i % 10 == 0:
            print(f"{i}/{args.reps} datasets ({time.time() - start:.0f}s)")
    runs = pd.DataFrame(runs)
    summary = summarise(runs)

    runs.round(6).to_csv(REPORTS / "coverage_study_runs.csv", index=False)
    summary.round(4).to_csv(REPORTS / "coverage_study_summary.csv", index=False)
    plot_study(runs)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        print(summary.to_string(index=False))
    print(f"\nDone in {time.time() - start:.0f}s; saved reports/coverage_study_*.csv")


if __name__ == "__main__":
    main()
