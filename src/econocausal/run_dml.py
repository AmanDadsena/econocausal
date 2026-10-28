"""Estimate price elasticities and store them.

On the synthetic data (known elasticity) the own DML and EconML are run as a check of the
method; their results feed the dashboard's "synthetic data" comparison.

On the real data, the expensive step (cross-fitting the two LightGBM nuisance models) runs once; its
residuals are shared by the average DML estimate, the per-category estimates and the
causal forest's per-product estimates. EconML LinearDML runs separately as a check.

Results go to the `results`, `segment_elasticity` and `product_elasticity` tables and to
CSV files in reports/.

Usage:
    python -m econocausal.run_dml
"""

import time
import warnings

import pandas as pd

from econocausal.data.db import (
    get_engine,
    load_panel,
    save_product_results,
    save_results,
    save_segment_results,
)
from econocausal.dml import estimate_by_segment, estimate_elasticity, residualize
from econocausal.forest import product_elasticities
from econocausal.models import estimate_elasticity_econml
from econocausal.paths import PROJECT_ROOT
from econocausal.simulate import simulate_panel

REPORTS = PROJECT_ROOT / "reports"


def timed(label: str, func, *args, **kwargs):
    start = time.time()
    out = func(*args, **kwargs)
    print(f"{label:<28} done in {time.time() - start:.0f}s")
    return out


def run_synthetic(engine) -> pd.DataFrame:
    """Own DML and EconML on the default synthetic panel (true elasticity -1.8)."""
    synthetic = simulate_panel()
    truth = synthetic.attrs["true_elasticity"]
    rows = []
    for method, fit in [("dml_manual", estimate_elasticity), ("dml_econml", estimate_elasticity_econml)]:
        result = timed(f"Synthetic {method}", fit, synthetic).to_dict()
        rows.append({"dataset": "synthetic", "method": method, **result,
                     "true_value": truth, "error": result["estimate"] - truth})
    table = pd.DataFrame(rows)
    save_results(table, engine)
    table.round(4).to_csv(REPORTS / "dml_results_synthetic.csv", index=False)
    for _, row in table.iterrows():
        print(f"synthetic {row['method']:<11} {row['estimate']:.3f}  "
              f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]  (true {truth:g})")
    return table


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    engine = get_engine()
    run_synthetic(engine)

    panel = load_panel().reset_index(drop=True)
    print(f"Real panel: {len(panel):,} rows, {panel['stock_code'].nunique():,} products")

    residualized = timed("Cross-fitted residuals", residualize, panel)

    # Average elasticity: own DML (shared residuals) and EconML.
    manual = estimate_elasticity(panel, residualized=residualized).to_dict()
    econml = timed("EconML LinearDML", estimate_elasticity_econml, panel).to_dict()
    overall = pd.DataFrame([
        {"dataset": "real", "method": "dml_manual", **manual},
        {"dataset": "real", "method": "dml_econml", **econml},
    ])
    save_results(overall, engine)
    overall.round(4).to_csv(REPORTS / "dml_results_real.csv", index=False)

    # Per category: separate final stage on the shared residuals.
    by_category = estimate_by_segment(panel, "category", residualized=residualized)
    by_category = by_category.assign(dataset="real", method="dml_manual")
    save_segment_results(by_category, engine)
    by_category.round(4).to_csv(REPORTS / "category_elasticities_real.csv", index=False)

    # Per product: causal forest on the shared residuals.
    products, importances = timed(
        "Causal forest", product_elasticities, panel, residuals=residualized[0]
    )
    products = products.assign(dataset="real", method="causal_forest")
    save_product_results(products, engine)
    products.round(4).to_csv(REPORTS / "product_elasticities_real.csv", index=False)
    importances.round(4).rename("importance").to_csv(REPORTS / "forest_feature_importance.csv")

    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 120):
        for _, row in overall.iterrows():
            print(f"{row['method']:<11} {row['estimate']:.3f}  [{row['ci_low']:.3f}, {row['ci_high']:.3f}]")
        print("\nElasticity by category (manual DML):")
        print(by_category[["value", "estimate", "ci_low", "ci_high", "n_groups"]].to_string(index=False))
        print(f"\nProduct elasticities (causal forest): {len(products):,} products, "
              f"median {products['estimate'].median():.3f}, "
              f"range {products['estimate'].min():.3f} to {products['estimate'].max():.3f}")
        print("Feature importance:", importances.head(4).round(3).to_dict())
    print("\nSaved to reports/ and the results / segment_elasticity / product_elasticity tables")


if __name__ == "__main__":
    main()
