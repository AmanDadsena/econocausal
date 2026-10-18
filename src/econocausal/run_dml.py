"""Estimate price elasticities on the real data and store them.

Runs the manual DML and EconML LinearDML for the average elasticity, and the manual DML
per product category. Results go to the `results` and `segment_elasticity` tables and to
CSV files in reports/.

Usage:
    python -m econocausal.run_dml
"""

import time
import warnings

import pandas as pd

from econocausal.data.db import get_engine, load_panel, save_results, save_segment_results
from econocausal.dml import estimate_by_segment, estimate_elasticity
from econocausal.models import estimate_elasticity_econml
from econocausal.paths import PROJECT_ROOT

REPORTS = PROJECT_ROOT / "reports"


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    panel = load_panel()
    engine = get_engine()
    print(f"Real panel: {len(panel):,} rows, {panel['stock_code'].nunique():,} products")

    rows = []
    for method, fit in [("dml_manual", estimate_elasticity), ("dml_econml", estimate_elasticity_econml)]:
        start = time.time()
        result = fit(panel).to_dict()
        rows.append({"dataset": "real", "method": method, **result})
        print(f"{method:<11} {result['estimate']:.3f}  "
              f"[{result['ci_low']:.3f}, {result['ci_high']:.3f}]  ({time.time() - start:.0f}s)")

    overall = pd.DataFrame(rows)
    save_results(overall, engine)
    overall.round(4).to_csv(REPORTS / "dml_results_real.csv", index=False)

    by_category = estimate_by_segment(panel, "category").assign(dataset="real", method="dml_manual")
    save_segment_results(by_category, engine)
    by_category.round(4).to_csv(REPORTS / "category_elasticities_real.csv", index=False)

    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 120):
        print("\nElasticity by category (manual DML):")
        print(by_category[["value", "estimate", "ci_low", "ci_high", "n_groups", "n_obs"]]
              .to_string(index=False))
    print("\nSaved to reports/ and the results / segment_elasticity tables")


if __name__ == "__main__":
    main()
