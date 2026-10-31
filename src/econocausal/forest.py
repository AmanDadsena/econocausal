"""Product-level price elasticities with a causal forest.

The forest is fitted on the **same cross-fitted, product-demeaned residuals** as the manual
DML in dml.py, so it stays consistent with that estimate. Instead of one elasticity θ, it
learns θ(X): how the elasticity changes with product characteristics X ("effect
modifiers"). It uses econml's generalized random forest, which splits on X where the
relationship between the price residual and the quantity residual differs most, and gives
honest confidence intervals (each tree uses separate samples to choose splits and to
estimate effects).

Effect modifiers are product-level, so every product gets exactly one estimate:
category, the product's typical (median) log price, and its average guest and UK shares.
"""

import numpy as np
import pandas as pd
from econml.grf import CausalForest

from econocausal.dml import GROUP, residualize


def effect_modifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Product-level features (constant within a product) the elasticity may depend on."""
    by_product = df.groupby(GROUP)
    X = pd.get_dummies(df["category"], prefix="category", dtype=float)
    X["log_price_level"] = by_product["log_price"].transform("median")
    for col in ["guest_share", "uk_share"]:
        if col in df:
            X[f"{col}_mean"] = by_product[col].transform("mean")
    return X


def default_forest(seed: int = 0) -> CausalForest:
    """Causal forest used on the real data: 400 honest trees, at least 50 rows per leaf
    (stable estimates for products with about 100 weeks), confidence intervals enabled."""
    return CausalForest(
        n_estimators=400,
        min_samples_leaf=50,
        max_samples=0.45,
        honest=True,
        inference=True,
        random_state=seed,
        n_jobs=4,
    )


def product_elasticities(
    df: pd.DataFrame,
    residuals: pd.DataFrame | None = None,
    forest: CausalForest | None = None,
    level: float = 0.95,
    **residualize_kwargs,
) -> tuple[pd.DataFrame, pd.Series]:
    """Elasticity with a confidence interval for every product.

    Parameters
    ----------
    df : modelling panel.
    residuals : output of `dml.residualize(df)`; computed here if not given, so the
        expensive ML step can be shared with the other estimators.
    forest : an unfitted econml CausalForest; `default_forest()` if not given.
    level : confidence level of the intervals.

    Returns
    -------
    A DataFrame with one row per product and the forest's feature importances.
    """
    df = df.reset_index(drop=True)
    if residuals is None:
        residuals, _, _ = residualize(df, **residualize_kwargs)

    X = effect_modifiers(df)
    # Explicit None check: an unfitted econml forest raises an error if used as a bool.
    if forest is None:
        forest = default_forest()
    forest.fit(X.to_numpy(), residuals["t"].to_numpy(), residuals["y"].to_numpy())

    products = X.groupby(df[GROUP]).first()
    estimate, low, high = forest.predict(products.to_numpy(), interval=True, alpha=1 - level)

    info = df.groupby(GROUP).agg(
        description=("description", "first") if "description" in df else ("category", "first"),
        category=("category", "first"),
        n_obs=(GROUP, "size"),
    )
    result = info.assign(
        estimate=estimate.ravel(), ci_low=low.ravel(), ci_high=high.ravel()
    ).reset_index()
    if "description" not in df:
        result["description"] = None
    importances = pd.Series(forest.feature_importances_, index=X.columns).sort_values(ascending=False)
    return result, importances


def summarise_by(products: pd.DataFrame, column: str) -> pd.DataFrame:
    """Average product elasticity per group, weighted by the number of product-weeks."""
    rows = []
    for value, part in products.groupby(column, observed=True):
        weights = part["n_obs"].to_numpy()
        rows.append({
            "value": value,
            "estimate": float(np.average(part["estimate"], weights=weights)),
            "n_products": len(part),
            "share_significant": float((part["ci_high"] < 0).mean()),
        })
    return pd.DataFrame(rows).sort_values("estimate").reset_index(drop=True)
