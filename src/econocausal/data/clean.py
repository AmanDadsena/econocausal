"""Clean raw transactions so that only real product sales remain.

Each step is a small function so it can be tested on its own. `clean_transactions`
runs them in order and returns a log of how many rows each step removed.
The reasoning behind each rule is in docs/data_dictionary.md.
"""

import pandas as pd

# Real products have a 5-digit code, optionally followed by letters for variants
# (e.g. 85123A). Everything else is postage, fees, vouchers, samples or adjustments.
PRODUCT_CODE_PATTERN = r"^\d{5}[A-Z]*$"

# A unit price more than this many times above or below the product's median
# price is treated as a data entry error (e.g. £26 for a glass that sells for £2).
MAX_PRICE_RATIO = 10.0

# Invoice lines above this quantile are one-off bulk orders (the largest, 80,995
# units, was cancelled straight away). They would dominate weekly totals.
QUANTITY_QUANTILE = 0.9999


def drop_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    """Remove cancelled invoices (C...) and accounting adjustments (A...)."""
    return df[~df["invoice"].str.startswith(("C", "A"))]


def drop_non_positive(df: pd.DataFrame) -> pd.DataFrame:
    """Remove lines with zero/negative quantity or price (write-offs, free items)."""
    return df[(df["quantity"] > 0) & (df["price"] > 0)]


def keep_product_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only stock codes that look like real products."""
    return df[df["stock_code"].str.match(PRODUCT_CODE_PATTERN)]


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove lines that are identical in every column (double-scanned lines)."""
    return df.drop_duplicates()


def drop_price_outliers(df: pd.DataFrame, max_ratio: float = MAX_PRICE_RATIO) -> pd.DataFrame:
    """Remove lines whose price is far from the product's usual price.

    The bound is wide on purpose: normal discounts and bulk prices must stay,
    because that price variation is exactly what the elasticity is estimated from.
    """
    ratio = df["price"] / df.groupby("stock_code")["price"].transform("median")
    return df[(ratio <= max_ratio) & (ratio >= 1 / max_ratio)]


def drop_bulk_outliers(df: pd.DataFrame, quantile: float = QUANTITY_QUANTILE) -> pd.DataFrame:
    """Remove extremely large single invoice lines."""
    limit = df["quantity"].quantile(quantile)
    return df[df["quantity"] <= limit]


def fill_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Give every stock code one description: the most frequent one.

    Some codes have missing descriptions or several spellings of the same name.
    """
    canonical = (
        df.dropna(subset=["description"])
        .groupby("stock_code")["description"]
        .agg(lambda s: s.value_counts().index[0])
    )
    df = df.copy()
    df["description"] = df["stock_code"].map(canonical).astype("string")
    return df


CLEANING_STEPS = [
    drop_cancellations,
    drop_non_positive,
    keep_product_codes,
    drop_duplicates,
    drop_price_outliers,
    drop_bulk_outliers,
]


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all cleaning steps.

    Returns the cleaned data and a small table with the number of rows
    removed by each step.
    """
    log = []
    for step in CLEANING_STEPS:
        before = len(df)
        df = step(df)
        log.append({"step": step.__name__, "rows_removed": before - len(df), "rows_left": len(df)})

    df = fill_descriptions(df)
    df = df.assign(revenue=df["quantity"] * df["price"]).reset_index(drop=True)
    return df, pd.DataFrame(log)
