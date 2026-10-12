"""Load the raw Online Retail II data and store it as parquet.

The Excel file has two sheets (2009-2010 and 2010-2011). Reading it takes a
couple of minutes, so it is converted once to parquet and every later step
reads the parquet file instead.

Usage:
    python -m econocausal.data.load
"""

import pandas as pd

from econocausal.paths import RAW_DIR

EXCEL_PATH = RAW_DIR / "online_retail_II.xlsx"
PARQUET_PATH = RAW_DIR / "online_retail_ii.parquet"

COLUMN_NAMES = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to snake_case and fix data types.

    Excel stores some invoice numbers and stock codes as numbers and others as
    text (e.g. 'C489449' for cancellations, '85123A'), so both are forced to
    strings. Customer ID is missing for some rows, so it uses a nullable int.
    """
    missing = set(COLUMN_NAMES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    df = df.rename(columns=COLUMN_NAMES)[list(COLUMN_NAMES.values())].copy()
    df["invoice"] = df["invoice"].astype(str).str.strip()
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    df["description"] = df["description"].astype("string").str.strip()
    df["quantity"] = df["quantity"].astype("int64")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["price"] = df["price"].astype("float64")
    df["customer_id"] = df["customer_id"].astype("Int64")
    df["country"] = df["country"].astype("string").str.strip()
    return df


def read_excel() -> pd.DataFrame:
    """Read both sheets of the Excel file into one DataFrame."""
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"{EXCEL_PATH} not found. Run `python -m econocausal.data.download` first."
        )
    sheets = pd.read_excel(EXCEL_PATH, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    return pd.concat(sheets.values(), ignore_index=True)


def build_parquet() -> pd.DataFrame:
    """Read the Excel file, standardize it and save it as parquet."""
    df = standardize(read_excel())
    df.to_parquet(PARQUET_PATH, index=False)
    print(f"Saved {len(df):,} rows to {PARQUET_PATH}")
    return df


def load_raw() -> pd.DataFrame:
    """Return the raw transactions, building the parquet file if needed."""
    if not PARQUET_PATH.exists():
        return build_parquet()
    return pd.read_parquet(PARQUET_PATH)


if __name__ == "__main__":
    build_parquet()
