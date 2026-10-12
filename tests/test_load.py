import pandas as pd
import pytest

from econocausal.data.load import standardize


def make_raw_rows():
    return pd.DataFrame(
        {
            "Invoice": [489434, "C489449"],
            "StockCode": [21232, "85123a "],
            "Description": [" STRAWBERRY CERAMIC TRINKET BOX", "WHITE HANGING HEART T-LIGHT HOLDER"],
            "Quantity": [12, -6],
            "InvoiceDate": ["2009-12-01 07:45:00", "2009-12-01 10:33:00"],
            "Price": [1.25, 2.55],
            "Customer ID": [13085.0, None],
            "Country": ["United Kingdom", "Australia"],
        }
    )


def test_standardize_renames_columns():
    df = standardize(make_raw_rows())
    assert list(df.columns) == [
        "invoice",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "price",
        "customer_id",
        "country",
    ]


def test_standardize_fixes_types():
    df = standardize(make_raw_rows())
    assert df["invoice"].tolist() == ["489434", "C489449"]
    assert df["stock_code"].tolist() == ["21232", "85123A"]
    assert df["description"].iloc[0] == "STRAWBERRY CERAMIC TRINKET BOX"
    assert pd.api.types.is_datetime64_any_dtype(df["invoice_date"])
    assert str(df["customer_id"].dtype) == "Int64"
    assert df["customer_id"].isna().iloc[1]


def test_standardize_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing expected columns"):
        standardize(make_raw_rows().drop(columns=["Price"]))
