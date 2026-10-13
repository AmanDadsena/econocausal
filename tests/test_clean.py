import pandas as pd

from econocausal.data.clean import (
    clean_transactions,
    drop_cancellations,
    drop_non_positive,
    drop_price_outliers,
    fill_descriptions,
    keep_product_codes,
)


def make_rows(**overrides):
    row = {
        "invoice": "489434",
        "stock_code": "21232",
        "description": "STRAWBERRY CERAMIC TRINKET BOX",
        "quantity": 12,
        "invoice_date": pd.Timestamp("2010-03-01 10:00"),
        "price": 1.25,
        "customer_id": 13085,
        "country": "United Kingdom",
    }
    row.update(overrides)
    return row


def test_drop_cancellations():
    df = pd.DataFrame([make_rows(), make_rows(invoice="C489449"), make_rows(invoice="A506401")])
    assert drop_cancellations(df)["invoice"].tolist() == ["489434"]


def test_drop_non_positive():
    df = pd.DataFrame([make_rows(), make_rows(quantity=-3), make_rows(price=0.0)])
    assert len(drop_non_positive(df)) == 1


def test_keep_product_codes():
    codes = ["21232", "85123A", "15056BL", "POST", "DOT", "M", "DCGS0058", "GIFT_0001_20"]
    df = pd.DataFrame([make_rows(stock_code=c) for c in codes])
    assert keep_product_codes(df)["stock_code"].tolist() == ["21232", "85123A", "15056BL"]


def test_drop_price_outliers_keeps_normal_discounts():
    prices = [2.0, 2.0, 1.5, 2.5, 30.0]  # 30.0 is 15x the median
    df = pd.DataFrame([make_rows(price=p) for p in prices])
    assert drop_price_outliers(df)["price"].tolist() == [2.0, 2.0, 1.5, 2.5]


def test_fill_descriptions_uses_most_frequent():
    df = pd.DataFrame(
        [
            make_rows(description="RED MUG"),
            make_rows(description="RED MUG"),
            make_rows(description="red mug "),
            make_rows(description=None),
        ]
    )
    df["description"] = df["description"].astype("string")
    assert fill_descriptions(df)["description"].tolist() == ["RED MUG"] * 4


def test_clean_transactions_logs_every_step():
    df = pd.DataFrame([make_rows(), make_rows(invoice="C1"), make_rows(stock_code="POST")])
    df["description"] = df["description"].astype("string")
    cleaned, log = clean_transactions(df)
    assert len(cleaned) == 1
    assert "revenue" in cleaned.columns
    assert log["rows_removed"].sum() == 2
