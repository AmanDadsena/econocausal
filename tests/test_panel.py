import numpy as np
import pandas as pd

from econocausal.data.panel import (
    add_features,
    aggregate_weekly,
    assign_category,
    filter_products,
    weeks_to_christmas,
)


def make_line(code, date, qty, price, invoice="1", customer=1, country="United Kingdom"):
    return {
        "invoice": invoice,
        "stock_code": code,
        "description": "RED MUG",
        "quantity": qty,
        "invoice_date": pd.Timestamp(date),
        "price": price,
        "customer_id": customer,
        "country": country,
        "revenue": qty * price,
    }


def test_aggregate_weekly_uses_median_price_and_sums_quantity():
    lines = [
        make_line("A", "2010-03-01", 10, 1.0, invoice="1"),   # Monday
        make_line("A", "2010-03-03", 100, 0.5, invoice="2"),  # bulk order, lower price
        make_line("A", "2010-03-04", 5, 1.0, invoice="3", customer=None, country="France"),
        make_line("A", "2010-03-15", 1, 1.0),                 # keeps the week above complete
    ]
    weekly = aggregate_weekly(pd.DataFrame(lines))
    first = weekly.iloc[0]
    assert first["week_start"] == pd.Timestamp("2010-03-01")
    assert first["price"] == 1.0          # median, not pulled down by the bulk order
    assert first["quantity"] == 115
    assert first["n_invoices"] == 3
    assert np.isclose(first["uk_share"], 2 / 3)
    assert np.isclose(first["guest_share"], 1 / 3)


def test_aggregate_weekly_drops_incomplete_last_week():
    lines = [
        make_line("A", "2010-03-01", 1, 1.0),
        make_line("A", "2010-03-09", 1, 1.0),  # Tuesday: last week is incomplete
    ]
    weekly = aggregate_weekly(pd.DataFrame(lines))
    assert weekly["week_start"].tolist() == [pd.Timestamp("2010-03-01")]


def test_filter_products_needs_weeks_and_price_variation():
    weeks = pd.date_range("2010-01-04", periods=6, freq="W-MON")
    varied = pd.DataFrame({"stock_code": "A", "week_start": weeks, "price": [1, 1.1, 1.2, 1.3, 1.4, 1.5]})
    flat = pd.DataFrame({"stock_code": "B", "week_start": weeks, "price": 2.0})
    short = pd.DataFrame({"stock_code": "C", "week_start": weeks[:3], "price": [1, 2, 3]})
    weekly = pd.concat([varied, flat, short])
    kept = filter_products(weekly, min_weeks=5, min_price_levels=5)
    assert kept["stock_code"].unique().tolist() == ["A"]


def test_weeks_to_christmas():
    weeks = pd.Series(pd.to_datetime(["2010-12-20", "2010-12-27", "2010-11-29"]))
    assert weeks_to_christmas(weeks).tolist() == [0, 51, 3]


def test_assign_category():
    assert assign_category("PAPER CHAIN KIT 50'S CHRISTMAS") == "christmas"
    assert assign_category("WHITE HANGING HEART T-LIGHT HOLDER") == "candles_lighting"
    assert assign_category("JUMBO BAG RED RETROSPOT") == "bags"
    assert assign_category("SOMETHING UNUSUAL") == "other"
    assert assign_category(None) == "other"


def test_add_features_creates_model_columns():
    weekly = pd.DataFrame(
        {
            "stock_code": ["A"] * 3,
            "week_start": pd.to_datetime(["2010-12-20", "2010-12-27", "2011-04-25"]),
            "description": ["RED MUG"] * 3,
            "price": [2.0, 2.5, 2.2],
            "quantity": [10, 20, 15],
        }
    )
    df = add_features(weekly)
    assert np.isclose(df["log_price"].iloc[0], np.log(2.0))
    # Christmas 2010 fell on a Saturday, so the holidays moved to 27-28 Dec.
    assert df["has_bank_holiday"].tolist() == [0, 1, 1]
    assert df["category"].iloc[0] == "kitchen_dining"
