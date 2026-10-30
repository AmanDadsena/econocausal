"""Tests for the pricing endpoints: /optimal-price, /simulate and /recommendations.

Test data (tests/conftest.py): product 22386 sold 100, 80 and 60 units at £1.95, £2.145
and £2.34 in three weeks, so its current price is the median £2.145 and its current weekly
units are 240 / 13. It has no product-level estimate, so it uses the bags category
elasticity (−1.37). Product 85123A has a product-level estimate (−1.62).
"""

import pandas as pd
import pytest

from econocausal.data.db import get_engine

P0 = 2.145
Q0 = 240 / 13
THETA = -1.37


# --- /optimal-price ---------------------------------------------------------------

def test_optimal_price_default_rules(client):
    r = client.post("/optimal-price", json={"stock_code": "22386"})
    assert r.status_code == 200
    b = r.json()
    assert b["current_price"] == pytest.approx(P0)
    assert b["current_units"] == pytest.approx(Q0)
    assert b["unit_cost"] == pytest.approx(0.6 * P0)
    assert (b["elasticity"], b["elasticity_level"]) == (THETA, "category")
    # θ = −1.37 puts the profit optimum far above the price, so the +20% limit decides.
    assert b["recommended_price"] == pytest.approx(P0 * 1.2)
    assert b["reason"] == "upper price limit"
    assert b["expected_units"] == pytest.approx(Q0 * 1.2**THETA)
    assert b["expected_profit"] == pytest.approx((P0 * 1.2 - 0.6 * P0) * Q0 * 1.2**THETA)
    assert b["window"] == "2009-12-21 to 2010-03-21"


def test_optimal_price_uses_product_level_elasticity(client):
    b = client.post("/optimal-price", json={"stock_code": "85123a"}).json()
    assert (b["elasticity"], b["elasticity_level"]) == (-1.62, "product")
    assert (b["elasticity_ci_low"], b["elasticity_ci_high"]) == (-1.80, -1.44)


def test_optimal_price_custom_rules(client):
    b = client.post("/optimal-price", json={"stock_code": "22386", "max_change": 0.1}).json()
    assert b["recommended_price"] == pytest.approx(P0 * 1.1)

    b = client.post("/optimal-price", json={"stock_code": "22386", "objective": "revenue"}).json()
    # Elastic demand (θ < −1): revenue is highest at the lowest allowed price.
    assert b["recommended_price"] == pytest.approx(P0 * 0.8)
    assert b["reason"] == "lower price limit"


def test_optimal_price_infeasible_rules(client):
    b = client.post(
        "/optimal-price", json={"stock_code": "22386", "cost_ratio": 0.95, "min_margin": 0.9}
    ).json()
    assert b["recommended_price"] is None
    assert b["reason"].startswith("infeasible")


@pytest.mark.parametrize(
    "body",
    [
        {"stock_code": "22386", "max_change": 0.9},          # above the 50% maximum
        {"stock_code": "22386", "max_change": 0},            # must be positive
        {"stock_code": "22386", "cost_ratio": -0.1},
        {"stock_code": "22386", "objective": "sales"},
        {"stock_code": "22386", "unexpected": 1},            # unknown fields are rejected
        {"stock_code": "22386'; DROP TABLE products; --"},  # only letters and digits allowed
        {"stock_code": ""},
        {},
    ],
)
def test_optimal_price_rejects_invalid_input(client, body):
    assert client.post("/optimal-price", json=body).status_code == 422


def test_optimal_price_unknown_product(client):
    assert client.post("/optimal-price", json={"stock_code": "NOPE"}).status_code == 404


def test_product_without_recent_sales_cannot_be_priced(client, api_db):
    engine = get_engine(api_db)
    old = pd.DataFrame({
        "stock_code": ["OLD1"], "week_start": [pd.Timestamp("2009-06-01")], "description": ["OLD ITEM"],
        "category": ["bags"], "price": [1.0], "quantity": [5], "revenue": [5.0], "guest_share": [0.0],
    })
    old.to_sql("weekly_sales", engine, if_exists="append", index=False)
    pd.DataFrame({
        "stock_code": ["OLD1"], "description": ["OLD ITEM"], "category": ["bags"], "n_weeks": [1],
        "n_prices": [1], "median_price": [1.0], "min_price": [1.0], "max_price": [1.0],
        "total_quantity": [5], "total_revenue": [5.0],
    }).to_sql("products", engine, if_exists="append", index=False)

    r = client.post("/optimal-price", json={"stock_code": "OLD1"})
    assert r.status_code == 422
    assert "no sales in the last 13 weeks" in r.json()["detail"]


# --- /simulate --------------------------------------------------------------------------

def test_simulate_returns_one_row_per_price(client):
    r = client.post("/simulate", json={"stock_code": "22386", "prices": [P0, 2.5]})
    assert r.status_code == 200
    results = r.json()["results"]
    assert [row["price"] for row in results] == [P0, 2.5]

    at_current, higher = results
    assert at_current["expected_units"] == pytest.approx(Q0)
    assert at_current["price_change_pct"] == pytest.approx(0)
    assert higher["expected_units"] == pytest.approx(Q0 * (2.5 / P0) ** THETA)
    assert higher["expected_revenue"] == pytest.approx(2.5 * higher["expected_units"])
    assert higher["expected_profit"] == pytest.approx((2.5 - 0.6 * P0) * higher["expected_units"])
    assert higher["margin_pct"] == pytest.approx(100 * (2.5 - 0.6 * P0) / 2.5)
    assert higher["units_low"] <= higher["expected_units"] <= higher["units_high"]


@pytest.mark.parametrize(
    "body",
    [
        {"stock_code": "22386", "prices": []},
        {"stock_code": "22386", "prices": [1.0] * 51},
        {"stock_code": "22386", "prices": [-1.0]},
        {"stock_code": "22386", "prices": [0]},
        {"stock_code": "22386", "prices": ["cheap"]},
        {"stock_code": "22386"},
    ],
)
def test_simulate_rejects_invalid_input(client, body):
    assert client.post("/simulate", json=body).status_code == 422


# --- /recommendations ---------------------------------------------------------------------

def test_recommendations_sorted_by_profit_gain(client):
    body = client.get("/recommendations").json()
    assert body["total"] == 2
    assert [i["stock_code"] for i in body["items"]] == ["22086", "22386"]  # gains 6.0 vs 1.5


def test_recommendations_filters_and_sorting(client):
    bags = client.get("/recommendations", params={"category": "bags"}).json()
    assert [i["stock_code"] for i in bags["items"]] == ["22386"]
    by_code = client.get("/recommendations", params={"sort": "stock_code"}).json()
    assert [i["stock_code"] for i in by_code["items"]] == ["22086", "22386"]
    none = client.get("/recommendations", params={"reason": "minimum margin"}).json()
    assert none["total"] == 0


def test_recommendations_reject_unknown_sort(client):
    assert client.get("/recommendations", params={"sort": "price; DROP TABLE x"}).status_code == 422


@pytest.mark.parametrize("price", [P0 * 0.19, P0 * 5.01])
def test_simulate_rejects_prices_far_from_the_current_price(client, price):
    r = client.post("/simulate", json={"stock_code": "22386", "prices": [P0, price]})
    assert r.status_code == 422
    assert "only meaningful near prices" in r.json()["detail"]


def test_simulate_accepts_the_edges_of_the_allowed_range(client):
    r = client.post("/simulate", json={"stock_code": "22386", "prices": [P0 * 0.2, P0 * 5.0]})
    assert r.status_code == 200


@pytest.mark.parametrize("raw", ['{"stock_code": "22386", "prices": [NaN]}',
                                 '{"stock_code": "22386", "prices": [Infinity]}',
                                 '{"stock_code": "22386", "cost_ratio": NaN}'])
def test_nan_and_infinity_are_rejected(client, raw):
    path = "/simulate" if "prices" in raw else "/optimal-price"
    r = client.post(path, content=raw, headers={"Content-Type": "application/json"})
    assert r.status_code == 422
