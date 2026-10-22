def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert {"products", "weekly_sales", "results", "segment_elasticity"} <= set(body["tables"])


def test_categories(client):
    r = client.get("/categories")
    assert r.status_code == 200
    assert {c["category"] for c in r.json()} == {"bags", "candles_lighting", "christmas"}
    assert all(c["n_products"] == 1 for c in r.json())


def test_products_sorted_by_revenue_with_paging(client):
    r = client.get("/products", params={"limit": 2})
    body = r.json()
    assert r.status_code == 200
    assert body["total"] == 3 and len(body["items"]) == 2
    revenues = [p["total_revenue"] for p in body["items"]]
    assert revenues == sorted(revenues, reverse=True)

    page2 = client.get("/products", params={"limit": 2, "offset": 2}).json()
    assert len(page2["items"]) == 1


def test_products_filter_by_category_and_search(client):
    bags = client.get("/products", params={"category": "bags"}).json()
    assert [p["stock_code"] for p in bags["items"]] == ["22386"]

    found = client.get("/products", params={"search": "heart"}).json()
    assert [p["stock_code"] for p in found["items"]] == ["85123A"]


def test_search_is_not_executed_as_sql(client):
    r = client.get("/products", params={"search": "x' OR '1'='1"})
    assert r.status_code == 200
    assert r.json()["total"] == 0


def test_products_rejects_bad_limit(client):
    assert client.get("/products", params={"limit": 0}).status_code == 422
    assert client.get("/products", params={"limit": 1000}).status_code == 422


def test_single_product_is_case_insensitive(client):
    r = client.get("/products/85123a")
    assert r.status_code == 200
    assert r.json()["description"] == "WHITE HANGING HEART T-LIGHT HOLDER"


def test_unknown_product_returns_404(client):
    assert client.get("/products/NOPE").status_code == 404
    assert client.get("/products/NOPE/sales").status_code == 404


def test_product_sales_history(client):
    r = client.get("/products/22386/sales")
    assert r.status_code == 200
    weeks = r.json()
    assert [w["week_start"] for w in weeks] == ["2010-03-01", "2010-03-08", "2010-03-15"]
    assert weeks[0]["quantity"] == 100


def test_elasticity_real_lists_methods_in_order(client):
    r = client.get("/elasticity")
    assert r.status_code == 200
    rows = r.json()
    assert [row["method"] for row in rows] == ["product_fe", "dml_manual"]
    assert all(row["dataset"] == "real" for row in rows)
    assert rows[1]["estimate"] == -1.28
    assert rows[1]["true_value"] is None


def test_elasticity_synthetic_includes_true_value(client):
    rows = client.get("/elasticity", params={"dataset": "synthetic"}).json()
    assert rows[0]["true_value"] == -1.8


def test_elasticity_rejects_unknown_dataset(client):
    assert client.get("/elasticity", params={"dataset": "other"}).status_code == 422


def test_category_elasticities_most_elastic_first(client):
    rows = client.get("/elasticity/categories").json()
    assert [r["value"] for r in rows] == ["candles_lighting", "bags", "christmas"]
    assert client.get("/elasticity/categories", params={"method": "nope"}).status_code == 404


def test_product_elasticity_uses_its_category(client):
    r = client.get("/products/22386/elasticity")
    assert r.status_code == 200
    body = r.json()
    assert body["category"] == "bags"
    assert body["elasticity"] == -1.37
    assert body["level"] == "category"
    assert body["overall_elasticity"] == -1.28
    assert body["interpretation"] == "A 1% price increase lowers weekly units sold by about 1.37%."


def test_product_elasticity_prefers_product_level_estimate(client):
    body = client.get("/products/85123A/elasticity").json()
    assert body["level"] == "product"
    assert body["elasticity"] == -1.62
    assert (body["ci_low"], body["ci_high"]) == (-1.80, -1.44)


def test_product_elasticity_unknown_product(client):
    assert client.get("/products/NOPE/elasticity").status_code == 404


def test_missing_tables_return_503(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from api.main import app, engine

    monkeypatch.setenv("ECONOCAUSAL_DB", str(tmp_path / "empty.db"))
    engine.cache_clear()
    r = TestClient(app).get("/products")
    engine.cache_clear()
    assert r.status_code == 503
    assert "Run the data and model pipelines" in r.json()["detail"]
