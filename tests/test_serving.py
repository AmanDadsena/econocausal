import sqlite3
from contextlib import closing

import pytest
from fastapi.testclient import TestClient

from econocausal.serving import SERVING_TABLES, export_serving_db


def table_columns(path, table):
    with closing(sqlite3.connect(path)) as conn:
        return [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]


def test_export_copies_only_the_api_tables(api_db, tmp_path):
    target = tmp_path / "serving.db"
    counts = export_serving_db(api_db, target)

    assert set(counts) == set(SERVING_TABLES)
    assert counts["products"] == 3 and counts["weekly_sales"] == 9
    with closing(sqlite3.connect(target)) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert tables == set(SERVING_TABLES)  # cleaning_log etc. are left out
    assert table_columns(target, "weekly_sales") == SERVING_TABLES["weekly_sales"]


def test_api_answers_are_identical_on_the_serving_db(api_db, tmp_path, monkeypatch):
    from api.main import app, engine

    calls = [
        ("get", "/products", None),
        ("get", "/products/22386/sales", None),
        ("get", "/elasticity", None),
        ("get", "/elasticity/categories", None),
        ("get", "/products/85123A/elasticity", None),
        ("get", "/recommendations", None),
        ("post", "/optimal-price", {"stock_code": "22386"}),
        ("post", "/simulate", {"stock_code": "22386", "prices": [2.0, 2.5]}),
    ]

    def answers():
        engine.cache_clear()
        client = TestClient(app)
        out = []
        for method, path, body in calls:
            r = client.post(path, json=body) if method == "post" else client.get(path)
            assert r.status_code == 200, path
            out.append(r.json())
        return out

    full = answers()
    serving = tmp_path / "serving.db"
    export_serving_db(api_db, serving)
    monkeypatch.setenv("ECONOCAUSAL_DB", str(serving))
    assert answers() == full
    engine.cache_clear()


def test_export_fails_clearly_when_tables_are_missing(tmp_path):
    source = tmp_path / "incomplete.db"
    with closing(sqlite3.connect(source)) as conn:
        conn.execute("CREATE TABLE products (stock_code TEXT)")
    with pytest.raises(ValueError, match="missing tables"):
        export_serving_db(source, tmp_path / "out.db")
    assert not (tmp_path / "out.db").exists()
    assert not (tmp_path / "out.tmp").exists()


def test_export_fails_clearly_without_source(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run the pipeline"):
        export_serving_db(tmp_path / "nope.db", tmp_path / "out.db")
