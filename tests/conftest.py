import pandas as pd
import pytest
from fastapi.testclient import TestClient

from econocausal.data.db import (
    get_engine,
    save_panel,
    save_product_results,
    save_results,
    save_segment_results,
)


def make_panel() -> pd.DataFrame:
    weeks = pd.to_datetime(["2010-03-01", "2010-03-08", "2010-03-15"])
    rows = []
    for code, desc, cat, base in [
        ("22386", "JUMBO BAG PINK POLKADOT", "bags", 1.95),
        ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", "candles_lighting", 2.55),
        ("22086", "PAPER CHAIN KIT 50'S CHRISTMAS", "christmas", 2.95),
    ]:
        for i, week in enumerate(weeks):
            price = base * (1 + 0.1 * i)
            quantity = 100 - 20 * i
            rows.append({
                "stock_code": code, "week_start": week, "description": desc, "category": cat,
                "price": price, "quantity": quantity, "revenue": price * quantity,
                "guest_share": 0.1 * i,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def api_db(tmp_path, monkeypatch):
    """A small SQLite database with every table the API reads, used through
    the ECONOCAUSAL_DB setting."""
    path = tmp_path / "api.db"
    engine = get_engine(path)
    save_panel(make_panel(), pd.DataFrame({"step": ["x"], "rows_removed": [0], "rows_left": [9]}), engine)
    save_results(
        pd.DataFrame({
            "dataset": ["real", "real", "synthetic"],
            "method": ["dml_manual", "product_fe", "dml_manual"],
            "estimate": [-1.28, -1.54, -1.81],
            "std_error": [0.02, 0.02, 0.02],
            "ci_low": [-1.32, -1.58, -1.85],
            "ci_high": [-1.23, -1.50, -1.76],
            "n_obs": [126672, 126672, 31200],
            "true_value": [None, None, -1.8],
        }),
        engine,
    )
    save_segment_results(
        pd.DataFrame({
            "dataset": "real", "method": "dml_manual", "segment": "category",
            "value": ["bags", "candles_lighting", "christmas"],
            "estimate": [-1.37, -1.53, -1.11], "std_error": [0.09, 0.07, 0.1],
            "ci_low": [-1.55, -1.66, -1.31], "ci_high": [-1.18, -1.39, -0.91],
            "n_obs": [7330, 10085, 2812], "n_groups": [94, 157, 60],
        }),
        engine,
    )
    # Product-level (causal forest) estimate for one product only, so both the
    # product-level and the category fallback paths of the API can be tested.
    save_product_results(
        pd.DataFrame({
            "dataset": ["real"], "method": ["causal_forest"], "stock_code": ["85123A"],
            "description": ["WHITE HANGING HEART T-LIGHT HOLDER"], "category": ["candles_lighting"],
            "estimate": [-1.62], "ci_low": [-1.80], "ci_high": [-1.44], "n_obs": [3],
        }),
        engine,
    )
    monkeypatch.setenv("ECONOCAUSAL_DB", str(path))
    return path


@pytest.fixture
def client(api_db):
    from api.main import app, engine

    engine.cache_clear()
    with TestClient(app) as c:
        yield c
    engine.cache_clear()
