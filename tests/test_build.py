"""End-to-end test of the data build step (clean -> weekly panel -> SQLite) on a small
made-up set of raw transactions."""

import pandas as pd
import pytest

from econocausal.data import build
from econocausal.data.db import get_engine, read_table
from econocausal.data.panel import MIN_PRICE_LEVELS, MIN_WEEKS, build_panel

PRICES = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]


def raw_transactions() -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2010-01-04 10:00")  # a Monday
    for week in range(40):
        day = start + pd.Timedelta(weeks=week)
        # Product A: sold every week at changing prices -> kept.
        rows.append((f"5{week:05d}", "21232", "RED MUG", 10, day, PRICES[week % len(PRICES)], 1, "United Kingdom"))
        # Product B: only 10 weeks of sales -> filtered out.
        if week < 10:
            rows.append((f"6{week:05d}", "22386", "JUMBO BAG", 5, day, PRICES[week % len(PRICES)], 2, "France"))
        # Postage line: not a product -> removed by cleaning.
        rows.append((f"5{week:05d}", "POST", "POSTAGE", 1, day, 18.0, 1, "United Kingdom"))
    # A cancellation -> removed by cleaning.
    rows.append(("C500000", "21232", "RED MUG", -10, start, 1.0, 1, "United Kingdom"))
    return pd.DataFrame(rows, columns=["invoice", "stock_code", "description", "quantity",
                                       "invoice_date", "price", "customer_id", "country"])


def test_build_panel_keeps_only_products_with_enough_history():
    from econocausal.data.clean import clean_transactions

    clean, _ = clean_transactions(raw_transactions())
    panel = build_panel(clean)
    assert panel["stock_code"].unique().tolist() == ["21232"]
    assert len(panel) >= MIN_WEEKS
    assert panel["price"].round(2).nunique() >= MIN_PRICE_LEVELS
    assert panel["week_start"].is_monotonic_increasing


def test_build_main_writes_all_tables(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "built.db"
    monkeypatch.setattr(build, "load_raw", raw_transactions)
    monkeypatch.setattr(build, "get_engine", lambda: get_engine(db_path))
    monkeypatch.setattr(build, "DB_PATH", db_path)

    build.main()

    engine = get_engine(db_path)
    panel = read_table("weekly_sales", engine)
    products = read_table("products", engine)
    log = read_table("cleaning_log", engine).set_index("step")
    assert products["stock_code"].tolist() == ["21232"]
    assert set(panel["stock_code"]) == {"21232"}
    assert log.loc["drop_cancellations", "rows_removed"] == 1
    assert log.loc["keep_product_codes", "rows_removed"] == 40
    assert "Saved to" in capsys.readouterr().out


def test_build_main_needs_raw_data(monkeypatch):
    def no_data():
        raise FileNotFoundError("run the download step first")

    monkeypatch.setattr(build, "load_raw", no_data)
    with pytest.raises(FileNotFoundError, match="download"):
        build.main()
