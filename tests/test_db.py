import pandas as pd
import pytest
from sqlalchemy import inspect

from econocausal.data.db import get_engine, load_panel, read_table, save_panel


@pytest.fixture
def engine(tmp_path):
    return get_engine(tmp_path / "test.db")


def make_panel():
    return pd.DataFrame(
        {
            "stock_code": ["A", "A", "B"],
            "week_start": pd.to_datetime(["2010-03-01", "2010-03-08", "2010-03-01"]),
            "description": ["RED MUG", "RED MUG", "JUMBO BAG"],
            "category": ["kitchen_dining", "kitchen_dining", "bags"],
            "price": [2.0, 2.5, 1.0],
            "quantity": [10, 6, 30],
            "revenue": [20.0, 15.0, 30.0],
        }
    )


def test_save_and_load_panel_round_trip(engine):
    panel = make_panel()
    log = pd.DataFrame({"step": ["drop_cancellations"], "rows_removed": [5], "rows_left": [3]})
    save_panel(panel, log, engine)

    loaded = load_panel(engine)
    pd.testing.assert_frame_equal(loaded, panel, check_dtype=False)
    assert set(inspect(engine).get_table_names()) == {"weekly_sales", "products", "cleaning_log"}


def test_products_summary(engine):
    save_panel(make_panel(), pd.DataFrame({"step": [], "rows_removed": [], "rows_left": []}), engine)
    products = read_table("products", engine).set_index("stock_code")
    assert products.loc["A", "n_weeks"] == 2
    assert products.loc["A", "total_quantity"] == 16
    assert products.loc["A", "max_price"] == 2.5


def test_read_missing_table_gives_clear_error(engine):
    with pytest.raises(ValueError, match="not found"):
        read_table("weekly_sales", engine)
