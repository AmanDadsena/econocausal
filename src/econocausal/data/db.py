"""SQLite storage for the cleaned data and, later, model results.

Tables written by the data pipeline:
    weekly_sales   product x week panel used for modelling
    products       one row per product with summary stats
    cleaning_log   rows removed by each cleaning step

Tables written by the models:
    results              one row per (dataset, method) elasticity estimate
    segment_elasticity   one row per (dataset, method, segment, value), e.g. per category
"""

from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, create_engine, inspect, text

from econocausal.paths import DATA_DIR

DB_PATH = DATA_DIR / "econocausal.db"


def get_engine(db_path: Path = DB_PATH) -> Engine:
    """Create a SQLAlchemy engine for the SQLite database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}")


def save_table(df: pd.DataFrame, name: str, engine: Engine) -> None:
    """Write a DataFrame to a table, replacing it if it already exists."""
    df.to_sql(name, engine, if_exists="replace", index=False, chunksize=10_000)


def read_table(name: str, engine: Engine, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Read a whole table into a DataFrame."""
    if not inspect(engine).has_table(name):
        raise ValueError(f"Table '{name}' not found. Run `python -m econocausal.data.build` first.")
    return pd.read_sql_table(name, engine, parse_dates=parse_dates)


def summarise_products(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per product with the stats shown in the API and dashboard."""
    return (
        panel.groupby("stock_code")
        .agg(
            description=("description", "first"),
            category=("category", "first"),
            n_weeks=("week_start", "size"),
            n_prices=("price", lambda p: p.round(2).nunique()),
            median_price=("price", "median"),
            min_price=("price", "min"),
            max_price=("price", "max"),
            total_quantity=("quantity", "sum"),
            total_revenue=("revenue", "sum"),
        )
        .reset_index()
    )


def save_panel(panel: pd.DataFrame, cleaning_log: pd.DataFrame, engine: Engine) -> None:
    """Store the panel, the product summary and the cleaning log."""
    save_table(panel, "weekly_sales", engine)
    save_table(summarise_products(panel), "products", engine)
    save_table(cleaning_log, "cleaning_log", engine)

    with engine.begin() as conn:
        conn.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_weekly_sales_product_week "
            "ON weekly_sales (stock_code, week_start)"
        ))
        conn.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_products_stock_code ON products (stock_code)"
        ))


RESULT_KEYS = ["dataset", "method"]
SEGMENT_KEYS = ["dataset", "method", "segment", "value"]


def upsert_table(df: pd.DataFrame, name: str, keys: list[str], engine: Engine) -> pd.DataFrame:
    """Add rows to a table; rows with the same key values replace the existing ones,
    so re-running a model updates its results instead of adding duplicates.
    Returns the full table after the update."""
    new = df.assign(created_at=pd.Timestamp.now().floor("s"))
    if inspect(engine).has_table(name):
        old = read_table(name, engine, parse_dates=["created_at"])
        old_keys = pd.MultiIndex.from_frame(old[keys].astype(str))
        new_keys = pd.MultiIndex.from_frame(new[keys].astype(str))
        new = pd.concat([old[~old_keys.isin(new_keys)], new], ignore_index=True)
    save_table(new, name, engine)
    return new


def save_results(results: pd.DataFrame, engine: Engine) -> pd.DataFrame:
    """Add overall estimates to the `results` table (key: dataset, method)."""
    return upsert_table(results, "results", RESULT_KEYS, engine)


def save_segment_results(results: pd.DataFrame, engine: Engine) -> pd.DataFrame:
    """Add per-segment estimates to `segment_elasticity` (key: dataset, method, segment, value)."""
    return upsert_table(results, "segment_elasticity", SEGMENT_KEYS, engine)


def load_panel(engine: Engine | None = None) -> pd.DataFrame:
    """Read the modelling panel back from the database."""
    return read_table("weekly_sales", engine or get_engine(), parse_dates=["week_start"])
