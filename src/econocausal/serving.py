"""Export a small, read-only "serving" database for the deployed API.

The full database (data/econocausal.db) is built by the pipeline and is not stored in git.
The API only needs a few tables and, from the weekly panel, a few columns, so this module
copies exactly those into deploy/econocausal_serving.db, which is small enough to commit.
A hosting service can then run the API straight from the repository, without the raw data
download or the machine-learning libraries.

Usage (after the pipeline has run):
    python -m econocausal.serving
"""

import sqlite3
from contextlib import closing
from pathlib import Path

from econocausal.data.db import DB_PATH
from econocausal.paths import PROJECT_ROOT

SERVING_DB_PATH = PROJECT_ROOT / "deploy" / "econocausal_serving.db"

# Bookkeeping columns the API never reads; left out so the file only changes when results do.
DROP_COLUMNS = {"created_at"}
# Floats are rounded so that differences in the last binary digits (from multi-threaded
# model fitting) do not change the file between otherwise identical runs.
FLOAT_DECIMALS = 10

# Table -> columns the API reads (None = all columns).
SERVING_TABLES: dict[str, list[str] | None] = {
    "products": None,
    "weekly_sales": ["stock_code", "week_start", "price", "quantity", "revenue", "guest_share"],
    "results": None,
    "segment_elasticity": None,
    "product_elasticity": None,
    "price_recommendations": None,
}

# Rows are written in a fixed order, so the file does not depend on the order in which the
# pipeline happened to save results (re-saving a result moves its rows to the end).
ORDER_BY = {
    "products": ["stock_code"],
    "weekly_sales": ["stock_code", "week_start"],
    "results": ["dataset", "method"],
    "segment_elasticity": ["dataset", "method", "segment", "value"],
    "product_elasticity": ["dataset", "method", "stock_code"],
    "price_recommendations": ["stock_code"],
}

INDEXES = [
    "CREATE UNIQUE INDEX ix_products_stock_code ON products (stock_code)",
    "CREATE UNIQUE INDEX ix_weekly_sales_product_week ON weekly_sales (stock_code, week_start)",
    "CREATE INDEX ix_product_elasticity_code ON product_elasticity (stock_code)",
]


def export_serving_db(source: Path = DB_PATH, target: Path = SERVING_DB_PATH) -> dict[str, int]:
    """Copy the API's tables from `source` into a fresh database at `target`.

    Returns the number of rows copied per table. Raises if a required table is missing.
    """
    if not source.exists():
        raise FileNotFoundError(f"{source} not found. Run the pipeline first (see README).")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)

    try:
        counts = {}
        # closing(): sqlite3's own context manager commits but does not close the file.
        with closing(sqlite3.connect(tmp)) as out:
            out.execute("ATTACH DATABASE ? AS src", (str(source),))
            existing = {row[0] for row in out.execute("SELECT name FROM src.sqlite_master WHERE type = 'table'")}
            missing = sorted(set(SERVING_TABLES) - existing)
            if missing:
                raise ValueError(f"Source database is missing tables {missing}. Run the full pipeline first.")
            for table, columns in SERVING_TABLES.items():
                # Table and column names come from the fixed SERVING_TABLES mapping and the
                # source schema, never from user input.
                types = {row[1]: (row[2] or "").upper()
                         for row in out.execute(f'PRAGMA src.table_info("{table}")')}
                names = [c for c in (columns or types) if c not in DROP_COLUMNS]
                cols = ", ".join(
                    f'ROUND("{c}", {FLOAT_DECIMALS}) AS "{c}"' if types[c] in {"FLOAT", "REAL", "DOUBLE"}
                    else f'"{c}"'
                    for c in names
                )
                order = ", ".join(f'"{c}"' for c in ORDER_BY[table])
                out.execute(f'CREATE TABLE "{table}" AS SELECT {cols} FROM src."{table}" ORDER BY {order}')
                counts[table] = out.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for statement in INDEXES:
                out.execute(statement)
            out.commit()
            out.execute("DETACH DATABASE src")
        with closing(sqlite3.connect(tmp)) as out:
            out.execute("VACUUM")
    except Exception:
        tmp.unlink(missing_ok=True)  # never leave a half-written file behind
        raise
    tmp.replace(target)
    return counts


def main() -> None:
    counts = export_serving_db()
    size_mb = SERVING_DB_PATH.stat().st_size / 1e6
    for table, n in counts.items():
        print(f"{table:<22} {n:>8,} rows")
    print(f"\nWrote {SERVING_DB_PATH.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
