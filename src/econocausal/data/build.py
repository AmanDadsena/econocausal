"""Run the whole data pipeline: raw data -> cleaning -> weekly panel -> SQLite.

Usage:
    python -m econocausal.data.build
"""

from econocausal.data.clean import clean_transactions
from econocausal.data.db import DB_PATH, get_engine, save_panel
from econocausal.data.load import load_raw
from econocausal.data.panel import build_panel


def main() -> None:
    raw = load_raw()
    print(f"Raw transactions:   {len(raw):>9,}")

    clean, log = clean_transactions(raw)
    print(log.to_string(index=False))
    print(f"Clean transactions: {len(clean):>9,}")

    panel = build_panel(clean)
    print(
        f"Weekly panel:       {len(panel):>9,} rows, "
        f"{panel['stock_code'].nunique():,} products, "
        f"{panel['week_start'].min():%Y-%m-%d} to {panel['week_start'].max():%Y-%m-%d}"
    )

    save_panel(panel, log, get_engine())
    print(f"Saved to {DB_PATH}")


if __name__ == "__main__":
    main()
