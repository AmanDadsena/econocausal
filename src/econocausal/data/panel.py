"""Turn cleaned transactions into a product x week panel.

One row = one product in one week (weeks start on Monday). This is the level at
which elasticity is estimated: weekly price (treatment) vs weekly quantity (outcome),
plus calendar and product features used as confounders.
"""

import re

import numpy as np
import pandas as pd

# Products need enough history and enough price changes to estimate anything.
MIN_WEEKS = 30
MIN_PRICE_LEVELS = 5

# UK (England) bank holidays in the data period, 2009-12 to 2011-12.
UK_BANK_HOLIDAYS = pd.to_datetime(
    [
        "2009-12-25", "2009-12-28",
        "2010-01-01", "2010-04-02", "2010-04-05", "2010-05-03", "2010-05-31",
        "2010-08-30", "2010-12-27", "2010-12-28",
        "2011-01-03", "2011-04-22", "2011-04-25", "2011-04-29", "2011-05-02",
        "2011-05-30", "2011-08-29",
    ]
)

# Keyword rules for a rough product category. The first matching rule wins,
# so seasonal categories come before generic ones (a Christmas candle is "christmas").
CATEGORY_KEYWORDS = {
    "christmas": ["CHRISTMAS", "XMAS", "ADVENT", "SANTA", "REINDEER", "SNOWFLAKE", "BAUBLE"],
    "easter": ["EASTER", "CHICK", "BUNNY"],
    "candles_lighting": ["CANDLE", "CANDLES", "T-LIGHT", "TLIGHT", "LANTERN", "LIGHT", "LIGHTS",
                         "LAMP", "INCENSE"],
    "jewellery": ["NECKLACE", "BRACELET", "EARRINGS", "RING", "CHARM", "BEAD", "DIAMANTE"],
    "bags": ["BAG", "BAGS", "PURSE", "TOTE", "SHOPPER"],
    "kitchen_dining": ["MUG", "CUP", "CUPS", "BOWL", "PLATE", "PLATES", "TEA", "CAKE", "JAR",
                       "BOTTLE", "TRAY", "LUNCH", "SPOON", "NAPKINS", "BAKING", "CUTLERY"],
    "stationery_cards": ["CARD", "CARDS", "PAPER", "WRAP", "NOTEBOOK", "PENCIL", "PENCILS",
                         "PEN", "STICKERS", "TAPE", "RIBBON", "RIBBONS"],
    "toys_kids": ["TOY", "DOLL", "DOLLY", "GAME", "JIGSAW", "PLAYHOUSE", "SPACEBOY",
                  "CHILDREN", "CHILDS", "KIDS"],
    "garden": ["GARDEN", "PLANT", "WATERING", "PLANTER"],
    "home_decor": ["FRAME", "MIRROR", "CLOCK", "CUSHION", "SIGN", "WALL", "DOORMAT", "HOOK",
                   "DRAWER", "VASE", "GARLAND", "DECORATION", "ORNAMENT", "HANGING"],
}


def assign_category(description: str | None) -> str:
    """Map a product description to a category using keyword rules."""
    if not isinstance(description, str):
        return "other"
    words = set(re.findall(r"[A-Z]+(?:-[A-Z]+)?", description.upper()))
    for category, keywords in CATEGORY_KEYWORDS.items():
        if words.intersection(keywords):
            return category
    return "other"


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions to one row per product per week.

    Weekly price is the **median** unit price over the week's invoice lines. The
    median is used instead of revenue / quantity because large wholesale orders get
    lower unit prices; a quantity-weighted price would partly be caused by the
    quantity itself, which is the reverse of the effect we want to measure.
    """
    df = df.assign(week_start=df["invoice_date"].dt.to_period("W-SUN").dt.start_time)

    weekly = (
        df.groupby(["stock_code", "week_start"])
        .agg(
            description=("description", "first"),
            price=("price", "median"),
            quantity=("quantity", "sum"),
            revenue=("revenue", "sum"),
            n_invoices=("invoice", "nunique"),
            n_customers=("customer_id", "nunique"),
            uk_share=("country", lambda s: (s == "United Kingdom").mean()),
        )
        .reset_index()
    )

    # The last week in the data is cut off (data ends on a Friday), so drop it.
    last_date = df["invoice_date"].max().normalize()
    last_week = weekly["week_start"].max()
    if last_date < last_week + pd.Timedelta(days=6):
        weekly = weekly[weekly["week_start"] < last_week]

    return weekly


def filter_products(
    weekly: pd.DataFrame, min_weeks: int = MIN_WEEKS, min_price_levels: int = MIN_PRICE_LEVELS
) -> pd.DataFrame:
    """Keep products with enough weeks of sales and enough different weekly prices."""
    stats = weekly.groupby("stock_code").agg(
        n_weeks=("week_start", "size"),
        n_prices=("price", lambda p: p.round(2).nunique()),
    )
    keep = stats[(stats["n_weeks"] >= min_weeks) & (stats["n_prices"] >= min_price_levels)].index
    return weekly[weekly["stock_code"].isin(keep)].reset_index(drop=True)


def weeks_to_christmas(week_start: pd.Series) -> pd.Series:
    """Number of weeks until the next Christmas Day."""
    christmas = pd.to_datetime(week_start.dt.year.astype(str) + "-12-25")
    passed = week_start > christmas
    christmas = christmas.where(~passed, christmas + pd.DateOffset(years=1))
    return ((christmas - week_start).dt.days // 7).astype("int64")


def add_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """Add model variables, calendar features and product category."""
    df = weekly.copy()
    df["log_price"] = np.log(df["price"])
    df["log_quantity"] = np.log(df["quantity"])

    week = df["week_start"]
    df["year"] = week.dt.year
    df["month"] = week.dt.month
    df["week_of_year"] = week.dt.isocalendar().week.astype("int64")
    df["weeks_to_christmas"] = weeks_to_christmas(week)
    df["is_peak_season"] = df["month"].isin([9, 10, 11]).astype("int64")

    week_end = week + pd.Timedelta(days=6)
    df["has_bank_holiday"] = [
        int(((UK_BANK_HOLIDAYS >= s) & (UK_BANK_HOLIDAYS <= e)).any())
        for s, e in zip(week, week_end)
    ]

    categories = {code: assign_category(desc) for code, desc in
                  df.groupby("stock_code")["description"].first().items()}
    df["category"] = df["stock_code"].map(categories)
    return df


def build_panel(clean: pd.DataFrame) -> pd.DataFrame:
    """Full step: cleaned transactions -> filtered product x week panel with features."""
    panel = add_features(filter_products(aggregate_weekly(clean)))
    return panel.sort_values(["stock_code", "week_start"]).reset_index(drop=True)
