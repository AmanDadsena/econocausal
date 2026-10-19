"""EconoCausal REST API.

Serves the cleaned product data and the elasticity estimates stored in SQLite by the
pipeline (`python -m econocausal.data.build`, `python -m econocausal.baselines`,
`python -m econocausal.run_dml`). The API is read-only.

Run locally:
    uvicorn api.main:app --reload
Interactive docs: http://localhost:8000/docs

The database location can be changed with the ECONOCAUSAL_DB environment variable.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import Engine, inspect, text

from econocausal.data.db import DB_PATH, get_engine

app = FastAPI(
    title="EconoCausal API",
    description="Price elasticities estimated with Double Machine Learning on the Online Retail II data.",
    version="0.1.0",
)


@lru_cache
def engine() -> Engine:
    return get_engine(Path(os.environ.get("ECONOCAUSAL_DB", DB_PATH)))


EngineDep = Annotated[Engine, Depends(engine)]


def query(db: Engine, sql: str, **params) -> pd.DataFrame:
    """Run a parameterised read query. Values are always passed as parameters, never
    formatted into the SQL string."""
    with db.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of dicts with NaN replaced by None (valid JSON)."""
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


def require_tables(db: Engine, *tables: str) -> None:
    missing = [t for t in tables if not inspect(db).has_table(t)]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Missing tables {missing}. Run the data and model pipelines first.",
        )


# --- response models -----------------------------------------------------------

class Health(BaseModel):
    status: str
    tables: list[str]


class Category(BaseModel):
    category: str
    n_products: int
    total_revenue: float


class Product(BaseModel):
    stock_code: str
    description: str | None
    category: str
    n_weeks: int
    n_prices: int
    median_price: float
    min_price: float
    max_price: float
    total_quantity: int
    total_revenue: float


class ProductList(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[Product]


class WeeklySales(BaseModel):
    week_start: str
    price: float
    quantity: int
    revenue: float
    guest_share: float


class Estimate(BaseModel):
    dataset: str
    method: str
    estimate: float
    std_error: float
    ci_low: float
    ci_high: float
    n_obs: int
    true_value: float | None = None
    outcome_r2: float | None = None
    treatment_r2: float | None = None


class SegmentEstimate(BaseModel):
    segment: str
    value: str
    estimate: float
    std_error: float
    ci_low: float
    ci_high: float
    n_obs: int
    n_groups: int


class ProductElasticity(BaseModel):
    stock_code: str
    description: str | None
    category: str
    elasticity: float
    ci_low: float
    ci_high: float
    level: str
    overall_elasticity: float
    interpretation: str


# --- endpoints ---------------------------------------------------------------------

@app.get("/health", response_model=Health, tags=["status"])
def health(db: EngineDep) -> Health:
    """Check the API is running and list the database tables it can see."""
    return Health(status="ok", tables=sorted(inspect(db).get_table_names()))


@app.get("/categories", response_model=list[Category], tags=["products"])
def categories(db: EngineDep) -> list[dict]:
    """Product categories with the number of products and their total revenue."""
    require_tables(db, "products")
    df = query(
        db,
        """
        SELECT category, COUNT(*) AS n_products, SUM(total_revenue) AS total_revenue
        FROM products GROUP BY category ORDER BY n_products DESC
        """,
    )
    return records(df)


@app.get("/products", response_model=ProductList, tags=["products"])
def products(
    db: EngineDep,
    category: str | None = None,
    search: Annotated[str | None, Query(max_length=100, description="Text to find in the description")] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict:
    """Products in the modelling panel, best sellers (by revenue) first."""
    require_tables(db, "products")
    where = "WHERE (:category IS NULL OR category = :category) " \
            "AND (:search IS NULL OR description LIKE '%' || :search || '%')"
    params = {"category": category, "search": search.upper() if search else None}

    total = int(query(db, f"SELECT COUNT(*) AS n FROM products {where}", **params)["n"].iloc[0])
    df = query(
        db,
        f"SELECT * FROM products {where} ORDER BY total_revenue DESC LIMIT :limit OFFSET :offset",
        **params, limit=limit, offset=offset,
    )
    return {"total": total, "limit": limit, "offset": offset, "items": records(df)}


@app.get("/products/{stock_code}", response_model=Product, tags=["products"])
def product(stock_code: str, db: EngineDep) -> dict:
    """One product's summary."""
    require_tables(db, "products")
    df = query(db, "SELECT * FROM products WHERE stock_code = :code", code=stock_code.upper())
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Product '{stock_code}' not found")
    return records(df)[0]


@app.get("/products/{stock_code}/sales", response_model=list[WeeklySales], tags=["products"])
def product_sales(stock_code: str, db: EngineDep) -> list[dict]:
    """Weekly price, units, revenue and guest share for one product."""
    require_tables(db, "weekly_sales")
    df = query(
        db,
        """
        SELECT week_start, price, quantity, revenue, guest_share
        FROM weekly_sales WHERE stock_code = :code ORDER BY week_start
        """,
        code=stock_code.upper(),
    )
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Product '{stock_code}' not found")
    df["week_start"] = pd.to_datetime(df["week_start"]).dt.strftime("%Y-%m-%d")
    return records(df)


# Methods in the order they are usually compared (baselines first, then DML).
METHOD_ORDER = [
    "pooled", "product_fe", "product_week_fe", "product_week_fe_controls", "dml_manual", "dml_econml",
]
DEFAULT_METHOD = "dml_manual"


@app.get("/elasticity", response_model=list[Estimate], tags=["elasticity"])
def elasticity(
    db: EngineDep,
    dataset: Annotated[str, Query(pattern="^(real|synthetic)$")] = "real",
) -> list[dict]:
    """Average price elasticity from every method (naive OLS baselines and DML).

    On `synthetic` data the true value is known and returned as `true_value`."""
    require_tables(db, "results")
    df = query(db, "SELECT * FROM results WHERE dataset = :dataset", dataset=dataset)
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    df = df.sort_values("method", key=lambda s: s.map(order).fillna(len(order)))
    return records(df)


@app.get("/elasticity/categories", response_model=list[SegmentEstimate], tags=["elasticity"])
def elasticity_by_category(
    db: EngineDep,
    method: str = DEFAULT_METHOD,
) -> list[dict]:
    """Price elasticity for each product category on the real data, most elastic first."""
    require_tables(db, "segment_elasticity")
    df = query(
        db,
        """
        SELECT segment, value, estimate, std_error, ci_low, ci_high, n_obs, n_groups
        FROM segment_elasticity
        WHERE dataset = 'real' AND segment = 'category' AND method = :method
        ORDER BY estimate
        """,
        method=method,
    )
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No category estimates for method '{method}'")
    return records(df)


@app.get("/products/{stock_code}/elasticity", response_model=ProductElasticity, tags=["elasticity"])
def product_elasticity(stock_code: str, db: EngineDep) -> dict:
    """Price elasticity for one product.

    Currently this is the DML estimate for the product's **category** (products in the
    same category share one elasticity). Product-level estimates will come from a causal
    forest model."""
    require_tables(db, "products", "segment_elasticity", "results")
    info = product(stock_code, db)
    seg = query(
        db,
        """
        SELECT estimate, ci_low, ci_high FROM segment_elasticity
        WHERE dataset = 'real' AND segment = 'category' AND method = :method AND value = :category
        """,
        method=DEFAULT_METHOD, category=info["category"],
    )
    overall = query(
        db, "SELECT estimate FROM results WHERE dataset = 'real' AND method = :method",
        method=DEFAULT_METHOD,
    )
    if seg.empty or overall.empty:
        raise HTTPException(status_code=503, detail="Elasticity results missing. Run `python -m econocausal.run_dml`.")

    e = float(seg["estimate"].iloc[0])
    return {
        "stock_code": info["stock_code"],
        "description": info["description"],
        "category": info["category"],
        "elasticity": e,
        "ci_low": float(seg["ci_low"].iloc[0]),
        "ci_high": float(seg["ci_high"].iloc[0]),
        "level": "category",
        "overall_elasticity": float(overall["estimate"].iloc[0]),
        "interpretation": (
            f"A 1% price increase {'lowers' if e < 0 else 'raises'} weekly units sold "
            f"by about {abs(e):.2f}%."
        ),
    }
