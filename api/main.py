"""EconoCausal REST API.

Serves the cleaned product data and the elasticity estimates stored in SQLite by the
pipeline (`python -m econocausal.data.build`, `python -m econocausal.baselines`,
`python -m econocausal.run_dml`, `python -m econocausal.optimize`), and calculates
price recommendations and what-if simulations on request. The API never writes to the
database.

Run locally:
    uvicorn api.main:app --reload
Interactive docs: http://localhost:8000/docs

The database location can be changed with the ECONOCAUSAL_DB environment variable.
"""

import logging
import os
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from sqlalchemy import Engine, inspect, text

import econocausal
from api.schemas import (
    Category,
    Estimate,
    Health,
    OptimalPriceRequest,
    OptimalPriceResponse,
    Product,
    ProductElasticity,
    ProductEstimateList,
    ProductList,
    RecommendationList,
    SegmentEstimate,
    SimulateRequest,
    SimulateResponse,
    WeeklySales,
)
from econocausal.data.db import DB_PATH, get_engine
from econocausal.log import setup_logging
from econocausal.optimize import PricingRules, demand, pricing_inputs, recommend_price

app = FastAPI(
    title="EconoCausal API",
    description="Price elasticities estimated with Double Machine Learning on the Online Retail II data.",
    version=econocausal.__version__,
)

setup_logging()
logger = logging.getLogger("econocausal.api")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """One log line per request: method, path, status and duration. A request ID is
    returned in the X-Request-ID header so a user's report can be matched to the log.
    Query strings and request bodies are not logged."""
    request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info("%s %s -> %d in %.0f ms [%s]", request.method, request.url.path,
                response.status_code, elapsed_ms, request_id)
    return response


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


@app.get("/elasticity/products", response_model=ProductEstimateList, tags=["elasticity"])
def elasticity_by_product(
    db: EngineDep,
    category: str | None = None,
    sort: Annotated[str, Query(pattern="^(most_elastic|least_elastic|stock_code)$")] = "most_elastic",
    limit: Annotated[int, Query(ge=1, le=2000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict:
    """Product-level elasticities from the causal forest (real data)."""
    require_tables(db, "product_elasticity")
    # Fixed, whitelisted ORDER BY clauses only; user input never enters the SQL text.
    order = {"most_elastic": "estimate ASC", "least_elastic": "estimate DESC", "stock_code": "stock_code ASC"}[sort]
    where = "WHERE dataset = 'real' AND method = 'causal_forest' AND (:category IS NULL OR category = :category)"
    total = int(query(db, f"SELECT COUNT(*) AS n FROM product_elasticity {where}", category=category)["n"].iloc[0])
    df = query(
        db,
        f"""
        SELECT stock_code, description, category, estimate, ci_low, ci_high, n_obs
        FROM product_elasticity {where} ORDER BY {order} LIMIT :limit OFFSET :offset
        """,
        category=category, limit=limit, offset=offset,
    )
    return {"total": total, "limit": limit, "offset": offset, "items": records(df)}


def elasticity_for(db: Engine, info: dict) -> tuple[float, float, float, str]:
    """(estimate, ci_low, ci_high, level) for a product: the causal forest's product-level
    estimate when available, otherwise the DML estimate for the product's category."""
    estimate, level = pd.DataFrame(), "product"
    if inspect(db).has_table("product_elasticity"):
        estimate = query(
            db,
            """
            SELECT estimate, ci_low, ci_high FROM product_elasticity
            WHERE dataset = 'real' AND method = 'causal_forest' AND stock_code = :code
            """,
            code=info["stock_code"],
        )
    if estimate.empty:
        level = "category"
        estimate = query(
            db,
            """
            SELECT estimate, ci_low, ci_high FROM segment_elasticity
            WHERE dataset = 'real' AND segment = 'category' AND method = :method AND value = :category
            """,
            method=DEFAULT_METHOD, category=info["category"],
        )
    if estimate.empty:
        raise HTTPException(status_code=503, detail="Elasticity results missing. Run `python -m econocausal.run_dml`.")
    row = estimate.iloc[0]
    return float(row["estimate"]), float(row["ci_low"]), float(row["ci_high"]), level


@app.get("/products/{stock_code}/elasticity", response_model=ProductElasticity, tags=["elasticity"])
def product_elasticity(stock_code: str, db: EngineDep) -> dict:
    """Price elasticity for one product.

    Uses the causal forest's product-level estimate (`level: "product"`) when available,
    otherwise the DML estimate for the product's category (`level: "category"`)."""
    require_tables(db, "products", "segment_elasticity", "results")
    info = product(stock_code, db)
    e, low, high, level = elasticity_for(db, info)
    overall = query(
        db, "SELECT estimate FROM results WHERE dataset = 'real' AND method = :method",
        method=DEFAULT_METHOD,
    )
    if overall.empty:
        raise HTTPException(status_code=503, detail="Elasticity results missing. Run `python -m econocausal.run_dml`.")
    return {
        "stock_code": info["stock_code"],
        "description": info["description"],
        "category": info["category"],
        "elasticity": e,
        "ci_low": low,
        "ci_high": high,
        "level": level,
        "overall_elasticity": float(overall["estimate"].iloc[0]),
        "interpretation": (
            f"A 1% price increase {'lowers' if e < 0 else 'raises'} weekly units sold "
            f"by about {abs(e):.2f}%."
        ),
    }


# --- pricing -----------------------------------------------------------------------

PRICING_WINDOW_WEEKS = 13


def pricing_basis(db: Engine, stock_code: str, cost_ratio: float) -> dict:
    """Current price, current weekly units and elasticity for one product.

    Uses the same 13-week window (ending at the last week in the data) as
    `python -m econocausal.optimize`, so the API and the stored recommendations agree."""
    require_tables(db, "products", "weekly_sales", "segment_elasticity")
    info = product(stock_code, db)
    last_week = pd.Timestamp(query(db, "SELECT MAX(week_start) AS w FROM weekly_sales")["w"].iloc[0])
    sales = query(
        db, "SELECT stock_code, week_start, price, quantity FROM weekly_sales WHERE stock_code = :code",
        code=info["stock_code"],
    )
    sales["week_start"] = pd.to_datetime(sales["week_start"])
    inputs = pricing_inputs(sales, recent_weeks=PRICING_WINDOW_WEEKS, last_week=last_week)
    if inputs.empty:
        raise HTTPException(
            status_code=422,
            detail=f"Product '{info['stock_code']}' has no sales in the last {PRICING_WINDOW_WEEKS} "
                   "weeks of data, so there is no current price to start from.",
        )
    e, low, high, level = elasticity_for(db, info)
    current_price = float(inputs["current_price"].iloc[0])
    start = last_week - pd.Timedelta(weeks=PRICING_WINDOW_WEEKS - 1)
    return {
        "stock_code": info["stock_code"],
        "description": info["description"],
        "category": info["category"],
        "current_price": current_price,
        "current_units": float(inputs["current_units"].iloc[0]),
        "unit_cost": cost_ratio * current_price,
        "elasticity": e,
        "elasticity_ci_low": low,
        "elasticity_ci_high": high,
        "elasticity_level": level,
        "window": f"{start:%Y-%m-%d} to {last_week + pd.Timedelta(days=6):%Y-%m-%d}",
    }


@app.post("/optimal-price", response_model=OptimalPriceResponse, tags=["pricing"])
def optimal_price(request: OptimalPriceRequest, db: EngineDep) -> dict:
    """Best price for one product under the given business rules.

    The price may move at most `max_change` from the current price and must keep at least
    `min_margin`. Unit cost is not in the data, so it is assumed to be `cost_ratio` times the
    current price. `price_at_ci_low` / `price_at_ci_high` repeat the calculation at both ends
    of the elasticity's 95% interval."""
    basis = pricing_basis(db, request.stock_code, request.cost_ratio)
    rules = PricingRules(max_change=request.max_change, min_margin=request.min_margin,
                         cost_ratio=request.cost_ratio, objective=request.objective)
    rec = recommend_price(basis["current_price"], basis["current_units"], basis["elasticity"], rules,
                          elasticity_low=basis["elasticity_ci_low"],
                          elasticity_high=basis["elasticity_ci_high"])
    rec.pop("elasticity", None)
    rec.pop("current_units", None)
    return {**basis, **rec}


@app.post("/simulate", response_model=SimulateResponse, tags=["pricing"])
def simulate(request: SimulateRequest, db: EngineDep) -> dict:
    """Expected weekly units, revenue and profit at each candidate price.

    `units_low` / `units_high` use the two ends of the elasticity's 95% interval."""
    basis = pricing_basis(db, request.stock_code, request.cost_ratio)
    p0, q0, cost = basis["current_price"], basis["current_units"], basis["unit_cost"]
    results = []
    for price in request.prices:
        units = float(demand(price, p0, q0, basis["elasticity"]))
        bounds = [float(demand(price, p0, q0, basis[k])) for k in ("elasticity_ci_low", "elasticity_ci_high")]
        results.append({
            "price": price,
            "price_change_pct": 100 * (price / p0 - 1),
            "expected_units": units,
            "units_low": min(bounds),
            "units_high": max(bounds),
            "expected_revenue": price * units,
            "expected_profit": (price - cost) * units,
            "margin_pct": 100 * (price - cost) / price,
        })
    return {**basis, "results": results}


@app.get("/recommendations", response_model=RecommendationList, tags=["pricing"])
def recommendations(
    db: EngineDep,
    category: str | None = None,
    reason: str | None = None,
    sort: Annotated[str, Query(pattern="^(profit_gain|price_change|stock_code)$")] = "profit_gain",
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict:
    """Stored recommendations from `python -m econocausal.optimize` (default rules),
    largest expected profit gain first."""
    require_tables(db, "price_recommendations")
    # Only fixed, whitelisted ORDER BY clauses are used; user input never enters the SQL text.
    order = {
        "profit_gain": "(expected_profit - current_profit) DESC",
        "price_change": "price_change_pct DESC",
        "stock_code": "stock_code ASC",
    }[sort]
    where = "WHERE (:category IS NULL OR category = :category) AND (:reason IS NULL OR reason = :reason)"
    params = {"category": category, "reason": reason}
    total = int(query(db, f"SELECT COUNT(*) AS n FROM price_recommendations {where}", **params)["n"].iloc[0])
    df = query(
        db,
        f"""
        SELECT stock_code, description, category, current_price, recommended_price, price_change_pct,
               elasticity, reason, expected_profit, current_profit, profit_change_pct
        FROM price_recommendations {where} ORDER BY {order} LIMIT :limit OFFSET :offset
        """,
        **params, limit=limit, offset=offset,
    )
    return {"total": total, "limit": limit, "offset": offset, "items": records(df)}
