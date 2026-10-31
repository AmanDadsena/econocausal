"""Request and response models for the EconoCausal API.

Request models validate every input (ranges, list sizes, allowed values), so invalid
requests are rejected with a 422 error before any calculation runs.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# --- products and elasticities -----------------------------------------------------

class Health(BaseModel):
    """Service status and the database tables the API can see."""

    status: str
    tables: list[str]


class Category(BaseModel):
    """A product category with its number of products and total revenue (GBP)."""

    category: str
    n_products: int
    total_revenue: float


class Product(BaseModel):
    """Summary of one product in the modelling panel (prices in GBP, totals over 2009-2011)."""

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
    """One page of products plus the total number matching the filters."""

    total: int
    limit: int
    offset: int
    items: list[Product]


class WeeklySales(BaseModel):
    """One week of a product: median price, units sold, revenue and guest share."""

    week_start: str
    price: float
    quantity: int
    revenue: float
    guest_share: float


class Estimate(BaseModel):
    """An average price elasticity from one method on one dataset. `true_value` is set only for synthetic data."""

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
    """Price elasticity for one segment (e.g. one product category) with its 95% interval."""

    segment: str
    value: str
    estimate: float
    std_error: float
    ci_low: float
    ci_high: float
    n_obs: int
    n_groups: int


class ProductElasticity(BaseModel):
    """Elasticity for one product: product level (causal forest) when available, else its category."""

    stock_code: str
    description: str | None
    category: str
    elasticity: float
    ci_low: float
    ci_high: float
    level: str
    overall_elasticity: float
    interpretation: str


# --- pricing -------------------------------------------------------------------------

StockCode = Annotated[str, Field(min_length=1, max_length=20, pattern=r"^[A-Za-z0-9]+$",
                                 examples=["22386"])]
# allow_inf_nan=False: Python's JSON parser accepts NaN and Infinity, which must be rejected.
Price = Annotated[float, Field(gt=0, le=10_000, allow_inf_nan=False)]


class PricingSettings(BaseModel):
    """Business rules for a recommendation (defaults match `python -m econocausal.optimize`)."""

    model_config = ConfigDict(extra="forbid")

    max_change: float = Field(0.20, gt=0, le=0.5, allow_inf_nan=False, description="Largest allowed price move, e.g. 0.2 = ±20%")
    min_margin: float = Field(0.20, ge=0, le=0.9, allow_inf_nan=False, description="Smallest allowed (price − cost) / price")
    cost_ratio: float = Field(0.60, ge=0, le=0.95, allow_inf_nan=False, description="Assumed unit cost as a share of the current price")
    objective: Literal["profit", "revenue"] = "profit"


class OptimalPriceRequest(PricingSettings):
    """Ask for the best price for one product under the given business rules."""

    stock_code: StockCode


class SimulateRequest(BaseModel):
    """Candidate prices (1 to 50, each 20%-500% of the current price) to evaluate for one product."""

    model_config = ConfigDict(extra="forbid")

    stock_code: StockCode
    prices: list[Price] = Field(min_length=1, max_length=50, description="Candidate prices in GBP",
                                examples=[[1.80, 2.10, 2.40]])
    cost_ratio: float = Field(0.60, ge=0, le=0.95, allow_inf_nan=False, description="Assumed unit cost as a share of the current price")


class PricingBasis(BaseModel):
    """What a recommendation or simulation is based on."""

    stock_code: str
    description: str | None
    category: str
    current_price: float
    current_units: float
    unit_cost: float
    elasticity: float
    elasticity_ci_low: float
    elasticity_ci_high: float
    elasticity_level: str
    window: str


class OptimalPriceResponse(PricingBasis):
    """Recommended price, the rule that decided it, and expected units, revenue and profit."""

    recommended_price: float | None
    reason: str
    price_change_pct: float | None = None
    expected_units: float | None = None
    units_change_pct: float | None = None
    current_revenue: float | None = None
    expected_revenue: float | None = None
    revenue_change_pct: float | None = None
    current_profit: float | None = None
    expected_profit: float | None = None
    profit_change_pct: float | None = None
    price_at_ci_low: float | None = None
    price_at_ci_high: float | None = None


class SimulatedPrice(BaseModel):
    """Expected weekly results at one candidate price; units_low/high use the elasticity's 95% interval."""

    price: float
    price_change_pct: float
    expected_units: float
    units_low: float
    units_high: float
    expected_revenue: float
    expected_profit: float
    margin_pct: float


class SimulateResponse(PricingBasis):
    """Pricing basis plus one result per candidate price, in the order given."""

    results: list[SimulatedPrice]


class Recommendation(BaseModel):
    """A stored catalogue recommendation (default rules) for one product."""

    stock_code: str
    description: str | None
    category: str
    current_price: float
    recommended_price: float | None
    price_change_pct: float | None
    elasticity: float
    reason: str
    expected_profit: float | None
    current_profit: float | None
    profit_change_pct: float | None


class RecommendationList(BaseModel):
    """One page of stored recommendations plus the total number matching the filters."""

    total: int
    limit: int
    offset: int
    items: list[Recommendation]


class ProductEstimate(BaseModel):
    """Causal-forest elasticity for one product with its 95% interval."""

    stock_code: str
    description: str | None
    category: str
    estimate: float
    ci_low: float
    ci_high: float
    n_obs: int


class ProductEstimateList(BaseModel):
    """One page of product elasticities plus the total number matching the filters."""

    total: int
    limit: int
    offset: int
    items: list[ProductEstimate]
