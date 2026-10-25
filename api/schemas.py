"""Request and response models for the EconoCausal API.

Request models validate every input (ranges, list sizes, allowed values), so invalid
requests are rejected with a 422 error before any calculation runs.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# --- products and elasticities -----------------------------------------------------

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


# --- pricing -------------------------------------------------------------------------

StockCode = Annotated[str, Field(min_length=1, max_length=20, pattern=r"^[A-Za-z0-9]+$",
                                 examples=["22386"])]
Price = Annotated[float, Field(gt=0, le=10_000)]


class PricingSettings(BaseModel):
    """Business rules for a recommendation (defaults match `python -m econocausal.optimize`)."""

    model_config = ConfigDict(extra="forbid")

    max_change: float = Field(0.20, gt=0, le=0.5, description="Largest allowed price move, e.g. 0.2 = ±20%")
    min_margin: float = Field(0.20, ge=0, le=0.9, description="Smallest allowed (price − cost) / price")
    cost_ratio: float = Field(0.60, ge=0, le=0.95, description="Assumed unit cost as a share of the current price")
    objective: Literal["profit", "revenue"] = "profit"


class OptimalPriceRequest(PricingSettings):
    stock_code: StockCode


class SimulateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stock_code: StockCode
    prices: list[Price] = Field(min_length=1, max_length=50, description="Candidate prices in GBP",
                                examples=[[1.80, 2.10, 2.40]])
    cost_ratio: float = Field(0.60, ge=0, le=0.95, description="Assumed unit cost as a share of the current price")


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
    price: float
    price_change_pct: float
    expected_units: float
    units_low: float
    units_high: float
    expected_revenue: float
    expected_profit: float
    margin_pct: float


class SimulateResponse(PricingBasis):
    results: list[SimulatedPrice]


class Recommendation(BaseModel):
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
    total: int
    limit: int
    offset: int
    items: list[Recommendation]


class ProductEstimate(BaseModel):
    stock_code: str
    description: str | None
    category: str
    estimate: float
    ci_low: float
    ci_high: float
    n_obs: int


class ProductEstimateList(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ProductEstimate]
