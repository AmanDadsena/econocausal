"""Synthetic product x week panel with a known price elasticity.

On real data the true elasticity is unknown, so there is no way to check whether an
estimator is right. Here the data is generated from a model where the answer is set
by us, with the same kinds of confounding found in the EDA:

* product popularity: popular products are also the cheap ones
  (biases the pooled slope towards zero)
* season: demand peaks in Oct-Nov and the shop raises prices when demand is high
* customer mix: weeks with more guest buyers have a higher (retail) price and lower
  quantity, with a step at 50% because the weekly price is a median

The seasonal and guest effects are nonlinear, so a linear regression with these
controls is still biased, while flexible ML-based adjustment (DML) is not.
Design notes and the exact equations: docs/synthetic_data.md
"""

import numpy as np
import pandas as pd

from econocausal.data.panel import weeks_to_christmas

TRUE_ELASTICITY = -1.8

# Category-level shifts of the elasticity, used when heterogeneous=True.
# Seasonal and gift items react more to price, everyday items less.
CATEGORY_ELASTICITY_SHIFT = {
    "christmas": -0.8,
    "jewellery": -0.4,
    "home_decor": -0.2,
    "kitchen_dining": 0.0,
    "bags": 0.3,
    "stationery_cards": 0.5,
}

START_WEEK = "2009-11-30"


def seasonal_demand(week_of_year: np.ndarray) -> np.ndarray:
    """Seasonal demand in log points: low in summer, strong peak in Oct-Nov."""
    peak = 0.45 * np.exp(-0.5 * ((week_of_year - 45) / 4.0) ** 2)
    yearly = 0.12 * np.cos(2 * np.pi * (week_of_year - 1) / 52)
    return peak + yearly


def guest_price_premium(guest_share: np.ndarray) -> np.ndarray:
    """Log price premium from guest buyers: a smooth part plus a jump to the
    retail tier (about 2x, log 2 = 0.69) once guests are the majority of lines."""
    return 0.1 * guest_share + 0.6 * (guest_share > 0.5)


def guest_demand_effect(guest_share: np.ndarray) -> np.ndarray:
    """Weeks bought mostly by guests sell fewer units (guests buy small quantities)."""
    return -1.2 * guest_share**2


def simulate_panel(
    n_products: int = 300,
    n_weeks: int = 104,
    elasticity: float = TRUE_ELASTICITY,
    heterogeneous: bool = False,
    hidden_confounding: float = 0.0,
    noise_sd: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a product x week panel with a known elasticity.

    Parameters
    ----------
    n_products, n_weeks : panel size.
    elasticity : true average price elasticity.
    heterogeneous : if True, the elasticity differs by category
        (see CATEGORY_ELASTICITY_SHIFT); the true value per row is in `true_elasticity`.
    hidden_confounding : strength of a demand shock that also raises price but is NOT
        included in the returned columns. 0 means every confounder is observed, which is
        the setting where DML should recover the true value. Values > 0 are for testing
        how estimates break when the no-unobserved-confounding assumption fails.
    noise_sd : standard deviation of the demand noise.
    seed : random seed, so the same arguments always give the same data.

    Returns
    -------
    DataFrame with the same main columns as the real `weekly_sales` table, plus
    `true_elasticity`. The average true elasticity is stored in `df.attrs`.
    """
    rng = np.random.default_rng(seed)

    # --- products -------------------------------------------------------------
    categories = np.array(list(CATEGORY_ELASTICITY_SHIFT))
    product_category = rng.choice(categories, size=n_products)
    base_log_price = rng.normal(0.75, 0.8, n_products)
    # Popular products are cheaper: base demand falls with base price.
    base_log_demand = 3.0 - 0.9 * base_log_price + rng.normal(0, 0.8, n_products)
    product_guest_rate = rng.beta(1.0, 2.5, n_products)

    product_elasticity = np.full(n_products, elasticity, dtype=float)
    if heterogeneous:
        shifts = np.array([CATEGORY_ELASTICITY_SHIFT[c] for c in product_category])
        # Centre the shifts so the average elasticity stays equal to `elasticity`.
        product_elasticity = elasticity + shifts - shifts.mean()

    # --- panel ----------------------------------------------------------------
    weeks = pd.date_range(START_WEEK, periods=n_weeks, freq="7D")
    product_idx = np.repeat(np.arange(n_products), n_weeks)
    week_start = pd.Series(np.tile(weeks, n_products))
    n = len(product_idx)

    week_of_year = week_start.dt.isocalendar().week.to_numpy().astype(int)
    month = week_start.dt.month.to_numpy()

    # Customer mix: many weeks with no guests, some with only guests (as in the real data).
    guest_share = np.clip(product_guest_rate[product_idx] + rng.normal(0, 0.25, n), 0, 1)
    mix_draw = rng.random(n)
    guest_share[mix_draw < 0.35] = 0.0                        # registered customers only
    guest_share[(mix_draw >= 0.35) & (mix_draw < 0.47)] = 1.0  # guests only
    uk_share = np.clip(rng.normal(0.92, 0.1, n), 0, 1)

    hidden_shock = rng.normal(0, 1, n)
    season = seasonal_demand(week_of_year)

    # --- treatment: log price ------------------------------------------------
    log_price = (
        base_log_price[product_idx]
        + 0.35 * season                       # higher prices when demand is high
        + guest_price_premium(guest_share)    # retail tier for guest buyers
        + hidden_confounding * 0.15 * hidden_shock
        + rng.normal(0, 0.12, n)              # independent price variation
    )

    # --- outcome: log quantity ----------------------------------------------
    true_elasticity = product_elasticity[product_idx]
    log_quantity = (
        base_log_demand[product_idx]
        + true_elasticity * (log_price - base_log_price[product_idx])
        + season
        + guest_demand_effect(guest_share)
        + 0.3 * (uk_share - 0.92)
        + hidden_confounding * 0.5 * hidden_shock
        + rng.normal(0, noise_sd, n)
    )

    df = pd.DataFrame(
        {
            "stock_code": np.char.add("SYN", np.char.zfill(product_idx.astype(str), 4)),
            "week_start": week_start,
            "category": product_category[product_idx],
            "price": np.exp(log_price),
            "quantity": np.exp(log_quantity),
            "log_price": log_price,
            "log_quantity": log_quantity,
            "year": week_start.dt.year.to_numpy(),
            "month": month,
            "week_of_year": week_of_year,
            "weeks_to_christmas": weeks_to_christmas(week_start).to_numpy(),
            "is_peak_season": np.isin(month, [9, 10, 11]).astype(int),
            "guest_share": guest_share,
            "uk_share": uk_share,
            "true_elasticity": true_elasticity,
        }
    )
    df.attrs["true_elasticity"] = float(product_elasticity.mean())
    return df
