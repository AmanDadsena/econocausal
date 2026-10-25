"""Optimizer: recommended price for one product under chosen rules, and the whole catalogue."""

import pandas as pd
import streamlit as st

from common import api_get, api_post, call_or_stop, categories_list, money, product_picker

st.title("Price optimizer")
st.caption(
    "The best price for each product given its estimated elasticity and business rules. "
    "Unit cost is not in the data, so it is an assumed share of the current price."
)

# --- one product ------------------------------------------------------------------------
st.subheader("Recommend a price")
left, right = st.columns([1, 2])
with left:
    stock_code = product_picker(categories_list())
    if stock_code is None:
        st.stop()
    max_change = st.slider("Largest allowed price change (±%)", 5, 50, 20, 5) / 100
    min_margin = st.slider("Minimum margin (%)", 0, 60, 20, 5) / 100
    cost_ratio = st.slider("Unit cost as % of current price", 30, 90, 60, 5) / 100
    objective = st.radio("Goal", ["profit", "revenue"], horizontal=True,
                         format_func=lambda o: "Maximise profit" if o == "profit" else "Maximise revenue")

rec = call_or_stop(api_post, "/optimal-price", {
    "stock_code": stock_code, "max_change": max_change, "min_margin": min_margin,
    "cost_ratio": cost_ratio, "objective": objective,
})

with right:
    if rec["recommended_price"] is None:
        st.error(f"No price satisfies these rules: {rec['reason']}. Lower the minimum margin or the cost share.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Current price", money(rec["current_price"]))
        c2.metric("Recommended price", money(rec["recommended_price"]), f"{rec['price_change_pct']:+.1f}%",
                  delta_color="off")
        c3.metric("Decided by", rec["reason"].capitalize())

        c4, c5, c6 = st.columns(3)
        c4.metric("Units / week", f"{rec['expected_units']:,.0f}", f"{rec['units_change_pct']:+.1f}%")
        c5.metric("Revenue / week", money(rec["expected_revenue"], 0), f"{rec['revenue_change_pct']:+.1f}%")
        c6.metric("Profit / week", money(rec["expected_profit"], 0), f"{rec['profit_change_pct']:+.1f}%")

        low, high = rec.get("price_at_ci_low"), rec.get("price_at_ci_high")
        level = "this product" if rec["elasticity_level"] == "product" else "its category"
        st.markdown(
            f"Elasticity {rec['elasticity']:.2f} for {level} "
            f"(95% CI {rec['elasticity_ci_low']:.2f} to {rec['elasticity_ci_high']:.2f})."
        )
        if low is not None and high is not None:
            if abs(low - high) < 0.005 and abs(low - rec["recommended_price"]) < 0.005:
                st.success("The recommendation is the same anywhere in the elasticity's 95% interval.")
            else:
                prices = sorted([low, high])
                st.info(f"Across the elasticity's 95% interval the best price ranges from {money(prices[0])} "
                        f"to {money(prices[1])}, so this recommendation is less certain.")
        st.caption(
            f"Based on the median price and average weekly units over {rec['window']}. Unit cost assumed "
            f"{money(rec['unit_cost'])} ({cost_ratio:.0%} of the current price)."
        )

# --- catalogue ----------------------------------------------------------------------------
st.divider()
st.subheader("Whole catalogue")
st.caption("Stored results from `python -m econocausal.optimize` with the default rules: ±20% price "
           "change, at least 20% margin, cost 60% of the current price, maximise profit.")


@st.cache_data(ttl=300, show_spinner=False)
def all_recommendations() -> pd.DataFrame:
    items, offset = [], 0
    while True:
        page = api_get("/recommendations", sort="profit_gain", limit=500, offset=offset)
        items += page["items"]
        offset += page["limit"]
        if offset >= page["total"]:
            return pd.DataFrame(items)


recs = call_or_stop(all_recommendations)
priced = recs.dropna(subset=["recommended_price"])
gain = priced["expected_profit"].sum() / priced["current_profit"].sum() - 1

c1, c2, c3, c4 = st.columns(4)
c1.metric("Products priced", f"{len(priced):,}")
c2.metric("Price rises", f"{(priced['price_change_pct'] > 0.5).sum():,}")
c3.metric("Price cuts", f"{(priced['price_change_pct'] < -0.5).sum():,}")
c4.metric("Weekly profit", money(priced["expected_profit"].sum(), 0), f"{gain:+.1%}")

st.markdown("**Why each recommendation was made**")
reasons = priced["reason"].value_counts().rename_axis("Reason").reset_index(name="Products")
st.dataframe(reasons, hide_index=True, use_container_width=False)

category = st.selectbox("Show category", ["All", *sorted(priced["category"].unique())], key="catalogue_category")
view = priced if category == "All" else priced[priced["category"] == category]
view = view.assign(profit_gain=view["expected_profit"] - view["current_profit"]).nlargest(50, "profit_gain")
st.markdown(f"**Largest expected profit gains** ({len(view)} shown)")
st.dataframe(
    view[["description", "category", "elasticity", "current_price", "recommended_price",
          "price_change_pct", "profit_gain", "reason"]],
    hide_index=True, use_container_width=True,
    column_config={
        "description": "Product", "category": "Category",
        "elasticity": st.column_config.NumberColumn("Elasticity", format="%.2f"),
        "current_price": st.column_config.NumberColumn("Current", format="£%.2f"),
        "recommended_price": st.column_config.NumberColumn("Recommended", format="£%.2f"),
        "price_change_pct": st.column_config.NumberColumn("Change", format="%+.1f%%"),
        "profit_gain": st.column_config.NumberColumn("Profit gain / week", format="£%.0f"),
        "reason": "Decided by",
    },
)
st.caption("Profit figures use the pre-Christmas window (Sep–Dec 2011), so they reflect peak-season volumes. "
           "See docs/pricing.md for the cost sensitivity and limitations.")
