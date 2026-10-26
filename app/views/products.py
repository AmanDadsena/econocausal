"""Products: weekly price and sales history with the product's elasticity."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import BLUE, api_get, call_or_stop, categories_list, money, product_picker, style

st.title("Product explorer")

left, right = st.columns([1, 2])
with left:
    stock_code = product_picker(categories_list())
if stock_code is None:
    st.stop()

info = call_or_stop(api_get, f"/products/{stock_code}")
sales = pd.DataFrame(call_or_stop(api_get, f"/products/{stock_code}/sales"))
elasticity = call_or_stop(api_get, f"/products/{stock_code}/elasticity")
sales["week_start"] = pd.to_datetime(sales["week_start"])

with left:
    st.metric("Median weekly price", money(info["median_price"]),
              help=f"Range {money(info['min_price'])} – {money(info['max_price'])}")
    st.metric("Total revenue", money(info["total_revenue"], 0))
    level = "this product" if elasticity["level"] == "product" else f"{elasticity['category'].replace('_', ' ')} category"
    st.metric(f"Elasticity ({level})", f"{elasticity['elasticity']:.2f}",
              help=f"95% CI {elasticity['ci_low']:.2f} to {elasticity['ci_high']:.2f}. "
                   f"Average across all products: {elasticity['overall_elasticity']:.2f}.")
    st.caption(elasticity["interpretation"])

with right:
    price = go.Figure(go.Scatter(x=sales["week_start"], y=sales["price"], mode="lines+markers",
                                 line={"color": BLUE, "width": 2}, marker={"size": 4},
                                 hovertemplate="%{x|%d %b %Y}<br>£%{y:.2f}<extra></extra>"))
    price.update_yaxes(title="Weekly median price (£)", tickprefix="£")
    st.plotly_chart(style(price, 240), use_container_width=True)

    units = go.Figure(go.Bar(x=sales["week_start"], y=sales["quantity"], marker_color=BLUE,
                             hovertemplate="%{x|%d %b %Y}<br>%{y:,} units<extra></extra>"))
    units.update_yaxes(title="Units sold per week")
    st.plotly_chart(style(units, 240), use_container_width=True)
