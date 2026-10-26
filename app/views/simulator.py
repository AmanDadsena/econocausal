"""Price simulator: expected units, revenue and profit at any price for one product."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (
    BLUE,
    ORANGE,
    REFERENCE,
    api_post,
    call_or_stop,
    categories_list,
    money,
    product_picker,
    style,
)

TRUSTED_CHANGE = 0.20   # the demand curve is only trusted within ±20% of the current price
SLIDER_CHANGE = 0.40    # the slider allows ±40%, with a warning outside the trusted range

st.title("Price simulator")
st.caption("Move the price and see the expected weekly units, revenue and profit, based on the "
           "product's estimated elasticity.")

left, right = st.columns([1, 2])
with left:
    stock_code = product_picker(categories_list())
    if stock_code is None:
        st.stop()
    cost_ratio = st.slider(
        "Unit cost as % of current price", min_value=30, max_value=90, value=60, step=5,
        help="Real costs are not in the data, so cost is an assumption.",
    ) / 100

basis = call_or_stop(api_post, "/optimal-price", {"stock_code": stock_code, "cost_ratio": cost_ratio})
p0 = basis["current_price"]

with left:
    price = st.slider(
        "Price (£)", min_value=round(p0 * (1 - SLIDER_CHANGE), 2), max_value=round(p0 * (1 + SLIDER_CHANGE), 2),
        value=round(p0, 2), step=0.01, key=f"price_{stock_code}",
    )

grid = np.round(np.linspace(p0 * (1 - SLIDER_CHANGE), p0 * (1 + SLIDER_CHANGE), 48), 4).tolist()  # + 2 prices = 50, the API maximum
sim = call_or_stop(api_post, "/simulate",
                   {"stock_code": stock_code, "prices": [p0, price, *grid], "cost_ratio": cost_ratio})
current, chosen, curve = sim["results"][0], sim["results"][1], pd.DataFrame(sim["results"][2:])

with left:
    level = "product" if basis["elasticity_level"] == "product" else "category"
    st.caption(f"Elasticity {basis['elasticity']:.2f} ({level} level, 95% CI "
               f"{basis['elasticity_ci_low']:.2f} to {basis['elasticity_ci_high']:.2f}). "
               f"Current price and units: median and weekly average over {basis['window']}.")

change = price / p0 - 1
if abs(change) > TRUSTED_CHANGE + 1e-9:
    st.warning(f"This price is {change:+.0%} from the current price. The demand curve is only reliable "
               f"within ±{TRUSTED_CHANGE:.0%}, so treat these numbers as rough.")

with right:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", money(price), f"{change:+.1%}", delta_color="off")
    c2.metric("Units / week", f"{chosen['expected_units']:,.0f}",
              f"{chosen['expected_units'] / current['expected_units'] - 1:+.1%}")
    c3.metric("Revenue / week", money(chosen["expected_revenue"], 0),
              f"{chosen['expected_revenue'] / current['expected_revenue'] - 1:+.1%}")
    c4.metric("Profit / week", money(chosen["expected_profit"], 0),
              f"{chosen['expected_profit'] / current['expected_profit'] - 1:+.1%}"
              if current["expected_profit"] > 0 else None)
    st.caption(f"Compared with the current price {money(p0)}: {current['expected_units']:,.0f} units, "
               f"{money(current['expected_revenue'], 0)} revenue and {money(current['expected_profit'], 0)} "
               f"profit per week. Margin at the chosen price: {chosen['margin_pct']:.0f}%.")

    def base_chart(title, fmt):
        fig = go.Figure()
        fig.add_vrect(x0=p0 * (1 - TRUSTED_CHANGE), x1=p0 * (1 + TRUSTED_CHANGE),
                      fillcolor="rgba(128,128,128,0.08)", line_width=0)
        fig.add_vline(x=p0, line_dash="dot", line_color=REFERENCE)
        fig.add_annotation(x=p0, y=1, yref="paper", yanchor="bottom", showarrow=False,
                           text="current", font={"color": REFERENCE})
        fig.update_xaxes(title="Price (£)", tickprefix="£",
                         range=[p0 * (1 - SLIDER_CHANGE), p0 * (1 + SLIDER_CHANGE)])
        fig.update_yaxes(title=title, tickformat=fmt)
        return fig

    profit_fig = base_chart("Expected profit per week (£)", ",.0f")
    profit_fig.add_trace(go.Scatter(x=curve["price"], y=curve["expected_profit"], mode="lines",
                                    line={"color": BLUE, "width": 2}, name="Profit",
                                    hovertemplate="£%{x:.2f}<br>Profit £%{y:,.0f}<extra></extra>"))
    profit_fig.add_trace(go.Scatter(x=[price], y=[chosen["expected_profit"]], mode="markers",
                                    marker={"size": 12, "color": ORANGE}, name="Chosen price",
                                    hovertemplate="Chosen £%{x:.2f}<br>Profit £%{y:,.0f}<extra></extra>"))
    st.plotly_chart(style(profit_fig, 260).update_layout(margin={"t": 30}), use_container_width=True)

    units_fig = base_chart("Expected units per week", ",.0f")
    units_fig.add_trace(go.Scatter(x=curve["price"], y=curve["units_high"], mode="lines", line={"width": 0},
                                   hoverinfo="skip", showlegend=False))
    units_fig.add_trace(go.Scatter(x=curve["price"], y=curve["units_low"], mode="lines", line={"width": 0},
                                   fill="tonexty", fillcolor="rgba(42,120,214,0.15)",
                                   name="95% CI of elasticity", hoverinfo="skip"))
    units_fig.add_trace(go.Scatter(x=curve["price"], y=curve["expected_units"], mode="lines",
                                   line={"color": BLUE, "width": 2}, name="Units",
                                   hovertemplate="£%{x:.2f}<br>%{y:,.0f} units<extra></extra>"))
    units_fig.add_trace(go.Scatter(x=[price], y=[chosen["expected_units"]], mode="markers",
                                   marker={"size": 12, "color": ORANGE}, name="Chosen price",
                                   hovertemplate="Chosen £%{x:.2f}<br>%{y:,.0f} units<extra></extra>"))
    st.plotly_chart(style(units_fig, 260).update_layout(margin={"t": 30}), use_container_width=True)
    st.caption("Shaded band in the background: ±20% around the current price, where the demand curve "
               "is trusted. Blue band on the units chart: range from the elasticity's 95% interval.")
