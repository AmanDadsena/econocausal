"""Shared helpers for the dashboard pages: API access, chart style and the product picker."""

import json
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.environ.get("ECONOCAUSAL_API", "http://localhost:8000").rstrip("/")

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#8a8984"
# Mid grey for reference lines: readable on both Streamlit's light and dark themes.
REFERENCE = "#9a9994"

METHOD_LABELS = {
    "pooled": "OLS, pooled",
    "product_fe": "OLS, product FE",
    "product_week_fe": "OLS, product + week FE",
    "product_week_fe_controls": "OLS, product + week FE + controls",
    "dml_manual": "DML (own implementation)",
    "dml_econml": "DML (EconML)",
}


class ApiError(Exception):
    pass


def _handle(response: requests.Response, path: str):
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text or response.reason
        raise ApiError(f"API error {response.status_code} on {path}: {detail}")
    return response.json()


@st.cache_data(ttl=300, show_spinner=False)
def api_get(path: str, **params):
    try:
        response = requests.get(f"{API_URL}{path}", params=params or None, timeout=15)
    except requests.RequestException as exc:
        raise ApiError(f"Cannot reach the API at {API_URL}. Start it with `uvicorn api.main:app`.") from exc
    return _handle(response, path)


@st.cache_data(ttl=300, show_spinner=False)
def _api_post_cached(path: str, body_json: str):
    try:
        response = requests.post(f"{API_URL}{path}", data=body_json,
                                 headers={"Content-Type": "application/json"}, timeout=15)
    except requests.RequestException as exc:
        raise ApiError(f"Cannot reach the API at {API_URL}. Start it with `uvicorn api.main:app`.") from exc
    return _handle(response, path)


def api_post(path: str, body: dict):
    """POST with caching (the body is turned into a stable JSON key)."""
    return _api_post_cached(path, json.dumps(body, sort_keys=True))


def call_or_stop(func, *args, **kwargs):
    """Run an API call; on failure show the error and stop the page."""
    try:
        return func(*args, **kwargs)
    except ApiError as exc:
        st.error(str(exc))
        st.stop()


def style(fig: go.Figure, height: int, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.2)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", zeroline=False)
    return fig


def estimate_chart(df: pd.DataFrame, labels: list[str], highlight: pd.Series,
                   reference: float | None = None, reference_label: str = "",
                   height: int = 320) -> go.Figure:
    """Dot-and-interval chart of elasticity estimates (one row per estimate)."""
    colors = [BLUE if h else GREY for h in highlight]
    fig = go.Figure(
        go.Scatter(
            x=df["estimate"],
            y=labels,
            mode="markers",
            marker=dict(size=11, color=colors),
            error_x=dict(
                type="data", symmetric=False,
                array=df["ci_high"] - df["estimate"], arrayminus=df["estimate"] - df["ci_low"],
                thickness=2.5, width=0, color="rgba(128,128,128,0.6)",
            ),
            customdata=df[["ci_low", "ci_high"]],
            hovertemplate="%{y}<br>Elasticity %{x:.3f}<br>95% CI %{customdata[0]:.3f} to %{customdata[1]:.3f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    # Fixed x range from the intervals, so labels never stretch the axis on narrow screens.
    low, high = float(df["ci_low"].min()), float(df["ci_high"].max())
    if reference is not None:
        low, high = min(low, reference), max(high, reference)
    pad = 0.05 * (high - low)
    fig.update_xaxes(title="Price elasticity (95% CI)", range=[low - pad, high + pad])
    style(fig, height)
    if reference is not None:
        fig.add_vline(x=reference, line_dash="dash", line_color=REFERENCE)
        # Label anchored to the top edge of the plot area, in the margin above the rows.
        fig.add_annotation(x=reference, y=1, yref="paper", yanchor="bottom", showarrow=False,
                           text=reference_label, font=dict(color=REFERENCE))
        fig.update_layout(margin=dict(t=30))
    return fig


def product_picker(categories: list[str]) -> str | None:
    """Category filter, search box and product list. The chosen product is kept in
    st.session_state["stock_code"] so it stays selected when switching pages."""
    chosen_category = st.selectbox("Category", ["All", *sorted(categories)], key="picker_category")
    search = st.text_input("Search description", placeholder="e.g. bag, candle, heart", key="picker_search")
    params = {"limit": 500}
    if chosen_category != "All":
        params["category"] = chosen_category
    if search.strip():
        params["search"] = search.strip()
    found = call_or_stop(api_get, "/products", **params)
    if not found["items"]:
        st.info("No products match.")
        return None

    codes = [p["stock_code"] for p in found["items"]]
    labels = {p["stock_code"]: f"{p['description']} ({p['stock_code']})" for p in found["items"]}
    current = st.session_state.get("stock_code")
    index = codes.index(current) if current in codes else 0
    stock_code = st.selectbox("Product", codes, index=index, format_func=labels.get, key="picker_product")
    st.session_state["stock_code"] = stock_code
    st.caption(f"{found['total']:,} products, best sellers first")
    return stock_code


def categories_list() -> list[str]:
    return [c["category"] for c in call_or_stop(api_get, "/categories")]


def money(value: float, decimals: int = 2) -> str:
    return f"£{value:,.{decimals}f}"
