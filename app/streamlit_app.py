"""EconoCausal dashboard (first version).

Reads everything from the EconoCausal API, so the API must be running:

    uvicorn api.main:app            # terminal 1
    streamlit run app/streamlit_app.py   # terminal 2

The API address can be changed with the ECONOCAUSAL_API environment variable.
"""

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.environ.get("ECONOCAUSAL_API", "http://localhost:8000").rstrip("/")

BLUE = "#2a78d6"
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

st.set_page_config(page_title="EconoCausal", page_icon="📈", layout="wide")


class ApiError(Exception):
    pass


@st.cache_data(ttl=300, show_spinner=False)
def api_get(path: str, **params):
    try:
        response = requests.get(f"{API_URL}{path}", params=params or None, timeout=10)
    except requests.RequestException as exc:
        raise ApiError(f"Cannot reach the API at {API_URL}. Start it with `uvicorn api.main:app`.") from exc
    if response.status_code != 200:
        detail = response.json().get("detail", response.text) if response.content else response.reason
        raise ApiError(f"API error {response.status_code} on {path}: {detail}")
    return response.json()


def style(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(size=13),
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.2)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", zeroline=False)
    return fig


def estimate_chart(df: pd.DataFrame, labels: list[str], highlight: pd.Series,
                   true_value: float | None = None, height: int = 320) -> go.Figure:
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
    fig.update_xaxes(title="Price elasticity (95% CI)")
    style(fig, height)
    if true_value is not None:
        fig.add_vline(x=true_value, line_dash="dash", line_color=REFERENCE)
        # Label anchored to the top edge of the plot area, in the margin above the rows.
        fig.add_annotation(x=true_value, y=1, yref="paper", yanchor="bottom", showarrow=False,
                           text=f"true value {true_value:g}", font=dict(color=REFERENCE))
        fig.update_layout(margin=dict(t=30))
    return fig


def main() -> None:
    st.title("EconoCausal")
    st.caption(
        "Causal price elasticities for an online gift retailer, estimated with Double Machine "
        "Learning on the UCI Online Retail II data (Dec 2009 – Dec 2011)."
    )

    try:
        products_page = api_get("/products", limit=1)
        real = pd.DataFrame(api_get("/elasticity", dataset="real"))
        synthetic = pd.DataFrame(api_get("/elasticity", dataset="synthetic"))
        categories = pd.DataFrame(api_get("/elasticity/categories"))
    except ApiError as exc:
        st.error(str(exc))
        st.stop()

    real_by_method = real.set_index("method")
    dml = real_by_method.loc["dml_manual"]
    naive = real_by_method.loc["product_week_fe_controls"]

    # --- headline numbers ----------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products analysed", f"{products_page['total']:,}")
    c2.metric("Weekly observations", f"{int(dml['n_obs']):,}")
    c3.metric("Price elasticity (DML)", f"{dml['estimate']:.2f}",
              help=f"95% CI {dml['ci_low']:.2f} to {dml['ci_high']:.2f}")
    c4.metric("Best naive OLS", f"{naive['estimate']:.2f}",
              delta=f"{naive['estimate'] - dml['estimate']:+.2f} vs DML", delta_color="off")
    st.markdown(
        f"On average, a **1% price increase lowers weekly units sold by about "
        f"{abs(dml['estimate']):.2f}%** (95% CI {abs(dml['ci_high']):.2f}% to {abs(dml['ci_low']):.2f}%)."
    )

    # --- method comparison -----------------------------------------------------------
    st.subheader("Naive regression vs Double ML")
    tab_real, tab_synth = st.tabs(["Real data", "Synthetic data (true value known)"])
    with tab_real:
        st.plotly_chart(
            estimate_chart(real, [METHOD_LABELS.get(m, m) for m in real["method"]],
                           real["method"].str.startswith("dml")),
            use_container_width=True,
        )
        st.caption("The naive estimates move a lot depending on which controls are included. "
                   "DML adjusts for season and customer mix with flexible ML models.")
    with tab_synth:
        true_value = float(synthetic["true_value"].dropna().iloc[0])
        st.plotly_chart(
            estimate_chart(synthetic, [METHOD_LABELS.get(m, m) for m in synthetic["method"]],
                           synthetic["method"].str.startswith("dml"), true_value=true_value),
            use_container_width=True,
        )
        st.caption(f"On simulated data with a known elasticity of {true_value:g}, only the DML "
                   "estimators have confidence intervals that contain the true value.")

    # --- categories ------------------------------------------------------------------
    st.subheader("Elasticity by category")
    labels = [f"{v.replace('_', ' ')} ({n})" for v, n in zip(categories["value"], categories["n_groups"])]
    st.plotly_chart(
        estimate_chart(categories, labels, pd.Series([True] * len(categories)), height=420),
        use_container_width=True,
    )
    st.caption("Number of products in brackets. Categories with few products have wider intervals.")

    # --- product explorer ------------------------------------------------------------
    st.subheader("Product explorer")
    left, right = st.columns([1, 2])
    with left:
        category_options = ["All"] + sorted(categories["value"].tolist())
        chosen_category = st.selectbox("Category", category_options)
        search = st.text_input("Search description", placeholder="e.g. bag, candle, heart")
        params = {"limit": 500}
        if chosen_category != "All":
            params["category"] = chosen_category
        if search.strip():
            params["search"] = search.strip()
        try:
            found = api_get("/products", **params)
        except ApiError as exc:
            st.error(str(exc))
            return
        if not found["items"]:
            st.info("No products match.")
            return
        options = {f"{p['description']} ({p['stock_code']})": p["stock_code"] for p in found["items"]}
        st.caption(f"{found['total']:,} products, best sellers first")
        stock_code = options[st.selectbox("Product", list(options))]

    try:
        info = api_get(f"/products/{stock_code}")
        sales = pd.DataFrame(api_get(f"/products/{stock_code}/sales"))
        elasticity = api_get(f"/products/{stock_code}/elasticity")
    except ApiError as exc:
        st.error(str(exc))
        return
    sales["week_start"] = pd.to_datetime(sales["week_start"])

    with left:
        st.metric("Median weekly price", f"£{info['median_price']:.2f}",
                  help=f"Range £{info['min_price']:.2f} – £{info['max_price']:.2f}")
        st.metric("Total revenue", f"£{info['total_revenue']:,.0f}")
        st.metric(f"Elasticity ({elasticity['category'].replace('_', ' ')} category)",
                  f"{elasticity['elasticity']:.2f}",
                  help=f"95% CI {elasticity['ci_low']:.2f} to {elasticity['ci_high']:.2f}. "
                       "Category-level estimate; product-level estimates come later.")
        st.caption(elasticity["interpretation"])

    with right:
        price = go.Figure(go.Scatter(x=sales["week_start"], y=sales["price"], mode="lines+markers",
                                     line=dict(color=BLUE, width=2), marker=dict(size=4),
                                     hovertemplate="%{x|%d %b %Y}<br>£%{y:.2f}<extra></extra>"))
        price.update_yaxes(title="Weekly median price (£)", tickprefix="£")
        st.plotly_chart(style(price, 240), use_container_width=True)

        units = go.Figure(go.Bar(x=sales["week_start"], y=sales["quantity"], marker_color=BLUE,
                                 hovertemplate="%{x|%d %b %Y}<br>%{y:,} units<extra></extra>"))
        units.update_yaxes(title="Units sold per week")
        st.plotly_chart(style(units, 240), use_container_width=True)


main()
