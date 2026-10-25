"""Elasticity: by category (DML) and by product (causal forest)."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import BLUE, REFERENCE, api_get, call_or_stop, estimate_chart, style

st.title("How price sensitivity varies")

overall = pd.DataFrame(call_or_stop(api_get, "/elasticity", dataset="real")).set_index("method")
average = float(overall.loc["dml_manual", "estimate"])
categories = pd.DataFrame(call_or_stop(api_get, "/elasticity/categories"))
products = pd.DataFrame(call_or_stop(api_get, "/elasticity/products", limit=2000)["items"])

# --- categories -----------------------------------------------------------------------
st.subheader("By category")
labels = [f"{v.replace('_', ' ')} ({n})" for v, n in zip(categories["value"], categories["n_groups"])]
st.plotly_chart(
    estimate_chart(categories, labels, pd.Series([True] * len(categories)),
                   reference=average, reference_label=f"average {average:.2f}", height=440),
    use_container_width=True,
)
st.caption("DML with a separate estimate per category; number of products in brackets. "
           "Categories with few products have wider intervals.")

# --- products -------------------------------------------------------------------------
st.subheader("By product")
if products.empty:
    st.info("No product-level estimates yet. Run `python -m econocausal.run_dml`.")
    st.stop()

significant = (products["ci_high"] < 0).mean()
width = (products["ci_high"] - products["ci_low"]).median()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Products", f"{len(products):,}")
c2.metric("Median elasticity", f"{products['estimate'].median():.2f}")
c3.metric("Clearly price sensitive", f"{significant:.1%}", help="95% interval entirely below zero")
c4.metric("Median interval width", f"{width:.2f}")

hist = go.Figure(go.Histogram(x=products["estimate"], nbinsx=45, marker_color=BLUE,
                              hovertemplate="Elasticity %{x}<br>%{y} products<extra></extra>"))
hist.add_vline(x=average, line_dash="dash", line_color=REFERENCE)
hist.add_annotation(x=average, y=1, yref="paper", yanchor="bottom", showarrow=False,
                    text=f"average (DML) {average:.2f}", font=dict(color=REFERENCE))
# Fixed range from the data, so the label never stretches the axis on narrow screens.
pad = 0.1
hist.update_xaxes(title="Product price elasticity (causal forest)",
                  range=[products["estimate"].min() - pad, products["estimate"].max() + pad])
hist.update_yaxes(title="Products")
style(hist, 300).update_layout(bargap=0.05, margin=dict(t=30))
st.plotly_chart(hist, use_container_width=True)
st.caption("Each product's estimate comes from a causal forest fitted on the same DML residuals. "
           "Individual intervals are wide, so the pattern is more reliable than any single value.")

col_most, col_least = st.columns(2)
columns = {"description": "Product", "category": "Category", "estimate": "Elasticity",
           "ci_low": "CI low", "ci_high": "CI high"}
with col_most:
    st.markdown("**Most price sensitive**")
    st.dataframe(products.nsmallest(10, "estimate")[list(columns)].rename(columns=columns),
                 hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.2f")
                                for c in ["Elasticity", "CI low", "CI high"]})
with col_least:
    st.markdown("**Least price sensitive**")
    st.dataframe(products.nlargest(10, "estimate")[list(columns)].rename(columns=columns),
                 hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.2f")
                                for c in ["Elasticity", "CI low", "CI high"]})
