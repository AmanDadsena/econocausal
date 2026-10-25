"""Overview: headline result and naive regression vs Double ML."""

import pandas as pd
import streamlit as st

from common import METHOD_LABELS, api_get, call_or_stop, estimate_chart

st.title("EconoCausal")
st.caption(
    "Causal price elasticities for an online gift retailer, estimated with Double Machine "
    "Learning on the UCI Online Retail II data (Dec 2009 – Dec 2011)."
)

products_page = call_or_stop(api_get, "/products", limit=1)
real = pd.DataFrame(call_or_stop(api_get, "/elasticity", dataset="real"))
synthetic = pd.DataFrame(call_or_stop(api_get, "/elasticity", dataset="synthetic"))

by_method = real.set_index("method")
dml = by_method.loc["dml_manual"]
naive = by_method.loc["product_week_fe_controls"]

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
                       synthetic["method"].str.startswith("dml"),
                       reference=true_value, reference_label=f"true value {true_value:g}"),
        use_container_width=True,
    )
    st.caption(f"On simulated data with a known elasticity of {true_value:g}, only the DML "
               "estimators have confidence intervals that contain the true value.")
