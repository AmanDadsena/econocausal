"""EconoCausal dashboard.

Reads everything from the EconoCausal API, so the API must be running:

    uvicorn api.main:app                 # terminal 1
    streamlit run app/streamlit_app.py   # terminal 2

The API address can be changed with the ECONOCAUSAL_API environment variable.
"""

import streamlit as st

st.set_page_config(page_title="EconoCausal", page_icon="📈", layout="wide")

pages = st.navigation(
    {
        "Results": [
            st.Page("views/overview.py", title="Overview", icon="📊", default=True),
            st.Page("views/elasticity.py", title="Elasticity", icon="📉"),
            st.Page("views/products.py", title="Products", icon="🛍️"),
        ],
        "Pricing": [
            st.Page("views/simulator.py", title="Price simulator", icon="🎚️"),
            st.Page("views/optimizer.py", title="Optimizer", icon="💷"),
        ],
    }
)
pages.run()
