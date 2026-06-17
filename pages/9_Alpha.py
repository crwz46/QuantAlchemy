import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from quantalchemy.alpha.discovery import AlphaDiscovery
from quantalchemy.data.yahoo import YahooLoader

st.title("9. Alpha Discovery")

symbol = st.text_input("Symbol", "AAPL")
col1, col2 = st.columns(2)
with col1:
    start = st.date_input("Start", pd.Timestamp("2020-01-01"), key="ad_start")
with col2:
    end = st.date_input("End", pd.Timestamp("2024-01-01"), key="ad_end")

model_type = st.selectbox("ML Model", ["xgboost", "lightgbm", "catboost"])

if st.button("Discover Alpha Factors"):
    with st.spinner(f"Training {model_type}..."):
        loader = YahooLoader()
        data = loader.load(symbol, str(start), str(end), "1d")
        ad = AlphaDiscovery(model_type=model_type)
        result = ad.discover(data.df)

        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{result['accuracy']:.2%}")
        col2.metric("Features Used", len(result["feature_names"]))

        st.subheader("Feature Importance")
        fi = pd.DataFrame(
            list(result["feature_importance"].items()),
            columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=True)
        fig = go.Figure(data=[go.Bar(x=fi["Importance"], y=fi["Feature"], orientation="h")])
        fig.update_layout(title=f"Feature Importance ({model_type})", height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Classification Report")
        cr = pd.DataFrame(result["classification_report"]).transpose()
        st.dataframe(cr)
