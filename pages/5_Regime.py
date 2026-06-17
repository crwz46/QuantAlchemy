import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from quantalchemy.regime.detector import HMMRegime, GMMRegime, KMeansRegime
from quantalchemy.data.yahoo import YahooLoader

st.title("5. Regime Detection")

symbol = st.text_input("Symbol", "SPY")
col1, col2 = st.columns(2)
with col1:
    start = st.date_input("Start", pd.Timestamp("2020-01-01"), key="rg_start")
with col2:
    end = st.date_input("End", pd.Timestamp("2024-01-01"), key="rg_end")

model_type = st.selectbox("Model", ["HMM", "GMM", "K-Means"])
n_regimes = st.slider("Number of Regimes", 2, 6, 3)

if st.button("Detect Regimes"):
    with st.spinner("Loading..."):
        loader = YahooLoader()
        data = loader.load(symbol, str(start), str(end), "1d")
        returns = data.returns.dropna()

        models = {
            "HMM": HMMRegime(n_regimes),
            "GMM": GMMRegime(n_regimes),
            "K-Means": KMeansRegime(n_regimes),
        }
        detector = models[model_type]
        labels = detector.fit(returns)
        df = data.df.loc[returns.index].copy()
        df["regime"] = labels

        colors = ["green", "red", "blue", "orange", "purple", "brown"]
        fig = go.Figure()
        for r in range(n_regimes):
            mask = df["regime"] == r
            fig.add_trace(go.Scatter(
                x=df.index[mask], y=df["close"][mask],
                mode="lines", name=f"Regime {r}",
                line=dict(color=colors[r % len(colors)]),
            ))
        fig.update_layout(title=f"Price Colored by Regime ({model_type})", height=500)
        st.plotly_chart(fig, use_container_width=True)

        stats = detector.regime_stats(returns, labels)
        st.subheader("Regime Statistics")
        st.dataframe(stats)
