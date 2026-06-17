import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from quantalchemy.factors.factors import (
    FactorAnalyzer,
    information_coefficient,
)
from quantalchemy.data.yahoo import YahooLoader

st.title("4. Factor Analysis")

symbol = st.text_input("Symbol", "AAPL")
col1, col2 = st.columns(2)
with col1:
    start = st.date_input("Start", pd.Timestamp("2023-01-01"), key="fa_start")
with col2:
    end = st.date_input("End", pd.Timestamp("2024-01-01"), key="fa_end")

if st.button("Run Factor Analysis"):
    with st.spinner("Loading..."):
        loader = YahooLoader()
        data = loader.load(symbol, str(start), str(end), "1d")
        df = data.df

        df["returns"] = df["close"].pct_change()
        df["sma_10"] = df["close"].rolling(10).mean()
        df["sma_30"] = df["close"].rolling(30).mean()
        df["rsi"] = 50
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        df["momentum"] = df["close"].pct_change(20)
        df["volatility_20"] = df["returns"].rolling(20).std()
        df["forward_1d"] = df["close"].pct_change().shift(-1)

        factors = {
            "SMA_10": df["sma_10"] / df["close"],
            "SMA_30": df["sma_30"] / df["close"],
            "RSI": df["rsi"],
            "Volume_Ratio": df["volume_ratio"],
            "Momentum_20d": df["momentum"],
            "Volatility_20d": df["volatility_20"],
        }

        analyzer = FactorAnalyzer(factors)
        fr = df["forward_1d"].dropna()
        ics = analyzer.compute_rank_ics(fr)

        st.subheader("Information Coefficients (Rank IC)")
        ic_df = pd.DataFrame(list(ics.items()), columns=["Factor", "Rank IC"]).sort_values("Rank IC", ascending=False)
        fig = go.Figure(data=[go.Bar(x=ic_df["Factor"], y=ic_df["Rank IC"])])
        fig.update_layout(title="Factor Rank IC", yaxis_range=[-1, 1], height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ic_df)

        st.subheader("Factor Correlation Matrix")
        corr = analyzer.factor_correlation_matrix()
        fig2 = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu"))
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
