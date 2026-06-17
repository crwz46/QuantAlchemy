import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from quantalchemy.walkforward.analyzer import WalkForwardAnalyzer
from quantalchemy.data.yahoo import YahooLoader

st.title("6. Walk Forward Analysis")

symbol = st.text_input("Symbol", "AAPL")
col1, col2 = st.columns(2)
with col1:
    start = st.date_input("Start", pd.Timestamp("2020-01-01"), key="wf_start")
with col2:
    end = st.date_input("End", pd.Timestamp("2024-01-01"), key="wf_end")
n_windows = st.slider("Windows", 3, 10, 5)

def sma_strategy(data, params):
    fast = params.get("fast", 20)
    slow = params.get("slow", 50)
    signals = pd.Series(0, index=data.index)
    f = data["close"].rolling(fast).mean()
    s = data["close"].rolling(slow).mean()
    signals[f > s] = 1
    signals[f <= s] = -1
    signals = signals.shift(1).fillna(0)
    pos = signals
    rets = pos * data["close"].pct_change()
    equity = (1 + rets).cumprod() * 10000
    return equity

if st.button("Run Walk Forward"):
    with st.spinner("Running..."):
        loader = YahooLoader()
        data = loader.load(symbol, str(start), str(end), "1d")
        analyzer = WalkForwardAnalyzer()
        param_grid = {"fast": [10, 20, 50], "slow": [50, 100, 200]}
        result = analyzer.run(data.df, sma_strategy, param_grid, n_windows)

        st.success(f"Completed {len(result.windows)} windows")
        for w in result.windows:
            st.write(f"**Window {w['window']}**: {w['train_start'].date()} - {w['val_end'].date()} | Params: {w['params']}")

        if result.combined_equity is not None and len(result.combined_equity) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=result.combined_equity.index, y=result.combined_equity.values, mode="lines", name="Combined Equity"))
            fig.update_layout(title="Walk Forward Combined Equity", height=400)
            st.plotly_chart(fig, use_container_width=True)
