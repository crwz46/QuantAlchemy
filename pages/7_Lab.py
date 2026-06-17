import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from quantalchemy.lab.strategy_lab import StrategyLab
from quantalchemy.backtest.strategy import SMACrossover
from quantalchemy.data.yahoo import YahooLoader

st.title("7. Strategy Lab")

symbol = st.text_input("Symbol", "AAPL")
col1, col2 = st.columns(2)
with col1:
    start = st.date_input("Start", pd.Timestamp("2023-01-01"), key="lab_start")
with col2:
    end = st.date_input("End", pd.Timestamp("2024-01-01"), key="lab_end")

st.subheader("Parameter Grid (SMA Crossover)")
fast_range = st.slider("Fast SMA range", 5, 50, (10, 30))
slow_range = st.slider("Slow SMA range", 20, 200, (50, 100))

if st.button("Run Lab"):
    with st.spinner("Running param sweeps..."):
        loader = YahooLoader()
        data = loader.load(symbol, str(start), str(end), "1d")
        lab = StrategyLab()
        param_grid = {
            "fast": list(range(fast_range[0], fast_range[1] + 1, 5)),
            "slow": list(range(slow_range[0], slow_range[1] + 1, 10)),
        }
        results = lab.run(SMACrossover, data.df, param_grid)
        comparison = lab.compare(results)
        st.success(f"Tested {len(results)} parameter combinations")
        st.dataframe(comparison.sort_values("sharpe_ratio", ascending=False).head(20))

        best = results[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=best.backtest_result.equity.index, y=best.backtest_result.equity.values, mode="lines", name=best.name))
        fig.update_layout(title=f"Best: {best.name}", height=400)
        st.plotly_chart(fig, use_container_width=True)
