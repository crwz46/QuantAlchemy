import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quantalchemy.backtest.engine import BacktestEngine
from quantalchemy.backtest.strategy import SMACrossover, RSIBased
from quantalchemy.backtest.metrics import PerformanceMetrics
from quantalchemy.data.yahoo import YahooLoader

st.title("2. Backtest Engine")

symbol = st.text_input("Symbol", "AAPL")
col1, col2 = st.columns(2)
with col1:
    start = st.date_input("Start", pd.Timestamp("2023-01-01"), key="bt_start")
with col2:
    end = st.date_input("End", pd.Timestamp("2024-01-01"), key="bt_end")

strategy_type = st.selectbox("Strategy", ["SMA Crossover", "RSI Based"])
if strategy_type == "SMA Crossover":
    fast = st.slider("Fast SMA", 5, 100, 20)
    slow = st.slider("Slow SMA", 10, 200, 50)
    strategy = SMACrossover(fast, slow)
    params_display = {"fast": fast, "slow": slow}
else:
    period = st.slider("RSI Period", 5, 30, 14)
    oversold = st.slider("Oversold", 10, 40, 30)
    overbought = st.slider("Overbought", 60, 90, 70)
    strategy = RSIBased(period, oversold, overbought)
    params_display = {"period": period, "oversold": oversold, "overbought": overbought}

capital = st.number_input("Initial Capital", 1000, 1000000, 10000)

if st.button("Run Backtest"):
    with st.spinner("Running..."):
        loader = YahooLoader()
        data = loader.load(symbol, str(start), str(end), "1d")
        engine = BacktestEngine(initial_capital=capital)
        result = engine.run(strategy, data.df)
        metrics = PerformanceMetrics.format(result.metrics)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Return", metrics["total_return"])
        col2.metric("Sharpe", metrics["sharpe_ratio"])
        col3.metric("Max DD", metrics["max_drawdown"])
        col4.metric("Winrate", metrics["winrate"])

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Scatter(x=result.equity.index, y=result.equity.values, mode="lines", name="Equity"), row=1, col=1)
        peak = result.equity.expanding().max()
        dd = (result.equity - peak) / peak * 100
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown", fill="tozeroy"), row=2, col=1)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Performance Metrics")
        st.json(metrics)
        if len(result.trades) > 0:
            st.subheader(f"Trades ({len(result.trades)})")
            st.dataframe(result.trades)
