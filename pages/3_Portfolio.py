import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from quantalchemy.portfolio.markowitz import MarkowitzOptimizer
from quantalchemy.portfolio.risk_parity import RiskParityOptimizer
from quantalchemy.portfolio.black_litterman import BlackLitterman

st.title("3. Portfolio Optimization")

opt_type = st.selectbox("Optimizer", ["Markowitz (Max Sharpe)", "Markowitz (Min Vol)", "Risk Parity", "Black-Litterman"])
symbols_input = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN")
symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]

col1, col2 = st.columns(2)
with col1:
    start = st.date_input("Start", pd.Timestamp("2023-01-01"), key="pf_start")
with col2:
    end = st.date_input("End", pd.Timestamp("2024-01-01"), key="pf_end")

if st.button("Optimize"):
    with st.spinner("Loading data..."):
        from quantalchemy.data.yahoo import YahooLoader
        loader = YahooLoader()
        returns = []
        for sym in symbols:
            data = loader.load(sym, str(start), str(end), "1d")
            returns.append(data.returns.rename(sym))
        returns_df = pd.concat(returns, axis=1).dropna()
        st.success(f"Loaded {len(returns_df)} rows for {len(symbols)} assets")

        if opt_type == "Markowitz (Max Sharpe)":
            opt = MarkowitzOptimizer()
            result = opt.max_sharpe(returns_df)
        elif opt_type == "Markowitz (Min Vol)":
            opt = MarkowitzOptimizer()
            result = opt.min_volatility(returns_df)
        elif opt_type == "Risk Parity":
            opt = RiskParityOptimizer()
            result = opt.optimize(returns_df)
        else:
            market_caps = [100e9] * len(symbols)
            views_assets = [0]
            views_returns = [0.20]
            opt = BlackLitterman(delta=1.0)
            result = opt.optimize(returns_df, market_caps, views_assets, views_returns)

        weights = result["weights"]
        fig = go.Figure(data=[go.Pie(labels=list(weights.keys()), values=list(weights.values()))])
        fig.update_layout(title="Portfolio Weights", height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Results")
        cols = st.columns(3)
        cols[0].metric("Expected Return", f"{result.get('return', 0):.2%}")
        cols[1].metric("Volatility", f"{result.get('volatility', 0):.2%}")
        cols[2].metric("Sharpe Ratio", f"{result.get('sharpe', 0):.2f}")

        st.dataframe(pd.Series(weights, name="Weight").to_frame())
