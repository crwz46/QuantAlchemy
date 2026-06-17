import streamlit as st

from quantalchemy import __version__

st.set_page_config(
    page_title="QuantAlchemy",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("⚗️ QuantAlchemy")
st.sidebar.caption(f"v{__version__}")
st.sidebar.page_link("main.py", label="Home")
st.sidebar.page_link("pages/1_Data.py", label="1. Data")
st.sidebar.page_link("pages/2_Backtest.py", label="2. Backtest")
st.sidebar.page_link("pages/3_Portfolio.py", label="3. Portfolio")
st.sidebar.page_link("pages/4_Factors.py", label="4. Factors")
st.sidebar.page_link("pages/5_Regime.py", label="5. Regime Detector")
st.sidebar.page_link("pages/6_WalkForward.py", label="6. Walk Forward")
st.sidebar.page_link("pages/7_Lab.py", label="7. Strategy Lab")
st.sidebar.page_link("pages/8_Tearsheet.py", label="8. Tearsheet")
st.sidebar.page_link("pages/9_Alpha.py", label="9. Alpha Discovery")

st.title("⚗️ QuantAlchemy")
st.markdown("""
QuantAlchemy is a quantitative trading platform for backtesting, portfolio optimization,
factor analysis, regime detection, and alpha discovery.

**Modules:**
- **Data Layer** — Fetch OHLCV from Binance, Bybit, Coinbase, Yahoo Finance, CSV
- **Backtest Engine** — Run and evaluate trading strategies
- **Portfolio Optimization** — Markowitz, Risk Parity, Black-Litterman
- **Factor Analysis** — IC, rank IC, factor returns, top/bottom spread
- **Regime Detection** — HMM, GMM, K-Means regime classification
- **Walk Forward Analysis** — Robust out-of-sample validation
- **Strategy Lab** — Parameter sweeps and comparison
- **Tearsheet** — HTML performance report
- **Alpha Discovery** — ML-driven alpha factor generation
""")
