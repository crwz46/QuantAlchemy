import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from quantalchemy.data.binance import BinanceLoader
from quantalchemy.data.bybit import BybitLoader
from quantalchemy.data.coinbase import CoinbaseLoader
from quantalchemy.data.yahoo import YahooLoader

st.title("1. Data Layer")

source = st.selectbox("Data Source", ["Binance", "Bybit", "Coinbase", "Yahoo Finance"])
symbol = st.text_input("Symbol", "BTCUSDT")
tf = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"], index=4)
col1, col2 = st.columns(2)
with col1:
    start = st.date_input("Start", pd.Timestamp("2024-01-01"))
with col2:
    end = st.date_input("End", pd.Timestamp("2025-01-01"))

if st.button("Load Data"):
    with st.spinner("Fetching..."):
        loaders = {
            "Binance": BinanceLoader(),
            "Bybit": BybitLoader(),
            "Coinbase": CoinbaseLoader(),
            "Yahoo Finance": YahooLoader(),
        }
        loader = loaders[source]
        data = loader.load(symbol, str(start), str(end), tf)
        st.success(f"Loaded {len(data.df)} rows")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.df.index,
            open=data.df["open"], high=data.df["high"],
            low=data.df["low"], close=data.df["close"],
            name=symbol,
        ))
        fig.update_layout(title=f"{symbol} - {tf}", height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(data.df.tail(20))
