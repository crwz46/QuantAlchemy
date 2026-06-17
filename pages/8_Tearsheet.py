import streamlit as st
import pandas as pd
from datetime import datetime

from quantalchemy.backtest.engine import BacktestEngine
from quantalchemy.backtest.strategy import SMACrossover
from quantalchemy.backtest.metrics import PerformanceMetrics
from quantalchemy.tearsheet.report import TearsheetReport
from quantalchemy.data.yahoo import YahooLoader

st.title("8. Tearsheet")

symbol = st.text_input("Symbol", "AAPL")
col1, col2 = st.columns(2)
with col1:
    start = st.date_input("Start", pd.Timestamp("2023-01-01"), key="ts_start")
with col2:
    end = st.date_input("End", pd.Timestamp("2024-01-01"), key="ts_end")
fast = st.slider("Fast SMA", 5, 100, 20, key="ts_fast")
slow = st.slider("Slow SMA", 10, 200, 50, key="ts_slow")

if st.button("Generate Tearsheet"):
    with st.spinner("Generating..."):
        loader = YahooLoader()
        data = loader.load(symbol, str(start), str(end), "1d")
        engine = BacktestEngine()
        result = engine.run(SMACrossover(fast, slow), data.df)
        metrics = PerformanceMetrics.format(result.metrics)
        report = TearsheetReport(
            equity=result.equity,
            returns=result.returns,
            metrics=metrics,
            trades=result.trades,
        )
        html = report.create_html()
        dt = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button("Download HTML Report", data=html, file_name=f"tearsheet_{symbol}_{dt}.html", mime="text/html")
        st.components.v1.html(html, height=800, scrolling=True)
