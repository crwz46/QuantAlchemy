import pytest
import pandas as pd
import numpy as np

from quantalchemy.tearsheet.report import TearsheetReport


class TestTearsheet:
    def test_create_html(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="d")
        equity = pd.Series(np.random.randn(100).cumsum() + 10000, index=idx)
        returns = equity.pct_change().dropna()
        metrics = {
            "total_return": "20.00%",
            "sharpe_ratio": "1.50",
            "max_drawdown": "-15.00%",
            "winrate": "55.00%",
        }
        report = TearsheetReport(equity=equity, returns=returns, metrics=metrics)
        html = report.create_html()
        assert "<!DOCTYPE html>" in html
        assert "Equity Curve" in html
        assert "Performance Metrics" in html
