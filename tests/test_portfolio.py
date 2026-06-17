import pytest
import pandas as pd
import numpy as np

from quantalchemy.portfolio.markowitz import MarkowitzOptimizer
from quantalchemy.portfolio.risk_parity import RiskParityOptimizer


def sample_returns() -> pd.DataFrame:
    n = 252
    idx = pd.date_range("2024-01-01", periods=n, freq="d")
    return pd.DataFrame({
        "AAPL": np.random.randn(n) * 0.02,
        "MSFT": np.random.randn(n) * 0.02,
        "GOOGL": np.random.randn(n) * 0.02,
    }, index=idx)


class TestMarkowitz:
    def test_max_sharpe(self):
        r = sample_returns()
        opt = MarkowitzOptimizer()
        result = opt.max_sharpe(r)
        weights = result["weights"]
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1) < 0.01
        assert result["return"] > -1
        assert result["volatility"] > 0

    def test_min_volatility(self):
        r = sample_returns()
        opt = MarkowitzOptimizer()
        result = opt.min_volatility(r)
        assert abs(sum(result["weights"].values()) - 1) < 0.01


class TestRiskParity:
    def test_optimize(self):
        r = sample_returns()
        opt = RiskParityOptimizer()
        result = opt.optimize(r)
        assert abs(sum(result["weights"].values()) - 1) < 0.01
