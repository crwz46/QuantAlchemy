import pytest
import pandas as pd
import numpy as np

from quantalchemy.backtest.strategy import SMACrossover, RSIBased
from quantalchemy.backtest.engine import BacktestEngine
from quantalchemy.backtest.metrics import PerformanceMetrics


def sample_data() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=200, freq="h")
    return pd.DataFrame({
        "open": np.random.randn(200).cumsum() + 100,
        "high": np.random.randn(200).cumsum() + 102,
        "low": np.random.randn(200).cumsum() + 98,
        "close": np.random.randn(200).cumsum() + 100,
        "volume": np.abs(np.random.randn(200) * 1000),
    }, index=idx)


class TestStrategies:
    def test_sma_crossover(self):
        df = sample_data()
        strat = SMACrossover(fast=10, slow=30)
        signals = strat.generate_signals(df)
        assert len(signals) == len(df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_rsi_based(self):
        df = sample_data()
        strat = RSIBased(period=14, oversold=30, overbought=70)
        signals = strat.generate_signals(df)
        assert len(signals) == len(df)
        assert set(signals.unique()).issubset({-1, 0, 1})


class TestBacktestEngine:
    def test_run(self):
        df = sample_data()
        engine = BacktestEngine(initial_capital=10000)
        result = engine.run(SMACrossover(10, 30), df)
        assert len(result.equity) > 0
        assert len(result.returns) > 0
        assert isinstance(result.metrics, dict)
        assert "sharpe_ratio" in result.metrics
        assert result.equity.iloc[0] == 10000


class TestPerformanceMetrics:
    def test_compute(self):
        equity = pd.Series([10000, 11000, 10500, 12000])
        ret = equity.pct_change().dropna()
        metrics = PerformanceMetrics.compute(equity, ret)
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "sortino_ratio" in metrics
        assert metrics["total_trades"] == 3

    def test_format(self):
        equity = pd.Series([10000, 11000, 10500, 12000])
        ret = equity.pct_change().dropna()
        metrics = PerformanceMetrics.compute(equity, ret)
        formatted = PerformanceMetrics.format(metrics)
        assert isinstance(formatted["total_return"], str)
        assert "%" in formatted["total_return"]
