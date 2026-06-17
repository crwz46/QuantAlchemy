import pytest
import pandas as pd
import numpy as np

from quantalchemy.lab.strategy_lab import StrategyLab
from quantalchemy.backtest.strategy import SMACrossover


def sample_data() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=200, freq="h")
    return pd.DataFrame({
        "close": np.random.randn(200).cumsum() + 100,
        "open": np.random.randn(200).cumsum() + 100,
        "high": np.random.randn(200).cumsum() + 102,
        "low": np.random.randn(200).cumsum() + 98,
        "volume": np.abs(np.random.randn(200) * 1000),
    }, index=idx)


class TestStrategyLab:
    def test_run(self):
        data = sample_data()
        lab = StrategyLab()
        param_grid = {"fast": [10, 20], "slow": [50, 100]}
        results = lab.run(SMACrossover, data, param_grid)
        assert len(results) == 4
        assert results[0].backtest_result.metrics["sharpe_ratio"] >= results[-1].backtest_result.metrics["sharpe_ratio"]

    def test_compare(self):
        data = sample_data()
        lab = StrategyLab()
        param_grid = {"fast": [10, 20], "slow": [50, 100]}
        results = lab.run(SMACrossover, data, param_grid)
        comp = lab.compare(results)
        assert len(comp) == 4
