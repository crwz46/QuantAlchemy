import pytest
import pandas as pd
import numpy as np

from quantalchemy.walkforward.analyzer import WalkForwardAnalyzer


def sma_strategy(data, params):
    fast = params.get("fast", 20)
    slow = params.get("slow", 50)
    signals = pd.Series(0, index=data.index)
    f = data["close"].rolling(fast).mean()
    s = data["close"].rolling(slow).mean()
    signals[f > s] = 1
    signals[f <= s] = -1
    signals = signals.shift(1).fillna(0)
    rets = signals * data["close"].pct_change()
    return (1 + rets).cumprod() * 10000


def sample_data() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=500, freq="h")
    return pd.DataFrame({
        "close": np.random.randn(500).cumsum() + 100,
        "open": np.random.randn(500).cumsum() + 100,
        "high": np.random.randn(500).cumsum() + 102,
        "low": np.random.randn(500).cumsum() + 98,
        "volume": np.abs(np.random.randn(500) * 1000),
    }, index=idx)


class TestWalkForward:
    def test_run(self):
        data = sample_data()
        analyzer = WalkForwardAnalyzer()
        param_grid = {"fast": [10, 20], "slow": [50, 100]}
        result = analyzer.run(data, sma_strategy, param_grid, n_windows=3)
        assert len(result.windows) > 0
