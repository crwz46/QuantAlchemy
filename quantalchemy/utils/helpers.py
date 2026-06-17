import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def resample_ohlcv(df: pd.DataFrame, timeframe: str = "1h") -> pd.DataFrame:
    required = ["open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in required):
        raise ValueError(f"DataFrame must have columns: {required}")
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    idx = pd.to_datetime(df.index)
    resampled = df.groupby(pd.Grouper(freq=timeframe)).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    resampled.index.name = "timestamp"
    return resampled


def compute_returns(prices: pd.Series, method: str = "simple") -> pd.Series:
    if method == "log":
        return np.log(prices / prices.shift(1))
    return prices.pct_change()


def compute_log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1))


def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    return dd.min()
