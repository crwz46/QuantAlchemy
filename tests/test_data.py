import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from quantalchemy.data.base import OHLCVData
from quantalchemy.data.csv_loader import CSVLoader
from quantalchemy.data.binance import BinanceLoader
from quantalchemy.data.yahoo import YahooLoader


def sample_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=100, freq="h")
    return pd.DataFrame({
        "open": np.random.randn(100).cumsum() + 100,
        "high": np.random.randn(100).cumsum() + 102,
        "low": np.random.randn(100).cumsum() + 98,
        "close": np.random.randn(100).cumsum() + 100,
        "volume": np.abs(np.random.randn(100) * 1000),
    }, index=idx)


class TestOHLCVData:
    def test_returns_property(self):
        df = sample_df()
        ohlcv = OHLCVData(symbol="TEST", timeframe="1h", df=df)
        assert isinstance(ohlcv.returns, pd.Series)
        assert len(ohlcv.returns) == len(df) - 1
        assert ohlcv.returns.index.name == "timestamp"


class TestCSVLoader:
    def test_load_csv(self):
        df = sample_df()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f.name)
            f.flush()
            loader = CSVLoader(f.name)
            result = loader.load()
            assert isinstance(result, OHLCVData)
            assert len(result.df) > 0
            assert all(c in result.df.columns for c in ["open", "high", "low", "close", "volume"])
        Path(f.name).unlink(missing_ok=True)
