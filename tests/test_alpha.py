import pytest
import pandas as pd
import numpy as np

from quantalchemy.alpha.discovery import AlphaDiscovery


def sample_data() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=300, freq="h")
    return pd.DataFrame({
        "open": np.random.randn(300).cumsum() + 100,
        "high": np.random.randn(300).cumsum() + 102,
        "low": np.random.randn(300).cumsum() + 98,
        "close": np.random.randn(300).cumsum() + 100,
        "volume": np.abs(np.random.randn(300) * 1000),
    }, index=idx)


class TestAlphaDiscovery:
    def test_discover_xgboost(self):
        data = sample_data()
        ad = AlphaDiscovery(model_type="xgboost")
        result = ad.discover(data)
        assert "accuracy" in result
        assert "feature_importance" in result
        assert "feature_names" in result
        assert 0 <= result["accuracy"] <= 1

    def test_discover_lightgbm(self):
        data = sample_data()
        ad = AlphaDiscovery(model_type="lightgbm")
        result = ad.discover(data)
        assert "accuracy" in result

    def test_predict(self):
        data = sample_data()
        ad = AlphaDiscovery(model_type="xgboost")
        ad.discover(data)
        preds = ad.predict(data)
        assert len(preds) > 0
