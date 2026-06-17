import pytest
import numpy as np
import pandas as pd

from quantalchemy.regime.detector import HMMRegime, GMMRegime, KMeansRegime


def sample_returns() -> pd.Series:
    return pd.Series(np.random.randn(500) * 0.02)


class TestHMMRegime:
    def test_fit(self):
        r = sample_returns()
        detector = HMMRegime(n_regimes=3)
        labels = detector.fit(r)
        assert len(labels) == len(r.dropna())
        assert len(np.unique(labels)) <= 3


class TestGMMRegime:
    def test_fit(self):
        r = sample_returns()
        detector = GMMRegime(n_regimes=3)
        labels = detector.fit(r)
        assert len(labels) == len(r.dropna())


class TestKMeansRegime:
    def test_fit(self):
        r = sample_returns()
        detector = KMeansRegime(n_regimes=3)
        labels = detector.fit(r)
        assert len(labels) == len(r.dropna()) - 19
