from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class RegimeDetector(ABC):
    @abstractmethod
    def fit(self, returns: pd.Series) -> np.ndarray:
        pass

    def label_regimes(self, returns: pd.Series, n_regimes: int = 3) -> Tuple[np.ndarray, object]:
        labels = self.fit(returns)
        return labels, self

    def regime_stats(self, returns: pd.Series, labels: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame({"return": returns, "regime": labels})
        stats = df.groupby("regime")["return"].agg(["mean", "std", "count"])
        stats.columns = ["avg_return", "volatility", "samples"]
        return stats


class HMMRegime(RegimeDetector):
    def __init__(self, n_regimes: int = 3, n_iter: int = 1000):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.model = None

    def fit(self, returns: pd.Series) -> np.ndarray:
        from hmmlearn import hmm
        X = returns.dropna().values.reshape(-1, 1)
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="diag",
            n_iter=self.n_iter,
            random_state=42,
        )
        self.model.fit(X)
        return self.model.predict(X)


class GMMRegime(RegimeDetector):
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.model = None

    def fit(self, returns: pd.Series) -> np.ndarray:
        X = returns.dropna().values.reshape(-1, 1)
        self.model = GaussianMixture(n_components=self.n_regimes, random_state=42)
        return self.model.fit_predict(X)


class KMeansRegime(RegimeDetector):
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.model = None

    def fit(self, returns: pd.Series) -> np.ndarray:
        features = pd.DataFrame({
            "return": returns,
            "volatility": returns.rolling(20).std(),
            "volume": returns.abs().rolling(20).mean(),
        }).dropna()
        self.model = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        return self.model.fit_predict(features)
