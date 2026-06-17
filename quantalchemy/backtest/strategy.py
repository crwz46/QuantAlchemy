from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class Strategy(ABC):
    def __init__(self, params: Dict = None):
        self.params = params or {}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        pass


class SMACrossover(Strategy):
    def __init__(self, fast: int = 20, slow: int = 50):
        super().__init__({"fast": fast, "slow": slow})

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        fast = data["close"].rolling(self.params["fast"]).mean()
        slow = data["close"].rolling(self.params["slow"]).mean()
        signals = pd.Series(0, index=data.index)
        signals[fast > slow] = 1
        signals[fast <= slow] = -1
        return signals.shift(1).fillna(0)


class RSIBased(Strategy):
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__({"period": period, "oversold": oversold, "overbought": overbought})

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        delta = data["close"].diff()
        gain = delta.clip(lower=0).rolling(self.params["period"]).mean()
        loss = (-delta.clip(upper=0)).rolling(self.params["period"]).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        signals = pd.Series(0, index=data.index)
        signals[rsi < self.params["oversold"]] = 1
        signals[rsi > self.params["overbought"]] = -1
        return signals.shift(1).fillna(0)


class MLStrategy(Strategy):
    def __init__(self, model, features: List[str], threshold: float = 0.0):
        super().__init__({"threshold": threshold})
        self.model = model
        self.features = features

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        X = data[self.features].dropna()
        preds = self.model.predict(X)
        signals = pd.Series(0, index=data.index)
        signals.loc[X.index] = np.where(preds > self.params["threshold"], 1, -1)
        return signals.shift(1).fillna(0)
