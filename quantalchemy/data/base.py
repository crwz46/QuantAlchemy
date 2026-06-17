from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass
class OHLCVData:
    symbol: str
    timeframe: str
    df: pd.DataFrame

    def __post_init__(self):
        self.df.index = pd.to_datetime(self.df.index)
        self.df.index.name = "timestamp"

    @property
    def returns(self) -> pd.Series:
        return self.df["close"].pct_change().dropna()


class DataLoader(ABC):
    @abstractmethod
    def load(
        self,
        symbol: str,
        start: str,
        end: str,
        timeframe: str = "1h",
    ) -> OHLCVData:
        pass

    def load_multi(
        self,
        symbols: List[str],
        start: str,
        end: str,
        timeframe: str = "1h",
    ) -> Dict[str, OHLCVData]:
        return {s: self.load(s, start, end, timeframe) for s in symbols}
