from pathlib import Path
from typing import Optional

import pandas as pd

from quantalchemy.data.base import DataLoader, OHLCVData


class CSVLoader(DataLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(
        self,
        symbol: str = "",
        start: str = "",
        end: str = "",
        timeframe: str = "",
    ) -> OHLCVData:
        df = pd.read_csv(self.file_path)
        ts_cols = ["timestamp", "date", "time", "datetime"]
        found = False
        for c in ts_cols:
            if c in df.columns:
                df = df.set_index(pd.to_datetime(df[c]))
                found = True
                break
        if not found:
            if df.index.name is None or "timestamp" not in df.index.name:
                df.index = pd.to_datetime(df.index)
        df.columns = [c.lower().strip() for c in df.columns]
        col_map = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        df = df.rename(columns=col_map)
        required = ["open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in required):
            raise ValueError(f"CSV must have columns: {required}. Found: {list(df.columns)}")
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        name = symbol or Path(self.file_path).stem
        return OHLCVData(symbol=name, timeframe=timeframe or "custom", df=df)
