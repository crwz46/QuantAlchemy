from typing import Optional

import pandas as pd

from quantalchemy.config import Config
from quantalchemy.data.base import DataLoader, OHLCVData


class CoinbaseLoader(DataLoader):
    def __init__(self, base_url: str = None):
        self.base_url = base_url or Config.COINBASE_BASE_URL

    def load(
        self,
        symbol: str,
        start: str,
        end: str,
        timeframe: str = "1h",
    ) -> OHLCVData:
        import httpx
        symbol = symbol.upper().replace("-", "-")
        tf_map = {
            "1m": 60, "5m": 300, "15m": 900, "1h": 3600,
            "6h": 21600, "1d": 86400,
        }
        granularity = tf_map.get(timeframe, 3600)
        start_iso = pd.Timestamp(start).isoformat()
        end_iso = pd.Timestamp(end).isoformat()
        rows = []
        resp = httpx.get(
            f"{self.base_url}/products/{symbol}/candles",
            params={
                "start": start_iso,
                "end": end_iso,
                "granularity": granularity,
            },
            timeout=30,
        )
        resp.raise_for_status()
        for k in resp.json():
            rows.append({
                "timestamp": pd.Timestamp(k[0], unit="s"),
                "open": float(k[3]),
                "high": float(k[2]),
                "low": float(k[1]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        return OHLCVData(symbol=symbol, timeframe=timeframe, df=df)
