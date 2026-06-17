import time
from typing import Optional

import pandas as pd

from quantalchemy.config import Config
from quantalchemy.data.base import DataLoader, OHLCVData


class BybitLoader(DataLoader):
    def __init__(self, base_url: str = None):
        self.base_url = base_url or Config.BYBIT_BASE_URL

    def load(
        self,
        symbol: str,
        start: str,
        end: str,
        timeframe: str = "1h",
    ) -> OHLCVData:
        import httpx
        symbol = symbol.upper().replace("-", "")
        tf_map = {
            "1m": "1", "5m": "5", "15m": "15", "30m": "30",
            "1h": "60", "4h": "240", "1d": "D", "1w": "W",
        }
        interval = tf_map.get(timeframe, "60")
        start_ms = int(pd.Timestamp(start).timestamp() * 1000)
        end_ms = int(pd.Timestamp(end).timestamp() * 1000)
        rows = []
        limit = 1000
        while start_ms < end_ms:
            resp = httpx.get(f"{self.base_url}/v5/market/kline", params={
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "start": start_ms,
                "limit": limit,
            }, timeout=30)
            resp.raise_for_status()
            data = resp.json().get("result", {}).get("list", [])
            if not data:
                break
            for k in reversed(data):
                ts = int(k[0])
                if ts > end_ms:
                    continue
                rows.append({
                    "timestamp": pd.Timestamp(ts, unit="ms"),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })
                start_ms = ts + 1
            if len(data) < limit:
                break
            time.sleep(0.1)
        df = pd.DataFrame(rows).set_index("timestamp")
        return OHLCVData(symbol=symbol, timeframe=timeframe, df=df)
