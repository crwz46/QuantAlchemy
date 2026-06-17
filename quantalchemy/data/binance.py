import time
from typing import List, Optional

import pandas as pd

from quantalchemy.config import Config
from quantalchemy.data.base import DataLoader, OHLCVData, OHLCV_COLUMNS


class BinanceLoader(DataLoader):
    def __init__(self, base_url: str = None):
        self.base_url = base_url or Config.BINANCE_BASE_URL

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
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w",
        }
        interval = tf_map.get(timeframe, "1h")
        start_ms = int(pd.Timestamp(start).timestamp() * 1000)
        end_ms = int(pd.Timestamp(end).timestamp() * 1000)
        rows = []
        limit = 1500
        while start_ms < end_ms:
            resp = httpx.get(f"{self.base_url}/fapi/v1/klines", params={
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "limit": limit,
            }, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            for k in data:
                rows.append({
                    "timestamp": pd.Timestamp(k[0], unit="ms"),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })
            start_ms = data[-1][0] + 1
            if len(data) < limit:
                break
            time.sleep(0.1)
        df = pd.DataFrame(rows).set_index("timestamp")
        return OHLCVData(symbol=symbol, timeframe=timeframe, df=df)

    def load_funding_rate(
        self, symbol: str, start: str, end: str,
    ) -> pd.DataFrame:
        import httpx
        symbol = symbol.upper().replace("-", "")
        start_ms = int(pd.Timestamp(start).timestamp() * 1000)
        end_ms = int(pd.Timestamp(end).timestamp() * 1000)
        rows = []
        limit = 1000
        while start_ms < end_ms:
            resp = httpx.get(f"{self.base_url}/fapi/v1/fundingRate", params={
                "symbol": symbol,
                "startTime": start_ms,
                "limit": limit,
            }, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            for r in data:
                rows.append({
                    "timestamp": pd.Timestamp(r["fundingTime"], unit="ms"),
                    "funding_rate": float(r["fundingRate"]),
                })
            start_ms = data[-1]["fundingTime"] + 1
            if len(data) < limit:
                break
            time.sleep(0.1)
        return pd.DataFrame(rows).set_index("timestamp")

    def load_open_interest(
        self, symbol: str, start: str, end: str,
    ) -> pd.DataFrame:
        import httpx
        symbol = symbol.upper().replace("-", "")
        start_ms = int(pd.Timestamp(start).timestamp() * 1000)
        end_ms = int(pd.Timestamp(end).timestamp() * 1000)
        rows = []
        limit = 500
        while start_ms < end_ms:
            resp = httpx.get(f"{self.base_url}/futures/data/openInterestHist", params={
                "symbol": symbol,
                "period": "1h",
                "startTime": start_ms,
                "limit": limit,
            }, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            for r in data:
                rows.append({
                    "timestamp": pd.Timestamp(r["timestamp"], unit="ms"),
                    "open_interest": float(r["sumOpenInterest"]),
                    "open_interest_value": float(r["sumOpenInterestValue"]),
                })
            start_ms = data[-1]["timestamp"] + 1
            if len(data) < limit:
                break
            time.sleep(0.1)
        return pd.DataFrame(rows).set_index("timestamp")
