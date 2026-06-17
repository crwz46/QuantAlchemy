from typing import Optional

import pandas as pd

from quantalchemy.data.base import DataLoader, OHLCVData


class YahooLoader(DataLoader):
    def load(
        self,
        symbol: str,
        start: str,
        end: str,
        timeframe: str = "1d",
    ) -> OHLCVData:
        import httpx
        symbol = symbol.upper()
        period1 = int(pd.Timestamp(start).timestamp())
        period2 = int(pd.Timestamp(end).timestamp())
        tf_map = {
            "1d": "1d", "1wk": "1wk", "1mo": "1mo",
        }
        interval = tf_map.get(timeframe, "1d")
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        resp = httpx.get(url, params={
            "period1": period1,
            "period2": period2,
            "interval": interval,
        }, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        data = resp.json()["chart"]["result"][0]
        timestamps = data["timestamp"]
        quote = data["indicators"]["quote"][0]
        rows = []
        for i, ts in enumerate(timestamps):
            o = quote.get("open", [None])[i]
            if o is None:
                continue
            rows.append({
                "timestamp": pd.Timestamp(ts, unit="s"),
                "open": float(o),
                "high": float(quote["high"][i]),
                "low": float(quote["low"][i]),
                "close": float(quote["close"][i]),
                "volume": float(quote["volume"][i]),
            })
        df = pd.DataFrame(rows).set_index("timestamp")
        return OHLCVData(symbol=symbol, timeframe=timeframe, df=df)
