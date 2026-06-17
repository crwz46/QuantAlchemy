import os


class Config:
    BINANCE_BASE_URL: str = os.getenv("BINANCE_BASE_URL", "https://fapi.binance.com")
    BYBIT_BASE_URL: str = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")
    COINBASE_BASE_URL: str = os.getenv("COINBASE_BASE_URL", "https://api.exchange.coinbase.com")
    DATA_DIR: str = os.getenv("DATA_DIR", "./data/market")
    DEFAULT_START: str = "2023-01-01"
    DEFAULT_END: str = "2024-01-01"
    RISK_FREE_RATE: float = 0.05
