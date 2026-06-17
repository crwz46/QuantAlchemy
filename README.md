# ⚗️ QuantAlchemy

**Quantitative trading platform** — backtesting, portfolio optimization, factor analysis, regime detection, walk-forward validation, alpha discovery, and interactive dashboards.

## Architecture

```
QuantAlchemy/
├── quantalchemy/               # Core library
│   ├── data/                   # Multi-exchange data loaders
│   │   ├── binance.py          # Binance Futures (HTTPS)
│   │   ├── bybit.py            # Bybit V5
│   │   ├── coinbase.py         # Coinbase Exchange
│   │   ├── yahoo.py            # Yahoo Finance
│   │   └── csv_loader.py       # Local CSV/Excel
│   ├── backtest/               # Backtesting engine
│   │   ├── engine.py           # Event-loop simulator
│   │   ├── strategy.py         # SMA, RSI, ML strategies
│   │   └── metrics.py          # Sharpe, Sortino, CAGR, Max DD, etc.
│   ├── portfolio/              # Portfolio optimization
│   │   ├── markowitz.py        # Mean-variance (min vol, max Sharpe)
│   │   ├── risk_parity.py      # Equal risk contribution
│   │   └── black_litterman.py  # Black-Litterman model
│   ├── factors/                # Factor analysis
│   ├── regime/                 # Regime detection (HMM, GMM, K-Means)
│   ├── walkforward/            # Walk-forward validation
│   ├── lab/                    # Strategy parameter sweeps
│   ├── tearsheet/              # HTML performance reports
│   └── alpha/                  # ML alpha discovery (XGBoost, LightGBM, CatBoost)
├── pages/                      # Streamlit dashboard
│   ├── 1_Data.py
│   ├── 2_Backtest.py
│   ├── 3_Portfolio.py
│   ├── 4_Factors.py
│   ├── 5_Regime.py
│   ├── 6_WalkForward.py
│   ├── 7_Lab.py
│   ├── 8_Tearsheet.py
│   └── 9_Alpha.py
├── tests/                      # 23 unit tests
├── main.py                     # Streamlit entry point
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
streamlit run main.py
```

## Test Suite

```bash
pytest tests -v
```

## Features

| Feature | Description |
|---------|-------------|
| **Data** | Fetch OHLCV + funding rate + open interest from 5 sources |
| **Backtest** | Event-loop simulator with commission, slippage, position sizing |
| **Portfolio** | Markowitz efficient frontier, Risk Parity, Black-Litterman |
| **Factors** | IC, rank IC, factor returns, correlation matrix, top/bottom spread |
| **Regime** | HMM, GMM, K-Means clustering on returns + volatility |
| **Walk Forward** | Rolling window optimization with out-of-sample validation |
| **Strategy Lab** | Grid search param sweeps with ranked comparison |
| **Tearsheet** | Self-contained HTML report with equity, drawdown, monthly heatmap |
| **Alpha ML** | XGBoost / LightGBM / CatBoost classifiers with feature importance |
| **Dashboard** | 9 interactive Streamlit pages with Plotly charts |

## License

MIT
