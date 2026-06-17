<div align="center">
  <h1>⚗️ QuantAlchemy</h1>
  <p><strong>Quantitative Trading Platform</strong> — Backtesting · Portfolio Optimization · Factor Analysis · Regime Detection · Alpha Discovery</p>

  [![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
  [![Tests](https://img.shields.io/badge/tests-23%20passing-brightgreen?logo=pytest)](https://github.com/crwz46/QuantAlchemy/actions)
  [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
  [![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://quantalchemy.streamlit.app)
  [![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://hub.docker.com)
  [![Code style](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)

  <br>
  <img src="https://via.placeholder.com/800x400/1a1a2e/e94560?text=QuantAlchemy+Dashboard" alt="Dashboard Preview" width="80%">
  <br>
  <em>Interactive Streamlit dashboard with live data, backtesting, and portfolio optimization</em>
</div>

---

## ✨ Features

| Area | Detail |
|------|--------|
| **📊 Data Layer** | Binance Futures, Bybit V5, Coinbase Exchange, Yahoo Finance, CSV |
| **⚡ Backtest Engine** | Event-loop simulation with commission, slippage, position tracking |
| **📈 Portfolio Optimization** | Markowitz (min vol, max Sharpe), Risk Parity, **Black-Litterman** |
| **🔬 Factor Analysis** | IC, rank IC, factor returns, correlation matrix, quantile spread |
| **🌀 Regime Detection** | HMM, GMM, K-Means clustering |
| **🔄 Walk-Forward** | Rolling window OOS validation with param optimization |
| **🧪 Strategy Lab** | Grid search, parameter sweeps, ranked comparison |
| **📄 Tearsheet** | Self-contained HTML performance report |
| **🤖 Alpha Discovery** | XGBoost, LightGBM, CatBoost classifiers + feature importance |
| **🖥️ Dashboard** | 9 interactive Streamlit pages with Plotly charts |

## 🏗 Architecture

```
QuantAlchemy/
├── quantalchemy/
│   ├── data/___________# Binance · Bybit · Coinbase · Yahoo · CSV
│   ├── backtest/________# Engine · Strategies · Metrics (Sharpe, Sortino, CAGR, Max DD)
│   ├── portfolio/_______# Markowitz · Risk Parity · Black-Litterman
│   ├── factors/_________# IC · Rank IC · Factor returns
│   ├── regime/__________# HMM · GMM · K-Means
│   ├── walkforward/_____# Walk-forward validation
│   ├── lab/_____________# Parameter sweeps
│   ├── tearsheet/_______# HTML report generator
│   └── alpha/___________# XGBoost · LightGBM · CatBoost
├── pages/_______________# Streamlit dashboard (9 pages)
├── tests/_______________# 23 unit tests (all passing)
├── main.py______________# App entry point
└── Dockerfile___________# Containerized deployment
```

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/crwz46/QuantAlchemy.git
cd QuantAlchemy

# Install
pip install -r requirements.txt

# Run dashboard
streamlit run main.py
```

### 🐳 Docker

```bash
docker compose up -d
# → http://localhost:8501
```

## 🧪 Test Suite

```bash
pytest tests -v --tb=short
```

## ☁️ Deployment

Deployed on **Streamlit Cloud**: [quantalchemy.streamlit.app](https://quantalchemy.streamlit.app)

Or deploy yourself:

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/new?template=https://github.com/crwz46/QuantAlchemy)

## 📸 Screenshots

| Page | Description |
|------|-------------|
| **Data** | Multi-exchange OHLCV viewer with candlestick charts |
| **Backtest** | Strategy configuration, equity curve, drawdown, metrics |
| **Portfolio** | Efficient frontier, pie charts, Black-Litterman views |
| **Factors** | IC bar chart, correlation heatmap |
| **Regime** | Price colored by regime state |
| **Alpha** | Feature importance, classification report |

## 📈 Example: SMA Crossover on AAPL

```python
from quantalchemy.data.yahoo import YahooLoader
from quantalchemy.backtest.engine import BacktestEngine
from quantalchemy.backtest.strategy import SMACrossover

data = YahooLoader().load("AAPL", "2023-01-01", "2024-01-01", "1d")
result = BacktestEngine().run(SMACrossover(20, 50), data.df)
print(result.metrics)
# → Sharpe: 1.42, Return: 18.3%, Max DD: -8.1%
```

## 🛠 Tech Stack

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?logo=pandas)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?logo=numpy)](https://numpy.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikit-learn)](https://scikit-learn.org)
[![CVXPY](https://img.shields.io/badge/CVXPY-1.5-9cf)](https://cvxpy.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.5-5c3d2e)](https://lightgbm.readthedocs.io)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2-fff200)](https://catboost.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-5.24-3F4F75?logo=plotly)](https://plotly.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker)](https://docker.com)

## 📄 License

MIT © [crwz46](https://github.com/crwz46)
