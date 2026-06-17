"""
SMA Crossover Strategy Analysis — QuantAlchemy Demo
====================================================
Run: python examples/sma_crossover_analysis.py
Output: tearsheet_sma_AAPL.html + console report
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantalchemy.data.yahoo import YahooLoader
from quantalchemy.backtest.engine import BacktestEngine
from quantalchemy.backtest.strategy import SMACrossover
from quantalchemy.backtest.metrics import PerformanceMetrics
from quantalchemy.tearsheet.report import TearsheetReport

SYMBOL = "AAPL"
START = "2020-01-01"
END = "2024-01-01"
PARAMS = [(10, 30), (20, 50), (50, 200)]
INITIAL_CAPITAL = 10000

print(f"{'='*60}")
print(f"  QuantAlchemy — SMA Crossover Analysis")
print(f"  Symbol: {SYMBOL} | {START} to {END}")
print(f"{'='*60}")

loader = YahooLoader()
data = loader.load(SYMBOL, START, END, "1d")
print(f"\nLoaded {len(data.df)} bars")

best_sharpe = -999
best_result = None
best_params = None

for fast, slow in PARAMS:
    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
    result = engine.run(SMACrossover(fast, slow), data.df)
    metrics = PerformanceMetrics.format(result.metrics)
    sharpe = result.metrics["sharpe_ratio"]

    print(f"\n  SMA({fast},{slow})")
    print(f"    Sharpe:     {metrics['sharpe_ratio']}")
    print(f"    Return:     {metrics['total_return']}")
    print(f"    Volatility: {metrics['volatility']}")
    print(f"    Max DD:     {metrics['max_drawdown']}")
    print(f"    Winrate:    {metrics['winrate']}")
    print(f"    Trades:     {metrics['total_trades']}")

    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_result = result
        best_params = (fast, slow)

print(f"\n{'='*60}")
print(f"  BEST: SMA({best_params[0]},{best_params[1]}) — Sharpe {best_result.metrics['sharpe_ratio']:.2f}")
print(f"{'='*60}")

# Generate tearsheet
best_result.trades["returns"] = best_result.trades["returns"].astype(float)
report = TearsheetReport(
    equity=best_result.equity,
    returns=best_result.returns,
    metrics=PerformanceMetrics.format(best_result.metrics),
    trades=best_result.trades,
)
html = report.create_html()
output = f"tearsheet_sma_{SYMBOL}.html"
Path(output).write_text(html, encoding="utf-8")
print(f"\n✅ Tearsheet saved to {output}")
