import numpy as np
import pandas as pd
from typing import Dict

from quantalchemy.config import Config


class PerformanceMetrics:
    @staticmethod
    def compute(equity: pd.Series, returns: pd.Series = None, rf: float = None) -> Dict:
        if returns is None:
            returns = equity.pct_change().dropna()
        rf = rf or Config.RISK_FREE_RATE
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1 if len(equity) > 1 else 0
        n_years = len(returns) / 252 if len(returns) > 252 else len(returns) / (365 * 24)
        if n_years <= 0:
            n_years = 1
        cagr = (1 + total_return) ** (1 / n_years) - 1
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        downside = returns[returns < 0]
        sortino = np.sqrt(252) * returns.mean() / downside.std() if len(downside) > 0 and downside.std() > 0 else 0
        peak = equity.expanding().max()
        dd = (equity - peak) / peak
        max_dd = dd.min()
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        win_trades = (returns > 0).sum()
        total_trades = len(returns)
        winrate = win_trades / total_trades if total_trades > 0 else 0
        avg_return = returns.mean()
        var_95 = returns.quantile(0.05)
        cumulative = (1 + returns).cumprod()
        monthly = pd.Series(dtype=float)
        monthly_avg = 0.0
        monthly_std = 0.0
        if isinstance(cumulative.index, pd.DatetimeIndex):
            monthly = cumulative.resample("ME").last().pct_change().dropna()
            monthly_avg = monthly.mean()
            monthly_std = monthly.std()
        return {
            "total_return": total_return,
            "cagr": cagr,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "winrate": winrate,
            "avg_return": avg_return,
            "volatility": returns.std() * np.sqrt(252),
            "var_95": var_95,
            "monthly_avg_return": monthly_avg,
            "monthly_volatility": monthly_std,
            "total_trades": total_trades,
        }

    @staticmethod
    def format(metrics: Dict) -> Dict:
        fmts = {
            "total_return": "{:.2%}",
            "cagr": "{:.2%}",
            "sharpe_ratio": "{:.2f}",
            "sortino_ratio": "{:.2f}",
            "max_drawdown": "{:.2%}",
            "calmar_ratio": "{:.2f}",
            "winrate": "{:.2%}",
            "avg_return": "{:.4%}",
            "volatility": "{:.2%}",
            "var_95": "{:.2%}",
            "monthly_avg_return": "{:.2%}",
            "monthly_volatility": "{:.2%}",
        }
        return {k: fmts.get(k, "{}").format(v) for k, v in metrics.items()}
