import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class MarkowitzOptimizer:
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate

    def efficient_frontier(self, returns: pd.DataFrame, points: int = 50) -> pd.DataFrame:
        mean = returns.mean() * 252
        cov = returns.cov() * 252
        n = len(mean)
        results = []
        for target in np.linspace(mean.min(), mean.max(), points):
            try:
                w = self.minimize_variance(mean, cov, target_return=target)
                ret = w @ mean
                vol = np.sqrt(w @ cov @ w)
                sr = (ret - self.risk_free_rate) / vol
                results.append({"return": ret, "volatility": vol, "sharpe": sr, **{f"w{i}": w[i] for i in range(n)}})
            except Exception:
                continue
        return pd.DataFrame(results)

    def minimize_variance(self, mean, cov, target_return: float = None):
        import cvxpy as cp
        n = len(mean)
        w = cp.Variable(n)
        risk = cp.quad_form(w, cov)
        constraints = [cp.sum(w) == 1, w >= 0]
        if target_return is not None:
            constraints.append(mean @ w == target_return)
        problem = cp.Problem(cp.Minimize(risk), constraints)
        problem.solve()
        return w.value

    def max_sharpe(self, returns: pd.DataFrame) -> Dict:
        ef = self.efficient_frontier(returns, points=100)
        if ef.empty:
            return self.min_volatility(returns)
        best = ef.loc[ef["sharpe"].idxmax()]
        n = len(returns.columns)
        weights = {returns.columns[i]: best.get(f"w{i}", 0) for i in range(n)}
        return {
            "weights": weights,
            "return": best["return"],
            "volatility": best["volatility"],
            "sharpe": best["sharpe"],
        }

    def min_volatility(self, returns: pd.DataFrame) -> Dict:
        cov_arr = returns.cov().values * 252
        n = len(cov_arr)
        import cvxpy as cp
        w = cp.Variable(n)
        risk = cp.quad_form(w, cov_arr)
        constraints = [cp.sum(w) == 1, w >= 0]
        problem = cp.Problem(cp.Minimize(risk), constraints)
        problem.solve()
        weights = pd.Series(np.maximum(w.value, 0), index=returns.columns)
        weights /= weights.sum()
        cov_df = returns.cov() * 252
        port_vol = np.sqrt(weights @ cov_df @ weights)
        return {
            "weights": weights.to_dict(),
            "return": weights @ returns.mean() * 252,
            "volatility": port_vol,
            "sharpe": (weights @ returns.mean() * 252 - self.risk_free_rate) / port_vol,
        }
