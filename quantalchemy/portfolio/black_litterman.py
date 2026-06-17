import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class BlackLitterman:
    def __init__(self, delta: float = 1.0, tau: float = 0.05):
        self.delta = delta
        self.tau = tau

    def implied_returns(self, cov: np.ndarray, market_caps: np.ndarray) -> np.ndarray:
        w_mkt = market_caps / market_caps.sum()
        return self.delta * cov @ w_mkt

    def blend(
        self,
        cov: np.ndarray,
       Implied_returns: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        omega: np.ndarray = None,
    ) -> np.ndarray:
        n = len(Implied_returns)
        pi = Implied_returns.reshape(-1, 1)
        if omega is None:
            omega = np.diag(np.diag(P @ (self.tau * cov) @ P.T))
        inv_tau_cov = np.linalg.inv(self.tau * cov)
        inv_omega = np.linalg.inv(omega)
        mid = inv_tau_cov + P.T @ inv_omega @ P
        rhs = inv_tau_cov @ pi + P.T @ inv_omega @ Q.reshape(-1, 1)
        return np.linalg.solve(mid, rhs).flatten()

    def optimize(
        self,
        returns_df: pd.DataFrame,
        market_caps: np.ndarray,
        views_assets: List[int],
        views_returns: List[float],
    ) -> Dict:
        cov = returns_df.cov() * 252
        mean = returns_df.mean() * 252
        pi = self.implied_returns(cov.values, market_caps)
        n = len(pi)
        P = np.zeros((len(views_assets), n))
        for i, idx in enumerate(views_assets):
            P[i, idx] = 1
        Q = np.array(views_returns)
        mu_bl = self.blend(cov.values, pi, P, Q)
        import cvxpy as cp
        w = cp.Variable(n)
        ret = mu_bl @ w
        risk = cp.quad_form(w, cov.values)
        constraints = [cp.sum(w) == 1, w >= 0]
        problem = cp.Problem(cp.Maximize(ret - 0.5 * self.delta * risk), constraints)
        problem.solve()
        weights = pd.Series(w.value, index=returns_df.columns)
        port_ret = weights @ mu_bl
        port_vol = np.sqrt(weights @ cov.values @ weights)
        return {
            "weights": weights.to_dict(),
            "implied_returns": dict(zip(returns_df.columns, pi)),
            "bl_returns": dict(zip(returns_df.columns, mu_bl)),
            "return": port_ret,
            "volatility": port_vol,
        }
