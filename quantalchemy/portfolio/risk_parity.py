import numpy as np
import pandas as pd
from typing import Dict
from scipy.optimize import minimize


class RiskParityOptimizer:
    def _risk_parity_objective(self, w: np.ndarray, cov: np.ndarray) -> float:
        port_var = w @ cov @ w
        sigma = np.sqrt(port_var)
        rc = w * (cov @ w) / sigma
        target = sigma / len(w)
        return np.sum((rc - target) ** 2)

    def optimize(self, returns: pd.DataFrame) -> Dict:
        cov_arr = returns.cov().values * 252
        n = len(cov_arr)
        w0 = np.ones(n) / n
        bounds = [(0, 1)] * n
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        result = minimize(
            self._risk_parity_objective, w0, args=(cov_arr,),
            bounds=bounds, constraints=constraints,
            method="SLSQP",
        )
        weights = pd.Series(result.x, index=returns.columns)
        cov_df = returns.cov() * 252
        port_vol = np.sqrt(weights @ cov_df @ weights)
        return {
            "weights": weights.to_dict(),
            "volatility": port_vol,
            "return": weights @ (returns.mean() * 252),
        }
