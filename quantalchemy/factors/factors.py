import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy.stats import pearsonr, spearmanr


def compute_factor_returns(
    returns: pd.DataFrame,
    factor_exposures: pd.DataFrame,
) -> pd.Series:
    aligned = returns.align(factor_exposures, join="inner")
    r, f = aligned
    coeffs = []
    for t in r.index:
        y = r.loc[t].values
        X = f.loc[t].values
        if len(y) < 2 or np.std(y) == 0:
            coeffs.append(np.nan)
        else:
            coeffs.append(np.polyfit(X, y, 1)[0])
    return pd.Series(coeffs, index=r.index)


def information_coefficient(
    factor_vals: pd.Series,
    forward_returns: pd.Series,
    method: str = "pearson",
) -> float:
    valid = factor_vals.notna() & forward_returns.notna()
    if valid.sum() < 2:
        return np.nan
    if method == "spearman":
        return spearmanr(factor_vals[valid], forward_returns[valid])[0]
    return pearsonr(factor_vals[valid], forward_returns[valid])[0]


def factor_rank_ic(
    factor_vals: pd.Series,
    forward_returns: pd.Series,
) -> float:
    return information_coefficient(factor_vals, forward_returns, method="spearman")


class FactorAnalyzer:
    def __init__(self, factor_data: Dict[str, pd.Series]):
        self.factor_data = factor_data

    def compute_ics(
        self,
        forward_returns: pd.Series,
        method: str = "pearson",
    ) -> Dict[str, float]:
        return {
            name: information_coefficient(fv, forward_returns, method)
            for name, fv in self.factor_data.items()
        }

    def compute_rank_ics(self, forward_returns: pd.Series) -> Dict[str, float]:
        return self.compute_ics(forward_returns, method="spearman")

    def factor_correlation_matrix(self) -> pd.DataFrame:
        df = pd.DataFrame(self.factor_data)
        return df.corr()

    def top_bottom_spread(
        self,
        factor_name: str,
        forward_returns: pd.Series,
        quantiles: int = 5,
    ) -> Dict:
        fv = self.factor_data[factor_name]
        labels = pd.qcut(fv, quantiles, labels=False)
        grouped = forward_returns.groupby(labels).mean()
        spread = grouped.iloc[-1] - grouped.iloc[0]
        return {"quantile_means": grouped.to_dict(), "spread": spread}
