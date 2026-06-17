import pytest
import pandas as pd
import numpy as np

from quantalchemy.factors.factors import (
    FactorAnalyzer,
    information_coefficient,
    compute_factor_returns,
)


class TestFactorFunctions:
    def test_information_coefficient(self):
        factor = pd.Series(np.random.randn(100))
        fwd = pd.Series(np.random.randn(100))
        ic = information_coefficient(factor, fwd)
        assert -1 <= ic <= 1

    def test_compute_factor_returns(self):
        returns = pd.DataFrame({"A": np.random.randn(100), "B": np.random.randn(100)})
        exposure = pd.DataFrame({"A": np.random.randn(100), "B": np.random.randn(100)})
        result = compute_factor_returns(returns, exposure)
        assert isinstance(result, pd.Series)

    def test_factor_analyzer(self):
        factors = {
            "momentum": pd.Series(np.random.randn(200)),
            "volatility": pd.Series(np.random.randn(200)),
        }
        analyzer = FactorAnalyzer(factors)
        fwd = pd.Series(np.random.randn(200))
        ics = analyzer.compute_rank_ics(fwd)
        assert len(ics) == 2
        corr = analyzer.factor_correlation_matrix()
        assert corr.shape == (2, 2)
