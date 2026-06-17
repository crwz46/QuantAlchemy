from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from quantalchemy.backtest.engine import BacktestEngine, BacktestResult
from quantalchemy.backtest.strategy import Strategy


@dataclass
class LabResult:
    backtest_result: BacktestResult
    params: Dict
    name: str


class StrategyLab:
    def __init__(self, initial_capital: float = 10000):
        self.engine = BacktestEngine(initial_capital=initial_capital)

    def run(
        self,
        strategy_class: type,
        data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
    ) -> List[LabResult]:
        results = []
        from itertools import product
        keys = list(param_grid.keys())
        for combo in product(*param_grid.values()):
            params = dict(zip(keys, combo))
            strategy = strategy_class(**params)
            bt = self.engine.run(strategy, data)
            name = f"{strategy_class.__name__}({params})"
            results.append(LabResult(backtest_result=bt, params=params, name=name))
        results.sort(key=lambda r: r.backtest_result.metrics.get("sharpe_ratio", -np.inf), reverse=True)
        return results

    def compare(self, results: List[LabResult]) -> pd.DataFrame:
        rows = []
        for r in results:
            row = {"name": r.name, **r.backtest_result.metrics}
            rows.append(row)
        return pd.DataFrame(rows).set_index("name")
