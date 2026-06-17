from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class WalkForwardResult:
    windows: List[Dict] = field(default_factory=list)
    combined_equity: pd.Series = None
    performance: Dict = None


class WalkForwardAnalyzer:
    def __init__(self, train_pct: float = 0.7, step_pct: float = 0.1):
        self.train_pct = train_pct
        self.step_pct = step_pct

    def run(
        self,
        data: pd.DataFrame,
        strategy_fn: Callable,
        param_grid: Dict[str, List],
        n_windows: int = 5,
    ) -> WalkForwardResult:
        result = WalkForwardResult()
        total = len(data)
        train_size = int(total * self.train_pct)
        step_size = int(total * self.step_pct)
        best_configs = []

        for w in range(n_windows):
            train_end = train_size + w * step_size
            if train_end >= total:
                break
            val_start = train_end
            val_end = min(val_start + step_size, total)
            train = data.iloc[:train_end]
            val = data.iloc[val_start:val_end]
            if len(val) < 10:
                break

            best_params = None
            best_sharpe = -np.inf
            for p_name, p_values in param_grid.items():
                for p_val in p_values:
                    params = {p_name: p_val}
                    try:
                        eq = strategy_fn(train, params)
                        ret = eq.pct_change().dropna()
                        sharpe = np.sqrt(252) * ret.mean() / ret.std() if ret.std() > 0 else -np.inf
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params = params
                    except Exception:
                        continue

            val_eq = strategy_fn(val, best_params) if best_params else pd.Series(dtype=float)
            best_configs.append(best_params)
            result.windows.append({
                "window": w,
                "train_start": data.index[0],
                "train_end": data.index[train_end - 1],
                "val_start": data.index[val_start],
                "val_end": data.index[val_end - 1],
                "params": best_params,
                "train_sharpe": best_sharpe,
                "val_equity": val_eq,
            })

        equities = [w["val_equity"] for w in result.windows if w["val_equity"] is not None and len(w["val_equity"]) > 0]
        if equities:
            combined = pd.concat(equities)
            combined = combined[~combined.index.duplicated(keep="first")].sort_index()
            result.combined_equity = combined

        return result
