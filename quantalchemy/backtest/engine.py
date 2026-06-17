from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from quantalchemy.backtest.strategy import Strategy
from quantalchemy.backtest.metrics import PerformanceMetrics


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    signals: pd.Series
    trades: pd.DataFrame
    metrics: Dict


class BacktestEngine:
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001, slippage: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        start: str = None,
        end: str = None,
    ) -> BacktestResult:
        df = data.copy()
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]

        signals = strategy.generate_signals(df)
        price = df["close"]
        position = 0
        cash = self.initial_capital
        equity = [self.initial_capital]
        entry_price = 0.0
        entry_time = None
        trades = []

        for i in range(1, len(df)):
            signal = signals.iloc[i]
            curr_price = price.iloc[i]
            new_position = 0
            if signal > 0:
                new_position = 1
            elif signal < 0:
                new_position = -1

            if new_position != position:
                if position != 0:
                    exit_price = curr_price * (1 + self.slippage * np.sign(position))
                    ret = position * (exit_price / entry_price - 1) - self.commission
                    cash = equity[-1] * (1 + ret)
                    trades.append({
                        "entry_time": entry_time,
                        "entry_price": entry_price,
                        "exit_time": df.index[i],
                        "exit_price": exit_price,
                        "returns": ret,
                    })
                if new_position != 0:
                    entry_price = curr_price * (1 + self.slippage * np.sign(new_position))
                    entry_time = df.index[i]
                position = new_position

            if position != 0 and entry_price > 0:
                unrealized = position * (curr_price / entry_price - 1)
                equity.append(cash * (1 + unrealized))
            else:
                equity.append(cash)

        if position != 0 and entry_price > 0:
            exit_price = price.iloc[-1] * (1 + self.slippage * np.sign(position))
            ret = position * (exit_price / entry_price - 1) - self.commission
            trades.append({
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_time": df.index[-1],
                "exit_price": exit_price,
                "returns": ret,
            })

        eq = pd.Series(equity, index=df.index[:len(equity)])
        rets = eq.pct_change().dropna()
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        metrics = PerformanceMetrics.compute(eq, rets)

        return BacktestResult(
            equity=eq,
            returns=rets,
            signals=signals,
            trades=trades_df,
            metrics=metrics,
        )
