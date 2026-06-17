from dataclasses import dataclass, field
from io import BytesIO
from typing import Dict, List, Optional
import base64

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class TearsheetReport:
    equity: pd.Series
    returns: pd.Series
    metrics: Dict
    trades: pd.DataFrame = None

    def _fig_equity(self) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.equity.index, y=self.equity.values, mode="lines", name="Equity"))
        fig.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Equity", height=400)
        return fig

    def _fig_drawdown(self) -> go.Figure:
        peak = self.equity.expanding().max()
        dd = (self.equity - peak) / peak
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values * 100, mode="lines", name="Drawdown", fill="tozeroy"))
        fig.update_layout(title="Drawdown (%)", xaxis_title="Date", yaxis_title="Drawdown %", height=300)
        return fig

    def _fig_returns_dist(self) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=self.returns.values * 100, nbinsx=50, name="Returns"))
        fig.update_layout(title="Returns Distribution", xaxis_title="Return %", yaxis_title="Frequency", height=300)
        return fig

    def _fig_monthly_heatmap(self) -> go.Figure:
        cum = (1 + self.returns).cumprod()
        monthly = cum.resample("ME").last().pct_change().dropna()
        if monthly.empty:
            return go.Figure()
        years = monthly.index.year
        months = monthly.index.month_name()
        heat = pd.DataFrame({"return": monthly.values, "year": years, "month": months})
        pivot = heat.pivot_table(index="month", columns="year", values="return", aggfunc="sum")
        fig = go.Figure(data=go.Heatmap(z=pivot.values * 100, x=pivot.columns, y=pivot.index,
                                         colorscale="RdYlGn", text=np.round(pivot.values * 100, 1),
                                         texttemplate="%{text}"))
        fig.update_layout(title="Monthly Returns (%)", height=400)
        return fig

    def create_html(self) -> str:
        import plotly.io as pio
        eq_div = pio.to_html(self._fig_equity(), include_plotlyjs=False, full_html=False)
        dd_div = pio.to_html(self._fig_drawdown(), include_plotlyjs=False, full_html=False)
        rd_div = pio.to_html(self._fig_returns_dist(), include_plotlyjs=False, full_html=False)
        mh_div = pio.to_html(self._fig_monthly_heatmap(), include_plotlyjs=False, full_html=False)

        metrics_html = "<table><tr><th>Metric</th><th>Value</th></tr>"
        for k, v in self.metrics.items():
            metrics_html += f"<tr><td>{k}</td><td>{v}</td></tr>"
        metrics_html += "</table>"

        trades_html = ""
        if self.trades is not None and len(self.trades) > 0:
            trades_html = self.trades.to_html(classes="table table-striped")

        html = f"""<!DOCTYPE html>
<html><head><script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>body{{font-family:Arial,sans-serif;margin:40px}}
table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ddd;padding:8px;text-align:left}}
th{{background-color:#f2f2f2}}
h2{{color:#333}}</style></head><body>
<h1>Tearsheet Report</h1>
<h2>Performance Metrics</h2>{metrics_html}
<h2>Equity Curve</h2>{eq_div}
<h2>Drawdown</h2>{dd_div}
<h2>Returns Distribution</h2>{rd_div}
<h2>Monthly Returns</h2>{mh_div}
<h2>Trades</h2>{trades_html}
</body></html>"""
        return html
