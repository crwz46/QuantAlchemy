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

    def _fig_combined(self) -> go.Figure:
        peak = self.equity.expanding().max()
        dd = (self.equity - peak) / peak * 100
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        fig.add_trace(go.Scatter(
            x=self.equity.index, y=self.equity.values,
            mode="lines", name="Equity",
            line=dict(color="#2563eb", width=2),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            mode="lines", name="Drawdown",
            fill="tozeroy", line=dict(color="#dc2626", width=1.5),
        ), row=2, col=1)
        fig.update_layout(
            title=dict(text="Equity & Drawdown", font=dict(size=18)),
            height=500, margin=dict(l=40, r=20, t=50, b=30),
            hovermode="x unified",
        )
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        return fig

    def _fig_drawdown(self) -> go.Figure:
        peak = self.equity.expanding().max()
        dd = (self.equity - peak) / peak * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            mode="lines", name="Drawdown",
            fill="tozeroy", line=dict(color="#dc2626", width=1.5),
        ))
        fig.update_layout(
            title="Drawdown (%)", xaxis_title="Date", yaxis_title="Drawdown %",
            height=300, margin=dict(l=40, r=20, t=40, b=30),
        )
        return fig

    def _fig_returns_dist(self) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=self.returns.values * 100, nbinsx=50,
            name="Returns", marker_color="#2563eb",
            marker_line_color="white", marker_line_width=0.5,
        ))
        fig.update_layout(
            title="Returns Distribution", xaxis_title="Return %",
            yaxis_title="Frequency", height=300,
            margin=dict(l=40, r=20, t=40, b=30),
            bargap=0.05,
        )
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
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values * 100, x=pivot.columns, y=pivot.index,
            colorscale="RdYlGn", zmid=0,
            text=np.round(pivot.values * 100, 1),
            texttemplate="%{text}",
            hovertemplate="Month: %{y}<br>Year: %{x}<br>Return: %{z:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            title="Monthly Returns (%)", height=400,
            margin=dict(l=80, r=20, t=40, b=30),
        )
        return fig

    def _fig_rolling_sharpe(self) -> go.Figure:
        rolling = self.returns.rolling(60).apply(
            lambda r: np.sqrt(252) * r.mean() / r.std() if r.std() > 0 else 0
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling.index, y=rolling.values,
            mode="lines", name="60-period Rolling Sharpe",
            line=dict(color="#8b5cf6", width=2),
        ))
        fig.add_hline(y=0, line=dict(color="gray", dash="dash"))
        fig.add_hline(y=1, line=dict(color="green", dash="dot"))
        fig.update_layout(
            title="Rolling Sharpe Ratio (60-period)", xaxis_title="Date",
            yaxis_title="Sharpe Ratio", height=300,
            margin=dict(l=40, r=20, t=40, b=30),
        )
        return fig

    def create_html(self) -> str:
        import plotly.io as pio
        combined_div = pio.to_html(self._fig_combined(), include_plotlyjs=False, full_html=False)
        rd_div = pio.to_html(self._fig_returns_dist(), include_plotlyjs=False, full_html=False)
        mh_div = pio.to_html(self._fig_monthly_heatmap(), include_plotlyjs=False, full_html=False)
        rs_div = pio.to_html(self._fig_rolling_sharpe(), include_plotlyjs=False, full_html=False)

        metrics_rows = ""
        for k, v in self.metrics.items():
            key = k.replace("_", " ").title()
            metrics_rows += f"<tr><td class='metric-label'>{key}</td><td class='metric-value'>{v}</td></tr>"

        trades_html = ""
        if self.trades is not None and len(self.trades) > 0:
            df = self.trades.copy()
            if "returns" in df.columns:
                df["returns"] = df["returns"].map("{:.2%}".format)
            trades_html = df.to_html(classes="table", index=False)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QuantAlchemy Tearsheet</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f8fafc; color: #1e293b; line-height: 1.6;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 40px 20px; }}
  h1 {{
    font-size: 28px; font-weight: 700; color: #0f172a;
    border-bottom: 3px solid #2563eb; padding-bottom: 12px; margin-bottom: 32px;
  }}
  h2 {{
    font-size: 20px; font-weight: 600; color: #1e293b;
    margin: 32px 0 16px; padding-left: 12px; border-left: 4px solid #2563eb;
  }}
  .metrics-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 16px; margin-bottom: 32px;
  }}
  .metric-card {{
    background: white; border-radius: 12px; padding: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); transition: box-shadow 0.2s;
  }}
  .metric-card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.12); }}
  .metric-card .label {{
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;
    color: #64748b; font-weight: 600;
  }}
  .metric-card .value {{
    font-size: 22px; font-weight: 700; color: #0f172a; margin-top: 4px;
  }}
  .metric-card .value.positive {{ color: #16a34a; }}
  .metric-card .value.negative {{ color: #dc2626; }}
  .chart-container {{
    background: white; border-radius: 12px; padding: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 24px;
  }}
  .table {{
    width: 100%; border-collapse: collapse; background: white;
    border-radius: 12px; overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }}
  .table th {{
    background: #f1f5f9; color: #475569; font-weight: 600;
    font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;
    padding: 12px 16px; text-align: left;
  }}
  .table td {{ padding: 10px 16px; border-top: 1px solid #e2e8f0; }}
  .table tr:hover td {{ background: #f8fafc; }}
  .footer {{ text-align: center; color: #94a3b8; font-size: 13px; margin-top: 40px; }}
</style>
</head>
<body>
<div class="container">
<h1>⚗️ QuantAlchemy — Performance Tearsheet</h1>

<h2>Performance Metrics</h2>
<div class="metrics-grid">
"""
        for k, v in self.metrics.items():
            metric_id = k.lower().replace(" ", "_")
            cls = "value"
            if "return" in k and k != "avg_return":
                try:
                    cls += " positive" if float(v.strip("%")) >= 0 else " negative"
                except Exception:
                    pass
            key = k.replace("_", " ").title()
            html += f'<div class="metric-card"><div class="label">{key}</div><div class="{cls}">{v}</div></div>'

        html += f"""
</div>

<h2>Equity & Drawdown</h2>
<div class="chart-container">{combined_div}</div>

<h2>Returns Distribution</h2>
<div class="chart-container">{rd_div}</div>

<h2>Rolling Sharpe Ratio</h2>
<div class="chart-container">{rs_div}</div>

<h2>Monthly Returns</h2>
<div class="chart-container">{mh_div}</div>
"""
        if trades_html:
            html += f'<h2>Trades</h2><div class="chart-container">{trades_html}</div>'

        html += f"""
<div class="footer">
  Generated by QuantAlchemy — Quantitative Trading Platform
</div>
</div>
</body>
</html>"""
        return html
