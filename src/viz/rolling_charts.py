"""
Additional Plotly visualizations for rolling horizon and demand forecasting.
Imports from charts.py are assumed; these extend the chart library.
"""

from typing import Dict, List, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

_BG = "#0d1117"
_CARD = "#161b22"
_BORDER = "#21262d"
_GRID = "#2a2a2a"
_TEXT = "#c9d1d9"
_MUTED = "#8b949e"
_GREEN = "#3fb950"
_RED = "#f85149"
_BLUE = "#58a6ff"
_AMBER = "#d29922"
_TEAL = "#39d353"


def _layout(**kwargs):
    d = dict(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(color=_TEXT, size=11),
        margin=dict(l=60, r=30, t=40, b=50),
        legend=dict(bgcolor=_CARD, bordercolor=_BORDER, borderwidth=1),
    )
    d.update(kwargs)
    return d


def _axis(**kw):
    return dict(gridcolor=_GRID, zerolinecolor=_GRID, **kw)


def rolling_margin_chart(execution_log) -> go.Figure:
    """
    Dual-line chart: planned vs. realized daily margin.
    Shaded area = execution gap (uncertainty cost).
    """
    days = [e.day for e in execution_log]
    planned = [e.planned_margin for e in execution_log]
    realized = [e.realized_margin for e in execution_log]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days, y=planned,
        mode="lines", name="Planned",
        line=dict(color=_BLUE, width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=days, y=realized,
        mode="lines+markers", name="Realized",
        line=dict(color=_GREEN, width=2),
        marker=dict(size=5),
    ))
    # Shaded gap
    fig.add_trace(go.Scatter(
        x=days + days[::-1],
        y=planned + realized[::-1],
        fill="toself",
        fillcolor=_AMBER + "25",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.update_layout(
        **_layout(height=320,
                  title=dict(text="Rolling Horizon: Planned vs. Realized Daily Margin", font=dict(size=14))),
        yaxis=_axis(tickformat="$,.0f", title="$/day"),
        xaxis=_axis(title="Simulation Day"),
    )
    return fig


def service_level_trend(execution_log) -> go.Figure:
    """Cumulative service level over simulation period."""
    days = []
    cum_svc = []
    total_demand_so_far = 0
    total_unmet_so_far = 0

    for e in execution_log:
        total_demand_so_far += sum(e.realized_demand.values())
        total_unmet_so_far += sum(e.unmet_demand.values())
        svc = 1 - total_unmet_so_far / max(total_demand_so_far, 1)
        days.append(e.day)
        cum_svc.append(svc * 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days, y=cum_svc,
        mode="lines+markers",
        line=dict(color=_GREEN, width=2),
        fill="tozeroy", fillcolor=_GREEN + "20",
        name="Cumulative Service Level",
    ))
    fig.add_hline(y=95, line=dict(color=_AMBER, dash="dash"),
                  annotation_text="95% target")
    fig.update_layout(
        **_layout(height=280,
                  title=dict(text="Cumulative Service Level (Rolling Horizon)", font=dict(size=14))),
        yaxis=_axis(ticksuffix="%", range=[80, 101]),
        xaxis=_axis(title="Simulation Day"),
    )
    return fig


def forecast_chart(market_id: str, forecast, history=None) -> go.Figure:
    """
    Fan chart: demand forecast with 80% and 95% prediction intervals.
    Optional: overlay last 14 days of history.
    """
    T = forecast.horizon
    days_fc = list(range(1, T + 1))

    fig = go.Figure()

    # 95% interval
    fig.add_trace(go.Scatter(
        x=days_fc + days_fc[::-1],
        y=forecast.upper_95 + forecast.lower_95[::-1],
        fill="toself", fillcolor=_BLUE + "15",
        line=dict(width=0), showlegend=True, name="95% interval",
        hoverinfo="skip",
    ))
    # 80% interval
    fig.add_trace(go.Scatter(
        x=days_fc + days_fc[::-1],
        y=forecast.upper_80 + forecast.lower_80[::-1],
        fill="toself", fillcolor=_BLUE + "30",
        line=dict(width=0), showlegend=True, name="80% interval",
        hoverinfo="skip",
    ))
    # Point forecast
    fig.add_trace(go.Scatter(
        x=days_fc, y=forecast.point_forecast,
        mode="lines+markers",
        line=dict(color=_BLUE, width=2),
        marker=dict(size=5),
        name="Forecast",
    ))

    # Historical overlay (last 14 days)
    if history:
        hist_vals = history.history[-14:]
        days_hist = list(range(-len(hist_vals) + 1, 1))
        fig.add_trace(go.Scatter(
            x=days_hist, y=hist_vals,
            mode="lines",
            line=dict(color=_MUTED, width=1.5, dash="dot"),
            name="Historical",
        ))

    fig.add_vline(x=0.5, line=dict(color=_MUTED, width=1, dash="dash"),
                  annotation_text="Forecast Start")

    fig.update_layout(
        **_layout(height=300,
                  title=dict(text=f"Demand Forecast — {market_id} ({forecast.method.value})",
                              font=dict(size=14))),
        yaxis=_axis(tickformat=",d", title="bbl/day"),
        xaxis=_axis(title="Days (negative = history, positive = forecast)"),
    )
    return fig


def replan_trigger_chart(execution_log) -> go.Figure:
    """Bar chart of crack spread realizations vs. planned, coloured by deviation magnitude."""
    days = [e.day for e in execution_log]
    crack_ratios = []
    for e in execution_log:
        if e.realized_crack_spread:
            realized_avg = sum(e.realized_crack_spread.values()) / len(e.realized_crack_spread)
            crack_ratios.append((realized_avg - 1.0) * 100)  # pct deviation from base
        else:
            crack_ratios.append(0.0)

    colors = [_RED if abs(r) > 15 else (_AMBER if abs(r) > 7 else _GREEN) for r in crack_ratios]

    fig = go.Figure(go.Bar(
        x=days, y=crack_ratios,
        marker_color=colors,
        text=[f"{r:+.1f}%" for r in crack_ratios],
        textposition="outside",
        hovertemplate="Day %{x}<br>Crack spread deviation: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=_MUTED, width=1))
    fig.add_hline(y=15,  line=dict(color=_RED, dash="dash", width=1),
                  annotation_text="+15% replan trigger")
    fig.add_hline(y=-15, line=dict(color=_RED, dash="dash", width=1))
    fig.update_layout(
        **_layout(height=280,
                  title=dict(text="Daily Crack Spread Realization (% vs. Base)", font=dict(size=14))),
        yaxis=_axis(ticksuffix="%"),
        xaxis=_axis(title="Day"),
    )
    return fig
