"""
Visualization layer. All charts return Plotly figures for Streamlit embedding.
Dark industrial theme; color logic is data-driven (red = stress, green = ok, blue = neutral).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.model.optimizer import MultiPeriodResult
from src.model.stochastic import StochasticResult
from src.model.supply_chain import NodeType, SupplyChainNetwork

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

NODE_COLOR = {
    NodeType.WELL: "#2E86AB",
    NodeType.STORAGE: "#F0A500",
    NodeType.REFINERY: "#E94F37",
    NodeType.DISTRIBUTION: "#44BBA4",
    NodeType.DEMAND: "#9775FA",
}
NODE_SYMBOL = {
    NodeType.WELL: "circle",
    NodeType.STORAGE: "square",
    NodeType.REFINERY: "diamond",
    NodeType.DISTRIBUTION: "triangle-up",
    NodeType.DEMAND: "pentagon",
}

GRADE_COLOR = {"WTI": _BLUE, "WTS": _AMBER, "HEAVY": "#8b5cf6"}


def _layout(**kwargs):
    defaults = dict(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color=_TEXT, size=11),
        margin=dict(l=60, r=30, t=40, b=50),
        legend=dict(bgcolor=_CARD, bordercolor=_BORDER, borderwidth=1),
    )
    defaults.update(kwargs)
    return defaults


def _axis(**kwargs):
    return dict(gridcolor=_GRID, zerolinecolor=_GRID, **kwargs)


# ── Network Flow Map ──────────────────────────────────────────────────────────

def network_flow_map(
    network: SupplyChainNetwork,
    result: MultiPeriodResult,
    period: int = 1,
    min_flow: float = 200,
) -> go.Figure:
    fig = go.Figure()

    # Arcs
    for (i, j) in network.arc_index:
        flow = result.flows_by_period.get((i, j, period), 0)
        if flow < min_flow:
            continue
        arc = network.arc_lookup(i, j)
        util = flow / arc.capacity if arc.capacity > 0 else 0
        color = _RED if util > 0.88 else (_AMBER if util > 0.65 else "#555")
        width = max(1.0, flow / 7_000)
        n_from, n_to = network.nodes[i], network.nodes[j]
        fig.add_trace(go.Scattergeo(
            lon=[n_from.longitude, n_to.longitude, None],
            lat=[n_from.latitude, n_to.latitude, None],
            mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Nodes
    for nt in NodeType:
        nodes = network.get_nodes_by_type(nt)
        if not nodes:
            continue
        hover = []
        for n in nodes:
            if n.node_type == NodeType.DEMAND:
                unmet = sum(result.unmet_demand.get((n.id, t), 0) for t in range(1, result.planning_horizon + 1))
                hover.append(f"{n.name}<br>Daily Demand: {n.demand:,.0f} bbl<br>Avg Unmet: {unmet/result.planning_horizon:,.0f} bbl/d")
            elif n.node_type == NodeType.STORAGE:
                inv = sum(result.inventories.get((n.id, period, g), 0) for g in network.grade_ids())
                hover.append(f"{n.name}<br>Inventory (D{period}): {inv:,.0f} bbl")
            elif n.node_type == NodeType.WELL:
                out = sum(result.flows_by_period.get((n.id, j, period), 0) for j in network.nodes)
                hover.append(f"{n.name}<br>Grade: {n.primary_grade}<br>Output D{period}: {out:,.0f} bbl/d")
            else:
                inf = sum(result.flows_by_period.get((i, n.id, period), 0) for i in network.nodes)
                hover.append(f"{n.name}<br>Inflow D{period}: {inf:,.0f} bbl/d")

        fig.add_trace(go.Scattergeo(
            lon=[n.longitude for n in nodes],
            lat=[n.latitude for n in nodes],
            mode="markers+text",
            marker=dict(size=13, color=NODE_COLOR[nt], symbol=NODE_SYMBOL[nt],
                        line=dict(width=1.5, color="white")),
            text=[n.name for n in nodes],
            textposition="top center",
            hovertext=hover,
            hoverinfo="text",
            name=nt.value.capitalize(),
        ))

    fig.update_layout(
        geo=dict(
            scope="usa",
            showland=True, landcolor="#1a1a1a",
            showocean=True, oceancolor=_BG,
            showlakes=False, showcountries=False,
            showsubunits=True, subunitcolor="#333",
        ),
        **_layout(height=500, margin=dict(l=0, r=0, t=30, b=0),
                  title=dict(text=f"Supply Chain Flow Network — Day {period}", font=dict(size=14))),
    )
    return fig


# ── Cost Waterfall ────────────────────────────────────────────────────────────

def cost_waterfall(result: MultiPeriodResult) -> go.Figure:
    T = result.planning_horizon
    labels = ["Gross Revenue", "Transport", "Operating", "Holding", "SOP Deficiency", "Unmet Penalty", "Net Margin"]

    opex = result.revenue - result.transport_cost - result.holding_cost - result.sop_cost - result.penalty_cost - result.objective_value
    values = [
        result.revenue,
        -result.transport_cost,
        -opex,
        -result.holding_cost,
        -result.sop_cost,
        -result.penalty_cost,
        result.objective_value,
    ]
    measures = ["relative"] * 6 + ["total"]

    fig = go.Figure(go.Waterfall(
        name="",
        measure=measures,
        x=labels,
        y=values,
        connector=dict(line=dict(color="#444", width=1, dash="dot")),
        increasing=dict(marker=dict(color=_GREEN)),
        decreasing=dict(marker=dict(color=_RED)),
        totals=dict(marker=dict(color=_BLUE)),
        texttemplate="%{y:$,.0f}",
        textposition="outside",
    ))
    fig.update_layout(
        **_layout(height=380,
                  title=dict(text=f"P&L Breakdown — {T}-Day Horizon", font=dict(size=14))),
        yaxis=_axis(tickformat="$,.0f"),
        xaxis=_axis(),
    )
    return fig


# ── Arc Utilization ───────────────────────────────────────────────────────────

def arc_utilization_chart(
    network: SupplyChainNetwork,
    result: MultiPeriodResult,
) -> go.Figure:
    T = result.planning_horizon
    rows = []
    for (i, j) in network.arc_index:
        arc = network.arc_lookup(i, j)
        if arc.capacity == 0:
            continue
        avg_flow = sum(result.flows_by_period.get((i, j, t), 0) for t in range(1, T + 1)) / T
        util = avg_flow / arc.capacity * 100
        if avg_flow < 100:
            continue
        rows.append({
            "arc": f"{network.nodes[i].name} → {network.nodes[j].name}",
            "util": util,
            "flow": avg_flow,
        })

    df = pd.DataFrame(rows).sort_values("util", ascending=True)
    colors = [_RED if u > 88 else (_AMBER if u > 65 else _GREEN) for u in df["util"]]

    fig = go.Figure(go.Bar(
        y=df["arc"], x=df["util"], orientation="h",
        marker=dict(color=colors),
        text=[f"{u:.1f}%" for u in df["util"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Avg Utilization: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **_layout(height=max(360, len(df) * 30), margin=dict(l=220, r=80, t=40, b=40),
                  title=dict(text="Arc Utilization (Avg over Planning Horizon)", font=dict(size=14))),
        xaxis=_axis(range=[0, 115], ticksuffix="%"),
        yaxis=_axis(),
    )
    return fig


# ── Grade Mix at Refineries ───────────────────────────────────────────────────

def refinery_grade_mix(
    network: SupplyChainNetwork,
    result: MultiPeriodResult,
    period: int = 1,
) -> go.Figure:
    refineries = network.get_nodes_by_type(NodeType.REFINERY)
    grade_ids = network.grade_ids()
    node_ids = list(network.nodes.keys())
    arc_keys = network.arc_index

    ref_names = [r.name for r in refineries]
    fig = go.Figure()
    for g in grade_ids:
        vals = []
        for r in refineries:
            vol = sum(
                result.flows_by_grade.get((i, r.id, period, g), 0)
                for i in node_ids if (i, r.id) in arc_keys
            )
            vals.append(vol)
        fig.add_trace(go.Bar(name=network.grades[g].name, x=ref_names, y=vals,
                             marker_color=GRADE_COLOR[g]))

    fig.update_layout(
        barmode="stack",
        **_layout(height=350, title=dict(text=f"Crude Grade Mix at Refineries — Day {period}", font=dict(size=14))),
        yaxis=_axis(tickformat=",d", title="bbl/day"),
        xaxis=_axis(),
    )
    return fig


# ── Storage Inventory Profile ─────────────────────────────────────────────────

def inventory_profile(
    network: SupplyChainNetwork,
    result: MultiPeriodResult,
) -> go.Figure:
    terminals = network.get_nodes_by_type(NodeType.STORAGE)
    grade_ids = network.grade_ids()
    T = result.planning_horizon

    fig = make_subplots(
        rows=1, cols=len(terminals),
        subplot_titles=[t.name for t in terminals],
        shared_yaxes=True,
    )

    for col_idx, term in enumerate(terminals, 1):
        for g in grade_ids:
            inv_series = [result.inventories.get((term.id, t, g), 0) for t in range(1, T + 1)]
            fig.add_trace(go.Scatter(
                x=list(range(1, T + 1)),
                y=inv_series,
                name=g if col_idx == 1 else None,
                legendgroup=g,
                showlegend=(col_idx == 1),
                line=dict(color=GRADE_COLOR[g], width=2),
                fill="tozeroy" if g == grade_ids[0] else "tonexty",
                fillcolor=GRADE_COLOR[g] + "40",
            ), row=1, col=col_idx)

        # Capacity line
        fig.add_hline(
            y=term.max_capacity,
            line_dash="dash", line_color=_RED, line_width=1,
            row=1, col=col_idx,
        )

    fig.update_layout(
        **_layout(height=320, title=dict(text="Storage Inventory by Grade (bbl)", font=dict(size=14))),
        yaxis=_axis(tickformat=",d"),
    )
    fig.update_xaxes(title_text="Day", gridcolor=_GRID)
    return fig


# ── Daily Margin Trend ────────────────────────────────────────────────────────

def daily_margin_trend(result: MultiPeriodResult) -> go.Figure:
    T = result.planning_horizon
    # Approximate daily contribution as total / T (flat, since single-period margins per day)
    daily = [result.objective_value / T] * T

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, T + 1)), y=daily,
        mode="lines+markers",
        line=dict(color=_BLUE, width=2),
        marker=dict(size=5),
        fill="tozeroy",
        fillcolor=_BLUE + "20",
        name="Net Margin",
    ))
    fig.update_layout(
        **_layout(height=280, title=dict(text="Estimated Daily Net Margin ($/day)", font=dict(size=14))),
        yaxis=_axis(tickformat="$,.0f"),
        xaxis=_axis(title="Day"),
    )
    return fig


# ── Demand Coverage ───────────────────────────────────────────────────────────

def demand_coverage(
    network: SupplyChainNetwork,
    result: MultiPeriodResult,
) -> go.Figure:
    T = result.planning_horizon
    demand_nodes = network.get_nodes_by_type(NodeType.DEMAND)
    names, met_vals, unmet_vals = [], [], []

    for n in demand_nodes:
        total_unmet = sum(result.unmet_demand.get((n.id, t), 0) for t in range(1, T + 1))
        avg_unmet = total_unmet / T
        avg_met = n.demand - avg_unmet
        names.append(n.name)
        met_vals.append(max(0, avg_met))
        unmet_vals.append(max(0, avg_unmet))

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Met", x=names, y=met_vals, marker_color=_GREEN))
    fig.add_trace(go.Bar(name="Unmet", x=names, y=unmet_vals, marker_color=_RED))
    fig.update_layout(
        barmode="stack",
        **_layout(height=330, title=dict(text="Avg Daily Demand Fulfillment", font=dict(size=14))),
        yaxis=_axis(tickformat=",d", title="bbl/day"),
        xaxis=_axis(),
    )
    return fig


# ── Scenario Comparison ───────────────────────────────────────────────────────

def scenario_comparison(
    names: List[str],
    objectives: List[float],
    base: float,
    categories: Optional[List[str]] = None,
) -> go.Figure:
    deltas = [v - base for v in objectives]
    cat_color_map = {
        "supply": _RED,
        "demand": _AMBER,
        "infrastructure": _BLUE,
        "cost": "#9775FA",
        "policy": _TEAL,
    }
    colors = [
        (cat_color_map.get(c, _BLUE) if d < 0 else _GREEN)
        for c, d in zip(categories or [""] * len(names), deltas)
    ]

    fig = go.Figure(go.Bar(
        x=names, y=deltas,
        marker=dict(color=colors),
        text=[f"${d:+,.0f}" for d in deltas],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Δ Net Margin: %{y:$,.0f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=_MUTED, width=1, dash="dash"))
    fig.update_layout(
        **_layout(height=380, title=dict(text="Scenario Impact on Net Margin (vs. Base)", font=dict(size=14))),
        yaxis=_axis(tickformat="$,.0f", title="Δ Net Margin ($)"),
        xaxis=_axis(),
    )
    return fig


# ── Stochastic Fan Chart ──────────────────────────────────────────────────────

def stochastic_fan(stochastic_result: StochasticResult) -> go.Figure:
    objs = sorted(stochastic_result.scenario_objectives.values())
    percentiles = [5, 25, 50, 75, 95]
    vals = np.percentile(objs, percentiles)

    labels = [f"P{p}" for p in percentiles]
    colors_p = [_RED, _AMBER, _BLUE, _AMBER, _GREEN]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=vals,
        marker=dict(color=colors_p),
        text=[f"${v:,.0f}" for v in vals],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Net Margin: %{y:$,.0f}<extra></extra>",
    ))
    fig.add_hline(y=stochastic_result.rp, line=dict(color=_BLUE, width=2, dash="dot"),
                  annotation_text="RP", annotation_position="top left")

    fig.update_layout(
        **_layout(height=360, title=dict(text="Crack Spread Scenario Distribution — Net Margin", font=dict(size=14))),
        yaxis=_axis(tickformat="$,.0f"),
        xaxis=_axis(title="Percentile"),
    )
    return fig


# ── EVPI / VSS Summary ────────────────────────────────────────────────────────

def evpi_vss_table(sr: StochasticResult) -> go.Figure:
    metrics = [
        ("RP — Recourse Problem",        sr.rp,    "Optimal under uncertainty"),
        ("WS — Wait and See",            sr.ws,    "Upper bound (perfect information)"),
        ("EEV — Mean-Value Solution",    sr.eev,   "Naïve deterministic approach"),
        ("EVPI — Value of Perfect Info", sr.evpi,  "Max you'd pay for a crystal ball"),
        ("VSS — Value of Stoch. Sol.",   sr.vss,   "Gain from modeling uncertainty"),
    ]
    fig = go.Figure(go.Table(
        header=dict(
            values=["Metric", "Value ($)", "Interpretation"],
            fill_color=_CARD,
            font=dict(color=_TEXT, size=12),
            align="left",
            line=dict(color=_BORDER),
        ),
        cells=dict(
            values=[
                [m[0] for m in metrics],
                [f"${m[1]:,.0f}" for m in metrics],
                [m[2] for m in metrics],
            ],
            fill_color=[[_BG, _BG, _BG, _CARD + "88", _CARD + "88"]],
            font=dict(color=_TEXT, size=11),
            align="left",
            line=dict(color=_BORDER),
            height=30,
        ),
    ))
    fig.update_layout(**_layout(height=250, margin=dict(l=0, r=0, t=10, b=0)))
    return fig


# ── Sensitivity Tornado ───────────────────────────────────────────────────────

def tornado_chart(entries, top_n: int = 10) -> go.Figure:
    top = entries[:top_n]
    names = [e.parameter for e in top]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="+10%",
        y=names,
        x=[e.swing_up for e in top],
        orientation="h",
        marker_color=_GREEN,
    ))
    fig.add_trace(go.Bar(
        name="−10%",
        y=names,
        x=[e.swing_down for e in top],
        orientation="h",
        marker_color=_RED,
    ))
    fig.update_layout(
        barmode="overlay",
        **_layout(height=max(350, len(top) * 35), margin=dict(l=200, r=80, t=50, b=40),
                  title=dict(text="Sensitivity Tornado — Net Margin Response (±10%)", font=dict(size=14))),
        xaxis=_axis(tickformat="$,.0f", title="Δ Net Margin"),
        yaxis=_axis(),
    )
    return fig


# ── Carbon Emissions by Route ─────────────────────────────────────────────────

def carbon_emissions_chart(
    network: SupplyChainNetwork,
    result: MultiPeriodResult,
) -> go.Figure:
    T = result.planning_horizon
    grade_ids = network.grade_ids()

    rows = []
    for (i, j) in network.arc_index:
        for g in grade_ids:
            flow = sum(result.flows_by_grade.get((i, j, t, g), 0) for t in range(1, T + 1)) / T
            if flow < 100:
                continue
            co2 = flow * network.grades[g].carbon_intensity / 1000  # tonnes/day
            label = f"{network.nodes[i].name} → {network.nodes[j].name}"
            rows.append({"route": label, "grade": g, "co2": co2})

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["route", "grade", "co2"])
    if df.empty:
        return go.Figure()

    agg = df.groupby(["route", "grade"])["co2"].sum().reset_index()
    pivoted = agg.pivot(index="route", columns="grade", values="co2").fillna(0)

    fig = go.Figure()
    for g in grade_ids:
        if g in pivoted.columns:
            fig.add_trace(go.Bar(
                name=g, x=pivoted.index.tolist(), y=pivoted[g].tolist(),
                marker_color=GRADE_COLOR[g],
            ))

    if network.carbon_budget_per_day:
        fig.add_hline(
            y=network.carbon_budget_per_day / len(set(df["route"])) if len(set(df["route"])) > 0 else network.carbon_budget_per_day,
            line=dict(color=_RED, dash="dash"),
            annotation_text=f"Budget {network.carbon_budget_per_day:.0f} t/d",
        )

    fig.update_layout(
        barmode="stack",
        **_layout(height=380, margin=dict(l=30, r=30, t=40, b=120),
                  title=dict(text="Avg Daily CO₂e Emissions by Route (tCO₂e/day)", font=dict(size=14))),
        yaxis=_axis(title="tCO₂e/day"),
        xaxis=_axis(tickangle=-40),
    )
    return fig


# ── Contract Status Table ─────────────────────────────────────────────────────

def contract_status(
    network: SupplyChainNetwork,
    result: MultiPeriodResult,
) -> go.Figure:
    T = result.planning_horizon
    rows = []
    for c in network.contracts:
        arc = network.arc_lookup(c.arc_origin, c.arc_dest)
        avg_flow = sum(result.flows_by_period.get((c.arc_origin, c.arc_dest, t), 0) for t in range(1, T + 1)) / T
        avg_def = sum(result.sop_deficits.get((c.arc_origin, c.arc_dest, t), 0) for t in range(1, T + 1)) / T
        total_def_cost = avg_def * c.deficiency_charge * T
        compliance = (avg_flow / c.min_daily_volume * 100) if c.min_daily_volume > 0 else 100
        rows.append({
            "Contract": f"{network.nodes[c.arc_origin].name} → {network.nodes[c.arc_dest].name}",
            "Committed (bbl/d)": f"{c.min_daily_volume:,.0f}",
            "Avg Flow (bbl/d)": f"{avg_flow:,.0f}",
            "Compliance": f"{min(compliance, 100):.1f}%",
            "Avg Deficit": f"{avg_def:,.0f}",
            "Period Deficiency Cost": f"${total_def_cost:,.0f}",
        })

    fig = go.Figure(go.Table(
        header=dict(
            values=list(rows[0].keys()) if rows else [],
            fill_color=_CARD,
            font=dict(color=_TEXT, size=11),
            align="left",
            line=dict(color=_BORDER),
        ),
        cells=dict(
            values=[[r[k] for r in rows] for k in (rows[0].keys() if rows else [])],
            fill_color=_BG,
            font=dict(color=_TEXT, size=11),
            align="left",
            line=dict(color=_BORDER),
            height=28,
        ),
    ))
    fig.update_layout(**_layout(height=200 + len(rows) * 30, margin=dict(l=0, r=0, t=10, b=0)))
    return fig
