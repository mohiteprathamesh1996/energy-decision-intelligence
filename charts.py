"""
Visualization helpers. Plotly-based; designed for embedding in Streamlit.
Keeps styling minimal and consistent — no decorative noise.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.model.optimizer import OptimizationResult
from src.model.supply_chain import NodeType, SupplyChainNetwork

# Consistent palette by node type
NODE_COLORS = {
    NodeType.WELL: "#2E86AB",
    NodeType.STORAGE: "#F0A500",
    NodeType.REFINERY: "#E94F37",
    NodeType.DISTRIBUTION: "#44BBA4",
    NodeType.DEMAND: "#6B4226",
}

NODE_SYMBOLS = {
    NodeType.WELL: "circle",
    NodeType.STORAGE: "square",
    NodeType.REFINERY: "diamond",
    NodeType.DISTRIBUTION: "triangle-up",
    NodeType.DEMAND: "pentagon",
}


def network_flow_map(
    network: SupplyChainNetwork,
    result: OptimizationResult,
    min_flow_threshold: float = 100,
) -> go.Figure:
    fig = go.Figure()

    # Draw arcs as lines; line width proportional to flow
    for (i, j), flow in result.flows.items():
        if flow < min_flow_threshold:
            continue
        n_from = network.nodes[i]
        n_to = network.nodes[j]
        arc = network.arc_lookup(i, j)
        utilization = flow / arc.capacity if arc.capacity > 0 else 0

        color = "#ff4444" if utilization > 0.85 else "#aaaaaa"
        width = max(1, flow / 8_000)

        fig.add_trace(go.Scattergeo(
            lon=[n_from.longitude, n_to.longitude, None],
            lat=[n_from.latitude, n_to.latitude, None],
            mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Draw nodes grouped by type
    for node_type in NodeType:
        nodes = network.get_nodes_by_type(node_type)
        if not nodes:
            continue

        node_texts = []
        for n in nodes:
            if n.node_type == NodeType.DEMAND:
                unmet = result.unmet_demand.get(n.id, 0)
                node_texts.append(
                    f"{n.name}<br>Demand: {n.demand:,.0f} bbl/d"
                    f"<br>Unmet: {unmet:,.0f} bbl/d"
                )
            elif n.node_type == NodeType.STORAGE:
                inv = result.inventories.get(n.id, 0)
                node_texts.append(f"{n.name}<br>Inventory: {inv:,.0f} bbl")
            else:
                inflow = sum(
                    v for (src, dst), v in result.flows.items() if dst == n.id
                )
                node_texts.append(f"{n.name}<br>Inflow: {inflow:,.0f} bbl/d")

        fig.add_trace(go.Scattergeo(
            lon=[n.longitude for n in nodes],
            lat=[n.latitude for n in nodes],
            mode="markers+text",
            marker=dict(
                size=12,
                color=NODE_COLORS[node_type],
                symbol=NODE_SYMBOLS[node_type],
                line=dict(width=1, color="white"),
            ),
            text=[n.name for n in nodes],
            textposition="top center",
            hovertext=node_texts,
            hoverinfo="text",
            name=node_type.value.capitalize(),
        ))

    fig.update_layout(
        geo=dict(
            scope="usa",
            showland=True,
            landcolor="#1a1a1a",
            showocean=True,
            oceancolor="#0d1117",
            showlakes=False,
            showcountries=False,
            showsubunits=True,
            subunitcolor="#333333",
        ),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#cccccc", size=11),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(
            bgcolor="#1a1a1a",
            bordercolor="#333",
            borderwidth=1,
        ),
        height=500,
        title=dict(text="Supply Chain Flow Network", font=dict(size=14)),
    )

    return fig


def cost_waterfall(result: OptimizationResult) -> go.Figure:
    categories = ["Revenue", "Transport Cost", "Holding Cost", "Unmet Penalty", "Net Margin"]
    values = [
        result.revenue,
        -result.transport_cost,
        -result.holding_cost,
        -result.penalty_cost,
        result.objective_value,
    ]
    measure = ["relative", "relative", "relative", "relative", "total"]
    colors = ["#44BBA4", "#E94F37", "#E94F37", "#E94F37", "#2E86AB"]

    fig = go.Figure(go.Waterfall(
        name="",
        measure=measure,
        x=categories,
        y=values,
        connector=dict(line=dict(color="#555", width=1, dash="dot")),
        increasing=dict(marker=dict(color="#44BBA4")),
        decreasing=dict(marker=dict(color="#E94F37")),
        totals=dict(marker=dict(color="#2E86AB")),
        texttemplate="%{y:$,.0f}",
        textposition="outside",
    ))

    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#cccccc"),
        yaxis=dict(gridcolor="#2a2a2a", tickformat="$,.0f"),
        xaxis=dict(gridcolor="#2a2a2a"),
        margin=dict(l=60, r=30, t=40, b=40),
        height=380,
        title=dict(text="Cost & Revenue Waterfall ($/day)", font=dict(size=14)),
    )
    return fig


def arc_utilization_chart(
    network: SupplyChainNetwork, result: OptimizationResult
) -> go.Figure:
    rows = []
    for (i, j), flow in result.flows.items():
        arc = network.arc_lookup(i, j)
        if arc is None or arc.capacity == 0:
            continue
        util = flow / arc.capacity * 100
        label = f"{network.nodes[i].name} → {network.nodes[j].name}"
        rows.append({"arc": label, "utilization": util, "flow": flow})

    df = pd.DataFrame(rows).sort_values("utilization", ascending=True)
    df = df[df["flow"] > 100]  # filter idle arcs

    colors = ["#E94F37" if u > 85 else "#2E86AB" if u > 50 else "#44BBA4" for u in df["utilization"]]

    fig = go.Figure(go.Bar(
        y=df["arc"],
        x=df["utilization"],
        orientation="h",
        marker=dict(color=colors),
        text=[f"{u:.1f}%" for u in df["utilization"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Utilization: %{x:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#cccccc", size=10),
        xaxis=dict(range=[0, 110], ticksuffix="%", gridcolor="#2a2a2a"),
        yaxis=dict(gridcolor="#2a2a2a"),
        margin=dict(l=200, r=80, t=40, b=40),
        height=max(350, len(df) * 28),
        title=dict(text="Arc Utilization", font=dict(size=14)),
    )
    return fig


def scenario_comparison_chart(
    scenario_names: List[str],
    objectives: List[float],
    base_value: float,
) -> go.Figure:
    deltas = [v - base_value for v in objectives]
    colors = ["#44BBA4" if d >= 0 else "#E94F37" for d in deltas]

    fig = go.Figure(go.Bar(
        x=scenario_names,
        y=deltas,
        marker=dict(color=colors),
        text=[f"${d:+,.0f}" for d in deltas],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Δ Net Margin: %{y:$,.0f}<extra></extra>",
    ))

    fig.add_hline(y=0, line=dict(color="#666", width=1, dash="dash"))

    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#cccccc"),
        yaxis=dict(tickformat="$,.0f", gridcolor="#2a2a2a", title="Δ Net Margin vs Base"),
        xaxis=dict(gridcolor="#2a2a2a"),
        margin=dict(l=80, r=30, t=40, b=80),
        height=380,
        title=dict(text="Scenario Impact on Net Margin", font=dict(size=14)),
    )
    return fig


def demand_coverage_chart(
    network: SupplyChainNetwork, result: OptimizationResult
) -> go.Figure:
    demand_nodes = network.get_nodes_by_type(NodeType.DEMAND)
    names, met, unmet_vals = [], [], []

    for n in demand_nodes:
        unmet = result.unmet_demand.get(n.id, 0)
        met_vol = n.demand - unmet
        names.append(n.name)
        met.append(met_vol)
        unmet_vals.append(unmet)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Met", x=names, y=met, marker_color="#44BBA4"))
    fig.add_trace(go.Bar(name="Unmet", x=names, y=unmet_vals, marker_color="#E94F37"))

    fig.update_layout(
        barmode="stack",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#cccccc"),
        yaxis=dict(tickformat=",d", title="Volume (bbl/day)", gridcolor="#2a2a2a"),
        xaxis=dict(gridcolor="#2a2a2a"),
        legend=dict(bgcolor="#1a1a1a"),
        margin=dict(l=60, r=30, t=40, b=60),
        height=360,
        title=dict(text="Demand Fulfillment by Market", font=dict(size=14)),
    )
    return fig
