"""
Oil Supply Chain Optimization Dashboard.
Minimal, business-oriented layout. Each panel has a clear decision-making purpose.
"""

import copy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

from data.generate_data import (
    apply_cost_shock,
    apply_demand_spike,
    apply_refinery_outage,
    apply_supply_disruption,
    build_base_network,
)
from src.analysis.scenario import ScenarioRunner, build_standard_scenarios
from src.model.optimizer import SupplyChainOptimizer
from src.model.supply_chain import NodeType
from src.viz.charts import (
    arc_utilization_chart,
    cost_waterfall,
    demand_coverage_chart,
    network_flow_map,
    scenario_comparison_chart,
)

st.set_page_config(
    page_title="Oil Supply Chain Optimizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #21262d;
    }
    .metric-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 16px 20px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #58a6ff;
        font-family: 'Courier New', monospace;
    }
    .metric-label {
        font-size: 12px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-delta { font-size: 13px; margin-top: 4px; }
    .delta-up { color: #3fb950; }
    .delta-down { color: #f85149; }
    h1, h2, h3 { color: #e6edf3; font-weight: 600; }
    .stButton > button {
        background: #1f6feb;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 20px;
        font-weight: 600;
    }
    .stButton > button:hover { background: #388bfd; }
    hr { border-color: #21262d; }
    .stSelectbox label, .stSlider label { color: #8b949e; font-size: 12px; }
    [data-testid="stMetric"] { background: #161b22; padding: 12px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_base_network():
    return build_base_network()


@st.cache_data
def run_base_optimization():
    net = get_base_network()
    opt = SupplyChainOptimizer(net)
    return opt.solve()


def metric_card(label: str, value: str, delta: str = None, delta_positive: bool = True):
    delta_html = ""
    if delta:
        cls = "delta-up" if delta_positive else "delta-down"
        arrow = "▲" if delta_positive else "▼"
        delta_html = f'<div class="metric-delta {cls}">{arrow} {delta}</div>'
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>""",
        unsafe_allow_html=True,
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙ Configuration")
    st.markdown("---")
    st.markdown("**Scenario Builder**")

    disruption_well = st.selectbox(
        "Well Disruption", ["None", "W1 Midland North", "W2 Midland South", "W3 Delaware", "W4 Wolfcamp", "W5 Bone Spring"]
    )
    disruption_pct = st.slider("Production Reduction %", 0, 100, 0, 5) / 100

    st.markdown("")
    demand_market = st.selectbox(
        "Demand Spike", ["None", "M1 Chicago", "M2 New York", "M3 Dallas", "M4 Export A", "M5 Los Angeles"]
    )
    demand_mult = st.slider("Demand Multiplier", 1.0, 2.0, 1.0, 0.05)

    st.markdown("")
    refinery_out = st.selectbox("Refinery Outage", ["None", "R1 Houston", "R2 Port Arthur", "R3 Beaumont"])

    st.markdown("")
    cost_shock_pct = st.slider("Transport Cost Shock %", 0, 50, 0, 5) / 100

    st.markdown("---")
    run_btn = st.button("▶  Run Optimization", use_container_width=True)
    run_scenarios_btn = st.button("📊  Run All Scenarios", use_container_width=True)

# ── Main Layout ───────────────────────────────────────────────────────────────

st.markdown("# Oil Supply Chain Optimizer")
st.markdown(
    '<span style="color:#8b949e;font-size:13px;">Permian Basin → Gulf Coast MILP Model | Single-Period, Multi-Node</span>',
    unsafe_allow_html=True,
)
st.markdown("---")

# Build scenario network
net = copy.deepcopy(get_base_network())

if disruption_well != "None" and disruption_pct > 0:
    well_id = disruption_well.split()[0]
    net = apply_supply_disruption(net, well_id, disruption_pct)

if demand_market != "None" and demand_mult > 1.0:
    market_id = demand_market.split()[0]
    net = apply_demand_spike(net, market_id, demand_mult)

if refinery_out != "None":
    ref_id = refinery_out.split()[0]
    net = apply_refinery_outage(net, ref_id)

if cost_shock_pct > 0:
    net = apply_cost_shock(net, cost_shock_pct)

is_modified = any([
    disruption_well != "None" and disruption_pct > 0,
    demand_market != "None" and demand_mult > 1.0,
    refinery_out != "None",
    cost_shock_pct > 0,
])

if run_btn or is_modified:
    with st.spinner("Solving MILP..."):
        opt = SupplyChainOptimizer(net)
        result = opt.solve()
    base_result = run_base_optimization()
else:
    with st.spinner("Loading base case..."):
        result = run_base_optimization()
    base_result = result

# ── KPI Row ──────────────────────────────────────────────────────────────────

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    metric_card("Net Margin / Day", f"${result.objective_value:,.0f}")
with k2:
    delta_rev = result.revenue - base_result.revenue
    metric_card(
        "Gross Revenue",
        f"${result.revenue:,.0f}",
        f"${abs(delta_rev):,.0f}",
        delta_rev >= 0,
    )
with k3:
    metric_card("Transport Cost", f"${result.transport_cost:,.0f}")
with k4:
    total_unmet = sum(result.unmet_demand.values())
    metric_card(
        "Unmet Demand",
        f"{total_unmet:,.0f} bbl",
        "CRITICAL" if total_unmet > 5000 else "OK",
        total_unmet <= 5000,
    )
with k5:
    total_demand = sum(n.demand for n in net.get_nodes_by_type(NodeType.DEMAND))
    svc = (1 - total_unmet / max(total_demand, 1)) * 100
    metric_card("Service Level", f"{svc:.1f}%")

st.markdown("")

# ── Network Map + Waterfall ───────────────────────────────────────────────────

col_map, col_wf = st.columns([3, 2])

with col_map:
    st.plotly_chart(
        network_flow_map(net, result), use_container_width=True, config={"displayModeBar": False}
    )

with col_wf:
    st.plotly_chart(
        cost_waterfall(result), use_container_width=True, config={"displayModeBar": False}
    )

# ── Utilization + Demand Coverage ────────────────────────────────────────────

col_util, col_dem = st.columns([3, 2])

with col_util:
    st.plotly_chart(
        arc_utilization_chart(net, result), use_container_width=True, config={"displayModeBar": False}
    )

with col_dem:
    st.plotly_chart(
        demand_coverage_chart(net, result), use_container_width=True, config={"displayModeBar": False}
    )

# ── Scenario Analysis ─────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Scenario Analysis")

if run_scenarios_btn:
    with st.spinner("Running all scenarios..."):
        base_net = get_base_network()
        runner = ScenarioRunner(base_net)
        scenarios = build_standard_scenarios(base_net)
        runner.run_base()
        scenario_results = runner.run_all(scenarios)

    names = [sr.scenario_name for sr in scenario_results]
    objs = [sr.result.objective_value for sr in scenario_results]

    st.plotly_chart(
        scenario_comparison_chart(names, objs, base_result.objective_value),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    rows = []
    for sr in scenario_results:
        total_unmet = sum(sr.result.unmet_demand.values())
        rows.append({
            "Scenario": sr.scenario_name,
            "Net Margin ($)": f"${sr.result.objective_value:,.0f}",
            "Δ vs Base ($)": f"${sr.delta_objective:+,.0f}",
            "Revenue ($)": f"${sr.result.revenue:,.0f}",
            "Transport ($)": f"${sr.result.transport_cost:,.0f}",
            "Unmet Demand (bbl)": f"{total_unmet:,.0f}",
            "Solver Time (s)": f"{sr.result.solver_time:.2f}",
        })

    st.dataframe(
        pd.DataFrame(rows).set_index("Scenario"),
        use_container_width=True,
    )
else:
    st.markdown(
        '<span style="color:#8b949e;">Click <b>Run All Scenarios</b> to compare disruption, demand spike, '
        'refinery outage, and freight cost scenarios against the base case.</span>',
        unsafe_allow_html=True,
    )

# ── Flow Table ────────────────────────────────────────────────────────────────

with st.expander("Raw Flow Data", expanded=False):
    flow_rows = []
    for (i, j), v in sorted(result.flows.items(), key=lambda x: -x[1]):
        if v < 100:
            continue
        arc = net.arc_lookup(i, j)
        util = v / arc.capacity * 100 if arc.capacity > 0 else 0
        flow_rows.append({
            "From": net.nodes[i].name,
            "To": net.nodes[j].name,
            "Flow (bbl/day)": f"{v:,.0f}",
            "Capacity (bbl/day)": f"{arc.capacity:,.0f}",
            "Utilization": f"{util:.1f}%",
            "Transport Cost ($/bbl)": f"${arc.transport_cost:.2f}",
            "Daily Cost ($)": f"${v * arc.transport_cost:,.0f}",
        })
    st.dataframe(pd.DataFrame(flow_rows), use_container_width=True)
