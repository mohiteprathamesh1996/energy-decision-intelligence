"""
Oil Supply Chain Optimization Dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Multi-period MILP | Crude Grade Differentiation | Ship-or-Pay Contracts
Two-Stage Stochastic | Rolling Horizon MPC | Demand Forecasting | Carbon Tracking
"""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

from data.generate_data import (
    apply_carbon_cap, apply_demand_spike, apply_disruption,
    apply_freight_shock, apply_refinery_outage, build_base_network,
)
from src.analysis.scenario import ScenarioRunner, build_standard_scenarios
from src.analysis.sensitivity import identify_bottlenecks, run_sensitivity
from src.model.optimizer import MultiPeriodOptimizer
from src.model.stochastic import run_stochastic_analysis
from src.model.supply_chain import NodeType
from src.viz.charts import (
    arc_utilization_chart, carbon_emissions_chart, contract_status,
    cost_waterfall, daily_margin_trend, demand_coverage,
    evpi_vss_table, inventory_profile, network_flow_map,
    refinery_grade_mix, scenario_comparison, stochastic_fan,
    tornado_chart,
)

st.set_page_config(
    page_title="Oil Supply Chain Optimizer",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛢️",
)

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { background-color: #0d1117; color: #c9d1d9; }
[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #21262d; }
h1,h2,h3 { color: #e6edf3; font-weight: 600; }
.kpi-box {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 6px; padding: 14px 18px; text-align: center;
}
.kpi-value { font-size: 26px; font-weight: 700; color: #58a6ff; font-family: 'Courier New', monospace; }
.kpi-label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 4px; }
.kpi-delta { font-size: 12px; margin-top: 4px; }
.kpi-green { color: #3fb950; } .kpi-red { color: #f85149; } .kpi-amber { color: #d29922; }
hr { border-color: #21262d; }
.stTabs [data-baseweb="tab"] { color: #8b949e; }
.stTabs [aria-selected="true"] { color: #e6edf3; border-bottom: 2px solid #58a6ff; }
[data-testid="stButton"] > button {
    background: #1f6feb; color: white; border: none; border-radius: 4px; font-weight: 600;
}
[data-testid="stButton"] > button:hover { background: #388bfd; }
</style>
""", unsafe_allow_html=True)


def kpi(label, value, delta=None, ok=True):
    cls = "kpi-green" if ok else "kpi-red"
    d_html = f'<div class="kpi-delta {cls}">{delta}</div>' if delta else ""
    st.markdown(
        f'<div class="kpi-box"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{d_html}</div>',
        unsafe_allow_html=True,
    )


@st.cache_resource
def _base_net():
    return build_base_network(horizon=14)


@st.cache_data
def _base_result():
    net = copy.deepcopy(_base_net())
    return MultiPeriodOptimizer(net).solve()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛢️ Optimizer Controls")
    st.markdown("---")

    st.markdown("**Planning Horizon**")
    horizon = st.slider("Days", 7, 30, 14, 1)

    st.markdown("---")
    st.markdown("**Supply Disruption**")
    well_opt = st.selectbox("Well", ["None", "W1 Midland North", "W2 Midland South",
                                      "W3 Delaware", "W4 Wolfcamp", "W5 Bone Spring"])
    well_pct = st.slider("Capacity Reduction %", 0, 100, 0, 5) / 100.0

    st.markdown("**Demand Shock**")
    mkt_opt = st.selectbox("Market", ["None", "M1 Chicago", "M2 New York",
                                       "M3 Dallas", "M4 Freeport", "M5 Los Angeles"])
    mkt_mult = st.slider("Demand Multiplier", 1.0, 2.0, 1.0, 0.05)

    st.markdown("**Infrastructure**")
    ref_opt = st.selectbox("Refinery Outage", ["None", "R1 Houston", "R2 Port Arthur", "R3 Beaumont"])

    st.markdown("**Cost & Policy**")
    freight_pct = st.slider("Freight Shock %", 0, 50, 0, 5) / 100.0
    carbon_cap = st.number_input("Carbon Cap (tCO₂e/day, 0=none)", 0, 2000, 0, 50)

    st.markdown("---")
    run_btn = st.button("▶  Run Optimization", use_container_width=True)

    st.markdown("---")
    st.markdown("**Advanced Analysis**")
    run_scenarios_btn = st.button("📊 Run All Scenarios", use_container_width=True)
    run_stoch_btn = st.button("🎲 Stochastic Analysis", use_container_width=True)
    run_sens_btn = st.button("🌪 Sensitivity Analysis", use_container_width=True)
    run_rh_btn = st.button("🔄 Rolling Horizon (30d)", use_container_width=True)
    run_forecast_btn = st.button("📈 Demand Forecast", use_container_width=True)


# ── Build scenario network ────────────────────────────────────────────────────

net = copy.deepcopy(_base_net())
net.planning_horizon = horizon
is_modified = False

if well_opt != "None" and well_pct > 0:
    apply_disruption(net, well_opt.split()[0], well_pct); is_modified = True
if mkt_opt != "None" and mkt_mult > 1.0:
    apply_demand_spike(net, mkt_opt.split()[0], mkt_mult); is_modified = True
if ref_opt != "None":
    apply_refinery_outage(net, ref_opt.split()[0]); is_modified = True
if freight_pct > 0:
    apply_freight_shock(net, freight_pct); is_modified = True
if carbon_cap > 0:
    apply_carbon_cap(net, float(carbon_cap)); is_modified = True

if run_btn or is_modified:
    with st.spinner("Solving MILP…"):
        result = MultiPeriodOptimizer(copy.deepcopy(net)).solve()
    base_result_obj = _base_result()
else:
    result = _base_result()
    base_result_obj = result

# ── Header KPIs ───────────────────────────────────────────────────────────────

st.markdown("# Oil Supply Chain Optimizer")
st.markdown(
    '<span style="color:#8b949e;font-size:13px;">'
    'Permian Basin → Gulf Coast · Multi-Period MILP · Crude Grade Differentiation · '
    'Ship-or-Pay Contracts · Two-Stage Stochastic · Rolling Horizon MPC</span>',
    unsafe_allow_html=True,
)
st.markdown("---")

T = result.planning_horizon
total_unmet = sum(result.unmet_demand.values())
total_demand = sum(n.demand for n in net.get_nodes_by_type(NodeType.DEMAND)) * T
svc = (1 - total_unmet / max(total_demand, 1)) * 100
avg_carbon = sum(result.carbon_by_period.values()) / T
sop_total = sum(result.sop_deficits.values())
delta_obj = result.objective_value - base_result_obj.objective_value

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: kpi("Net Margin (Total)",   f"${result.objective_value:,.0f}",  f"vs base: ${delta_obj:+,.0f}", delta_obj >= 0)
with c2: kpi("Gross Revenue",        f"${result.revenue:,.0f}")
with c3: kpi("Transport Cost",       f"${result.transport_cost:,.0f}")
with c4: kpi("Service Level",        f"{svc:.1f}%",      "✓ FULL" if svc >= 99 else ("⚠ PARTIAL" if svc >= 90 else "✗ CRITICAL"), svc >= 95)
with c5: kpi("Avg Carbon",           f"{avg_carbon:.0f} t/d", None)
with c6: kpi("SOP Deficits",         f"{sop_total:,.0f} bbl", "✓ Clear" if sop_total < 100 else "⚠ Deficit", sop_total < 100)

st.markdown("")

# ── Tabs ──────────────────────────────────────────────────────────────────────

(tab_ov, tab_ops, tab_sc, tab_stoch,
 tab_rh, tab_fc, tab_carbon) = st.tabs([
    "📍 Overview", "⚙ Operations", "📊 Scenarios",
    "🎲 Stochastic", "🔄 Rolling Horizon", "📈 Forecasting", "🌿 Carbon",
])

# ─── OVERVIEW ────────────────────────────────────────────────────────────────
with tab_ov:
    day_sel = st.slider("Map Day", 1, T, 1, key="mapday")
    col_map, col_wf = st.columns([3, 2])
    with col_map:
        st.plotly_chart(network_flow_map(net, result, period=day_sel),
                        use_container_width=True, config={"displayModeBar": False})
    with col_wf:
        st.plotly_chart(cost_waterfall(result),
                        use_container_width=True, config={"displayModeBar": False})
    col_dem, col_mg = st.columns(2)
    with col_dem:
        st.plotly_chart(demand_coverage(net, result),
                        use_container_width=True, config={"displayModeBar": False})
    with col_mg:
        st.plotly_chart(daily_margin_trend(result),
                        use_container_width=True, config={"displayModeBar": False})

# ─── OPERATIONS ──────────────────────────────────────────────────────────────
with tab_ops:
    st.plotly_chart(arc_utilization_chart(net, result),
                    use_container_width=True, config={"displayModeBar": False})
    col_inv, col_grade = st.columns(2)
    with col_inv:
        st.plotly_chart(inventory_profile(net, result),
                        use_container_width=True, config={"displayModeBar": False})
    with col_grade:
        gday = st.slider("Grade Mix Day", 1, T, 1, key="gradeday")
        st.plotly_chart(refinery_grade_mix(net, result, period=gday),
                        use_container_width=True, config={"displayModeBar": False})
    st.markdown("**Ship-or-Pay Contract Status**")
    st.plotly_chart(contract_status(net, result),
                    use_container_width=True, config={"displayModeBar": False})
    with st.expander("Raw Flow Table", expanded=False):
        rows = []
        for (i, j) in net.arc_index:
            arc = net.arc_lookup(i, j)
            for t in range(1, T + 1):
                flow = result.flows_by_period.get((i, j, t), 0)
                if flow < 100: continue
                util = flow / arc.capacity * 100
                rows.append({"Day": t, "From": net.nodes[i].name, "To": net.nodes[j].name,
                              "Flow (bbl/d)": f"{flow:,.0f}", "Cap (bbl/d)": f"{arc.capacity:,.0f}",
                              "Util %": f"{util:.1f}%", "$/bbl": f"${arc.transport_cost:.2f}",
                              "Daily Cost": f"${flow * arc.transport_cost:,.0f}"})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ─── SCENARIOS ───────────────────────────────────────────────────────────────
with tab_sc:
    if run_scenarios_btn or st.session_state.get("sc_results"):
        if run_scenarios_btn:
            with st.spinner("Running 9 scenarios…"):
                runner = ScenarioRunner(copy.deepcopy(_base_net()), solver="cbc")
                runner._base_result = _base_result()
                sc_results = runner.run_all(build_standard_scenarios(_base_net()))
            st.session_state["sc_results"] = sc_results
        else:
            sc_results = st.session_state["sc_results"]

        base_obj_v = _base_result().objective_value
        names = [s.scenario_name for s in sc_results]
        objs = [s.result.objective_value for s in sc_results]
        cats = [s.category for s in sc_results]

        st.plotly_chart(scenario_comparison(names, objs, base_obj_v, cats),
                        use_container_width=True, config={"displayModeBar": False})

        sc_rows = []
        for s in sc_results:
            T_s = s.result.planning_horizon
            unmet_s = sum(s.result.unmet_demand.values())
            dem_s = sum(n.demand for n in _base_net().get_nodes_by_type(NodeType.DEMAND)) * T_s
            svc_s = (1 - unmet_s / max(dem_s, 1)) * 100
            sc_rows.append({
                "Scenario": s.scenario_name, "Category": s.category.title(),
                "Net Margin ($)": f"${s.result.objective_value:,.0f}",
                "Δ vs Base ($)": f"${s.delta_objective:+,.0f}",
                "Service Level": f"{svc_s:.1f}%",
                "Avg Carbon (t/d)": f"{sum(s.result.carbon_by_period.values())/T_s:.0f}",
                "Solve (s)": f"{s.result.solver_time:.1f}",
            })
        st.dataframe(pd.DataFrame(sc_rows).set_index("Scenario"), use_container_width=True)

        if run_sens_btn:
            with st.spinner("Running sensitivity analysis…"):
                entries = run_sensitivity(copy.deepcopy(_base_net()), _base_result())
            st.plotly_chart(tornado_chart(entries), use_container_width=True,
                            config={"displayModeBar": False})
    else:
        st.info("Click **Run All Scenarios** in the sidebar to compare supply disruptions, "
                "demand spikes, refinery outages, cost shocks, and policy constraints.")

# ─── STOCHASTIC ───────────────────────────────────────────────────────────────
with tab_stoch:
    if run_stoch_btn or st.session_state.get("stoch_result"):
        if run_stoch_btn:
            with st.spinner("Running two-stage stochastic analysis (10 crack spread scenarios)…"):
                sr = run_stochastic_analysis(
                    copy.deepcopy(_base_net()), n_scenarios=10,
                    use_extensive_form=False,   # fast mode for UI
                )
            st.session_state["stoch_result"] = sr
        else:
            sr = st.session_state["stoch_result"]

        st.markdown("### Stochastic Programming Metrics")
        st.plotly_chart(evpi_vss_table(sr), use_container_width=True,
                        config={"displayModeBar": False})

        col_fan, col_text = st.columns([2, 1])
        with col_fan:
            st.plotly_chart(stochastic_fan(sr), use_container_width=True,
                            config={"displayModeBar": False})
        with col_text:
            st.markdown("**Risk Metrics**")
            risk = sr.risk
            st.markdown(f"""
| Metric | Value |
|--------|-------|
| EVPI | ${sr.evpi:,.0f} |
| VSS  | ${sr.vss:,.0f}  |
| VaR 95% | ${risk.var_95:,.0f} |
| CVaR 95% | ${risk.cvar_95:,.0f} |
| VaR 99% | ${risk.var_99:,.0f} |
| Std Dev | ${risk.std_obj:,.0f} |

**EVPI** — Max worth paying for a perfect crack spread forecast.

**VSS** — Gain from stochastic planning over deterministic mean-price approach.

**VaR95** — Net margin at risk: 5% of scenarios fall below this level.

**CVaR95** — Expected margin in the worst 5% of scenarios (tail risk).
""")

        # Scenario distribution table
        st.markdown("**Scenario Breakdown**")
        sc_df = pd.DataFrame([
            {"Scenario": k, "Net Margin ($)": f"${v:,.0f}",
             "vs. RP ($)": f"${v - sr.rp:+,.0f}"}
            for k, v in sorted(sr.scenario_objectives.items(),
                                key=lambda x: -x[1])
        ])
        st.dataframe(sc_df, use_container_width=True, hide_index=True)
    else:
        st.info("Click **Stochastic Analysis** in the sidebar to compute EVPI, VSS, "
                "VaR/CVaR, and the scenario distribution under crack spread uncertainty.")

# ─── ROLLING HORIZON ─────────────────────────────────────────────────────────
with tab_rh:
    if run_rh_btn or st.session_state.get("rh_result"):
        if run_rh_btn:
            from src.analysis.rolling_horizon import RollingHorizonOptimizer
            with st.spinner("Running 30-day rolling horizon simulation…"):
                rh = RollingHorizonOptimizer(
                    copy.deepcopy(_base_net()),
                    rolling_horizon=7,
                    simulation_days=30,
                    replan_trigger="always",
                    solver="cbc",
                    noise_level=0.05,
                )
                rh_result = rh.run()
            st.session_state["rh_result"] = rh_result
        else:
            rh_result = st.session_state["rh_result"]

        from src.viz.rolling_charts import (
            rolling_margin_chart, service_level_trend, replan_trigger_chart
        )

        # Summary KPIs
        rk1, rk2, rk3, rk4 = st.columns(4)
        with rk1:
            kpi("Realized Margin (30d)", f"${rh_result.total_realized_margin:,.0f}")
        with rk2:
            kpi("Replanning Value", f"${rh_result.replanning_value:,.0f}",
                ok=rh_result.replanning_value >= 0)
        with rk3:
            kpi("Avg Service Level", f"{rh_result.avg_service_level:.1%}",
                ok=rh_result.avg_service_level >= 0.95)
        with rk4:
            kpi("Execution Gap (avg/d)", f"${rh_result.plan_execution_gap:,.0f}")

        st.markdown("")
        st.plotly_chart(rolling_margin_chart(rh_result.execution_log),
                        use_container_width=True, config={"displayModeBar": False})

        col_svc, col_crack = st.columns(2)
        with col_svc:
            st.plotly_chart(service_level_trend(rh_result.execution_log),
                            use_container_width=True, config={"displayModeBar": False})
        with col_crack:
            st.plotly_chart(replan_trigger_chart(rh_result.execution_log),
                            use_container_width=True, config={"displayModeBar": False})

        # Day-by-day log
        with st.expander("Day-by-Day Execution Log", expanded=False):
            log_rows = []
            for e in rh_result.execution_log:
                log_rows.append({
                    "Day": e.day, "Date": str(e.date),
                    "Planned Margin ($)": f"${e.planned_margin:,.0f}",
                    "Realized Margin ($)": f"${e.realized_margin:,.0f}",
                    "Gap ($)": f"${e.realized_margin - e.planned_margin:+,.0f}",
                    "Unmet Demand (bbl)": f"{sum(e.unmet_demand.values()):,.0f}",
                    "Replanned": "✓" if e.replanned else "–",
                })
            st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)

        st.markdown("""
**What this shows:** Each day the optimizer solves a fresh 7-day lookahead plan.
Only Day 1 is executed; noise is injected to simulate real production variability,
demand uncertainty, and crack spread movements. The **Replanning Value** is the
gain from re-optimizing daily vs. holding the initial static plan fixed.
""")
    else:
        st.info("Click **Rolling Horizon** in the sidebar to simulate 30 days of MPC-style "
                "replanning with production noise, demand uncertainty, and price shocks.")

# ─── DEMAND FORECASTING ───────────────────────────────────────────────────────
with tab_fc:
    if run_forecast_btn or st.session_state.get("forecasts"):
        if run_forecast_btn:
            from src.analysis.demand_forecast import (
                DemandForecaster, generate_synthetic_history, MARKET_PROFILES
            )
            with st.spinner("Generating demand forecasts…"):
                forecaster = DemandForecaster()
                forecaster.fit_from_defaults(
                    list(MARKET_PROFILES.keys()), n_history_days=180
                )
                forecasts = forecaster.forecast(horizon=horizon)
                histories = {
                    mid: generate_synthetic_history(mid, n_days=180)
                    for mid in MARKET_PROFILES
                }
            st.session_state["forecasts"] = (forecasts, histories, forecaster)
        else:
            forecasts, histories, forecaster = st.session_state["forecasts"]

        from src.viz.rolling_charts import forecast_chart
        from src.analysis.demand_forecast import MARKET_PROFILES

        st.markdown("### Demand Forecasts — All Markets")
        st.markdown(
            "Ensemble of Holt-Winters triple exponential smoothing and naive seasonal baseline. "
            "Calibrated to EIA PADD 2/3 weekly petroleum supply data patterns."
        )

        # Summary table
        summary = forecaster.forecast_summary()
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
        st.markdown("")

        # Individual fan charts, 2-up
        market_ids = list(forecasts.keys())
        for i in range(0, len(market_ids), 2):
            cols = st.columns(2)
            for k, mid in enumerate(market_ids[i:i+2]):
                fc = forecasts[mid]
                hist = histories.get(mid)
                with cols[k]:
                    name = MARKET_PROFILES.get(mid, {}).get("name", mid)
                    st.plotly_chart(
                        forecast_chart(name, fc, hist),
                        use_container_width=True, config={"displayModeBar": False},
                    )

        st.markdown("**Forecast → Optimizer Integration**")
        st.markdown(
            "These forecasts feed directly into the optimizer: the point forecast "
            "replaces static flat demand parameters, and the 80%/95% intervals drive "
            "the stochastic scenario bounds. Enable via the Python API:"
        )
        st.code("""
from src.analysis.demand_forecast import DemandForecaster
from data.generate_data import build_base_network

forecaster = DemandForecaster()
forecaster.fit_from_defaults(['M1','M2','M3','M4','M5'])
demand_params = forecaster.predict_demand_params(horizon=14)

net = build_base_network(horizon=14)
for market_id, periods in demand_params.items():
    avg = sum(periods.values()) / len(periods)
    net.nodes[market_id].demand = avg

result = MultiPeriodOptimizer(net).solve()
""", language="python")
    else:
        st.info("Click **Demand Forecast** in the sidebar to generate Holt-Winters ensemble "
                "forecasts with prediction intervals for all 5 market nodes.")

# ─── CARBON ──────────────────────────────────────────────────────────────────
with tab_carbon:
    col_emit, col_tbl = st.columns([2, 1])
    with col_emit:
        st.plotly_chart(carbon_emissions_chart(net, result),
                        use_container_width=True, config={"displayModeBar": False})
    with col_tbl:
        st.markdown("**Carbon Intensity by Grade**")
        grade_rows = [
            {"Grade": g.name, "API": g.api_gravity, "Sulfur %": g.sulfur_content,
             "Intensity (kg/bbl)": g.carbon_intensity,
             "Price Diff ($/bbl)": f"{g.price_differential:+.2f}"}
            for g in net.grades.values()
        ]
        st.dataframe(pd.DataFrame(grade_rows), use_container_width=True, hide_index=True)
        st.markdown("")
        if net.carbon_budget_per_day:
            budget_use = avg_carbon / net.carbon_budget_per_day * 100
            st.markdown(f"**Budget:** {net.carbon_budget_per_day:.0f} tCO₂e/day")
            st.markdown(f"**Avg Emissions:** {avg_carbon:.1f} tCO₂e/day")
            color = "red" if budget_use > 95 else "orange" if budget_use > 80 else "green"
            st.progress(min(1.0, budget_use / 100), text=f"{budget_use:.1f}% of budget")
        else:
            st.markdown(f"**Avg Daily Emissions:** {avg_carbon:.1f} tCO₂e/day")
            st.caption("Enable a carbon cap in the sidebar to model regulatory constraints.")

    st.markdown("**Period-by-Period Carbon Footprint**")
    import plotly.express as px
    c_fig = px.bar(
        pd.DataFrame({"Day": list(result.carbon_by_period.keys()),
                      "tCO₂e": list(result.carbon_by_period.values())}),
        x="Day", y="tCO₂e", color_discrete_sequence=["#44BBA4"], template="plotly_dark",
    )
    c_fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                        height=250, margin=dict(l=50, r=30, t=30, b=40))
    if net.carbon_budget_per_day:
        c_fig.add_hline(y=net.carbon_budget_per_day, line_dash="dash",
                        line_color="#f85149", annotation_text="Budget")
    st.plotly_chart(c_fig, use_container_width=True, config={"displayModeBar": False})

    # Carbon Pareto: what if we tighten the budget incrementally?
    if st.button("📉 Generate Carbon–Margin Pareto Curve", use_container_width=False):
        with st.spinner("Solving at 6 carbon budget levels…"):
            import copy
            base_carbon = avg_carbon
            budgets = [base_carbon * f for f in [1.0, 0.92, 0.85, 0.78, 0.70, 0.62]]
            pareto_rows = []
            for budget in budgets:
                n_copy = copy.deepcopy(net)
                apply_carbon_cap(n_copy, budget)
                r_copy = MultiPeriodOptimizer(n_copy).solve()
                c_copy = sum(r_copy.carbon_by_period.values()) / T
                pareto_rows.append({
                    "Carbon Budget (t/d)": f"{budget:.0f}",
                    "Realized Carbon (t/d)": f"{c_copy:.1f}",
                    "Net Margin ($)": f"${r_copy.objective_value:,.0f}",
                    "Margin Cost ($)": f"${result.objective_value - r_copy.objective_value:,.0f}",
                    "Implicit Carbon Price ($/t)": f"${(result.objective_value - r_copy.objective_value) / max((avg_carbon - c_copy)*T, 1):.0f}",
                })

        st.markdown("**Carbon–Margin Pareto Frontier**")
        st.dataframe(pd.DataFrame(pareto_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Implicit carbon price = marginal dollar cost per tonne CO₂e reduction. "
            "This is the internal shadow price of the carbon constraint — analogous to "
            "an EU ETS allowance price under the equivalent regulation."
        )
