"""
Two-stage stochastic programming — PuLP implementation.

Stage 1 (here-and-now): arc activation binaries y_{ij} — shared across scenarios.
Stage 2 (recourse):     routing flows x^ω — per scenario.

Metrics
───────
  WS   = Σ_ω p_ω · z*(ω)      Wait-and-See (perfect information upper bound)
  RP   = z*(EF)                Recourse Problem (non-anticipative, exact EF)
  EEV  = Σ_ω p_ω · Q(x̄, ω)   EV solution evaluated stochastically
  EVPI = WS − RP  ≥ 0
  VSS  = RP − EEV ≥ 0

Chain: WS ≥ RP ≥ EEV (Jensen's inequality).

Crack spread scenarios: log-normal GBM, σ_annual=22%, Itô-corrected so E[mult]=1.
"""

import copy
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pulp

from .optimizer import MultiPeriodOptimizer, MultiPeriodResult, _val
from .supply_chain import NodeType, SupplyChainNetwork

logger = logging.getLogger(__name__)


@dataclass
class CrackSpreadScenario:
    id: int
    probability: float
    multipliers: Dict[Tuple[str, int], float]
    label: str


@dataclass
class RiskMetrics:
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    min_obj: float
    max_obj: float
    mean_obj: float
    std_obj: float
    pct_below_zero: float


@dataclass
class StochasticResult:
    rp: float
    ws: float
    ev: float
    eev: float
    evpi: float
    vss: float
    scenario_objectives: Dict[str, float]
    scenario_results: List[MultiPeriodResult]
    mean_value_result: Optional[MultiPeriodResult]
    risk: RiskMetrics
    num_scenarios: int
    solver_time: float


# ── Scenario generation ───────────────────────────────────────────────────────

def generate_crack_spread_scenarios(
    network: SupplyChainNetwork,
    n_scenarios: int = 10,
    annual_vol: float = 0.22,
    seed: int = 42,
) -> List[CrackSpreadScenario]:
    rng = np.random.default_rng(seed)
    T = network.planning_horizon
    refinery_ids = [n for n, nd in network.nodes.items() if nd.node_type == NodeType.REFINERY]
    daily_vol = annual_vol / np.sqrt(252)
    drift = -0.5 * daily_vol ** 2
    prob = 1.0 / n_scenarios
    scenarios = []

    for s in range(n_scenarios):
        mults: Dict[Tuple[str, int], float] = {}
        for ref_id in refinery_ids:
            z = rng.normal(0.0, 1.0, T)
            path = np.exp(np.cumsum(drift + daily_vol * z))
            path /= path.mean()
            for t_idx, mult in enumerate(path):
                mults[(ref_id, t_idx + 1)] = float(mult)
        term = np.mean([v for (_, t), v in mults.items() if t == T])
        tag = "Bull" if term > 1.15 else ("Bear" if term < 0.85 else "Base")
        scenarios.append(CrackSpreadScenario(
            id=s, probability=prob, multipliers=mults,
            label=f"S{s+1:02d}-{tag}",
        ))
    return scenarios


# ── Wait-and-See ──────────────────────────────────────────────────────────────

def _wait_and_see(
    network: SupplyChainNetwork,
    scenarios: List[CrackSpreadScenario],
    time_limit: int,
) -> Tuple[float, List[MultiPeriodResult]]:
    ws = 0.0
    results = []
    for sc in scenarios:
        opt = MultiPeriodOptimizer(copy.deepcopy(network), time_limit=time_limit)
        opt.build(crack_spread_multipliers=sc.multipliers)
        r = opt.solve()
        ws += sc.probability * r.objective_value
        results.append(r)
        logger.debug(f"WS ω={sc.id}: ${r.objective_value:,.0f}")
    return ws, results


# ── Mean-value problem ────────────────────────────────────────────────────────

def _solve_ev(
    network: SupplyChainNetwork,
    scenarios: List[CrackSpreadScenario],
    time_limit: int,
) -> Tuple[float, MultiPeriodResult, Dict[Tuple[str, str], int]]:
    """
    Solve the mean-value (EV) problem.
    Returns (ev_objective, ev_result, y_vals) where y_vals are the
    Stage-1 arc activation decisions to be fixed when computing EEV.
    """
    T = network.planning_horizon
    refinery_ids = [n for n, nd in network.nodes.items() if nd.node_type == NodeType.REFINERY]
    mean_mults: Dict[Tuple[str, int], float] = {}
    for ref_id in refinery_ids:
        for t in range(1, T + 1):
            mean_mults[(ref_id, t)] = sum(
                sc.probability * sc.multipliers.get((ref_id, t), 1.0)
                for sc in scenarios
            )
    opt = MultiPeriodOptimizer(copy.deepcopy(network), time_limit=time_limit)
    opt.build(crack_spread_multipliers=mean_mults)
    r = opt.solve()
    # Extract Stage-1 binary decisions (arc activation) from the EV solution
    y_vals: Dict[Tuple[str, str], int] = {
        k: round(_val(v)) for k, v in opt._y.items()
    }
    return r.objective_value, r, y_vals


def _compute_eev(
    network: SupplyChainNetwork,
    scenarios: List[CrackSpreadScenario],
    time_limit: int,
    y_vals: Dict[Tuple[str, str], int],
) -> float:
    """
    EEV: evaluate the EV Stage-1 solution under each scenario.

    Stage-1 arc activation binaries (y) are fixed to the values from
    the mean-value solution. Only Stage-2 routing variables are free.
    This correctly measures the cost of ignoring uncertainty — using
    deterministic mean-price arc commitments then operating under
    realized crack spread scenarios.
    """
    eev = 0.0
    for sc in scenarios:
        opt = MultiPeriodOptimizer(copy.deepcopy(network), time_limit=time_limit)
        opt.build(crack_spread_multipliers=sc.multipliers)
        # Fix Stage-1 binaries to the EV solution values
        for key, val in y_vals.items():
            if key in opt._y:
                opt._y[key].lowBound = val
                opt._y[key].upBound = val
        r = opt.solve()
        eev += sc.probability * r.objective_value
    return eev


# ── Recourse Problem — Extensive Form (PuLP) ──────────────────────────────────

def _recourse_problem_ef(
    network: SupplyChainNetwork,
    scenarios: List[CrackSpreadScenario],
    time_limit: int,
) -> Tuple[float, List[MultiPeriodResult]]:
    """
    Exact two-stage extensive form via PuLP.

    Shared Stage-1: y_{ij}  (arc activation binaries — one set)
    Replicated Stage-2: x^ω, s^ω, u^ω, deficit^ω  (per scenario)

    Non-anticipativity is automatically satisfied because all scenarios share y.
    """
    net = network
    T = net.planning_horizon
    arc_keys = net.arc_index
    node_ids = list(net.nodes.keys())
    grade_ids = net.grade_ids()
    periods = list(range(1, T + 1))
    N = len(scenarios)

    well_ids     = [n for n in node_ids if net.nodes[n].node_type == NodeType.WELL]
    storage_ids  = [n for n in node_ids if net.nodes[n].node_type == NodeType.STORAGE]
    refinery_ids = [n for n in node_ids if net.nodes[n].node_type == NodeType.REFINERY]
    dist_ids     = [n for n in node_ids if net.nodes[n].node_type == NodeType.DISTRIBUTION]
    demand_ids   = [n for n in node_ids if net.nodes[n].node_type == NodeType.DEMAND]
    fixed_arcs   = [k for k in arc_keys if net.arc_lookup(*k).fixed_cost > 0]
    sop_arcs     = [(c.arc_origin, c.arc_dest) for c in net.contracts]
    sour_grades  = [g for g in grade_ids if net.grades[g].sulfur_content > 0.5]
    sweet_arcs   = [(a.origin, a.destination) for a in net.arcs if not a.accepts_sour]

    prob = pulp.LpProblem("EF_2SSP", pulp.LpMaximize)

    # Shared Stage-1 binary
    y = pulp.LpVariable.dicts("y", fixed_arcs, cat="Binary") if fixed_arcs else {}

    # Per-scenario Stage-2 variables
    x = {}; s = {}; u = {}; deficit = {}

    for sc in scenarios:
        w = sc.id
        for (i, j) in arc_keys:
            for t in periods:
                for g in grade_ids:
                    x[(w, i, j, t, g)] = pulp.LpVariable(f"x_{w}_{i}_{j}_{t}_{g}", lowBound=0)
        for nid in storage_ids:
            for t in periods:
                for g in grade_ids:
                    s[(w, nid, t, g)] = pulp.LpVariable(f"s_{w}_{nid}_{t}_{g}", lowBound=0)
        for nid in demand_ids:
            for t in periods:
                u[(w, nid, t)] = pulp.LpVariable(f"u_{w}_{nid}_{t}", lowBound=0)
        for (i, j) in sop_arcs:
            for t in periods:
                deficit[(w, i, j, t)] = pulp.LpVariable(f"def_{w}_{i}_{j}_{t}", lowBound=0)

    # Helpers
    def inf_(w, nid, t):
        return pulp.lpSum(x[(w, i, nid, t, g)] for i in node_ids if (i, nid) in arc_keys for g in grade_ids)

    def out_(w, nid, t):
        return pulp.lpSum(x[(w, nid, j, t, g)] for j in node_ids if (nid, j) in arc_keys for g in grade_ids)

    def inf_g(w, nid, t, g):
        return pulp.lpSum(x[(w, i, nid, t, g)] for i in node_ids if (i, nid) in arc_keys)

    def out_g(w, nid, t, g):
        return pulp.lpSum(x[(w, nid, j, t, g)] for j in node_ids if (nid, j) in arc_keys)

    # ── Objective ─────────────────────────────────────────────────────────────
    stage2_obj = pulp.lpSum(
        sc.probability * (
            pulp.lpSum(
                net.nodes[r].refinery_margin * sc.multipliers.get((r, t), 1.0) * inf_(sc.id, r, t)
                for r in refinery_ids for t in periods
            )
            - pulp.lpSum(
                net.arc_lookup(i, j).transport_cost * x[(sc.id, i, j, t, g)]
                for (i, j) in arc_keys for t in periods for g in grade_ids
            )
            - pulp.lpSum(
                net.nodes[n].holding_cost * s[(sc.id, n, t, g)]
                for n in storage_ids for t in periods for g in grade_ids
            )
            - pulp.lpSum(
                net.contract_lookup(i, j).deficiency_charge * deficit[(sc.id, i, j, t)]
                for (i, j) in sop_arcs for t in periods
            )
            - pulp.lpSum(
                net.nodes[n].unmet_penalty * u[(sc.id, n, t)]
                for n in demand_ids for t in periods
            )
        )
        for sc in scenarios
    )
    stage1_obj = (
        pulp.lpSum(net.arc_lookup(i, j).fixed_cost * y[(i, j)] for (i, j) in fixed_arcs)
        if fixed_arcs else 0
    )
    prob += stage2_obj - stage1_obj

    # ── Constraints (replicated per scenario) ─────────────────────────────────
    for sc in scenarios:
        w = sc.id

        # Grade lock
        for nid in well_ids:
            pg = net.nodes[nid].primary_grade
            for t in periods:
                for g in grade_ids:
                    if g != pg:
                        prob += out_g(w, nid, t, g) == 0, f"gl_{w}_{nid}_{t}_{g}"

        # Well capacity
        for nid in well_ids:
            for t in periods:
                prob += out_(w, nid, t) <= net.well_capacity(nid, t), f"wc_{w}_{nid}_{t}"

        # Storage balance
        for nid in storage_ids:
            for t in periods:
                for g in grade_ids:
                    prev = (net.nodes[nid].initial_inv_by_grade.get(g, 0.0)
                            if t == 1 else s[(w, nid, t-1, g)])
                    prob += (prev + inf_g(w, nid, t, g) == out_g(w, nid, t, g) + s[(w, nid, t, g)],
                             f"sb_{w}_{nid}_{t}_{g}")

        # Storage capacity
        for nid in storage_ids:
            for t in periods:
                prob += (pulp.lpSum(s[(w, nid, t, g)] for g in grade_ids) <= net.nodes[nid].max_capacity,
                         f"sc_{w}_{nid}_{t}")

        # Refinery bounds
        for nid in refinery_ids:
            for t in periods:
                prob += inf_(w, nid, t) >= net.nodes[nid].min_throughput, f"rlo_{w}_{nid}_{t}"
                prob += inf_(w, nid, t) <= net.nodes[nid].max_capacity,   f"rhi_{w}_{nid}_{t}"

        # Crude diet API
        for nid in refinery_ids:
            d = net.nodes[nid].crude_diet
            if d is None:
                continue
            for t in periods:
                pairs = [(i, g) for i in node_ids if (i, nid) in arc_keys for g in grade_ids]
                if not pairs:
                    continue
                wapi  = pulp.lpSum(net.grades[g].api_gravity * x[(w, i, nid, t, g)] for i, g in pairs)
                total = pulp.lpSum(x[(w, i, nid, t, g)] for i, g in pairs)
                prob += wapi >= d.api_min * total, f"alo_{w}_{nid}_{t}"
                prob += wapi <= d.api_max * total, f"ahi_{w}_{nid}_{t}"

        # Crude diet sulfur
        for nid in refinery_ids:
            d = net.nodes[nid].crude_diet
            if d is None:
                continue
            for t in periods:
                pairs = [(i, g) for i in node_ids if (i, nid) in arc_keys for g in grade_ids]
                if not pairs:
                    continue
                ws2  = pulp.lpSum(net.grades[g].sulfur_content * x[(w, i, nid, t, g)] for i, g in pairs)
                total = pulp.lpSum(x[(w, i, nid, t, g)] for i, g in pairs)
                prob += ws2 <= d.sulfur_max * total, f"sulf_{w}_{nid}_{t}"

        # Demand
        for nid in demand_ids:
            for t in periods:
                prob += inf_(w, nid, t) + u[(w, nid, t)] >= net.nodes[nid].demand, f"dem_{w}_{nid}_{t}"

        # Arc capacity
        for (i, j) in arc_keys:
            cap = net.arc_lookup(i, j).capacity
            for t in periods:
                prob += (pulp.lpSum(x[(w, i, j, t, g)] for g in grade_ids) <= cap,
                         f"ac_{w}_{i}_{j}_{t}")

        # Activation (shared y)
        for (i, j) in fixed_arcs:
            cap = net.arc_lookup(i, j).capacity
            for t in periods:
                prob += (pulp.lpSum(x[(w, i, j, t, g)] for g in grade_ids) <= cap * y[(i, j)],
                         f"act_{w}_{i}_{j}_{t}")

        # Ship-or-pay
        for (i, j) in sop_arcs:
            vmin = net.contract_lookup(i, j).min_daily_volume
            for t in periods:
                prob += (pulp.lpSum(x[(w, i, j, t, g)] for g in grade_ids) + deficit[(w, i, j, t)] >= vmin,
                         f"sop_{w}_{i}_{j}_{t}")

        # Distribution conservation
        for nid in dist_ids:
            for t in periods:
                prob += inf_(w, nid, t) == out_(w, nid, t), f"dist_{w}_{nid}_{t}"

        # Sweet-only pipelines
        for (i, j) in sweet_arcs:
            if (i, j) not in arc_keys:
                continue
            for t in periods:
                for g in sour_grades:
                    prob += x[(w, i, j, t, g)] == 0, f"sw_{w}_{i}_{j}_{t}_{g}"

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=0.005)
    prob.solve(solver)

    if prob.status not in (1,):
        raise RuntimeError(f"EF solver status: {pulp.LpStatus[prob.status]}")

    raw_rp = pulp.value(prob.objective)
    rp = float(raw_rp) if raw_rp is not None else 0.0

    # Build per-scenario summary results
    sc_results = []
    for sc in scenarios:
        w = sc.id
        flows_p = {
            (i, j, t): sum(_val(x[(w, i, j, t, g)]) for g in grade_ids)
            for (i, j) in arc_keys for t in periods
        }
        carbon = {
            t: sum(net.grades[g].carbon_intensity * _val(x[(w, i, j, t, g)])
                   for (i, j) in arc_keys for g in grade_ids) / 1000.0
            for t in periods
        }
        rev = sum(
            net.nodes[nid].refinery_margin * sc.multipliers.get((nid, t), 1.0)
            * flows_p.get((i, nid, t), 0)
            for nid in refinery_ids
            for i in node_ids if (i, nid) in arc_keys
            for t in periods
        )
        tc = sum(net.arc_lookup(i, j).transport_cost * flows_p[(i, j, t)]
                 for (i, j) in arc_keys for t in periods)
        sc_results.append(MultiPeriodResult(
            status="optimal", objective_value=rev - tc,
            flows_by_grade={}, flows_by_period=flows_p,
            inventories={},
            unmet_demand={(nid, t): _val(u[(w, nid, t)]) for nid in demand_ids for t in periods},
            sop_deficits={(c.arc_origin, c.arc_dest, t): _val(deficit[(w, c.arc_origin, c.arc_dest, t)])
                          for c in net.contracts for t in periods},
            revenue=rev, transport_cost=tc, holding_cost=0, penalty_cost=0, sop_cost=0,
            opex_cost=0, fixed_cost=0,
            carbon_by_period=carbon, solver_time=0, planning_horizon=T,
        ))

    return rp, sc_results


# ── Risk metrics ──────────────────────────────────────────────────────────────

def compute_risk_metrics(objectives: List[float]) -> RiskMetrics:
    arr = np.sort(np.array(objectives))
    n = len(arr)

    def _vc(cl):
        idx = max(0, int((1 - cl) * n) - 1)
        return float(arr[idx]), float(arr[:max(1, idx)].mean())

    v95, c95 = _vc(0.95)
    v99, c99 = _vc(0.99)
    return RiskMetrics(
        var_95=v95, cvar_95=c95, var_99=v99, cvar_99=c99,
        min_obj=float(arr.min()), max_obj=float(arr.max()),
        mean_obj=float(arr.mean()), std_obj=float(arr.std(ddof=1)),
        pct_below_zero=float((arr < 0).mean() * 100),
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def run_stochastic_analysis(
    network: SupplyChainNetwork,
    n_scenarios: int = 10,
    annual_vol: float = 0.22,
    seed: int = 42,
    time_limit: int = 300,
    use_extensive_form: bool = True,
) -> StochasticResult:
    """
    Full two-stage stochastic analysis.

    Parameters
    ----------
    use_extensive_form : bool
        True  → exact EF (shared Stage-1 y, correct RP)
        False → approximate RP (fast, equals WS — for diagnostics only)
    """
    t0 = time.time()
    scenarios = generate_crack_spread_scenarios(network, n_scenarios, annual_vol, seed)
    logger.info(f"Generated {n_scenarios} scenarios (σ_ann={annual_vol:.0%})")

    ws_val, ws_results = _wait_and_see(network, scenarios, time_limit)
    logger.info(f"WS  = ${ws_val:,.0f}")

    ev_val, ev_result, y_vals = _solve_ev(network, scenarios, time_limit)
    logger.info(f"EV  = ${ev_val:,.0f}")

    eev_val = _compute_eev(network, scenarios, time_limit, y_vals)
    logger.info(f"EEV = ${eev_val:,.0f}")

    if use_extensive_form:
        try:
            rp_val, rp_sc = _recourse_problem_ef(network, scenarios, time_limit)
            logger.info(f"RP  = ${rp_val:,.0f} (exact EF)")
        except Exception as e:
            logger.warning(f"EF failed ({e}); using EEV as RP lower bound")
            rp_val, rp_sc = eev_val, ws_results
    else:
        rp_val, rp_sc = eev_val, ws_results
        logger.info(f"RP  = ${rp_val:,.0f} (approx)")

    evpi = max(0.0, ws_val - rp_val)
    vss  = max(0.0, rp_val - eev_val)
    risk = compute_risk_metrics([r.objective_value for r in ws_results])

    elapsed = time.time() - t0
    logger.info(f"Done {elapsed:.1f}s | EVPI=${evpi:,.0f} | VSS=${vss:,.0f}")

    return StochasticResult(
        rp=rp_val, ws=ws_val, ev=ev_val, eev=eev_val,
        evpi=evpi, vss=vss,
        scenario_objectives={sc.label: r.objective_value for sc, r in zip(scenarios, ws_results)},
        scenario_results=ws_results,
        mean_value_result=ev_result,
        risk=risk,
        num_scenarios=n_scenarios,
        solver_time=elapsed,
    )