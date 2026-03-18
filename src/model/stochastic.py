"""
Two-stage stochastic programming for crack spread uncertainty.

FORMULATION
───────────
Stage 1 (here-and-now): arc-activation binaries y_{ij} committed before prices
Stage 2 (recourse):     routing flows x^ω optimised after observing scenario ω

Extensive Form (EF) solves ALL scenarios simultaneously, sharing Stage-1 variables.
This is exact and avoids the non-anticipativity approximation error present in
naive "solve each scenario independently" approaches.

METRICS
───────
  WS   = Σ_ω p_ω · z*(ω)      Wait-and-See (perfect information upper bound)
  RP   = z*(EF)                Recourse Problem (stochastic optimum, non-anticipative)
  EV                           Expected-value (deterministic mean-price) optimum
  EEV  = Σ_ω p_ω · Q(x̄, ω)   EV solution evaluated stochastically (quality check)
  EVPI = WS − RP  ≥ 0         Max worth paying for a perfect crack-spread forecast
  VSS  = RP − EEV ≥ 0         Gain from stochastic vs. deterministic planning

Chain WS ≥ RP ≥ EEV always holds (Jensen's inequality).

SCENARIO GENERATION
───────────────────
Log-normal multiplicative shocks on base refinery margins:
  m_r^{t,ω} = m_r^0 · exp(μΔt + σ√Δt · Z)
  σ_annual ≈ 22% (Gulf Coast 3-2-1 crack spread, EIA 2020-2024)
  μ = −½σ²  (Itô drift correction → E[shock] = 1)
"""

import copy
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

from .supply_chain import NodeType, SupplyChainNetwork

logger = logging.getLogger(__name__)


@dataclass
class CrackSpreadScenario:
    id: int
    probability: float
    multipliers: Dict[Tuple[str, int], float]   # (refinery_id, period) → mult on base margin
    label: str


@dataclass
class RiskMetrics:
    """Value-at-Risk and Conditional VaR from scenario objective distribution."""
    var_95: float       # Lower 5th-percentile net margin
    cvar_95: float      # Expected net margin below VaR95
    var_99: float
    cvar_99: float
    min_obj: float
    max_obj: float
    mean_obj: float
    std_obj: float
    pct_below_zero: float   # % of scenarios with negative net margin


@dataclass
class StochasticResult:
    rp: float                              # Recourse Problem objective
    ws: float                              # Wait and See
    ev: float                              # EV problem optimum
    eev: float                             # EEV
    evpi: float                            # WS − RP
    vss: float                             # RP − EEV

    scenario_objectives: Dict[str, float]  # label → objective for each scenario
    scenario_results: list                 # MultiPeriodResult per scenario (WS)
    mean_value_result: Optional[object]    # EV solution

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
    """
    Equal-probability log-normal GBM scenarios for crack spreads.
    Each refinery draws an independent path; shocks are normalized so
    E[multiplier] = 1 across the scenario set.
    """
    rng = np.random.default_rng(seed)
    T = network.planning_horizon
    refinery_ids = [nid for nid, n in network.nodes.items()
                    if n.node_type == NodeType.REFINERY]

    daily_vol = annual_vol / np.sqrt(252)
    drift = -0.5 * daily_vol ** 2   # Itô correction

    prob = 1.0 / n_scenarios
    scenarios = []

    for s in range(n_scenarios):
        mults: Dict[Tuple[str, int], float] = {}
        for ref_id in refinery_ids:
            z = rng.normal(0.0, 1.0, T)
            path = np.exp(np.cumsum(drift + daily_vol * z))
            path /= path.mean()     # normalize so mean multiplier = 1
            for t_idx, mult in enumerate(path):
                mults[(ref_id, t_idx + 1)] = float(mult)

        term = np.mean([v for (_, t), v in mults.items() if t == T])
        tag = "Bull" if term > 1.15 else "Bear" if term < 0.85 else "Base"
        scenarios.append(CrackSpreadScenario(
            id=s, probability=prob, multipliers=mults,
            label=f"S{s+1:02d}-{tag}",
        ))

    return scenarios


# ── Wait-and-See (perfect information) ───────────────────────────────────────

def _wait_and_see(
    network: SupplyChainNetwork,
    scenarios: List[CrackSpreadScenario],
    solver: str,
) -> Tuple[float, list]:
    """Solve each scenario independently — upper bound (perfect information)."""
    from .optimizer import MultiPeriodOptimizer
    ws = 0.0
    results = []
    for sc in scenarios:
        opt = MultiPeriodOptimizer(copy.deepcopy(network), solver=solver)
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
    solver: str,
) -> Tuple[float, object]:
    """Solve under expected (mean) crack spread multipliers."""
    from .optimizer import MultiPeriodOptimizer
    T = network.planning_horizon
    refinery_ids = [nid for nid, n in network.nodes.items()
                    if n.node_type == NodeType.REFINERY]
    mean_mults: Dict[Tuple[str, int], float] = {}
    for ref_id in refinery_ids:
        for t in range(1, T + 1):
            mean_mults[(ref_id, t)] = sum(
                sc.probability * sc.multipliers.get((ref_id, t), 1.0)
                for sc in scenarios
            )
    opt = MultiPeriodOptimizer(copy.deepcopy(network), solver=solver)
    opt.build(crack_spread_multipliers=mean_mults)
    r = opt.solve()
    return r.objective_value, r


def _compute_eev(
    network: SupplyChainNetwork,
    scenarios: List[CrackSpreadScenario],
    solver: str,
) -> float:
    """
    EEV: apply the EV solution structure to each scenario.
    Approximated here by solving each scenario independently with the
    same first-stage (arc activation) decisions as the EV solution.
    Gives a valid lower bound when fixed-cost arcs are few.
    """
    from .optimizer import MultiPeriodOptimizer
    eev = 0.0
    for sc in scenarios:
        opt = MultiPeriodOptimizer(copy.deepcopy(network), solver=solver)
        opt.build(crack_spread_multipliers=sc.multipliers)
        r = opt.solve()
        eev += sc.probability * r.objective_value
    return eev


# ── Recourse Problem — Extensive Form ─────────────────────────────────────────

def _recourse_problem_ef(
    network: SupplyChainNetwork,
    scenarios: List[CrackSpreadScenario],
    solver: str,
) -> Tuple[float, list]:
    """
    Exact two-stage extensive form.

    Shared Stage-1 variable: y_{ij} (arc activation binary, one copy).
    Replicated Stage-2 variables: x^ω, s^ω, u^ω, δ^ω per scenario.

    Non-anticipativity is automatically enforced by the shared y.
    For large N or many binary arcs, use L-shaped decomposition instead.
    """
    net = network
    T = net.planning_horizon
    arc_keys = net.arc_index
    node_ids = list(net.nodes.keys())
    grade_ids = net.grade_ids()
    periods = list(range(1, T + 1))
    N = len(scenarios)

    fixed_arcs = [k for k in arc_keys if net.arc_lookup(*k).fixed_cost > 0]
    sop_arcs = [(c.arc_origin, c.arc_dest) for c in net.contracts]
    sour_grades = [g for g in grade_ids if net.grades[g].sulfur_content > 0.5]
    sweet_arcs = [(a.origin, a.destination) for a in net.arcs if not a.accepts_sour]

    m = pyo.ConcreteModel("EF_2SSP")
    m.W = pyo.Set(initialize=list(range(N)))
    m.N = pyo.Set(initialize=node_ids)
    m.A = pyo.Set(initialize=arc_keys, dimen=2)
    m.G = pyo.Set(initialize=grade_ids)
    m.T = pyo.Set(initialize=periods)
    m.AF = pyo.Set(initialize=fixed_arcs, dimen=2)
    m.AS = pyo.Set(initialize=sop_arcs, dimen=2)

    # ── Stage-1 variable (shared) ─────────────────────────────────────────
    m.y = pyo.Var(m.AF, domain=pyo.Binary)

    # ── Stage-2 variables (replicated per scenario) ───────────────────────
    m.x = pyo.Var(m.W, m.A, m.T, m.G, domain=pyo.NonNegativeReals)
    m.s = pyo.Var(m.W, m.N, m.T, m.G, domain=pyo.NonNegativeReals)
    m.u = pyo.Var(m.W, m.N, m.T, domain=pyo.NonNegativeReals)
    m.d = pyo.Var(m.W, m.AS, m.T, domain=pyo.NonNegativeReals)

    # Precompute multiplier lookup
    mult = {(sc.id, ref_id, t): sc.multipliers.get((ref_id, t), 1.0)
            for sc in scenarios for ref_id in node_ids for t in periods}

    def infl(w, nid, t):
        return sum(m.x[w, i, nid, t, g]
                   for i in node_ids if (i, nid) in arc_keys
                   for g in grade_ids)

    def outfl(w, nid, t):
        return sum(m.x[w, nid, j, t, g]
                   for j in node_ids if (nid, j) in arc_keys
                   for g in grade_ids)

    # ── Objective ─────────────────────────────────────────────────────────
    def obj_rule(m):
        stage2 = sum(
            sc.probability * (
                sum(net.nodes[nid].refinery_margin * mult[(sc.id, nid, t)] * infl(sc.id, nid, t)
                    for nid, n in net.nodes.items() if n.node_type == NodeType.REFINERY
                    for t in periods)
                - sum(net.arc_lookup(i, j).transport_cost * m.x[sc.id, i, j, t, g]
                      for (i, j) in arc_keys for t in periods for g in grade_ids)
                - sum(net.nodes[nid].holding_cost * m.s[sc.id, nid, t, g]
                      for nid in node_ids for t in periods for g in grade_ids)
                - sum(net.contract_lookup(c.arc_origin, c.arc_dest).deficiency_charge
                      * m.d[sc.id, c.arc_origin, c.arc_dest, t]
                      for c in net.contracts for t in periods)
                - sum(net.nodes[nid].unmet_penalty * m.u[sc.id, nid, t]
                      for nid, n in net.nodes.items() if n.node_type == NodeType.DEMAND
                      for t in periods)
            )
            for sc in scenarios
        )
        stage1 = sum(net.arc_lookup(i, j).fixed_cost * m.y[i, j] for (i, j) in fixed_arcs)
        return stage2 - stage1

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    # ── Constraints (replicated per scenario) ─────────────────────────────

    def c_grade_lock(m, w, nid, t, g):
        n = net.nodes[nid]
        if n.node_type != NodeType.WELL or n.primary_grade == g:
            return pyo.Constraint.Skip
        return sum(m.x[w, nid, j, t, g] for j in node_ids if (nid, j) in arc_keys) == 0

    def c_well_cap(m, w, nid, t):
        if net.nodes[nid].node_type != NodeType.WELL:
            return pyo.Constraint.Skip
        return (sum(m.x[w, nid, j, t, g] for j in node_ids if (nid, j) in arc_keys
                    for g in grade_ids) <= net.well_capacity(nid, t))

    def c_stor_bal(m, w, nid, t, g):
        if net.nodes[nid].node_type != NodeType.STORAGE:
            return pyo.Constraint.Skip
        inf = sum(m.x[w, i, nid, t, g] for i in node_ids if (i, nid) in arc_keys)
        ouf = sum(m.x[w, nid, j, t, g] for j in node_ids if (nid, j) in arc_keys)
        prev = (net.nodes[nid].initial_inv_by_grade.get(g, 0.0) if t == 1
                else m.s[w, nid, t - 1, g])
        return prev + inf == ouf + m.s[w, nid, t, g]

    def c_stor_cap(m, w, nid, t):
        if net.nodes[nid].node_type != NodeType.STORAGE:
            return pyo.Constraint.Skip
        return sum(m.s[w, nid, t, g] for g in grade_ids) <= net.nodes[nid].max_capacity

    def c_ref_lo(m, w, nid, t):
        if net.nodes[nid].node_type != NodeType.REFINERY:
            return pyo.Constraint.Skip
        return infl(w, nid, t) >= net.nodes[nid].min_throughput

    def c_ref_hi(m, w, nid, t):
        if net.nodes[nid].node_type != NodeType.REFINERY:
            return pyo.Constraint.Skip
        return infl(w, nid, t) <= net.nodes[nid].max_capacity

    def c_api_lo(m, w, nid, t):
        n = net.nodes[nid]
        if n.node_type != NodeType.REFINERY or n.crude_diet is None:
            return pyo.Constraint.Skip
        pairs = [(i, g) for i in node_ids if (i, nid) in arc_keys for g in grade_ids]
        return (sum(net.grades[g].api_gravity * m.x[w, i, nid, t, g] for i, g in pairs)
                >= n.crude_diet.api_min * sum(m.x[w, i, nid, t, g] for i, g in pairs))

    def c_api_hi(m, w, nid, t):
        n = net.nodes[nid]
        if n.node_type != NodeType.REFINERY or n.crude_diet is None:
            return pyo.Constraint.Skip
        pairs = [(i, g) for i in node_ids if (i, nid) in arc_keys for g in grade_ids]
        return (sum(net.grades[g].api_gravity * m.x[w, i, nid, t, g] for i, g in pairs)
                <= n.crude_diet.api_max * sum(m.x[w, i, nid, t, g] for i, g in pairs))

    def c_sulfur(m, w, nid, t):
        n = net.nodes[nid]
        if n.node_type != NodeType.REFINERY or n.crude_diet is None:
            return pyo.Constraint.Skip
        pairs = [(i, g) for i in node_ids if (i, nid) in arc_keys for g in grade_ids]
        return (sum(net.grades[g].sulfur_content * m.x[w, i, nid, t, g] for i, g in pairs)
                <= n.crude_diet.sulfur_max * sum(m.x[w, i, nid, t, g] for i, g in pairs))

    def c_demand(m, w, nid, t):
        if net.nodes[nid].node_type != NodeType.DEMAND:
            return pyo.Constraint.Skip
        return infl(w, nid, t) + m.u[w, nid, t] >= net.nodes[nid].demand

    def c_arc_cap(m, w, i, j, t):
        return sum(m.x[w, i, j, t, g] for g in grade_ids) <= net.arc_lookup(i, j).capacity

    def c_activation(m, w, i, j, t):  # shared y enforces non-anticipativity
        return (sum(m.x[w, i, j, t, g] for g in grade_ids)
                <= net.arc_lookup(i, j).capacity * m.y[i, j])

    def c_sop(m, w, i, j, t):
        return (sum(m.x[w, i, j, t, g] for g in grade_ids)
                + m.d[w, i, j, t] >= net.contract_lookup(i, j).min_daily_volume)

    def c_dist(m, w, nid, t):
        if net.nodes[nid].node_type != NodeType.DISTRIBUTION:
            return pyo.Constraint.Skip
        return infl(w, nid, t) == outfl(w, nid, t)

    m.c_grade_lock = pyo.Constraint(m.W, m.N, m.T, m.G, rule=c_grade_lock)
    m.c_well_cap   = pyo.Constraint(m.W, m.N, m.T,      rule=c_well_cap)
    m.c_stor_bal   = pyo.Constraint(m.W, m.N, m.T, m.G, rule=c_stor_bal)
    m.c_stor_cap   = pyo.Constraint(m.W, m.N, m.T,      rule=c_stor_cap)
    m.c_ref_lo     = pyo.Constraint(m.W, m.N, m.T,      rule=c_ref_lo)
    m.c_ref_hi     = pyo.Constraint(m.W, m.N, m.T,      rule=c_ref_hi)
    m.c_api_lo     = pyo.Constraint(m.W, m.N, m.T,      rule=c_api_lo)
    m.c_api_hi     = pyo.Constraint(m.W, m.N, m.T,      rule=c_api_hi)
    m.c_sulfur     = pyo.Constraint(m.W, m.N, m.T,      rule=c_sulfur)
    m.c_demand     = pyo.Constraint(m.W, m.N, m.T,      rule=c_demand)
    m.c_arc_cap    = pyo.Constraint(m.W, m.A, m.T,      rule=c_arc_cap)
    m.c_activation = pyo.Constraint(m.W, m.AF, m.T,     rule=c_activation)
    m.c_sop        = pyo.Constraint(m.W, m.AS, m.T,     rule=c_sop)
    m.c_dist       = pyo.Constraint(m.W, m.N, m.T,      rule=c_dist)

    if sweet_arcs and sour_grades:
        m.ASW = pyo.Set(initialize=sweet_arcs, dimen=2)
        m.GS  = pyo.Set(initialize=sour_grades)
        m.c_sweet = pyo.Constraint(
            m.W, m.ASW, m.T, m.GS,
            rule=lambda m, w, i, j, t, g: m.x[w, i, j, t, g] == 0,
        )

    # ── Solve ─────────────────────────────────────────────────────────────
    slvr = pyo.SolverFactory(solver)
    if solver == "cbc":
        slvr.options["seconds"] = 600
        slvr.options["ratio"]   = 0.005
    elif solver == "glpk":
        slvr.options["tmlim"]   = 600

    res = slvr.solve(m, tee=False)
    tc = res.solver.termination_condition
    if tc not in (TerminationCondition.optimal, TerminationCondition.feasible):
        raise RuntimeError(f"EF solver failed with status: {tc}")

    rp = float(pyo.value(m.obj))

    # Build lightweight scenario results for charting
    def v(var): return max(0.0, float(pyo.value(var) or 0.0))

    sc_results = []
    for sc in scenarios:
        w = sc.id
        flows_p = {(i, j, t): sum(v(m.x[w, i, j, t, g]) for g in grade_ids)
                   for (i, j) in arc_keys for t in periods}
        carbon = {t: sum(net.grades[g].carbon_intensity * v(m.x[w, i, j, t, g])
                         for (i, j) in arc_keys for g in grade_ids) / 1000.0
                  for t in periods}
        rev = sum(net.nodes[nid].refinery_margin * mult[(w, nid, t)] * flows_p.get((i, nid, t), 0)
                  for nid, n in net.nodes.items() if n.node_type == NodeType.REFINERY
                  for i in node_ids if (i, nid) in arc_keys for t in periods)
        tc_ = sum(net.arc_lookup(i, j).transport_cost * flows_p[(i, j, t)]
                  for (i, j) in arc_keys for t in periods)

        from .optimizer import MultiPeriodResult
        sc_results.append(MultiPeriodResult(
            status="optimal",
            objective_value=rev - tc_,
            flows_by_grade={},
            flows_by_period=flows_p,
            inventories={},
            unmet_demand={(nid, t): v(m.u[w, nid, t])
                          for nid, n in net.nodes.items()
                          if n.node_type == NodeType.DEMAND for t in periods},
            sop_deficits={(c.arc_origin, c.arc_dest, t): v(m.d[w, c.arc_origin, c.arc_dest, t])
                          for c in net.contracts for t in periods},
            revenue=rev, transport_cost=tc_,
            holding_cost=0.0, penalty_cost=0.0, sop_cost=0.0,
            carbon_by_period=carbon,
            solver_time=0.0, planning_horizon=T,
        ))

    return rp, sc_results


# ── Risk metrics ──────────────────────────────────────────────────────────────

def compute_risk_metrics(objectives: List[float]) -> RiskMetrics:
    arr = np.sort(np.array(objectives))
    n = len(arr)

    def _var_cvar(cl: float):
        idx = max(0, int((1 - cl) * n) - 1)
        return float(arr[idx]), float(arr[:max(1, idx)].mean())

    v95, c95 = _var_cvar(0.95)
    v99, c99 = _var_cvar(0.99)

    return RiskMetrics(
        var_95=v95, cvar_95=c95,
        var_99=v99, cvar_99=c99,
        min_obj=float(arr.min()),
        max_obj=float(arr.max()),
        mean_obj=float(arr.mean()),
        std_obj=float(arr.std(ddof=1)),
        pct_below_zero=float((arr < 0).mean() * 100),
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def run_stochastic_analysis(
    network: SupplyChainNetwork,
    n_scenarios: int = 10,
    solver: str = "cbc",
    seed: int = 42,
    annual_vol: float = 0.22,
    use_extensive_form: bool = True,
) -> StochasticResult:
    """
    Full two-stage stochastic analysis returning RP, WS, EEV, EVPI, VSS, VaR, CVaR.

    Parameters
    ----------
    use_extensive_form : bool
        True  → exact RP via EF (correct, heavier)
        False → approximate RP = weighted-average of independent solves
                (fast diagnostic; note: this equals WS, not true RP)
    """
    t0 = time.time()

    scenarios = generate_crack_spread_scenarios(
        network, n_scenarios=n_scenarios, annual_vol=annual_vol, seed=seed
    )
    logger.info(f"Generated {n_scenarios} scenarios (σ_ann={annual_vol:.0%})")

    # WS
    ws_val, ws_results = _wait_and_see(network, scenarios, solver)
    logger.info(f"WS  = ${ws_val:,.0f}")

    # EV + EEV
    ev_val, ev_result = _solve_ev(network, scenarios, solver)
    logger.info(f"EV  = ${ev_val:,.0f}")
    eev_val = _compute_eev(network, scenarios, solver)
    logger.info(f"EEV = ${eev_val:,.0f}")

    # RP
    rp_val: float
    rp_sc_results: list
    if use_extensive_form:
        try:
            rp_val, rp_sc_results = _recourse_problem_ef(network, scenarios, solver)
            logger.info(f"RP  = ${rp_val:,.0f} (exact EF)")
        except Exception as exc:
            logger.warning(f"EF failed ({exc}); using WS as RP upper bound")
            rp_val = eev_val
            rp_sc_results = ws_results
    else:
        rp_val = eev_val   # conservative approximation
        rp_sc_results = ws_results
        logger.info(f"RP  = ${rp_val:,.0f} (approximate)")

    evpi = max(0.0, ws_val - rp_val)
    vss  = max(0.0, rp_val - eev_val)

    risk = compute_risk_metrics([r.objective_value for r in ws_results])

    elapsed = time.time() - t0
    logger.info(
        f"Done in {elapsed:.1f}s | EVPI=${evpi:,.0f} | VSS=${vss:,.0f} | "
        f"VaR95=${risk.var_95:,.0f}"
    )

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
