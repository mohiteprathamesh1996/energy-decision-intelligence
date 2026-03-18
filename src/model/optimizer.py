"""
Multi-period MILP for oil supply chain optimization.

Formulation highlights:
  - Grade-indexed flows and inventories (T × G × A variables)
  - Crude diet enforcement via linearized bilinear blending constraints
  - Ship-or-pay contract deficiency tracking
  - Time-varying crack spreads for scenario/stochastic integration
  - Optional carbon budget constraint
  - Well production decline over planning horizon

Solver: CBC (open source). For production: Gurobi or CPLEX via same interface.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

from .supply_chain import NodeType, SupplyChainNetwork

logger = logging.getLogger(__name__)


@dataclass
class MultiPeriodResult:
    status: str
    objective_value: float

    # Flows: (origin, dest, period, grade) -> bbl/day
    flows_by_grade: Dict[Tuple[str, str, int, str], float]
    # Aggregated: (origin, dest, period) -> bbl/day
    flows_by_period: Dict[Tuple[str, str, int], float]

    # Inventory: (node_id, period, grade) -> bbl
    inventories: Dict[Tuple[str, int, str], float]

    # Unmet demand: (node_id, period) -> bbl/day
    unmet_demand: Dict[Tuple[str, int], float]

    # Ship-or-pay deficits: (origin, dest, period) -> bbl/day
    sop_deficits: Dict[Tuple[str, str, int], float]

    # Cost components (total over all periods)
    revenue: float
    transport_cost: float
    holding_cost: float
    penalty_cost: float
    sop_cost: float

    # Carbon: period -> tonnes CO2e/day
    carbon_by_period: Dict[int, float]

    solver_time: float
    planning_horizon: int


class MultiPeriodOptimizer:
    """
    Single class handles both base case and scenario runs.
    Pass crack_spread_multipliers to vary margins by (refinery, period).
    """

    def __init__(self, network: SupplyChainNetwork, solver: str = "cbc"):
        self.network = network
        self.solver_name = solver
        self.model: Optional[pyo.ConcreteModel] = None

    def build(
        self,
        crack_spread_multipliers: Optional[Dict[Tuple[str, int], float]] = None,
    ) -> pyo.ConcreteModel:

        m = pyo.ConcreteModel("OilSupplyChain")
        net = self.network

        T = net.planning_horizon
        arc_keys = net.arc_index
        node_ids = list(net.nodes.keys())
        grade_ids = net.grade_ids()
        periods = list(range(1, T + 1))
        mults = crack_spread_multipliers or {}

        # ── Sets ─────────────────────────────────────────────────────────────

        m.NODES = pyo.Set(initialize=node_ids)
        m.ARCS = pyo.Set(initialize=arc_keys, dimen=2)
        m.GRADES = pyo.Set(initialize=grade_ids)
        m.PERIODS = pyo.Set(initialize=periods)

        fixed_arcs = [k for k in arc_keys if net.arc_lookup(*k).fixed_cost > 0]
        sop_arcs = [(c.arc_origin, c.arc_dest) for c in net.contracts]

        m.ARCS_FIXED = pyo.Set(initialize=fixed_arcs, dimen=2)
        m.ARCS_SOP = pyo.Set(initialize=sop_arcs, dimen=2)

        # ── Parameters ───────────────────────────────────────────────────────

        m.tau = pyo.Param(m.ARCS, initialize={net.arc_key(a): a.transport_cost for a in net.arcs})
        m.cap = pyo.Param(m.ARCS, initialize={net.arc_key(a): a.capacity for a in net.arcs})
        m.fixed_cost = pyo.Param(
            m.ARCS_FIXED,
            initialize={net.arc_key(a): a.fixed_cost for a in net.arcs if a.fixed_cost > 0},
        )

        m.node_cap = pyo.Param(m.NODES, initialize={n: net.nodes[n].max_capacity for n in node_ids})
        m.min_tp = pyo.Param(m.NODES, initialize={n: net.nodes[n].min_throughput for n in node_ids})
        m.op_cost = pyo.Param(m.NODES, initialize={n: net.nodes[n].operating_cost for n in node_ids})
        m.hold_cost = pyo.Param(m.NODES, initialize={n: net.nodes[n].holding_cost for n in node_ids})
        m.demand = pyo.Param(m.NODES, initialize={n: net.nodes[n].demand for n in node_ids})
        m.penalty = pyo.Param(m.NODES, initialize={n: net.nodes[n].unmet_penalty for n in node_ids})

        # Grade physical properties
        m.API = pyo.Param(m.GRADES, initialize={g: net.grades[g].api_gravity for g in grade_ids})
        m.sulfur = pyo.Param(m.GRADES, initialize={g: net.grades[g].sulfur_content for g in grade_ids})
        m.carbon_int = pyo.Param(m.GRADES, initialize={g: net.grades[g].carbon_intensity for g in grade_ids})

        # Refinery crude diet bounds
        m.API_lo = pyo.Param(m.NODES, initialize={
            n: (net.nodes[n].crude_diet.api_min if net.nodes[n].crude_diet else 0.0)
            for n in node_ids
        })
        m.API_hi = pyo.Param(m.NODES, initialize={
            n: (net.nodes[n].crude_diet.api_max if net.nodes[n].crude_diet else 100.0)
            for n in node_ids
        })
        m.sulfur_max = pyo.Param(m.NODES, initialize={
            n: (net.nodes[n].crude_diet.sulfur_max if net.nodes[n].crude_diet else 10.0)
            for n in node_ids
        })

        # Time-varying refinery margins
        def margin_init(m, nid, t):
            n = net.nodes[nid]
            if n.node_type != NodeType.REFINERY:
                return 0.0
            return n.refinery_margin * mults.get((nid, t), 1.0)

        m.margin = pyo.Param(m.NODES, m.PERIODS, initialize=margin_init)

        # Initial inventory by grade
        m.init_inv = pyo.Param(m.NODES, m.GRADES, initialize={
            (nid, g): net.nodes[nid].initial_inv_by_grade.get(g, 0.0)
            for nid in node_ids
            for g in grade_ids
        })

        # Well capacity with decline
        m.well_cap = pyo.Param(m.NODES, m.PERIODS, initialize={
            (nid, t): (
                net.well_capacity(nid, t)
                if net.nodes[nid].node_type == NodeType.WELL else 0.0
            )
            for nid in node_ids
            for t in periods
        })

        # Ship-or-pay
        m.sop_min = pyo.Param(m.ARCS_SOP, initialize={
            (c.arc_origin, c.arc_dest): c.min_daily_volume for c in net.contracts
        })
        m.sop_def_charge = pyo.Param(m.ARCS_SOP, initialize={
            (c.arc_origin, c.arc_dest): c.deficiency_charge for c in net.contracts
        })

        # ── Decision Variables ────────────────────────────────────────────────

        m.x = pyo.Var(m.ARCS, m.PERIODS, m.GRADES, domain=pyo.NonNegativeReals)
        m.s = pyo.Var(m.NODES, m.PERIODS, m.GRADES, domain=pyo.NonNegativeReals)
        m.u = pyo.Var(m.NODES, m.PERIODS, domain=pyo.NonNegativeReals)
        m.deficit = pyo.Var(m.ARCS_SOP, m.PERIODS, domain=pyo.NonNegativeReals)
        m.y = pyo.Var(m.ARCS_FIXED, domain=pyo.Binary)

        # ── Objective ────────────────────────────────────────────────────────

        def total_inflow(nid, t):
            return sum(
                m.x[i, nid, t, g]
                for i in node_ids if (i, nid) in arc_keys
                for g in grade_ids
            )

        def total_outflow(nid, t):
            return sum(
                m.x[nid, j, t, g]
                for j in node_ids if (nid, j) in arc_keys
                for g in grade_ids
            )

        def obj_rule(m):
            revenue = sum(
                m.margin[nid, t] * total_inflow(nid, t)
                for nid, n in net.nodes.items()
                if n.node_type == NodeType.REFINERY
                for t in periods
            )
            trans = sum(
                m.tau[i, j] * m.x[i, j, t, g]
                for (i, j) in arc_keys
                for t in periods
                for g in grade_ids
            )
            opex = sum(
                m.op_cost[nid] * total_outflow(nid, t)
                for nid in node_ids
                for t in periods
            )
            holding = sum(
                m.hold_cost[nid] * m.s[nid, t, g]
                for nid in node_ids for t in periods for g in grade_ids
            )
            fixed = sum(m.fixed_cost[i, j] * m.y[i, j] for (i, j) in fixed_arcs)
            sop = sum(
                m.sop_def_charge[i, j] * m.deficit[i, j, t]
                for (i, j) in sop_arcs for t in periods
            )
            pen = sum(
                m.penalty[nid] * m.u[nid, t]
                for nid, n in net.nodes.items()
                if n.node_type == NodeType.DEMAND
                for t in periods
            )
            return revenue - trans - opex - holding - fixed - sop - pen

        m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

        # ── Constraints ──────────────────────────────────────────────────────

        # (C1) Wells produce only their primary grade
        def grade_lock(m, nid, t, g):
            n = net.nodes[nid]
            if n.node_type != NodeType.WELL or n.primary_grade == g:
                return pyo.Constraint.Skip
            out = sum(m.x[nid, j, t, g] for j in node_ids if (nid, j) in arc_keys)
            return out == 0

        m.c_grade_lock = pyo.Constraint(m.NODES, m.PERIODS, m.GRADES, rule=grade_lock)

        # (C2) Well production capacity (period-specific, with decline)
        def well_capacity(m, nid, t):
            if net.nodes[nid].node_type != NodeType.WELL:
                return pyo.Constraint.Skip
            return (
                sum(m.x[nid, j, t, g] for j in node_ids if (nid, j) in arc_keys for g in grade_ids)
                <= m.well_cap[nid, t]
            )

        m.c_well_cap = pyo.Constraint(m.NODES, m.PERIODS, rule=well_capacity)

        # (C3) Storage flow balance — grade-indexed, period-linked
        def storage_balance(m, nid, t, g):
            if net.nodes[nid].node_type != NodeType.STORAGE:
                return pyo.Constraint.Skip
            inflow = sum(m.x[i, nid, t, g] for i in node_ids if (i, nid) in arc_keys)
            outflow = sum(m.x[nid, j, t, g] for j in node_ids if (nid, j) in arc_keys)
            prev = m.init_inv[nid, g] if t == 1 else m.s[nid, t - 1, g]
            return prev + inflow == outflow + m.s[nid, t, g]

        m.c_storage_bal = pyo.Constraint(m.NODES, m.PERIODS, m.GRADES, rule=storage_balance)

        # (C4) Storage tank capacity ceiling (mixed grades share tank volume)
        def storage_cap(m, nid, t):
            if net.nodes[nid].node_type != NodeType.STORAGE:
                return pyo.Constraint.Skip
            return sum(m.s[nid, t, g] for g in grade_ids) <= m.node_cap[nid]

        m.c_storage_cap = pyo.Constraint(m.NODES, m.PERIODS, rule=storage_cap)

        # (C5/C6) Refinery throughput bounds
        def ref_min(m, nid, t):
            if net.nodes[nid].node_type != NodeType.REFINERY:
                return pyo.Constraint.Skip
            return total_inflow(nid, t) >= m.min_tp[nid]

        def ref_max(m, nid, t):
            if net.nodes[nid].node_type != NodeType.REFINERY:
                return pyo.Constraint.Skip
            return total_inflow(nid, t) <= m.node_cap[nid]

        m.c_ref_min = pyo.Constraint(m.NODES, m.PERIODS, rule=ref_min)
        m.c_ref_max = pyo.Constraint(m.NODES, m.PERIODS, rule=ref_max)

        # (C7/C8) Crude diet: API gravity bounds
        # Linearized bilinear: API_lo * Σx ≤ Σ(API_g * x) ≤ API_hi * Σx
        def ref_api_lo(m, nid, t):
            n = net.nodes[nid]
            if n.node_type != NodeType.REFINERY or n.crude_diet is None:
                return pyo.Constraint.Skip
            inflows = [(i, g) for i in node_ids if (i, nid) in arc_keys for g in grade_ids]
            weighted = sum(m.API[g] * m.x[i, nid, t, g] for i, g in inflows)
            total = sum(m.x[i, nid, t, g] for i, g in inflows)
            return weighted >= m.API_lo[nid] * total

        def ref_api_hi(m, nid, t):
            n = net.nodes[nid]
            if n.node_type != NodeType.REFINERY or n.crude_diet is None:
                return pyo.Constraint.Skip
            inflows = [(i, g) for i in node_ids if (i, nid) in arc_keys for g in grade_ids]
            weighted = sum(m.API[g] * m.x[i, nid, t, g] for i, g in inflows)
            total = sum(m.x[i, nid, t, g] for i, g in inflows)
            return weighted <= m.API_hi[nid] * total

        m.c_api_lo = pyo.Constraint(m.NODES, m.PERIODS, rule=ref_api_lo)
        m.c_api_hi = pyo.Constraint(m.NODES, m.PERIODS, rule=ref_api_hi)

        # (C9) Crude diet: sulfur tolerance
        def ref_sulfur(m, nid, t):
            n = net.nodes[nid]
            if n.node_type != NodeType.REFINERY or n.crude_diet is None:
                return pyo.Constraint.Skip
            inflows = [(i, g) for i in node_ids if (i, nid) in arc_keys for g in grade_ids]
            weighted_s = sum(m.sulfur[g] * m.x[i, nid, t, g] for i, g in inflows)
            total = sum(m.x[i, nid, t, g] for i, g in inflows)
            return weighted_s <= m.sulfur_max[nid] * total

        m.c_sulfur = pyo.Constraint(m.NODES, m.PERIODS, rule=ref_sulfur)

        # (C10) Demand satisfaction with unmet slack
        def demand_sat(m, nid, t):
            if net.nodes[nid].node_type != NodeType.DEMAND:
                return pyo.Constraint.Skip
            return total_inflow(nid, t) + m.u[nid, t] >= m.demand[nid]

        m.c_demand = pyo.Constraint(m.NODES, m.PERIODS, rule=demand_sat)

        # (C11) Arc capacity (all grades share pipe capacity)
        def arc_cap(m, i, j, t):
            return sum(m.x[i, j, t, g] for g in grade_ids) <= m.cap[i, j]

        m.c_arc_cap = pyo.Constraint(m.ARCS, m.PERIODS, rule=arc_cap)

        # (C12) Fixed-cost arc activation (big-M)
        def activation_link(m, i, j, t):
            return sum(m.x[i, j, t, g] for g in grade_ids) <= m.cap[i, j] * m.y[i, j]

        m.c_activation = pyo.Constraint(m.ARCS_FIXED, m.PERIODS, rule=activation_link)

        # (C13) Ship-or-pay commitment
        def sop_commitment(m, i, j, t):
            return sum(m.x[i, j, t, g] for g in grade_ids) + m.deficit[i, j, t] >= m.sop_min[i, j]

        m.c_sop = pyo.Constraint(m.ARCS_SOP, m.PERIODS, rule=sop_commitment)

        # (C14) Distribution node flow conservation
        def dist_balance(m, nid, t):
            if net.nodes[nid].node_type != NodeType.DISTRIBUTION:
                return pyo.Constraint.Skip
            return total_inflow(nid, t) == total_outflow(nid, t)

        m.c_dist = pyo.Constraint(m.NODES, m.PERIODS, rule=dist_balance)

        # (C15) Carbon budget (optional hard constraint)
        if net.carbon_budget_per_day is not None:
            def carbon_budget(m, t):
                return (
                    sum(
                        m.carbon_int[g] * m.x[i, j, t, g]
                        for (i, j) in arc_keys
                        for g in grade_ids
                    ) / 1000.0  # kg -> tonnes
                    <= net.carbon_budget_per_day
                )
            m.c_carbon = pyo.Constraint(m.PERIODS, rule=carbon_budget)

        # (C16) Sour-restricted pipelines: no WTS/HEAVY on sweet-only arcs
        sour_grades = [g for g in grade_ids if net.grades[g].sulfur_content > 0.5]
        sweet_arcs = [(a.origin, a.destination) for a in net.arcs if not a.accepts_sour]

        if sweet_arcs and sour_grades:
            m.ARCS_SWEET = pyo.Set(initialize=sweet_arcs, dimen=2)
            m.GRADES_SOUR = pyo.Set(initialize=sour_grades)

            def sweet_only(m, i, j, t, g):
                return m.x[i, j, t, g] == 0

            m.c_sweet_only = pyo.Constraint(
                m.ARCS_SWEET, m.PERIODS, m.GRADES_SOUR, rule=sweet_only
            )

        self.model = m
        return m

    def solve(self, tee: bool = False) -> MultiPeriodResult:
        if self.model is None:
            self.build()

        solver = pyo.SolverFactory(self.solver_name)
        if self.solver_name == "cbc":
            solver.options["seconds"] = 300
            solver.options["ratio"] = 0.001
        elif self.solver_name == "glpk":
            solver.options["tmlim"] = 300

        t0 = time.time()
        res = solver.solve(self.model, tee=tee)
        elapsed = time.time() - t0

        status = str(res.solver.termination_condition)
        if res.solver.termination_condition not in (
            TerminationCondition.optimal, TerminationCondition.feasible
        ):
            logger.warning(f"Solver returned: {status}")

        return self._extract(elapsed, status)

    def _extract(self, elapsed: float, status: str) -> MultiPeriodResult:
        m = self.model
        net = self.network
        arc_keys = net.arc_index
        grade_ids = net.grade_ids()
        node_ids = list(net.nodes.keys())
        T = net.planning_horizon
        periods = list(range(1, T + 1))

        def v(var): return max(0.0, pyo.value(var) or 0.0)

        flows_by_grade = {
            (i, j, t, g): v(m.x[i, j, t, g])
            for (i, j) in arc_keys for t in periods for g in grade_ids
        }
        flows_by_period = {
            (i, j, t): sum(flows_by_grade[(i, j, t, g)] for g in grade_ids)
            for (i, j) in arc_keys for t in periods
        }
        inventories = {
            (nid, t, g): v(m.s[nid, t, g])
            for nid in node_ids for t in periods for g in grade_ids
        }
        unmet = {
            (nid, t): v(m.u[nid, t])
            for nid, n in net.nodes.items()
            if n.node_type == NodeType.DEMAND
            for t in periods
        }
        sop_def = {
            (c.arc_origin, c.arc_dest, t): v(m.deficit[c.arc_origin, c.arc_dest, t])
            for c in net.contracts for t in periods
        }

        # Revenue uses time-varying margins
        revenue = sum(
            (net.nodes[nid].refinery_margin * flows_by_period.get((i, nid, t), 0))
            for nid, n in net.nodes.items() if n.node_type == NodeType.REFINERY
            for i in node_ids if (i, nid) in arc_keys
            for t in periods
        )
        transport = sum(
            net.arc_lookup(i, j).transport_cost * flows_by_period[(i, j, t)]
            for (i, j) in arc_keys for t in periods
        )
        holding = sum(
            net.nodes[nid].holding_cost * inventories[(nid, t, g)]
            for nid in node_ids for t in periods for g in grade_ids
        )
        penalty = sum(
            net.nodes[nid].unmet_penalty * unmet.get((nid, t), 0)
            for nid, n in net.nodes.items() if n.node_type == NodeType.DEMAND
            for t in periods
        )
        sop_cost = sum(
            net.contract_lookup(c.arc_origin, c.arc_dest).deficiency_charge * sop_def.get((c.arc_origin, c.arc_dest, t), 0)
            for c in net.contracts for t in periods
        )
        carbon_by_period = {
            t: sum(
                net.grades[g].carbon_intensity * flows_by_grade.get((i, j, t, g), 0)
                for (i, j) in arc_keys for g in grade_ids
            ) / 1000.0
            for t in periods
        }

        return MultiPeriodResult(
            status=status,
            objective_value=v(m.obj),
            flows_by_grade=flows_by_grade,
            flows_by_period=flows_by_period,
            inventories=inventories,
            unmet_demand=unmet,
            sop_deficits=sop_def,
            revenue=revenue,
            transport_cost=transport,
            holding_cost=holding,
            penalty_cost=penalty,
            sop_cost=sop_cost,
            carbon_by_period=carbon_by_period,
            solver_time=elapsed,
            planning_horizon=T,
        )
