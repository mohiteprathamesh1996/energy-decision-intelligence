"""
Multi-period MILP for oil supply chain optimization — PuLP implementation.

Key advantage over Pyomo: `pip install pulp` bundles CBC automatically.
No system-level solver installation required.

Formulation: 16 constraint families covering
  - Well production with decline
  - Grade-indexed flows (WTI / WTS / Heavy)
  - Linearised crude diet blending (API gravity + sulfur)
  - Storage inventory balance and capacity
  - Ship-or-pay contract deficiency tracking
  - Sweet-only pipeline restrictions
  - Optional carbon budget
  - Arc activation binaries (fixed-cost routes)

Solver: PuLP bundled CBC  (pulp.PULP_CBC_CMD)
Upgrade path: swap to Gurobi via pulp.GUROBI_CMD() — same API.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pulp

from .supply_chain import NodeType, SupplyChainNetwork

logger = logging.getLogger(__name__)


@dataclass
class MultiPeriodResult:
    status: str
    objective_value: float

    # (origin, dest, period, grade) -> bbl/day
    flows_by_grade: Dict[Tuple[str, str, int, str], float]
    # (origin, dest, period) -> bbl/day
    flows_by_period: Dict[Tuple[str, str, int], float]
    # (node_id, period, grade) -> bbl
    inventories: Dict[Tuple[str, int, str], float]
    # (node_id, period) -> bbl/day
    unmet_demand: Dict[Tuple[str, int], float]
    # (origin, dest, period) -> bbl/day
    sop_deficits: Dict[Tuple[str, str, int], float]

    revenue: float
    transport_cost: float
    holding_cost: float
    penalty_cost: float
    sop_cost: float
    opex_cost: float
    fixed_cost: float

    # period -> tonnes CO2e/day
    carbon_by_period: Dict[int, float]

    solver_time: float
    planning_horizon: int


def _val(v) -> float:
    """Safe variable value extraction."""
    try:
        r = pulp.value(v)
        return max(0.0, float(r)) if r is not None else 0.0
    except Exception:
        return 0.0


class MultiPeriodOptimizer:
    """
    Build and solve the multi-period MILP using PuLP + bundled CBC.

    Parameters
    ----------
    network : SupplyChainNetwork
    time_limit : int
        Solver time limit in seconds (default 300).
    gap : float
        Relative MIP gap tolerance (default 0.001 = 0.1%).
    verbose : bool
        Print CBC solver log to stdout.
    """

    def __init__(
        self,
        network: SupplyChainNetwork,
        time_limit: int = 300,
        gap: float = 0.001,
        verbose: bool = False,
    ):
        self.network = network
        self.time_limit = time_limit
        self.gap = gap
        self.verbose = verbose
        self.prob: Optional[pulp.LpProblem] = None
        self._x: dict = {}
        self._s: dict = {}
        self._u: dict = {}
        self._deficit: dict = {}
        self._y: dict = {}

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(
        self,
        crack_spread_multipliers: Optional[Dict[Tuple[str, int], float]] = None,
    ) -> pulp.LpProblem:

        net = self.network
        T = net.planning_horizon
        arc_keys = net.arc_index
        node_ids = list(net.nodes.keys())
        grade_ids = net.grade_ids()
        periods = list(range(1, T + 1))
        mults = crack_spread_multipliers or {}

        # Node type groups
        well_ids     = [n for n in node_ids if net.nodes[n].node_type == NodeType.WELL]
        storage_ids  = [n for n in node_ids if net.nodes[n].node_type == NodeType.STORAGE]
        refinery_ids = [n for n in node_ids if net.nodes[n].node_type == NodeType.REFINERY]
        dist_ids     = [n for n in node_ids if net.nodes[n].node_type == NodeType.DISTRIBUTION]
        demand_ids   = [n for n in node_ids if net.nodes[n].node_type == NodeType.DEMAND]

        fixed_arcs = [k for k in arc_keys if net.arc_lookup(*k).fixed_cost > 0]
        sop_arcs   = [(c.arc_origin, c.arc_dest) for c in net.contracts]
        sour_grades  = [g for g in grade_ids if net.grades[g].sulfur_content > 0.5]
        sweet_arcs   = [(a.origin, a.destination) for a in net.arcs if not a.accepts_sour]

        # ── Variables ─────────────────────────────────────────────────────────

        prob = pulp.LpProblem("OilSupplyChain", pulp.LpMaximize)

        x_idx = [(i, j, t, g) for (i, j) in arc_keys for t in periods for g in grade_ids]
        x = pulp.LpVariable.dicts("x", x_idx, lowBound=0)

        s_idx = [(n, t, g) for n in storage_ids for t in periods for g in grade_ids]
        s = pulp.LpVariable.dicts("s", s_idx, lowBound=0)

        u_idx = [(n, t) for n in demand_ids for t in periods]
        u = pulp.LpVariable.dicts("u", u_idx, lowBound=0)

        d_idx = [(i, j, t) for (i, j) in sop_arcs for t in periods]
        deficit = pulp.LpVariable.dicts("deficit", d_idx, lowBound=0)

        y = pulp.LpVariable.dicts("y", fixed_arcs, cat="Binary") if fixed_arcs else {}

        self._x = x; self._s = s; self._u = u
        self._deficit = deficit; self._y = y

        # ── Helpers ───────────────────────────────────────────────────────────

        def inflow(nid, t):
            return pulp.lpSum(
                x[(i, nid, t, g)]
                for i in node_ids if (i, nid) in arc_keys
                for g in grade_ids
            )

        def outflow(nid, t):
            return pulp.lpSum(
                x[(nid, j, t, g)]
                for j in node_ids if (nid, j) in arc_keys
                for g in grade_ids
            )

        def inflow_g(nid, t, g):
            return pulp.lpSum(x[(i, nid, t, g)] for i in node_ids if (i, nid) in arc_keys)

        def outflow_g(nid, t, g):
            return pulp.lpSum(x[(nid, j, t, g)] for j in node_ids if (nid, j) in arc_keys)

        # ── Objective ─────────────────────────────────────────────────────────

        revenue = pulp.lpSum(
            net.nodes[r].refinery_margin * mults.get((r, t), 1.0) * inflow(r, t)
            for r in refinery_ids for t in periods
        )
        transport = pulp.lpSum(
            net.arc_lookup(i, j).transport_cost * x[(i, j, t, g)]
            for (i, j) in arc_keys for t in periods for g in grade_ids
        )
        opex = pulp.lpSum(
            net.nodes[n].operating_cost * outflow(n, t)
            for n in node_ids for t in periods
            if net.nodes[n].operating_cost > 0
        )
        holding = pulp.lpSum(
            net.nodes[n].holding_cost * s[(n, t, g)]
            for n in storage_ids for t in periods for g in grade_ids
        )
        fixed_cost = (
            pulp.lpSum(net.arc_lookup(i, j).fixed_cost * y[(i, j)] for (i, j) in fixed_arcs)
            if fixed_arcs else 0
        )
        sop_cost = pulp.lpSum(
            net.contract_lookup(i, j).deficiency_charge * deficit[(i, j, t)]
            for (i, j) in sop_arcs for t in periods
        )
        penalty = pulp.lpSum(
            net.nodes[n].unmet_penalty * u[(n, t)]
            for n in demand_ids for t in periods
        )

        prob += revenue - transport - opex - holding - fixed_cost - sop_cost - penalty

        # ── Constraints ───────────────────────────────────────────────────────

        # C1 — Well grade lock
        for nid in well_ids:
            pg = net.nodes[nid].primary_grade
            for t in periods:
                for g in grade_ids:
                    if g == pg:
                        continue
                    prob += outflow_g(nid, t, g) == 0, f"grade_lock_{nid}_{t}_{g}"

        # C2 — Well production capacity with decline
        for nid in well_ids:
            for t in periods:
                cap = net.well_capacity(nid, t)
                prob += outflow(nid, t) <= cap, f"well_cap_{nid}_{t}"

        # C3 — Storage flow balance (grade-indexed, period-linked)
        for nid in storage_ids:
            for t in periods:
                for g in grade_ids:
                    prev = (
                        net.nodes[nid].initial_inv_by_grade.get(g, 0.0)
                        if t == 1 else s[(nid, t - 1, g)]
                    )
                    prob += (
                        prev + inflow_g(nid, t, g) == outflow_g(nid, t, g) + s[(nid, t, g)],
                        f"stor_bal_{nid}_{t}_{g}",
                    )

        # C4 — Storage capacity (all grades share tank volume)
        for nid in storage_ids:
            cap = net.nodes[nid].max_capacity
            for t in periods:
                prob += (
                    pulp.lpSum(s[(nid, t, g)] for g in grade_ids) <= cap,
                    f"stor_cap_{nid}_{t}",
                )

        # C5/C6 — Refinery throughput bounds
        for nid in refinery_ids:
            lo = net.nodes[nid].min_throughput
            hi = net.nodes[nid].max_capacity
            for t in periods:
                prob += inflow(nid, t) >= lo, f"ref_lo_{nid}_{t}"
                prob += inflow(nid, t) <= hi, f"ref_hi_{nid}_{t}"

        # C7/C8 — Crude diet: API gravity (linearised bilinear)
        for nid in refinery_ids:
            d = net.nodes[nid].crude_diet
            if d is None:
                continue
            for t in periods:
                pairs = [(i, g) for i in node_ids if (i, nid) in arc_keys for g in grade_ids]
                if not pairs:
                    continue
                weighted = pulp.lpSum(net.grades[g].api_gravity * x[(i, nid, t, g)] for i, g in pairs)
                total    = pulp.lpSum(x[(i, nid, t, g)] for i, g in pairs)
                prob += weighted >= d.api_min * total, f"api_lo_{nid}_{t}"
                prob += weighted <= d.api_max * total, f"api_hi_{nid}_{t}"

        # C9 — Crude diet: sulfur tolerance
        for nid in refinery_ids:
            d = net.nodes[nid].crude_diet
            if d is None:
                continue
            for t in periods:
                pairs = [(i, g) for i in node_ids if (i, nid) in arc_keys for g in grade_ids]
                if not pairs:
                    continue
                weighted_s = pulp.lpSum(net.grades[g].sulfur_content * x[(i, nid, t, g)] for i, g in pairs)
                total      = pulp.lpSum(x[(i, nid, t, g)] for i, g in pairs)
                prob += weighted_s <= d.sulfur_max * total, f"sulfur_{nid}_{t}"

        # C10 — Demand satisfaction (soft, with penalty slack)
        for nid in demand_ids:
            d = net.nodes[nid].demand
            for t in periods:
                prob += inflow(nid, t) + u[(nid, t)] >= d, f"demand_{nid}_{t}"

        # C11 — Arc capacity (shared across all grades)
        for (i, j) in arc_keys:
            cap = net.arc_lookup(i, j).capacity
            for t in periods:
                prob += (
                    pulp.lpSum(x[(i, j, t, g)] for g in grade_ids) <= cap,
                    f"arc_cap_{i}_{j}_{t}",
                )

        # C12 — Fixed-cost arc activation (big-M)
        for (i, j) in fixed_arcs:
            cap = net.arc_lookup(i, j).capacity
            for t in periods:
                prob += (
                    pulp.lpSum(x[(i, j, t, g)] for g in grade_ids) <= cap * y[(i, j)],
                    f"activation_{i}_{j}_{t}",
                )

        # C13 — Ship-or-pay commitment
        for (i, j) in sop_arcs:
            vmin = net.contract_lookup(i, j).min_daily_volume
            for t in periods:
                prob += (
                    pulp.lpSum(x[(i, j, t, g)] for g in grade_ids) + deficit[(i, j, t)] >= vmin,
                    f"sop_{i}_{j}_{t}",
                )

        # C14 — Distribution node flow conservation
        for nid in dist_ids:
            for t in periods:
                prob += inflow(nid, t) == outflow(nid, t), f"dist_bal_{nid}_{t}"

        # C15 — Carbon budget (optional)
        if net.carbon_budget_per_day is not None:
            budget = net.carbon_budget_per_day
            for t in periods:
                prob += (
                    pulp.lpSum(
                        net.grades[g].carbon_intensity * x[(i, j, t, g)] / 1000.0
                        for (i, j) in arc_keys for g in grade_ids
                    ) <= budget,
                    f"carbon_{t}",
                )

        # C16 — Sweet-only pipeline restrictions
        for (i, j) in sweet_arcs:
            if (i, j) not in arc_keys:
                continue
            for t in periods:
                for g in sour_grades:
                    prob += x[(i, j, t, g)] == 0, f"sweet_{i}_{j}_{t}_{g}"

        self.prob = prob
        return prob

    # ── Solve ─────────────────────────────────────────────────────────────────

    def solve(self) -> MultiPeriodResult:
        if self.prob is None:
            self.build()

        solver = pulp.PULP_CBC_CMD(
            msg=1 if self.verbose else 0,
            timeLimit=self.time_limit,
            gapRel=self.gap,
        )

        t0 = time.time()
        self.prob.solve(solver)
        elapsed = time.time() - t0

        status_map = {
            1: "optimal", 0: "not_solved",
            -1: "infeasible", -2: "unbounded", -3: "undefined",
        }
        status = status_map.get(self.prob.status, "unknown")

        if self.prob.status != 1:
            logger.warning(f"Solver status: {status}")

        return self._extract(elapsed, status)

    # ── Extract results ───────────────────────────────────────────────────────

    def _extract(self, elapsed: float, status: str) -> MultiPeriodResult:
        net = self.network
        arc_keys = net.arc_index
        grade_ids = net.grade_ids()
        node_ids = list(net.nodes.keys())
        T = net.planning_horizon
        periods = list(range(1, T + 1))

        x = self._x; s = self._s; u = self._u
        deficit = self._deficit

        flows_by_grade = {
            (i, j, t, g): _val(x[(i, j, t, g)])
            for (i, j) in arc_keys for t in periods for g in grade_ids
        }
        flows_by_period = {
            (i, j, t): sum(flows_by_grade[(i, j, t, g)] for g in grade_ids)
            for (i, j) in arc_keys for t in periods
        }
        inventories = {
            (nid, t, g): _val(s[(nid, t, g)])
            for nid in node_ids
            if net.nodes[nid].node_type == NodeType.STORAGE
            for t in periods for g in grade_ids
        }
        unmet = {
            (nid, t): _val(u[(nid, t)])
            for nid in node_ids
            if net.nodes[nid].node_type == NodeType.DEMAND
            for t in periods
        }
        sop_deficits = {
            (c.arc_origin, c.arc_dest, t): _val(deficit[(c.arc_origin, c.arc_dest, t)])
            for c in net.contracts for t in periods
        }

        # Cost accounting
        revenue = sum(
            net.nodes[nid].refinery_margin
            * sum(flows_by_period.get((i, nid, t), 0) for i in node_ids if (i, nid) in arc_keys)
            for nid, n in net.nodes.items() if n.node_type == NodeType.REFINERY
            for t in periods
        )
        transport = sum(
            net.arc_lookup(i, j).transport_cost * flows_by_period[(i, j, t)]
            for (i, j) in arc_keys for t in periods
        )
        holding = sum(
            net.nodes[nid].holding_cost * inventories.get((nid, t, g), 0)
            for nid, n in net.nodes.items() if n.node_type == NodeType.STORAGE
            for t in periods for g in grade_ids
        )
        pen = sum(
            net.nodes[nid].unmet_penalty * unmet.get((nid, t), 0)
            for nid, n in net.nodes.items() if n.node_type == NodeType.DEMAND
            for t in periods
        )
        sop_cost = sum(
            net.contract_lookup(c.arc_origin, c.arc_dest).deficiency_charge
            * sop_deficits.get((c.arc_origin, c.arc_dest, t), 0)
            for c in net.contracts for t in periods
        )
        opex = sum(
            net.nodes[nid].operating_cost
            * sum(flows_by_period.get((nid, j, t), 0) for j in node_ids if (nid, j) in arc_keys)
            for nid, n in net.nodes.items()
            if n.operating_cost > 0
            for t in periods
        )
        fixed = sum(
            net.arc_lookup(i, j).fixed_cost * round(_val(self._y[(i, j)]))
            for (i, j) in self._y
        )
        carbon_by_period = {
            t: sum(
                net.grades[g].carbon_intensity * flows_by_grade.get((i, j, t, g), 0)
                for (i, j) in arc_keys for g in grade_ids
            ) / 1000.0
            for t in periods
        }

        raw_obj = pulp.value(self.prob.objective)
        obj_val = float(raw_obj) if raw_obj is not None else 0.0

        return MultiPeriodResult(
            status=status,
            objective_value=obj_val,
            flows_by_grade=flows_by_grade,
            flows_by_period=flows_by_period,
            inventories=inventories,
            unmet_demand=unmet,
            sop_deficits=sop_deficits,
            revenue=revenue,
            transport_cost=transport,
            holding_cost=holding,
            penalty_cost=pen,
            sop_cost=sop_cost,
            opex_cost=opex,
            fixed_cost=fixed,
            carbon_by_period=carbon_by_period,
            solver_time=elapsed,
            planning_horizon=T,
        )