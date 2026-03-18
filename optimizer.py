"""
MILP formulation of the oil supply chain optimization problem.

Objective: maximize net margin = revenue - transport costs - holding costs - unmet demand penalties
Subject to: flow balance, capacity limits, demand satisfaction, arc capacities.

Uses Pyomo with CBC (open-source) as default solver. Easily swappable for CPLEX/Gurobi.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

from .supply_chain import NodeType, SupplyChainNetwork

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    status: str
    objective_value: float
    flows: Dict[Tuple[str, str], float]
    inventories: Dict[str, float]
    unmet_demand: Dict[str, float]
    revenue: float
    transport_cost: float
    holding_cost: float
    penalty_cost: float
    solver_time: float


class SupplyChainOptimizer:
    """
    Single-period MILP. For multi-period (rolling horizon), wrap in scenario runner.
    Arc activation (fixed costs) is modeled via binary variables on arcs with fixed_cost > 0.
    """

    def __init__(self, network: SupplyChainNetwork, solver: str = "cbc"):
        self.network = network
        self.solver_name = solver
        self.model: Optional[pyo.ConcreteModel] = None

    def build(self) -> pyo.ConcreteModel:
        m = pyo.ConcreteModel("OilSupplyChain")
        net = self.network

        arc_keys = net.arc_index
        node_ids = list(net.nodes.keys())

        # --- Sets ---
        m.NODES = pyo.Set(initialize=node_ids)
        m.ARCS = pyo.Set(initialize=arc_keys, dimen=2)
        m.ARCS_FIXED = pyo.Set(
            initialize=[k for k in arc_keys if net.arc_lookup(*k).fixed_cost > 0],
            dimen=2
        )

        # --- Parameters ---
        m.transport_cost = pyo.Param(
            m.ARCS,
            initialize={net.arc_key(a): a.transport_cost for a in net.arcs}
        )
        m.arc_capacity = pyo.Param(
            m.ARCS,
            initialize={net.arc_key(a): a.capacity for a in net.arcs}
        )
        m.fixed_cost = pyo.Param(
            m.ARCS_FIXED,
            initialize={net.arc_key(a): a.fixed_cost for a in net.arcs if a.fixed_cost > 0}
        )
        m.node_capacity = pyo.Param(
            m.NODES,
            initialize={nid: n.max_capacity for nid, n in net.nodes.items()}
        )
        m.operating_cost = pyo.Param(
            m.NODES,
            initialize={nid: n.operating_cost for nid, n in net.nodes.items()}
        )
        m.demand = pyo.Param(
            m.NODES,
            initialize={nid: n.demand for nid, n in net.nodes.items()}
        )
        m.unmet_penalty = pyo.Param(
            m.NODES,
            initialize={nid: n.unmet_penalty for nid, n in net.nodes.items()}
        )
        m.holding_cost = pyo.Param(
            m.NODES,
            initialize={nid: n.holding_cost for nid, n in net.nodes.items()}
        )
        m.refinery_margin = pyo.Param(
            m.NODES,
            initialize={nid: n.refinery_margin for nid, n in net.nodes.items()}
        )
        m.crude_yield = pyo.Param(
            m.NODES,
            initialize={nid: n.crude_yield for nid, n in net.nodes.items()}
        )

        # --- Decision Variables ---
        m.flow = pyo.Var(m.ARCS, domain=pyo.NonNegativeReals)
        m.inventory = pyo.Var(m.NODES, domain=pyo.NonNegativeReals)
        m.unmet = pyo.Var(m.NODES, domain=pyo.NonNegativeReals)

        # Binary activation for fixed-cost arcs
        m.activate = pyo.Var(m.ARCS_FIXED, domain=pyo.Binary)

        # --- Objective ---
        def objective_rule(m):
            revenue = sum(
                m.refinery_margin[nid] * sum(
                    m.flow[i, nid] for i in net.nodes
                    if (i, nid) in arc_keys
                )
                for nid, n in net.nodes.items()
                if n.node_type == NodeType.REFINERY
            )
            trans_cost = sum(m.transport_cost[i, j] * m.flow[i, j] for (i, j) in arc_keys)
            op_cost = sum(
                m.operating_cost[nid] * sum(
                    m.flow[nid, j] for j in net.nodes
                    if (nid, j) in arc_keys
                )
                for nid in node_ids
            )
            hold_cost = sum(m.holding_cost[nid] * m.inventory[nid] for nid in node_ids)
            fix_cost = sum(m.fixed_cost[i, j] * m.activate[i, j] for (i, j) in m.ARCS_FIXED)
            penalty = sum(
                m.unmet_penalty[nid] * m.unmet[nid]
                for nid, n in net.nodes.items()
                if n.node_type == NodeType.DEMAND
            )
            return revenue - trans_cost - op_cost - hold_cost - fix_cost - penalty

        m.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

        # --- Constraints ---

        # Production / source capacity
        def well_capacity(m, nid):
            n = net.nodes[nid]
            if n.node_type != NodeType.WELL:
                return pyo.Constraint.Skip
            outflow = sum(m.flow[nid, j] for j in node_ids if (nid, j) in arc_keys)
            return outflow <= m.node_capacity[nid]

        m.well_cap = pyo.Constraint(m.NODES, rule=well_capacity)

        # Refinery throughput bounds
        def refinery_min(m, nid):
            n = net.nodes[nid]
            if n.node_type != NodeType.REFINERY:
                return pyo.Constraint.Skip
            inflow = sum(m.flow[i, nid] for i in node_ids if (i, nid) in arc_keys)
            return inflow >= n.min_throughput

        def refinery_max(m, nid):
            n = net.nodes[nid]
            if n.node_type != NodeType.REFINERY:
                return pyo.Constraint.Skip
            inflow = sum(m.flow[i, nid] for i in node_ids if (i, nid) in arc_keys)
            return inflow <= m.node_capacity[nid]

        m.ref_min = pyo.Constraint(m.NODES, rule=refinery_min)
        m.ref_max = pyo.Constraint(m.NODES, rule=refinery_max)

        # Flow balance at storage nodes: inflow + initial_inventory = outflow + inventory
        def storage_balance(m, nid):
            n = net.nodes[nid]
            if n.node_type != NodeType.STORAGE:
                return pyo.Constraint.Skip
            inflow = sum(m.flow[i, nid] for i in node_ids if (i, nid) in arc_keys)
            outflow = sum(m.flow[nid, j] for j in node_ids if (nid, j) in arc_keys)
            return inflow + n.initial_inventory == outflow + m.inventory[nid]

        m.storage_bal = pyo.Constraint(m.NODES, rule=storage_balance)

        # Storage capacity ceiling
        def storage_cap(m, nid):
            n = net.nodes[nid]
            if n.node_type != NodeType.STORAGE:
                return pyo.Constraint.Skip
            return m.inventory[nid] <= m.node_capacity[nid]

        m.storage_cap = pyo.Constraint(m.NODES, rule=storage_cap)

        # Demand satisfaction (with slack)
        def demand_satisfy(m, nid):
            n = net.nodes[nid]
            if n.node_type != NodeType.DEMAND:
                return pyo.Constraint.Skip
            inflow = sum(m.flow[i, nid] for i in node_ids if (i, nid) in arc_keys)
            return inflow + m.unmet[nid] >= m.demand[nid]

        m.demand_sat = pyo.Constraint(m.NODES, rule=demand_satisfy)

        # Arc capacity
        def arc_cap(m, i, j):
            return m.flow[i, j] <= m.arc_capacity[i, j]

        m.arc_cap_con = pyo.Constraint(m.ARCS, rule=arc_cap)

        # Fixed-cost arc activation coupling (big-M)
        def activation_link(m, i, j):
            return m.flow[i, j] <= m.arc_capacity[i, j] * m.activate[i, j]

        m.activation = pyo.Constraint(m.ARCS_FIXED, rule=activation_link)

        # Flow conservation at intermediate (distribution) nodes
        def distribution_balance(m, nid):
            n = net.nodes[nid]
            if n.node_type != NodeType.DISTRIBUTION:
                return pyo.Constraint.Skip
            inflow = sum(m.flow[i, nid] for i in node_ids if (i, nid) in arc_keys)
            outflow = sum(m.flow[nid, j] for j in node_ids if (nid, j) in arc_keys)
            return inflow == outflow

        m.dist_bal = pyo.Constraint(m.NODES, rule=distribution_balance)

        self.model = m
        return m

    def solve(self, tee: bool = False) -> OptimizationResult:
        if self.model is None:
            self.build()

        solver = pyo.SolverFactory(self.solver_name)
        if self.solver_name == "cbc":
            solver.options["seconds"] = 120
            solver.options["ratio"] = 0.001

        import time
        t0 = time.time()
        results = solver.solve(self.model, tee=tee)
        elapsed = time.time() - t0

        status = results.solver.termination_condition
        if status not in (TerminationCondition.optimal, TerminationCondition.feasible):
            logger.warning(f"Solver returned {status}")

        m = self.model
        net = self.network
        arc_keys = net.arc_index

        flows = {(i, j): pyo.value(m.flow[i, j]) for (i, j) in arc_keys}
        inventories = {nid: pyo.value(m.inventory[nid]) for nid in net.nodes}
        unmet = {
            nid: pyo.value(m.unmet[nid])
            for nid, n in net.nodes.items()
            if n.node_type == NodeType.DEMAND
        }

        revenue = sum(
            net.nodes[nid].refinery_margin * sum(
                flows.get((i, nid), 0)
                for i in net.nodes
                if (i, nid) in arc_keys
            )
            for nid, n in net.nodes.items()
            if n.node_type == NodeType.REFINERY
        )
        trans_cost = sum(
            net.arc_lookup(i, j).transport_cost * v
            for (i, j), v in flows.items()
        )
        hold_cost = sum(
            net.nodes[nid].holding_cost * v
            for nid, v in inventories.items()
        )
        penalty = sum(
            net.nodes[nid].unmet_penalty * v
            for nid, v in unmet.items()
        )

        return OptimizationResult(
            status=str(status),
            objective_value=pyo.value(m.obj),
            flows=flows,
            inventories=inventories,
            unmet_demand=unmet,
            revenue=revenue,
            transport_cost=trans_cost,
            holding_cost=hold_cost,
            penalty_cost=penalty,
            solver_time=elapsed
        )
