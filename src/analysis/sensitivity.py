"""
Sensitivity analysis for the supply chain optimizer.

Two approaches:
1. Parametric perturbation — re-solve with ±δ on key parameters, compute objective gradient.
   Clean, solver-agnostic, correct for nonlinear sensitivities.
2. Shadow price extraction from LP relaxation — only valid at the LP optimum but gives
   marginal values per constraint. Requires an LP solve (relax binaries).

Both feed the tornado chart visualization.
"""

import copy
import logging
from dataclasses import dataclass
from typing import List

from src.model.optimizer import MultiPeriodOptimizer, MultiPeriodResult
from src.model.supply_chain import NodeType, SupplyChainNetwork

logger = logging.getLogger(__name__)


@dataclass
class SensitivityEntry:
    parameter: str
    description: str
    base_value: float
    delta_pct: float
    objective_up: float     # objective when parameter increases by delta_pct
    objective_down: float   # objective when parameter decreases by delta_pct
    base_objective: float

    @property
    def swing_up(self) -> float:
        return self.objective_up - self.base_objective

    @property
    def swing_down(self) -> float:
        return self.objective_down - self.base_objective

    @property
    def total_swing(self) -> float:
        return abs(self.swing_up) + abs(self.swing_down)


def run_sensitivity(
    base_network: SupplyChainNetwork,
    base_result: MultiPeriodResult,
    delta_pct: float = 0.10,
) -> List[SensitivityEntry]:
    """
    Perturb key parameters ±10% and measure objective change.
    Returns entries sorted by total swing (for tornado chart).
    """
    base_obj = base_result.objective_value

    def solve_with(net: SupplyChainNetwork) -> float:
        return MultiPeriodOptimizer(net).solve().objective_value

    entries = []

    # ── Refinery margins ─────────────────────────────────────────────────────
    for nid, n in base_network.nodes.items():
        if n.node_type != NodeType.REFINERY:
            continue

        def _margin_obj(margin_mult, _nid=nid):
            net = copy.deepcopy(base_network)
            net.nodes[_nid].refinery_margin *= margin_mult
            return solve_with(net)

        entries.append(SensitivityEntry(
            parameter=f"{n.name} Crack Spread",
            description=f"±{int(delta_pct*100)}% on {n.name} refinery margin",
            base_value=n.refinery_margin,
            delta_pct=delta_pct,
            objective_up=_margin_obj(1 + delta_pct),
            objective_down=_margin_obj(1 - delta_pct),
            base_objective=base_obj,
        ))

    # ── Well production capacities ────────────────────────────────────────────
    for nid, n in base_network.nodes.items():
        if n.node_type != NodeType.WELL:
            continue

        def _well_obj(cap_mult, _nid=nid):
            net = copy.deepcopy(base_network)
            net.nodes[_nid].max_capacity *= cap_mult
            return solve_with(net)

        entries.append(SensitivityEntry(
            parameter=f"{n.name} Capacity",
            description=f"±{int(delta_pct*100)}% on {n.name} max production",
            base_value=n.max_capacity,
            delta_pct=delta_pct,
            objective_up=_well_obj(1 + delta_pct),
            objective_down=_well_obj(1 - delta_pct),
            base_objective=base_obj,
        ))

    # ── Pipeline transport costs (aggregate shock) ────────────────────────────
    def _transport_obj(cost_mult):
        net = copy.deepcopy(base_network)
        for arc in net.arcs:
            arc.transport_cost *= cost_mult
        return solve_with(net)

    entries.append(SensitivityEntry(
        parameter="Pipeline Tariff (All)",
        description=f"±{int(delta_pct*100)}% on all arc transport costs",
        base_value=sum(a.transport_cost for a in base_network.arcs) / len(base_network.arcs),
        delta_pct=delta_pct,
        objective_up=_transport_obj(1 + delta_pct),
        objective_down=_transport_obj(1 - delta_pct),
        base_objective=base_obj,
    ))

    # ── Market demand ─────────────────────────────────────────────────────────
    def _demand_obj(demand_mult):
        net = copy.deepcopy(base_network)
        for n in net.nodes.values():
            if n.node_type == NodeType.DEMAND:
                n.demand *= demand_mult
        return solve_with(net)

    entries.append(SensitivityEntry(
        parameter="Market Demand (All)",
        description=f"±{int(delta_pct*100)}% on all demand nodes",
        base_value=sum(n.demand for n in base_network.nodes.values() if n.node_type == NodeType.DEMAND),
        delta_pct=delta_pct,
        objective_up=_demand_obj(1 + delta_pct),
        objective_down=_demand_obj(1 - delta_pct),
        base_objective=base_obj,
    ))

    return sorted(entries, key=lambda e: e.total_swing, reverse=True)


@dataclass
class BottleneckReport:
    arc_label: str
    origin: str
    dest: str
    avg_utilization: float
    max_utilization: float
    shadow_value_estimate: float  # $/bbl-day of additional capacity
    periods_at_capacity: int


def identify_bottlenecks(
    network: SupplyChainNetwork,
    result: MultiPeriodResult,
    capacity_expand_bbl: float = 5_000,
) -> List[BottleneckReport]:
    """
    For arcs that are near-saturated (>85% utilization), estimate the marginal value
    of adding capacity by solving with +5,000 bbl/day and computing objective delta.
    """
    T = network.planning_horizon
    reports = []

    for arc in network.arcs:
        i, j = arc.origin, arc.destination
        utils = [
            result.flows_by_period.get((i, j, t), 0) / arc.capacity
            for t in range(1, T + 1)
            if arc.capacity > 0
        ]
        if not utils:
            continue

        avg_util = sum(utils) / len(utils)
        max_util = max(utils)
        at_cap = sum(1 for u in utils if u > 0.90)

        if avg_util < 0.70:
            continue

        # Marginal value: re-solve with expanded capacity
        net_exp = copy.deepcopy(network)
        exp_arc = net_exp.arc_lookup(i, j)
        exp_arc.capacity += capacity_expand_bbl
        opt = MultiPeriodOptimizer(net_exp)
        exp_result = opt.solve()
        shadow_val = (exp_result.objective_value - result.objective_value) / capacity_expand_bbl

        reports.append(BottleneckReport(
            arc_label=f"{network.nodes[i].name} → {network.nodes[j].name}",
            origin=i, dest=j,
            avg_utilization=avg_util,
            max_utilization=max_util,
            shadow_value_estimate=shadow_val,
            periods_at_capacity=at_cap,
        ))

    return sorted(reports, key=lambda r: r.shadow_value_estimate, reverse=True)
