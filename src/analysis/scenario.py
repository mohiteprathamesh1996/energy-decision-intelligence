"""
Scenario runner for deterministic what-if analysis.
Each scenario is a pure function that modifies a deep-copied network.
Results include delta analysis vs. the base case.
"""

import copy
import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

from src.model.optimizer import MultiPeriodOptimizer, MultiPeriodResult
from src.model.supply_chain import SupplyChainNetwork

logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    name: str
    description: str
    category: str    # "supply", "demand", "infrastructure", "cost", "policy"
    modifier: Callable[[SupplyChainNetwork], SupplyChainNetwork]


@dataclass
class ScenarioResult:
    scenario_name: str
    description: str
    category: str
    result: MultiPeriodResult
    base_obj: float

    @property
    def delta_objective(self) -> float:
        return self.result.objective_value - self.base_obj

    @property
    def total_unmet(self) -> float:
        return sum(self.result.unmet_demand.values())

    @property
    def avg_service_level(self) -> float:
        return 1.0 - self.total_unmet / max(
            sum(self.result.unmet_demand.values()) + 1, 1
        )

    @property
    def total_carbon(self) -> float:
        return sum(self.result.carbon_by_period.values())


class ScenarioRunner:
    def __init__(self, base_network: SupplyChainNetwork):
        self.base_network = base_network
        self._base_result: Optional[MultiPeriodResult] = None

    def run_base(self) -> MultiPeriodResult:
        opt = MultiPeriodOptimizer(copy.deepcopy(self.base_network))
        self._base_result = opt.solve()
        logger.info(f"Base case: ${self._base_result.objective_value:,.0f} | "
                    f"solver time {self._base_result.solver_time:.2f}s")
        return self._base_result

    def run_scenario(self, scenario: Scenario) -> ScenarioResult:
        modified = scenario.modifier(copy.deepcopy(self.base_network))
        opt = MultiPeriodOptimizer(modified)
        result = opt.solve()
        sr = ScenarioResult(
            scenario_name=scenario.name,
            description=scenario.description,
            category=scenario.category,
            result=result,
            base_obj=self._base_result.objective_value,
        )
        logger.info(f"Scenario '{scenario.name}': Δobj = ${sr.delta_objective:+,.0f}")
        return sr

    def run_all(self, scenarios: List[Scenario]) -> List[ScenarioResult]:
        if self._base_result is None:
            self.run_base()
        return [self.run_scenario(s) for s in scenarios]


# ── Modifier functions ────────────────────────────────────────────────────────

def supply_disruption(well_id: str, reduction: float) -> Callable:
    def _mod(net):
        net.nodes[well_id].max_capacity *= (1 - reduction)
        return net
    return _mod


def demand_spike(market_id: str, multiplier: float) -> Callable:
    def _mod(net):
        net.nodes[market_id].demand *= multiplier
        return net
    return _mod


def refinery_outage(ref_id: str) -> Callable:
    def _mod(net):
        net.nodes[ref_id].max_capacity = 0
        net.nodes[ref_id].min_throughput = 0
        return net
    return _mod


def transport_cost_shock(pct: float) -> Callable:
    def _mod(net):
        for arc in net.arcs:
            arc.transport_cost *= (1 + pct)
        return net
    return _mod


def carbon_budget_constraint(budget_tco2_per_day: float) -> Callable:
    def _mod(net):
        net.carbon_budget_per_day = budget_tco2_per_day
        return net
    return _mod


def crack_spread_shock(ref_id: str, delta_pct: float) -> Callable:
    def _mod(net):
        net.nodes[ref_id].refinery_margin *= (1 + delta_pct)
        return net
    return _mod


def pipeline_capacity_expansion(origin: str, dest: str, new_cap: float) -> Callable:
    def _mod(net):
        arc = net.arc_lookup(origin, dest)
        if arc:
            arc.capacity = new_cap
        return net
    return _mod


def build_standard_scenarios(base_net: SupplyChainNetwork) -> List[Scenario]:
    return [
        Scenario(
            name="Hurricane – Delaware Basin",
            description="W3 Delaware Basin reduced 60% (weather/infrastructure damage)",
            category="supply",
            modifier=supply_disruption("W3", 0.60),
        ),
        Scenario(
            name="Chicago Demand Surge",
            description="Chicago market +40% (winter heating season + cold snap)",
            category="demand",
            modifier=demand_spike("M1", 1.40),
        ),
        Scenario(
            name="Port Arthur Refinery Outage",
            description="R2 Port Arthur offline for 14-day turnaround",
            category="infrastructure",
            modifier=refinery_outage("R2"),
        ),
        Scenario(
            name="Freight Cost +25%",
            description="Pipeline tariff increase (FERC rate case + diesel surcharge)",
            category="cost",
            modifier=transport_cost_shock(0.25),
        ),
        Scenario(
            name="Carbon Budget – 850 t/day",
            description="Regulatory carbon cap imposed at 850 tCO2e/day",
            category="policy",
            modifier=carbon_budget_constraint(850.0),
        ),
        Scenario(
            name="Wolfcamp Decline Acceleration",
            description="W4 Wolfcamp shale decline rate doubles (over-fracking)",
            category="supply",
            modifier=lambda net: [
                setattr(net.nodes["W4"], "decline_rate", 0.008), net
            ][-1],
        ),
        Scenario(
            name="Houston Crack Spread +20%",
            description="Gasoline/distillate spread widens at R1 (refinery bottleneck downstream)",
            category="cost",
            modifier=crack_spread_shock("R1", 0.20),
        ),
        Scenario(
            name="T1–R1 Pipeline Expansion",
            description="Midland Hub → Houston capacity expanded from 80K to 120K bbl/day",
            category="infrastructure",
            modifier=pipeline_capacity_expansion("T1", "R1", 120_000),
        ),
        Scenario(
            name="Dual Disruption",
            description="W3 −50% AND Chicago +35% simultaneously (stress test)",
            category="supply",
            modifier=lambda net: demand_spike("M1", 1.35)(supply_disruption("W3", 0.50)(net)),
        ),
    ]
