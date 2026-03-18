"""
Scenario analysis wrapper. Runs the optimizer under multiple operational scenarios
and aggregates results for comparative analysis.
"""

import copy
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List

from src.model.optimizer import OptimizationResult, SupplyChainOptimizer
from src.model.supply_chain import SupplyChainNetwork

logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    name: str
    description: str
    modifier: Callable[[SupplyChainNetwork], SupplyChainNetwork]


@dataclass
class ScenarioResult:
    scenario_name: str
    description: str
    result: OptimizationResult
    delta_objective: float = 0.0
    delta_revenue: float = 0.0
    delta_transport: float = 0.0


class ScenarioRunner:
    def __init__(self, base_network: SupplyChainNetwork, solver: str = "cbc"):
        self.base_network = base_network
        self.solver = solver
        self._base_result: OptimizationResult = None

    def run_base(self) -> OptimizationResult:
        opt = SupplyChainOptimizer(copy.deepcopy(self.base_network), solver=self.solver)
        self._base_result = opt.solve()
        logger.info(f"Base case solved: objective = ${self._base_result.objective_value:,.0f}")
        return self._base_result

    def run_scenario(self, scenario: Scenario) -> ScenarioResult:
        modified_net = scenario.modifier(copy.deepcopy(self.base_network))
        opt = SupplyChainOptimizer(modified_net, solver=self.solver)
        result = opt.solve()

        base = self._base_result
        sr = ScenarioResult(
            scenario_name=scenario.name,
            description=scenario.description,
            result=result,
            delta_objective=result.objective_value - base.objective_value,
            delta_revenue=result.revenue - base.revenue,
            delta_transport=result.transport_cost - base.transport_cost,
        )
        logger.info(
            f"Scenario '{scenario.name}': "
            f"Δobj = ${sr.delta_objective:+,.0f} | "
            f"Δrevenue = ${sr.delta_revenue:+,.0f}"
        )
        return sr

    def run_all(self, scenarios: List[Scenario]) -> List[ScenarioResult]:
        if self._base_result is None:
            self.run_base()
        return [self.run_scenario(s) for s in scenarios]


def build_standard_scenarios(base_net: SupplyChainNetwork) -> List[Scenario]:
    from data.generate_data import (
        apply_cost_shock,
        apply_demand_spike,
        apply_refinery_outage,
        apply_supply_disruption,
    )

    return [
        Scenario(
            name="Hurricane Disruption",
            description="W3 Delaware Basin reduced 60% (weather/infrastructure damage)",
            modifier=lambda net: apply_supply_disruption(net, "W3", 0.60)
        ),
        Scenario(
            name="Chicago Demand Spike",
            description="Chicago market demand increases 40% (winter heating surge)",
            modifier=lambda net: apply_demand_spike(net, "M1", 1.40)
        ),
        Scenario(
            name="Port Arthur Outage",
            description="R2 refinery offline for maintenance/emergency",
            modifier=lambda net: apply_refinery_outage(net, "R2")
        ),
        Scenario(
            name="Freight Cost +25%",
            description="Pipeline/vessel transport costs increase 25% across all arcs",
            modifier=lambda net: apply_cost_shock(net, 0.25)
        ),
        Scenario(
            name="Dual Disruption",
            description="W3 reduced 40% AND Chicago demand +30% simultaneously",
            modifier=lambda net: apply_demand_spike(
                apply_supply_disruption(net, "W3", 0.40), "M1", 1.30
            )
        ),
    ]
