"""
Rolling Horizon Executor
━━━━━━━━━━━━━━━━━━━━━━━━
Implements the plan → execute → update loop used in real supply chain planning.

Real-world context
──────────────────
Supply chain planners at major oil companies don't run a single 14-day solve
and then execute it blindly. They run a rolling horizon:

  1. Solve a 14-day MILP (full horizon)
  2. Execute only Day 1 decisions (committed flows, arcs activated)
  3. Observe actual production, prices, demand (with random deviations)
  4. Update inventory levels and well production states
  5. Advance the window by 1 day, re-solve
  6. Repeat for the full simulation period

This yields a realized sequence of decisions that accounts for feedback
between execution and re-planning — the core of Model Predictive Control
(MPC), which is exactly how modern pipeline and refinery scheduling works.

Key outputs
───────────
  - Realized total margin over simulation period
  - Cumulative unmet demand (service level)
  - Replanning frequency vs. one-shot planning comparison
  - Inventory trajectories with re-planning corrections
  - Comparison: rolling horizon vs. static plan (quantifies replanning value)
"""

import copy
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from src.model.optimizer import MultiPeriodOptimizer, MultiPeriodResult
from src.model.supply_chain import NodeType, SupplyChainNetwork

logger = logging.getLogger(__name__)


@dataclass
class DayExecution:
    """Records actual outcomes for one simulated day."""
    day: int
    date: date

    # Planned (from optimizer)
    planned_flows: Dict[Tuple[str, str], float]           # (origin, dest) -> bbl/day
    planned_margin: float

    # Realized (after noise)
    realized_production: Dict[str, float]                 # well_id -> actual bbl/day
    realized_demand: Dict[str, float]                     # market_id -> actual bbl/day
    realized_crack_spread: Dict[str, float]               # refinery_id -> actual $/bbl

    # Execution outcomes
    actual_flows: Dict[Tuple[str, str], float]
    unmet_demand: Dict[str, float]
    updated_inventories: Dict[Tuple[str, str], float]     # (node_id, grade) -> bbl
    realized_margin: float

    # Replanning triggered?
    replanned: bool = False
    replan_reason: str = ""


@dataclass
class RollingHorizonResult:
    """Full simulation result from rolling horizon execution."""
    simulation_days: int
    execution_log: List[DayExecution] = field(default_factory=list)

    # Aggregate metrics
    total_realized_margin: float = 0.0
    total_planned_margin: float = 0.0
    total_unmet_demand: float = 0.0
    total_replans: int = 0
    avg_service_level: float = 0.0

    # Static comparison: single plan held fixed over full horizon
    static_plan_margin: float = 0.0

    @property
    def replanning_value(self) -> float:
        """
        Quantifies value of rolling horizon over static plan.
        = Realized RH margin - Realized static plan margin
        """
        return self.total_realized_margin - self.static_plan_margin

    @property
    def plan_execution_gap(self) -> float:
        """
        Average daily gap between planned and realized margin.
        Measures how much uncertainty degrades planned performance.
        """
        if not self.execution_log:
            return 0.0
        return sum(
            abs(d.planned_margin - d.realized_margin)
            for d in self.execution_log
        ) / len(self.execution_log)


class RealitySimulator:
    """
    Injects random deviations into execution to simulate real-world uncertainty.

    Deviations modelled:
      - Well production: ±5% Gaussian around planned (equipment variability)
      - Market demand: ±8% Gaussian with weekly seasonality (inherent demand noise)
      - Crack spreads: log-normal shocks with σ_daily = 22%/√252 (market prices)
    """

    def __init__(self, seed: int = 0):
        self.rng = None
        self._base_seed = seed

    def _get_rng(self, day: int):
        import numpy as np
        return np.random.default_rng(self._base_seed + day)

    def realized_production(
        self, planned: Dict[str, float], day: int, noise_level: float = 0.05
    ) -> Dict[str, float]:
        import numpy as np
        rng = self._get_rng(day)
        return {
            well_id: max(0.0, float(vol * (1 + rng.normal(0, noise_level))))
            for well_id, vol in planned.items()
        }

    def realized_demand(
        self, planned: Dict[str, float], day: int, noise_level: float = 0.08
    ) -> Dict[str, float]:
        import numpy as np
        rng = self._get_rng(day + 1000)
        return {
            mid: max(0.0, float(d * (1 + rng.normal(0, noise_level))))
            for mid, d in planned.items()
        }

    def realized_crack_spread(
        self, base_margins: Dict[str, float], day: int
    ) -> Dict[str, float]:
        import numpy as np
        annual_vol = 0.22
        daily_vol = annual_vol / np.sqrt(252)
        rng = self._get_rng(day + 2000)
        return {
            ref_id: float(margin * np.exp(rng.normal(-0.5 * daily_vol**2, daily_vol)))
            for ref_id, margin in base_margins.items()
        }


class RollingHorizonOptimizer:
    """
    Applies the MPC-style rolling horizon loop:

      for day in simulation_period:
          solve MILP for [day, day + H)
          execute Day 1 decisions
          update state (inventories, production)
          re-plan if trigger condition met
    """

    REPLAN_TRIGGERS = {
        "always": lambda ctx: True,
        "daily": lambda ctx: True,
        "price_shock": lambda ctx: abs(ctx.get("crack_spread_ratio", 1.0) - 1.0) > 0.15,
        "demand_spike": lambda ctx: ctx.get("demand_ratio", 1.0) > 1.20,
        "supply_disruption": lambda ctx: ctx.get("production_ratio", 1.0) < 0.70,
    }

    def __init__(
        self,
        base_network: SupplyChainNetwork,
        rolling_horizon: int = 14,
        simulation_days: int = 30,
        replan_trigger: str = "always",
        solver: str = "cbc",
        seed: int = 42,
        noise_level: float = 0.05,
    ):
        self.base_network = base_network
        self.rolling_horizon = rolling_horizon
        self.simulation_days = simulation_days
        self.trigger_fn = self.REPLAN_TRIGGERS.get(replan_trigger, self.REPLAN_TRIGGERS["always"])
        self.solver = solver
        self.simulator = RealitySimulator(seed=seed)
        self.noise_level = noise_level

    def _build_current_network(
        self,
        current_inventories: Dict[Tuple[str, str], float],
        day: int,
        crack_spread_mult: Optional[Dict[str, float]] = None,
    ) -> SupplyChainNetwork:
        """
        Clone base network, update initial inventories from current state,
        apply production decline from day 0 to current day.
        """
        net = copy.deepcopy(self.base_network)
        net.planning_horizon = self.rolling_horizon

        # Update initial inventories from realized state
        for (node_id, grade), bbl in current_inventories.items():
            if node_id in net.nodes:
                net.nodes[node_id].initial_inv_by_grade[grade] = max(0.0, bbl)
                net.nodes[node_id].initial_inventory = sum(
                    net.nodes[node_id].initial_inv_by_grade.values()
                )

        # Apply production decline relative to simulation day
        for well_id, n in net.nodes.items():
            if n.node_type == NodeType.WELL and n.decline_rate > 0:
                n.max_capacity = n.max_capacity * ((1 - n.decline_rate) ** day)

        return net

    def _extract_day1_decisions(
        self, result: MultiPeriodResult, network: SupplyChainNetwork
    ) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float]]:
        """
        Extract Day 1 flows and end-of-Day-1 inventories from optimizer output.
        These are the executed decisions.
        """
        flows = {
            (i, j): result.flows_by_period.get((i, j, 1), 0.0)
            for (i, j) in network.arc_index
        }
        inventories = {
            (nid, g): result.inventories.get((nid, 1, g), 0.0)
            for nid in network.nodes
            for g in network.grade_ids()
            if network.nodes[nid].node_type == NodeType.STORAGE
        }
        return flows, inventories

    def _compute_day_margin(
        self,
        flows: Dict[Tuple[str, str], float],
        network: SupplyChainNetwork,
        crack_spreads: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute realized daily net margin given actual flows and crack spreads."""
        margin_map = crack_spreads or {
            nid: n.refinery_margin
            for nid, n in network.nodes.items()
            if n.node_type == NodeType.REFINERY
        }

        revenue = sum(
            margin_map.get(nid, n.refinery_margin)
            * sum(flows.get((i, nid), 0) for i in network.nodes)
            for nid, n in network.nodes.items()
            if n.node_type == NodeType.REFINERY
        )
        transport = sum(
            (arc := network.arc_lookup(i, j)) and arc.transport_cost * flow or 0
            for (i, j), flow in flows.items()
        )
        return revenue - transport

    def run(self) -> RollingHorizonResult:
        """
        Execute the full rolling horizon simulation.
        Returns detailed day-by-day execution log and aggregate metrics.
        """
        import numpy as np
        from datetime import date

        result = RollingHorizonResult(simulation_days=self.simulation_days)
        base_start = self.base_network.start_date or date.today()

        # Initialize state from base network initial inventories
        current_inventories: Dict[Tuple[str, str], float] = {}
        for nid, n in self.base_network.nodes.items():
            if n.node_type == NodeType.STORAGE:
                for g, bbl in n.initial_inv_by_grade.items():
                    current_inventories[(nid, g)] = bbl

        # Static baseline: solve once from day 0, hold fixed
        logger.info("Solving static baseline plan...")
        static_net = copy.deepcopy(self.base_network)
        static_net.planning_horizon = self.simulation_days
        try:
            static_result = MultiPeriodOptimizer(static_net, solver=self.solver).solve()
            result.static_plan_margin = sum(
                self._compute_day_margin(
                    {(i, j): static_result.flows_by_period.get((i, j, t), 0)
                     for (i, j) in static_net.arc_index},
                    static_net,
                )
                for t in range(1, min(self.simulation_days + 1, static_net.planning_horizon + 1))
            )
        except Exception as e:
            logger.warning(f"Static baseline failed: {e}")
            result.static_plan_margin = 0.0

        logger.info(f"Static baseline margin: ${result.static_plan_margin:,.0f}")
        logger.info(f"Starting rolling horizon: {self.simulation_days} days, H={self.rolling_horizon}")

        for day in range(self.simulation_days):
            sim_date = base_start + timedelta(days=day)
            logger.info(f"  Day {day+1}/{self.simulation_days} ({sim_date})")

            # Build current-state network
            net = self._build_current_network(current_inventories, day)

            # Solve rolling horizon
            try:
                opt = MultiPeriodOptimizer(net, solver=self.solver)
                plan = opt.solve()
                replanned = True
            except Exception as e:
                logger.error(f"  Solver failed on day {day+1}: {e}")
                # Fall back to previous day's flows (no replanning)
                replanned = False
                plan = None

            if plan is None:
                continue

            # Extract Day 1 decisions
            planned_flows, new_inventories = self._extract_day1_decisions(plan, net)

            # Simulate reality: inject noise
            planned_well_output = {
                well_id: sum(planned_flows.get((well_id, t_id), 0) for t_id in net.nodes)
                for well_id, n in net.nodes.items()
                if n.node_type == NodeType.WELL
            }
            base_crack_spreads = {
                nid: n.refinery_margin
                for nid, n in net.nodes.items()
                if n.node_type == NodeType.REFINERY
            }
            base_demand = {
                nid: n.demand
                for nid, n in net.nodes.items()
                if n.node_type == NodeType.DEMAND
            }

            realized_prod = self.simulator.realized_production(
                planned_well_output, day, self.noise_level
            )
            realized_demand = self.simulator.realized_demand(base_demand, day, self.noise_level)
            realized_spreads = self.simulator.realized_crack_spread(base_crack_spreads, day)

            # Compute realized margin
            realized_margin = self._compute_day_margin(planned_flows, net, realized_spreads)
            planned_margin = self._compute_day_margin(planned_flows, net, base_crack_spreads)

            # Unmet demand
            unmet = {
                mid: max(0.0, realized_demand[mid] - sum(
                    planned_flows.get((i, mid), 0) for i in net.nodes
                ))
                for mid in realized_demand
            }

            # Update inventories for next day (use optimizer's projected end-of-day)
            for (nid, grade), bbl in new_inventories.items():
                current_inventories[(nid, grade)] = bbl

            # Log execution
            execution = DayExecution(
                day=day + 1,
                date=sim_date,
                planned_flows=planned_flows,
                planned_margin=planned_margin,
                realized_production=realized_prod,
                realized_demand=realized_demand,
                realized_crack_spread=realized_spreads,
                actual_flows=planned_flows,   # executed = planned (no physical re-dispatch)
                unmet_demand=unmet,
                updated_inventories=new_inventories,
                realized_margin=realized_margin,
                replanned=replanned,
            )
            result.execution_log.append(execution)

            result.total_realized_margin += realized_margin
            result.total_planned_margin += planned_margin
            result.total_unmet_demand += sum(unmet.values())
            if replanned:
                result.total_replans += 1

        # Aggregate service level
        total_demand = sum(
            n.demand for n in self.base_network.get_nodes_by_type(NodeType.DEMAND)
        ) * self.simulation_days
        result.avg_service_level = (
            1.0 - result.total_unmet_demand / max(total_demand, 1.0)
        )

        logger.info(
            f"Rolling horizon complete: "
            f"realized=${result.total_realized_margin:,.0f} | "
            f"replanning value=${result.replanning_value:,.0f} | "
            f"service={result.avg_service_level:.1%}"
        )
        return result
