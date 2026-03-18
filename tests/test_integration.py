"""
Integration tests — require CBC solver to be installed.
Run with: pytest tests/test_integration.py -v

These tests verify that the optimizer produces economically sensible solutions:
  - Objective is positive (profitable operation)
  - Service levels above 90% in base case
  - Ship-or-pay constraints are tracked correctly
  - Scenario results satisfy inequality chains
  - Stochastic metrics: WS >= RP >= EEV (Jensen's inequality)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import pytest
import pyomo.environ as pyo

# Skip all tests if CBC is not available
cbc_available = pyo.SolverFactory("cbc").available()
pytestmark = pytest.mark.skipif(not cbc_available, reason="CBC solver not installed")


from data.generate_data import build_base_network
from src.model.optimizer import MultiPeriodOptimizer
from src.model.supply_chain import NodeType


@pytest.fixture(scope="module")
def base_result():
    net = build_base_network(horizon=7)
    opt = MultiPeriodOptimizer(net, solver="cbc")
    return net, opt.solve()


class TestBaseSolution:
    def test_solver_succeeds(self, base_result):
        _, r = base_result
        assert r.status in ("optimal", "feasible")

    def test_positive_objective(self, base_result):
        _, r = base_result
        assert r.objective_value > 0

    def test_positive_revenue(self, base_result):
        _, r = base_result
        assert r.revenue > 0

    def test_service_level_above_90_pct(self, base_result):
        net, r = base_result
        demand_nodes = net.get_nodes_by_type(NodeType.DEMAND)
        T = r.planning_horizon
        total_demand = sum(n.demand for n in demand_nodes) * T
        total_unmet = sum(r.unmet_demand.values())
        svc = 1.0 - total_unmet / max(total_demand, 1.0)
        assert svc >= 0.90

    def test_flows_respect_arc_capacity(self, base_result):
        net, r = base_result
        T = r.planning_horizon
        for (i, j) in net.arc_index:
            arc = net.arc_lookup(i, j)
            for t in range(1, T + 1):
                flow = r.flows_by_period.get((i, j, t), 0)
                assert flow <= arc.capacity * 1.001, \
                    f"Arc {i}->{j} at t={t}: flow={flow:.0f} > cap={arc.capacity:.0f}"

    def test_inventory_nonnegative(self, base_result):
        net, r = base_result
        for (nid, t, g), inv in r.inventories.items():
            assert inv >= -1.0, f"Negative inventory at {nid} t={t} g={g}: {inv:.1f}"

    def test_all_flows_nonnegative(self, base_result):
        _, r = base_result
        for val in r.flows_by_period.values():
            assert val >= -1.0

    def test_carbon_computed_positive(self, base_result):
        _, r = base_result
        total = sum(r.carbon_by_period.values())
        assert total > 0.0


class TestGradeRouting:
    def test_wts_only_routes_to_coking_refinery(self, base_result):
        """
        WTS (sour grade, sulfur=1.52%) should not flow to R1 Houston (sulfur_max=0.50%).
        Verify that T_x -> R1 flows for WTS are essentially zero.
        """
        net, r = base_result
        T = r.planning_horizon
        for t in range(1, T + 1):
            for storage in net.get_nodes_by_type(NodeType.STORAGE):
                # R1 sweet refinery should have near-zero WTS inflow
                flow_wts_to_r1 = r.flows_by_grade.get((storage.id, "R1", t, "WTS"), 0)
                assert flow_wts_to_r1 < 500, \
                    f"WTS flowing to R1 (sweet) at t={t}: {flow_wts_to_r1:.0f} bbl/d"

    def test_heavy_grade_has_carbon_signature(self, base_result):
        net, r = base_result
        # HEAVY grade has highest carbon_intensity — verify it contributes to carbon total
        heavy_intensity = net.grades["HEAVY"].carbon_intensity
        assert heavy_intensity > net.grades["WTI"].carbon_intensity


class TestShipOrPayTracking:
    def test_sop_deficits_are_nonnegative(self, base_result):
        _, r = base_result
        for val in r.sop_deficits.values():
            assert val >= -1.0

    def test_sop_cost_is_nonnegative(self, base_result):
        _, r = base_result
        assert r.sop_cost >= 0.0


class TestScenarioSolvability:
    @pytest.mark.parametrize("well_id,pct", [("W3", 0.60), ("W1", 0.30)])
    def test_disruption_scenario_solves(self, well_id, pct):
        import copy
        from src.analysis.scenario import supply_disruption
        net = build_base_network(horizon=7)
        modified = supply_disruption(well_id, pct)(net)
        opt = MultiPeriodOptimizer(modified, solver="cbc")
        r = opt.solve()
        assert r.status in ("optimal", "feasible")
        assert r.objective_value > 0 or sum(r.unmet_demand.values()) > 0

    def test_carbon_cap_constraint_binding(self):
        """With very tight carbon budget, emissions should be at or below budget."""
        net = build_base_network(horizon=7)
        net.carbon_budget_per_day = 600.0   # tight enough to be binding
        opt = MultiPeriodOptimizer(net, solver="cbc")
        r = opt.solve()
        for t, co2 in r.carbon_by_period.items():
            assert co2 <= 600.0 + 5.0, f"Carbon exceeded budget at t={t}: {co2:.1f} t/d"

    def test_disruption_reduces_objective(self):
        from src.analysis.scenario import supply_disruption
        net_base = build_base_network(horizon=7)
        net_disrupt = supply_disruption("W3", 0.80)(copy.deepcopy(net_base))

        r_base = MultiPeriodOptimizer(net_base, solver="cbc").solve()
        r_disrupt = MultiPeriodOptimizer(net_disrupt, solver="cbc").solve()

        # Major supply disruption must reduce net margin
        assert r_disrupt.objective_value < r_base.objective_value


class TestStochasticMetrics:
    @pytest.mark.timeout(90)
    def test_ws_ge_rp_ge_eev(self):
        """
        Jensen's inequality: WS >= RP >= EEV must hold.
        Uses fast approximate mode (use_extensive_form=False) for CI speed.
        """
        from src.model.stochastic import run_stochastic_analysis
        net = build_base_network(horizon=5)
        sr = run_stochastic_analysis(net, n_scenarios=4, solver="cbc",
                                     use_extensive_form=False)
        assert sr.ws >= sr.rp - 1.0    # allow tiny numerical tolerance
        assert sr.rp >= sr.eev - 1.0
        assert sr.evpi >= 0.0
        assert sr.vss >= 0.0

    def test_risk_metrics_ordering(self):
        from src.model.stochastic import compute_risk_metrics
        import numpy as np
        objs = list(np.random.default_rng(42).normal(1e6, 1e5, 20))
        risk = compute_risk_metrics(objs)
        assert risk.min_obj <= risk.var_99 <= risk.var_95 <= risk.mean_obj
        assert risk.cvar_95 <= risk.var_95
