"""
Integration tests — require PuLP's bundled CBC.
Run with: pytest tests/test_integration.py -v

CBC ships with `pip install pulp`, so these should work
in any environment where PuLP is installed.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import pulp
import pytest

# Verify bundled CBC is usable
cbc_ok = pulp.PULP_CBC_CMD(msg=0).available()
pytestmark = pytest.mark.skipif(not cbc_ok, reason="PuLP bundled CBC not available — pip install pulp")

from data.generate_data import build_base_network
from src.model.optimizer import MultiPeriodOptimizer
from src.model.supply_chain import NodeType


@pytest.fixture(scope="module")
def base():
    net = build_base_network(horizon=7)
    r   = MultiPeriodOptimizer(net).solve()
    return net, r


class TestBaseSolution:
    def test_status_optimal(self, base):
        _, r = base
        assert r.status == "optimal"

    def test_objective_positive(self, base):
        _, r = base
        assert r.objective_value > 0

    def test_revenue_exceeds_costs(self, base):
        _, r = base
        assert r.revenue > r.transport_cost + r.holding_cost

    def test_service_level_above_90(self, base):
        net, r = base
        T = r.planning_horizon
        total = sum(n.demand for n in net.get_nodes_by_type(NodeType.DEMAND)) * T
        unmet = sum(r.unmet_demand.values())
        assert (1 - unmet / max(total, 1)) >= 0.90

    def test_flows_within_arc_capacity(self, base):
        net, r = base
        T = r.planning_horizon
        for (i, j) in net.arc_index:
            cap = net.arc_lookup(i, j).capacity
            for t in range(1, T + 1):
                flow = r.flows_by_period.get((i, j, t), 0)
                assert flow <= cap * 1.001, f"Arc {i}->{j} t={t}: {flow:.0f} > {cap:.0f}"

    def test_inventories_nonnegative(self, base):
        _, r = base
        for v in r.inventories.values():
            assert v >= -1.0

    def test_carbon_positive(self, base):
        _, r = base
        assert sum(r.carbon_by_period.values()) > 0


class TestGradeRouting:
    def test_wts_not_routed_to_sweet_refinery(self, base):
        """WTS (sulfur 1.52%) must not reach R1 Houston (sulfur_max 0.50%)."""
        net, r = base
        T = r.planning_horizon
        node_ids = list(net.nodes.keys())
        for t in range(1, T + 1):
            for s_id in [n.id for n in net.get_nodes_by_type(NodeType.STORAGE)]:
                flow = r.flows_by_grade.get((s_id, "R1", t, "WTS"), 0)
                assert flow < 500, f"WTS flowing to R1 at t={t}: {flow:.0f} bbl/d"


class TestScenarios:
    @pytest.mark.parametrize("well,pct", [("W3", 0.60), ("W1", 0.30)])
    def test_disruption_solves(self, well, pct):
        from src.analysis.scenario import supply_disruption
        net = build_base_network(horizon=7)
        mod = supply_disruption(well, pct)(net)
        r   = MultiPeriodOptimizer(mod).solve()
        assert r.status in ("optimal",)

    def test_carbon_cap_binding(self):
        net = build_base_network(horizon=7)
        net.carbon_budget_per_day = 600.0
        r   = MultiPeriodOptimizer(net).solve()
        for t, co2 in r.carbon_by_period.items():
            assert co2 <= 600.0 + 5.0, f"Carbon exceeded budget at t={t}: {co2:.1f}"

    def test_disruption_lowers_objective(self):
        from src.analysis.scenario import supply_disruption
        net_b = build_base_network(horizon=7)
        net_d = supply_disruption("W3", 0.80)(copy.deepcopy(net_b))
        r_b = MultiPeriodOptimizer(net_b).solve()
        r_d = MultiPeriodOptimizer(net_d).solve()
        assert r_d.objective_value < r_b.objective_value


class TestStochasticMetrics:
    @pytest.mark.timeout(120)
    def test_jensen_chain(self):
        """WS >= RP >= EEV must hold (Jensen's inequality)."""
        from src.model.stochastic import run_stochastic_analysis
        net = build_base_network(horizon=5)
        sr  = run_stochastic_analysis(net, n_scenarios=4, use_extensive_form=False)
        assert sr.ws  >= sr.rp  - 1.0
        assert sr.rp  >= sr.eev - 1.0
        assert sr.evpi >= 0.0
        assert sr.vss  >= 0.0

    def test_risk_metric_ordering(self):
        from src.model.stochastic import compute_risk_metrics
        import numpy as np
        objs = list(np.random.default_rng(0).normal(1e6, 1e5, 20))
        risk = compute_risk_metrics(objs)
        assert risk.min_obj <= risk.var_99 <= risk.var_95 <= risk.mean_obj
        assert risk.cvar_95 <= risk.var_95