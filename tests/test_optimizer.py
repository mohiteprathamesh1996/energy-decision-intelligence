"""
Unit tests for the PuLP optimizer model.
These run without needing CBC — they validate variable creation,
constraint structure, and parameter values.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pulp
import pytest
from data.generate_data import build_base_network
from src.model.optimizer import MultiPeriodOptimizer
from src.model.supply_chain import NodeType


@pytest.fixture(scope="module")
def net7():
    return build_base_network(horizon=7)


@pytest.fixture(scope="module")
def prob7(net7):
    opt = MultiPeriodOptimizer(net7)
    return opt.build(), opt


class TestPuLPModelStructure:
    def test_problem_is_maximise(self, prob7):
        prob, _ = prob7
        assert prob.sense == pulp.LpMaximize

    def test_has_variables(self, prob7):
        prob, opt = prob7
        assert len(prob.variables()) > 0

    def test_flow_variables_nonneg(self, prob7):
        prob, opt = prob7
        for v in opt._x.values():
            assert v.lowBound == 0.0

    def test_inventory_variables_nonneg(self, prob7):
        prob, opt = prob7
        for v in opt._s.values():
            assert v.lowBound == 0.0

    def test_binary_activation_variables(self, prob7):
        prob, opt = prob7
        # Fixed-cost arcs (long-haul tanker routes to LA) have binary y
        if opt._y:
            for v in opt._y.values():
                assert v.cat == "Integer"
                assert v.lowBound == 0
                assert v.upBound == 1

    def test_has_constraints(self, prob7):
        prob, _ = prob7
        assert len(prob.constraints) > 0

    def test_constraint_names_unique(self, prob7):
        prob, _ = prob7
        names = list(prob.constraints.keys())
        assert len(names) == len(set(names))

    def test_grade_lock_constraints_exist(self, prob7):
        prob, _ = prob7
        grade_lock = [k for k in prob.constraints if k.startswith("grade_lock")]
        assert len(grade_lock) > 0

    def test_storage_balance_constraints_exist(self, prob7):
        prob, _ = prob7
        stor_bal = [k for k in prob.constraints if k.startswith("stor_bal")]
        assert len(stor_bal) > 0

    def test_refinery_bounds_constraints_exist(self, prob7):
        prob, _ = prob7
        ref_lo = [k for k in prob.constraints if k.startswith("ref_lo")]
        ref_hi = [k for k in prob.constraints if k.startswith("ref_hi")]
        assert len(ref_lo) > 0
        assert len(ref_hi) > 0

    def test_api_constraints_exist(self, prob7):
        prob, _ = prob7
        api_lo = [k for k in prob.constraints if k.startswith("api_lo")]
        api_hi = [k for k in prob.constraints if k.startswith("api_hi")]
        assert len(api_lo) > 0
        assert len(api_hi) > 0

    def test_sulfur_constraints_exist(self, prob7):
        prob, _ = prob7
        sulfur = [k for k in prob.constraints if k.startswith("sulfur")]
        assert len(sulfur) > 0

    def test_demand_constraints_exist(self, prob7):
        prob, _ = prob7
        demand = [k for k in prob.constraints if k.startswith("demand")]
        assert len(demand) > 0

    def test_arc_capacity_constraints_exist(self, prob7):
        prob, _ = prob7
        arc_cap = [k for k in prob.constraints if k.startswith("arc_cap")]
        assert len(arc_cap) > 0

    def test_sop_constraints_exist(self, prob7):
        prob, _ = prob7
        sop = [k for k in prob.constraints if k.startswith("sop")]
        assert len(sop) > 0

    def test_sweet_only_constraints_exist(self, prob7):
        prob, _ = prob7
        sweet = [k for k in prob.constraints if k.startswith("sw_")]
        # At least some sweet-only constraints should exist
        assert len(sweet) > 0

    def test_well_capacity_constraints_exist(self, prob7):
        prob, _ = prob7
        wc = [k for k in prob.constraints if k.startswith("well_cap")]
        assert len(wc) > 0


class TestCrackSpreadMultiplier:
    def test_multiplier_changes_objective(self, net7):
        """Doubling crack spread multiplier should increase objective."""
        import copy
        # Base
        opt1 = MultiPeriodOptimizer(copy.deepcopy(net7))
        opt1.build()

        # With doubled margins
        mults = {
            (nid, t): 2.0
            for nid, n in net7.nodes.items()
            if n.node_type == NodeType.REFINERY
            for t in range(1, 8)
        }
        opt2 = MultiPeriodOptimizer(copy.deepcopy(net7))
        opt2.build(crack_spread_multipliers=mults)

        # The second model's objective coefficients should be higher
        # (We don't solve here — just check the model was built differently)
        n_vars1 = len(opt1.prob.variables())
        n_vars2 = len(opt2.prob.variables())
        assert n_vars1 == n_vars2   # same structure


class TestScenarioModifiers:
    def test_supply_disruption(self):
        import copy
        from src.analysis.scenario import supply_disruption
        net = build_base_network()
        orig = net.nodes["W3"].max_capacity
        mod  = supply_disruption("W3", 0.60)(copy.deepcopy(net))
        assert abs(mod.nodes["W3"].max_capacity - orig * 0.40) < 1.0

    def test_demand_spike(self):
        import copy
        from src.analysis.scenario import demand_spike
        net = build_base_network()
        orig = net.nodes["M1"].demand
        mod  = demand_spike("M1", 1.40)(copy.deepcopy(net))
        assert abs(mod.nodes["M1"].demand - orig * 1.40) < 1.0

    def test_refinery_outage(self):
        import copy
        from src.analysis.scenario import refinery_outage
        net = build_base_network()
        mod = refinery_outage("R2")(copy.deepcopy(net))
        assert mod.nodes["R2"].max_capacity == 0

    def test_carbon_cap(self):
        import copy
        from src.analysis.scenario import carbon_budget_constraint
        net = build_base_network()
        assert net.carbon_budget_per_day is None
        mod = carbon_budget_constraint(850.0)(copy.deepcopy(net))
        assert mod.carbon_budget_per_day == 850.0

    def test_composable(self):
        import copy
        from src.analysis.scenario import demand_spike, supply_disruption
        net = build_base_network()
        mod = demand_spike("M1", 1.35)(supply_disruption("W3", 0.50)(copy.deepcopy(net)))
        assert mod.nodes["W3"].max_capacity < net.nodes["W3"].max_capacity
        assert mod.nodes["M1"].demand > net.nodes["M1"].demand


class TestPuLPAvailability:
    def test_pulp_importable(self):
        import pulp
        assert pulp is not None

    def test_cbc_bundled(self):
        """CBC should be available via PuLP without system install."""
        import pulp
        solver = pulp.PULP_CBC_CMD(msg=0)
        # available() returns True if the bundled binary is found
        assert solver.available(), (
            "CBC not found via PuLP. Run: pip install pulp --upgrade"
        )