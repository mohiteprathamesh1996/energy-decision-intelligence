"""
Unit tests for optimizer model construction.
Tests that the Pyomo model is built correctly — constraint counts,
variable domains, parameter values — without running the solver.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pyomo.environ as pyo
from data.generate_data import build_base_network
from src.model.optimizer import MultiPeriodOptimizer
from src.model.supply_chain import NodeType


@pytest.fixture(scope="module")
def net_7():
    return build_base_network(horizon=7)


@pytest.fixture(scope="module")
def model_7(net_7):
    opt = MultiPeriodOptimizer(net_7)
    return opt.build()


class TestModelSets:
    def test_period_set(self, model_7):
        assert list(model_7.PERIODS) == list(range(1, 8))

    def test_grade_set(self, model_7):
        assert set(model_7.GRADES) == {"WTI", "WTS", "HEAVY"}

    def test_arc_set_nonempty(self, model_7):
        assert len(list(model_7.ARCS)) > 0

    def test_fixed_arc_set(self, model_7):
        # Some dist->demand arcs have fixed_cost > 0 (long-haul tanker routes)
        assert len(list(model_7.ARCS_FIXED)) > 0

    def test_sop_arc_set_count(self, model_7):
        assert len(list(model_7.ARCS_SOP)) == 4   # 4 ship-or-pay contracts


class TestModelVariables:
    def test_flow_variables_exist(self, model_7):
        assert hasattr(model_7, "x")

    def test_inventory_variables_exist(self, model_7):
        assert hasattr(model_7, "s")

    def test_unmet_variables_exist(self, model_7):
        assert hasattr(model_7, "u")

    def test_deficit_variables_exist(self, model_7):
        assert hasattr(model_7, "deficit")

    def test_binary_activation_variables(self, model_7):
        assert hasattr(model_7, "y")
        for (i, j) in model_7.ARCS_FIXED:
            assert model_7.y[i, j].domain == pyo.Binary

    def test_flow_variables_nonnegative(self, net_7, model_7):
        arc = list(model_7.ARCS)[0]
        i, j = arc
        assert model_7.x[i, j, 1, "WTI"].lb == 0.0


class TestModelConstraints:
    def test_well_capacity_constraints_exist(self, model_7):
        assert hasattr(model_7, "c_well_cap")

    def test_storage_balance_constraints_exist(self, model_7):
        assert hasattr(model_7, "c_storage_bal")

    def test_refinery_min_constraints_exist(self, model_7):
        assert hasattr(model_7, "c_ref_min")

    def test_api_constraints_exist(self, model_7):
        assert hasattr(model_7, "c_api_lo")
        assert hasattr(model_7, "c_api_hi")

    def test_sulfur_constraint_exists(self, model_7):
        assert hasattr(model_7, "c_sulfur")

    def test_demand_constraints_exist(self, model_7):
        assert hasattr(model_7, "c_demand")

    def test_arc_capacity_constraints_exist(self, model_7):
        assert hasattr(model_7, "c_arc_cap")

    def test_sop_constraints_exist(self, model_7):
        assert hasattr(model_7, "c_sop")

    def test_sweet_only_constraints_exist(self, model_7):
        # Model should generate sweet-only constraints for sour grades on restricted pipelines
        assert hasattr(model_7, "c_sweet_only")

    def test_grade_lock_constraints_exist(self, model_7):
        assert hasattr(model_7, "c_grade_lock")


class TestModelParameters:
    def test_well_capacity_declines(self, net_7, model_7):
        # At period 7, Wolfcamp (W4, decline=0.004) should be lower than period 1
        cap_d1 = pyo.value(model_7.well_cap["W4", 1])
        cap_d7 = pyo.value(model_7.well_cap["W4", 7])
        assert cap_d7 < cap_d1

    def test_refinery_api_lo_positive(self, model_7):
        for nid in model_7.NODES:
            lo = pyo.value(model_7.API_lo[nid])
            assert lo >= 0.0

    def test_carbon_intensities_positive(self, model_7):
        for g in model_7.GRADES:
            assert pyo.value(model_7.carbon_int[g]) > 0.0

    def test_base_margin_positive_at_refineries(self, net_7, model_7):
        for nid in model_7.NODES:
            if net_7.nodes[nid].node_type == NodeType.REFINERY:
                assert pyo.value(model_7.margin[nid, 1]) > 0.0

    def test_crack_spread_multiplier_applied(self, net_7):
        """With multiplier=2.0, margin should double."""
        mults = {(ref_id, t): 2.0
                 for ref_id, n in net_7.nodes.items()
                 if n.node_type == NodeType.REFINERY
                 for t in range(1, 8)}
        opt = MultiPeriodOptimizer(net_7)
        m_base = opt.build()
        opt2 = MultiPeriodOptimizer(build_base_network(horizon=7))
        m_2x = opt2.build(crack_spread_multipliers=mults)
        for ref_id, n in net_7.nodes.items():
            if n.node_type == NodeType.REFINERY:
                base_margin = pyo.value(m_base.margin[ref_id, 1])
                scaled_margin = pyo.value(m_2x.margin[ref_id, 1])
                assert abs(scaled_margin - 2.0 * base_margin) < 1e-6


class TestObjectiveStructure:
    def test_objective_is_maximize(self, model_7):
        assert model_7.obj.sense == pyo.maximize

    def test_objective_is_single(self, model_7):
        # Only one Objective declared
        obj_count = sum(1 for _ in model_7.component_objects(pyo.Objective))
        assert obj_count == 1


class TestScenarioModifiers:
    """Test scenario modifier functions from the analysis layer."""

    def test_supply_disruption_reduces_capacity(self):
        import copy
        from src.analysis.scenario import supply_disruption
        net = build_base_network()
        original_cap = net.nodes["W3"].max_capacity
        modified = supply_disruption("W3", 0.60)(copy.deepcopy(net))
        assert abs(modified.nodes["W3"].max_capacity - original_cap * 0.40) < 1.0

    def test_demand_spike_increases_demand(self):
        import copy
        from src.analysis.scenario import demand_spike
        net = build_base_network()
        original_demand = net.nodes["M1"].demand
        modified = demand_spike("M1", 1.40)(copy.deepcopy(net))
        assert abs(modified.nodes["M1"].demand - original_demand * 1.40) < 1.0

    def test_refinery_outage_zeroes_capacity(self):
        import copy
        from src.analysis.scenario import refinery_outage
        net = build_base_network()
        modified = refinery_outage("R2")(copy.deepcopy(net))
        assert modified.nodes["R2"].max_capacity == 0

    def test_carbon_constraint_sets_budget(self):
        import copy
        from src.analysis.scenario import carbon_budget_constraint
        net = build_base_network()
        assert net.carbon_budget_per_day is None
        modified = carbon_budget_constraint(850.0)(copy.deepcopy(net))
        assert modified.carbon_budget_per_day == 850.0

    def test_modifiers_are_composable(self):
        import copy
        from src.analysis.scenario import demand_spike, supply_disruption
        net = build_base_network()
        # Compose two modifiers
        modified = demand_spike("M1", 1.35)(supply_disruption("W3", 0.50)(copy.deepcopy(net)))
        assert modified.nodes["W3"].max_capacity < net.nodes["W3"].max_capacity
        assert modified.nodes["M1"].demand > net.nodes["M1"].demand
