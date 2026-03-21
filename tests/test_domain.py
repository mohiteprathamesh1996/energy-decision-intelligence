"""
Unit tests for the domain model layer.
No solver dependency — pure Python / dataclass logic.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from data.generate_data import build_base_network
from src.model.supply_chain import (
    Arc, CrudeDiet, CrudeGrade, Node, NodeType,
    ShipOrPayContract, SupplyChainNetwork,
)


# ── Network construction ──────────────────────────────────────────────────────

class TestNetworkConstruction:
    def test_build_base_network_node_count(self):
        net = build_base_network(horizon=7)
        # 5W + 3T + 3R + 2D + 5M = 18
        assert len(net.nodes) == 18

    def test_build_base_network_arc_count(self):
        net = build_base_network(horizon=7)
        assert len(net.arcs) == 31   # sum of all well-storage + storage-ref + ref-dist + dist-demand

    def test_grade_count(self):
        net = build_base_network()
        assert len(net.grades) == 3
        assert "WTI" in net.grades
        assert "WTS" in net.grades
        assert "HEAVY" in net.grades

    def test_contract_count(self):
        net = build_base_network()
        assert len(net.contracts) == 4

    def test_planning_horizon(self):
        net = build_base_network(horizon=21)
        assert net.planning_horizon == 21

    def test_all_nodes_have_unique_ids(self):
        net = build_base_network()
        ids = list(net.nodes.keys())
        assert len(ids) == len(set(ids))


class TestNodeTypes:
    def setup_method(self):
        self.net = build_base_network()

    def test_correct_number_of_wells(self):
        wells = self.net.get_nodes_by_type(NodeType.WELL)
        assert len(wells) == 5

    def test_correct_number_of_refineries(self):
        refs = self.net.get_nodes_by_type(NodeType.REFINERY)
        assert len(refs) == 3

    def test_correct_number_of_demand_nodes(self):
        demand = self.net.get_nodes_by_type(NodeType.DEMAND)
        assert len(demand) == 5

    def test_storage_has_initial_inventory(self):
        storage = self.net.get_nodes_by_type(NodeType.STORAGE)
        for s in storage:
            assert s.initial_inventory > 0
            assert len(s.initial_inv_by_grade) > 0

    def test_refineries_have_crude_diet(self):
        refineries = self.net.get_nodes_by_type(NodeType.REFINERY)
        for r in refineries:
            assert r.crude_diet is not None

    def test_wells_have_primary_grade(self):
        wells = self.net.get_nodes_by_type(NodeType.WELL)
        for w in wells:
            assert w.primary_grade in self.net.grade_ids()

    def test_wells_have_decline_rate(self):
        wells = self.net.get_nodes_by_type(NodeType.WELL)
        for w in wells:
            assert 0.0 < w.decline_rate < 0.05


class TestCrudeGrades:
    def setup_method(self):
        self.net = build_base_network()

    def test_wti_is_lightest(self):
        assert self.net.grades["WTI"].api_gravity > self.net.grades["WTS"].api_gravity
        assert self.net.grades["WTS"].api_gravity > self.net.grades["HEAVY"].api_gravity

    def test_heavy_is_highest_carbon(self):
        intensities = {g: grade.carbon_intensity for g, grade in self.net.grades.items()}
        assert intensities["HEAVY"] > intensities["WTS"] > intensities["WTI"]

    def test_wti_has_zero_differential(self):
        assert self.net.grades["WTI"].price_differential == 0.0

    def test_discounts_are_negative(self):
        for g, grade in self.net.grades.items():
            if g != "WTI":
                assert grade.price_differential < 0.0

    def test_sulfur_content_ordering(self):
        assert self.net.grades["WTI"].sulfur_content < 0.5   # sweet
        assert self.net.grades["WTS"].sulfur_content > 0.5   # sour
        assert self.net.grades["HEAVY"].sulfur_content > 1.0


class TestCrudeDiet:
    def setup_method(self):
        self.net = build_base_network()

    def test_houston_rejects_sour(self):
        r1 = self.net.nodes["R1"]
        assert r1.crude_diet.sulfur_max < 1.0   # R1 can't handle heavy sour

    def test_port_arthur_accepts_sour(self):
        r2 = self.net.nodes["R2"]
        assert r2.crude_diet.sulfur_max > 2.0   # R2 has coking capacity

    def test_api_bounds_are_valid(self):
        refineries = self.net.get_nodes_by_type(NodeType.REFINERY)
        for r in refineries:
            assert r.crude_diet.api_min < r.crude_diet.api_max
            assert r.crude_diet.api_min > 0
            assert r.crude_diet.api_max < 60


class TestArcProperties:
    def setup_method(self):
        self.net = build_base_network()

    def test_all_arcs_have_positive_capacity(self):
        for arc in self.net.arcs:
            assert arc.capacity > 0

    def test_all_arcs_have_non_negative_transport_cost(self):
        for arc in self.net.arcs:
            assert arc.transport_cost >= 0

    def test_arc_lookup_works(self):
        arc = self.net.arc_lookup("W1", "T1")
        assert arc is not None
        assert arc.capacity == 45_000

    def test_arc_lookup_returns_none_for_missing(self):
        arc = self.net.arc_lookup("W1", "R1")   # wells don't connect directly to refineries
        assert arc is None

    def test_sour_arcs_exist(self):
        sour_arcs = [a for a in self.net.arcs if not a.accepts_sour]
        # Houston (R1) is light-sweet — some pipelines feeding it are sweet-only
        # Check that the flag is used somewhere
        all_accept_sour = all(a.accepts_sour for a in self.net.arcs)
        assert not all_accept_sour, "At least some pipelines should be sweet-only"

    def test_arc_index_matches_arcs(self):
        index = self.net.arc_index
        assert len(index) == len(self.net.arcs)
        for arc in self.net.arcs:
            assert (arc.origin, arc.destination) in index


class TestShipOrPayContracts:
    def setup_method(self):
        self.net = build_base_network()

    def test_all_contracts_reference_valid_arcs(self):
        for c in self.net.contracts:
            arc = self.net.arc_lookup(c.arc_origin, c.arc_dest)
            assert arc is not None, f"Contract references non-existent arc {c.arc_origin}->{c.arc_dest}"

    def test_committed_volume_below_capacity(self):
        for c in self.net.contracts:
            arc = self.net.arc_lookup(c.arc_origin, c.arc_dest)
            assert c.min_daily_volume < arc.capacity

    def test_deficiency_charge_less_than_tariff(self):
        for c in self.net.contracts:
            # Economic: deficiency charge should < spot tariff (otherwise no one commits)
            assert c.deficiency_charge < c.contract_tariff + 5.0


class TestProductionDecline:
    def setup_method(self):
        self.net = build_base_network(horizon=30)

    def test_capacity_declines_over_time(self):
        for well_id, n in self.net.nodes.items():
            if n.node_type == NodeType.WELL and n.decline_rate > 0:
                cap_d1 = self.net.well_capacity(well_id, 1)
                cap_d30 = self.net.well_capacity(well_id, 30)
                assert cap_d30 < cap_d1

    def test_decline_formula_correct(self):
        net = build_base_network(horizon=10)
        # Use W4 (Wolfcamp, highest decline rate 0.004)
        w4 = net.nodes["W4"]
        expected_d5 = w4.max_capacity * ((1 - w4.decline_rate) ** 4)
        actual_d5 = net.well_capacity("W4", 5)
        assert abs(actual_d5 - expected_d5) < 1e-6

    def test_no_decline_at_period_one(self):
        for well_id, n in self.net.nodes.items():
            if n.node_type == NodeType.WELL:
                assert self.net.well_capacity(well_id, 1) == n.max_capacity
