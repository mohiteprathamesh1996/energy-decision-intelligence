"""
Unit tests for stochastic scenario generation and risk metrics.
No solver required.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from data.generate_data import build_base_network
from src.model.stochastic import (
    compute_risk_metrics, generate_crack_spread_scenarios,
)
from src.model.supply_chain import NodeType


class TestScenarioGeneration:
    def setup_method(self):
        self.net = build_base_network(horizon=14)
        self.scenarios = generate_crack_spread_scenarios(
            self.net, n_scenarios=20, annual_vol=0.22, seed=42
        )

    def test_correct_number_of_scenarios(self):
        assert len(self.scenarios) == 20

    def test_probabilities_sum_to_one(self):
        total_prob = sum(sc.probability for sc in self.scenarios)
        assert abs(total_prob - 1.0) < 1e-10

    def test_equal_probabilities(self):
        probs = [sc.probability for sc in self.scenarios]
        assert all(abs(p - 1/20) < 1e-12 for p in probs)

    def test_unique_scenario_ids(self):
        ids = [sc.id for sc in self.scenarios]
        assert len(ids) == len(set(ids))

    def test_multipliers_cover_all_refineries(self):
        refinery_ids = {nid for nid, n in self.net.nodes.items()
                        if n.node_type == NodeType.REFINERY}
        for sc in self.scenarios:
            for ref_id in refinery_ids:
                for t in range(1, self.net.planning_horizon + 1):
                    assert (ref_id, t) in sc.multipliers, \
                        f"Scenario {sc.id} missing multiplier for ({ref_id}, {t})"

    def test_multipliers_cover_all_periods(self):
        T = self.net.planning_horizon
        for sc in self.scenarios:
            for (ref_id, t), _ in sc.multipliers.items():
                assert 1 <= t <= T

    def test_multipliers_are_positive(self):
        for sc in self.scenarios:
            for (ref_id, t), mult in sc.multipliers.items():
                assert mult > 0.0, f"Negative multiplier in scenario {sc.id}"

    def test_mean_multiplier_close_to_one(self):
        """Itô correction should make E[multiplier] ≈ 1."""
        all_mults = [mult
                     for sc in self.scenarios
                     for mult in sc.multipliers.values()]
        avg = np.mean(all_mults)
        assert abs(avg - 1.0) < 0.15, f"Mean multiplier {avg:.3f} too far from 1.0"

    def test_labels_are_strings(self):
        for sc in self.scenarios:
            assert isinstance(sc.label, str)
            assert len(sc.label) > 0

    def test_reproducibility(self):
        sc2 = generate_crack_spread_scenarios(self.net, n_scenarios=20, seed=42)
        for a, b in zip(self.scenarios, sc2):
            for key in a.multipliers:
                assert abs(a.multipliers[key] - b.multipliers[key]) < 1e-12

    def test_different_seeds_produce_different_scenarios(self):
        sc_42 = generate_crack_spread_scenarios(self.net, n_scenarios=10, seed=42)
        sc_99 = generate_crack_spread_scenarios(self.net, n_scenarios=10, seed=99)
        # At least some multipliers should differ
        diff = sum(
            abs(a.multipliers[k] - b.multipliers[k])
            for a, b in zip(sc_42, sc_99)
            for k in a.multipliers
        )
        assert diff > 0.1

    def test_vol_parameter_affects_spread(self):
        low_vol = generate_crack_spread_scenarios(self.net, n_scenarios=50, annual_vol=0.05, seed=0)
        hi_vol  = generate_crack_spread_scenarios(self.net, n_scenarios=50, annual_vol=0.50, seed=0)

        all_low = [m for sc in low_vol for m in sc.multipliers.values()]
        all_hi  = [m for sc in hi_vol  for m in sc.multipliers.values()]

        assert np.std(all_hi) > np.std(all_low), "Higher vol should produce wider spread"


class TestRiskMetrics:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.objectives = list(rng.normal(loc=1_000_000, scale=200_000, size=100))
        self.risk = compute_risk_metrics(self.objectives)

    def test_var95_is_lower_than_mean(self):
        assert self.risk.var_95 < self.risk.mean_obj

    def test_cvar95_le_var95(self):
        assert self.risk.cvar_95 <= self.risk.var_95

    def test_var99_le_var95(self):
        assert self.risk.var_99 <= self.risk.var_95

    def test_cvar99_le_var99(self):
        assert self.risk.cvar_99 <= self.risk.var_99

    def test_min_le_var99(self):
        assert self.risk.min_obj <= self.risk.var_99

    def test_max_ge_mean(self):
        assert self.risk.max_obj >= self.risk.mean_obj

    def test_std_positive(self):
        assert self.risk.std_obj > 0

    def test_pct_below_zero_in_range(self):
        assert 0.0 <= self.risk.pct_below_zero <= 100.0

    def test_constant_distribution(self):
        """All identical values — VaR = CVaR = mean."""
        risk = compute_risk_metrics([1_000.0] * 20)
        assert abs(risk.var_95 - 1_000.0) < 1e-6
        assert abs(risk.mean_obj - 1_000.0) < 1e-6
        assert risk.std_obj == 0.0

    def test_single_element(self):
        """Single scenario — should not crash."""
        risk = compute_risk_metrics([500_000.0])
        assert risk.mean_obj == 500_000.0

    def test_all_negative_objectives(self):
        """All scenarios unprofitable — pct_below_zero should be 100."""
        risk = compute_risk_metrics([-1000.0] * 10)
        assert risk.pct_below_zero == 100.0
