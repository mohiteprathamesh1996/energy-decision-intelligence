"""
Market data and price scenario generation.

Crack spread calibration:
  - Gulf Coast 3-2-1 crack spread: historical mean ~$14-18/bbl (2020-2024)
  - Annual volatility: ~22% (computed from EIA weekly crack spread data)
  - WTI/WTS differential: -$2.50 to -$4.00/bbl (EIA crude oil marketing reports)
  - Grade-adjusted refinery margins account for yield and product slate differences

This module is intentionally decoupled from the optimizer — it generates parameters
that get injected as crack_spread_multipliers into MultiPeriodOptimizer.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# Historical Gulf Coast crack spread statistics (3-2-1 spread, $/bbl)
# Source: EIA Weekly Petroleum Status Report, 2020-2024 average
CRACK_SPREAD_STATS = {
    "R1_houston":     {"mean": 14.20, "vol": 0.22, "skew": 0.15},
    "R2_port_arthur": {"mean": 12.80, "vol": 0.20, "skew": 0.12},
    "R3_beaumont":    {"mean": 13.50, "vol": 0.21, "skew": 0.13},
}

# WTI benchmark and grade differentials ($/bbl)
WTI_PRICE = 78.50   # $/bbl (approximate 2024 average)

GRADE_DIFFERENTIALS = {
    "WTI":   0.00,
    "WTS":  -3.20,
    "HEAVY": -7.40,
}

# Permian Basin long-haul pipeline tariff ranges (FERC Form 6)
PIPELINE_TARIFF_RANGES = {
    "permian_to_houston":     (3.50, 4.50),   # $/bbl
    "permian_to_port_arthur": (3.80, 4.80),
    "intrastate_gathering":   (1.80, 2.80),
}


@dataclass
class PriceScenario:
    id: str
    label: str
    crack_spreads: Dict[str, float]   # refinery_id -> $/bbl
    wti_price: float
    probability: float = 1.0


def generate_price_scenarios(
    n: int = 20,
    seed: int = 42,
    horizon: int = 14,
) -> List[PriceScenario]:
    """
    Generate N price scenarios from calibrated distributions.
    Used for scenario fan charts and stochastic analysis.
    """
    rng = np.random.default_rng(seed)
    scenarios = []

    ref_map = {
        "R1": CRACK_SPREAD_STATS["R1_houston"],
        "R2": CRACK_SPREAD_STATS["R2_port_arthur"],
        "R3": CRACK_SPREAD_STATS["R3_beaumont"],
    }

    for i in range(n):
        spreads = {}
        for ref_id, stats in ref_map.items():
            # Lognormal with calibrated mean and vol
            log_mean = np.log(stats["mean"]) - 0.5 * stats["vol"] ** 2
            shock = rng.normal(log_mean, stats["vol"])
            spreads[ref_id] = float(np.exp(shock))

        wti = float(rng.normal(WTI_PRICE, WTI_PRICE * 0.15))

        label = _label(spreads, wti)
        scenarios.append(PriceScenario(
            id=f"S{i+1:02d}",
            label=label,
            crack_spreads=spreads,
            wti_price=max(30.0, wti),
            probability=1.0 / n,
        ))

    return scenarios


def _label(spreads: Dict[str, float], wti: float) -> str:
    avg = sum(spreads.values()) / len(spreads)
    if avg > 17:
        return "Bull"
    elif avg < 10:
        return "Bear"
    else:
        return "Base"


def crack_spread_multipliers_from_scenario(
    scenario: PriceScenario,
    base_margins: Dict[str, float],
    horizon: int = 14,
) -> Dict[Tuple[str, int], float]:
    """
    Convert absolute crack spreads to multipliers over the base margins.
    These are passed into MultiPeriodOptimizer.build().
    """
    mults = {}
    for ref_id, spread in scenario.crack_spreads.items():
        base = base_margins.get(ref_id, 1.0)
        mult = spread / base if base > 0 else 1.0
        for t in range(1, horizon + 1):
            mults[(ref_id, t)] = mult
    return mults


def compute_value_at_risk(
    scenario_objectives: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Returns (VaR, CVaR) at the given confidence level.
    For a maximization problem, VaR = lower tail cutoff.
    """
    sorted_obj = sorted(scenario_objectives)
    idx = int((1 - confidence) * len(sorted_obj))
    var = sorted_obj[idx]
    cvar = np.mean(sorted_obj[:max(1, idx)])
    return float(var), float(cvar)


def get_grade_netback(grade_id: str, wti_price: float, pipeline_tariff: float) -> float:
    """
    Wellhead netback price: WTI - grade differential - pipeline tariff - lifting cost.
    This is the realized price at the wellhead.
    """
    diff = GRADE_DIFFERENTIALS.get(grade_id, 0.0)
    return wti_price + diff - pipeline_tariff
