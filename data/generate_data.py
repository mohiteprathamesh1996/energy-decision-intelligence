"""
Builds the Permian Basin → Gulf Coast supply chain network.

Calibration basis:
  - Well capacities: EIA Permian production data (2023 avg ~6.2 mmb/d across basin)
  - Pipeline tariffs: FERC Form 6 public data for Permian-to-Gulf pipelines (~$3-6/bbl)
  - Refinery margins: EIA crack spread data (3-2-1 crack spread, Gulf Coast, $12-18/bbl range)
  - Demand: EIA PADD 2/3 regional petroleum product consumption
  - Carbon intensities: ICF International / API upstream carbon intensity studies
  - Storage capacities: EIA Cushing/Gulf Coast commercial crude stocks (pro-rated per terminal)
"""

import copy
from datetime import date
from typing import Optional

from src.model.supply_chain import (
    Arc, CrudeDiet, CrudeGrade, Node, NodeType,
    ShipOrPayContract, SupplyChainNetwork,
)


# ── Crude grade definitions ───────────────────────────────────────────────────

WTI_LIGHT = CrudeGrade(
    id="WTI",
    name="West Texas Intermediate (Light Sweet)",
    api_gravity=40.8,
    sulfur_content=0.24,
    price_differential=0.0,       # benchmark
    carbon_intensity=8.5,         # kg CO2e/bbl (upstream, OPGEE model reference)
)

WTS_SOUR = CrudeGrade(
    id="WTS",
    name="West Texas Sour",
    api_gravity=34.0,
    sulfur_content=1.52,
    price_differential=-3.20,     # WTS trades at ~$3/bbl discount to WTI
    carbon_intensity=10.2,        # higher flaring rate in sour Permian zones
)

PERMIAN_HEAVY = CrudeGrade(
    id="HEAVY",
    name="Permian Heavy Blend",
    api_gravity=27.5,
    sulfur_content=2.10,
    price_differential=-7.40,     # heavy sour discount
    carbon_intensity=11.8,        # highest carbon intensity; more water cut
)


def build_base_network(horizon: int = 14, start: Optional[date] = None) -> SupplyChainNetwork:
    net = SupplyChainNetwork(
        planning_horizon=horizon,
        start_date=start or date.today(),
    )

    net.add_grade(WTI_LIGHT)
    net.add_grade(WTS_SOUR)
    net.add_grade(PERMIAN_HEAVY)

    # ── Production Wells ──────────────────────────────────────────────────────
    # Permian Basin; coordinates are approximate geographic centers of sub-plays

    wells = [
        Node("W1", "Midland Basin North",  NodeType.WELL, 31.90, -102.08,
             max_capacity=45_000, operating_cost=18.50,
             primary_grade="WTI", decline_rate=0.002, flare_rate=0.35),
        Node("W2", "Midland Basin South",  NodeType.WELL, 31.52, -102.38,
             max_capacity=38_000, operating_cost=19.20,
             primary_grade="WTI", decline_rate=0.003, flare_rate=0.28),
        Node("W3", "Delaware Basin",       NodeType.WELL, 31.18, -103.10,
             max_capacity=52_000, operating_cost=17.80,
             primary_grade="WTS", decline_rate=0.001, flare_rate=0.42),
        Node("W4", "Wolfcamp Shale",       NodeType.WELL, 32.08, -101.92,
             max_capacity=28_000, operating_cost=22.10,
             primary_grade="WTI", decline_rate=0.004, flare_rate=0.31),
        Node("W5", "Bone Spring",          NodeType.WELL, 31.70, -103.52,
             max_capacity=33_000, operating_cost=20.40,
             primary_grade="HEAVY", decline_rate=0.002, flare_rate=0.55),
    ]

    # ── Midstream Storage Terminals ───────────────────────────────────────────

    terminals = [
        Node("T1", "Midland Hub",     NodeType.STORAGE, 31.99, -102.07,
             max_capacity=3_200_000, holding_cost=0.08, operating_cost=0.50,
             initial_inventory=480_000,
             initial_inv_by_grade={"WTI": 360_000, "WTS": 80_000, "HEAVY": 40_000}),
        Node("T2", "Odessa Terminal", NodeType.STORAGE, 31.84, -102.36,
             max_capacity=2_100_000, holding_cost=0.10, operating_cost=0.45,
             initial_inventory=300_000,
             initial_inv_by_grade={"WTI": 240_000, "WTS": 40_000, "HEAVY": 20_000}),
        Node("T3", "Crane Facility",  NodeType.STORAGE, 31.40, -102.35,
             max_capacity=1_600_000, holding_cost=0.09, operating_cost=0.52,
             initial_inventory=220_000,
             initial_inv_by_grade={"WTI": 80_000, "WTS": 100_000, "HEAVY": 40_000}),
    ]

    # ── Refineries (Gulf Coast) ───────────────────────────────────────────────
    # Crude diets reflect light-sweet vs. complex refinery configurations

    refineries = [
        Node("R1", "Houston Refinery (Light Sweet)",  NodeType.REFINERY, 29.68, -95.37,
             max_capacity=120_000, min_throughput=42_000,
             operating_cost=8.50, crude_yield=0.93, refinery_margin=14.20,
             crude_diet=CrudeDiet(api_min=35.0, api_max=45.0, sulfur_max=0.50, optimal_api=40.0)),
        Node("R2", "Port Arthur Complex (Coking)",    NodeType.REFINERY, 29.90, -93.93,
             max_capacity=95_000, min_throughput=32_000,
             operating_cost=9.40, crude_yield=0.91, refinery_margin=12.80,
             crude_diet=CrudeDiet(api_min=24.0, api_max=43.0, sulfur_max=2.80, optimal_api=34.0)),
        Node("R3", "Beaumont Complex (Mid-Range)",    NodeType.REFINERY, 30.08, -94.15,
             max_capacity=80_000, min_throughput=28_000,
             operating_cost=8.80, crude_yield=0.92, refinery_margin=13.50,
             crude_diet=CrudeDiet(api_min=30.0, api_max=44.0, sulfur_max=1.20, optimal_api=37.0)),
    ]

    # ── Distribution Hubs ─────────────────────────────────────────────────────

    dist_nodes = [
        Node("D1", "Houston Products Hub",     NodeType.DISTRIBUTION, 29.73, -95.01),
        Node("D2", "Gulf Coast Pipeline Hub",  NodeType.DISTRIBUTION, 30.02, -94.52),
    ]

    # ── Demand Markets ────────────────────────────────────────────────────────

    demand_nodes = [
        Node("M1", "Chicago (PADD 2)",       NodeType.DEMAND, 41.88, -87.63,
             demand=55_000, unmet_penalty=520),
        Node("M2", "New York (PADD 1)",      NodeType.DEMAND, 40.71, -74.01,
             demand=48_000, unmet_penalty=560),
        Node("M3", "Dallas / Fort Worth",    NodeType.DEMAND, 32.78, -96.80,
             demand=35_000, unmet_penalty=490),
        Node("M4", "Freeport Export (LNG)",  NodeType.DEMAND, 28.94, -95.37,
             demand=40_000, unmet_penalty=440),
        Node("M5", "Los Angeles (PADD 5)",   NodeType.DEMAND, 33.94, -118.41,
             demand=42_000, unmet_penalty=530),
    ]

    for n in wells + terminals + refineries + dist_nodes + demand_nodes:
        net.add_node(n)

    # ── Arcs: Wells → Storage ─────────────────────────────────────────────────
    # Gathering systems + trunkline tariffs

    well_to_storage = [
        Arc("W1", "T1", 45_000, 2.10),
        Arc("W1", "T2", 20_000, 2.40),
        Arc("W2", "T2", 38_000, 1.90),
        Arc("W2", "T3", 15_000, 2.60),
        Arc("W3", "T1", 30_000, 3.20, accepts_sour=True),
        Arc("W3", "T3", 52_000, 2.30, accepts_sour=True),
        Arc("W4", "T1", 28_000, 2.80),
        Arc("W4", "T2", 18_000, 3.10),
        Arc("W5", "T2", 25_000, 2.50, accepts_sour=True),
        Arc("W5", "T3", 33_000, 2.20, accepts_sour=True),
    ]

    # ── Arcs: Storage → Refineries ────────────────────────────────────────────
    # Long-haul pipeline tariffs (FERC-regulated Permian-to-Gulf lines)

    storage_to_ref = [
        Arc("T1", "R1",  80_000, 3.80),
        Arc("T1", "R2",  60_000, 4.20, accepts_sour=True),
        Arc("T1", "R3",  45_000, 4.10),
        Arc("T2", "R1",  70_000, 3.60),
        Arc("T2", "R2",  55_000, 3.90, accepts_sour=True),
        Arc("T2", "R3",  40_000, 4.30),
        Arc("T3", "R2",  50_000, 4.50, accepts_sour=True),
        Arc("T3", "R3",  40_000, 4.00),
    ]

    # ── Arcs: Refineries → Distribution ──────────────────────────────────────

    ref_to_dist = [
        Arc("R1", "D1", 120_000, 1.20),
        Arc("R1", "D2",  80_000, 1.50),
        Arc("R2", "D1",  60_000, 1.80),
        Arc("R2", "D2",  95_000, 1.30),
        Arc("R3", "D2",  80_000, 1.40),
        Arc("R3", "D1",  50_000, 1.60),
    ]

    # ── Arcs: Distribution → Demand ───────────────────────────────────────────
    # Tanker / pipeline to market. Long-haul arcs have fixed vessel costs.

    dist_to_demand = [
        Arc("D1", "M1",  70_000, 5.50, is_pipeline=False, fixed_cost=0),
        Arc("D1", "M2",  60_000, 6.20, is_pipeline=False, fixed_cost=0),
        Arc("D1", "M3",  45_000, 3.80),
        Arc("D1", "M4",  50_000, 2.10),
        Arc("D1", "M5",  50_000, 8.80, is_pipeline=False, fixed_cost=14_500),
        Arc("D2", "M1",  55_000, 5.80, is_pipeline=False, fixed_cost=0),
        Arc("D2", "M2",  50_000, 6.50, is_pipeline=False, fixed_cost=0),
        Arc("D2", "M3",  45_000, 3.40),
        Arc("D2", "M4",  40_000, 2.40),
        Arc("D2", "M5",  55_000, 8.20, is_pipeline=False, fixed_cost=12_000),
    ]

    for a in well_to_storage + storage_to_ref + ref_to_dist + dist_to_demand:
        net.add_arc(a)

    # ── Ship-or-Pay Contracts ─────────────────────────────────────────────────
    # Key long-haul pipeline commitments; below-minimum throughput triggers deficiency charge

    contracts = [
        ShipOrPayContract("T1", "R1", min_daily_volume=35_000,
                          contract_tariff=3.20, deficiency_charge=1.80),
        ShipOrPayContract("T2", "R2", min_daily_volume=25_000,
                          contract_tariff=3.40, deficiency_charge=1.60),
        ShipOrPayContract("T3", "R2", min_daily_volume=20_000,
                          contract_tariff=3.90, deficiency_charge=1.50),
        ShipOrPayContract("T1", "R3", min_daily_volume=18_000,
                          contract_tariff=3.50, deficiency_charge=1.70),
    ]
    for c in contracts:
        net.add_contract(c)

    return net


# ── Scenario modifier functions (used by both CLI and dashboard) ──────────────

def apply_disruption(net: SupplyChainNetwork, well_id: str, pct: float) -> SupplyChainNetwork:
    net.nodes[well_id].max_capacity *= (1 - pct)
    return net

def apply_demand_spike(net: SupplyChainNetwork, market_id: str, mult: float) -> SupplyChainNetwork:
    net.nodes[market_id].demand *= mult
    return net

def apply_refinery_outage(net: SupplyChainNetwork, ref_id: str) -> SupplyChainNetwork:
    net.nodes[ref_id].max_capacity = 0
    net.nodes[ref_id].min_throughput = 0
    return net

def apply_freight_shock(net: SupplyChainNetwork, pct: float) -> SupplyChainNetwork:
    for arc in net.arcs:
        arc.transport_cost *= (1 + pct)
    return net

def apply_carbon_cap(net: SupplyChainNetwork, budget_tco2: float) -> SupplyChainNetwork:
    net.carbon_budget_per_day = budget_tco2
    return net
