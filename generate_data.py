"""
Generates a realistic Gulf Coast / Permian Basin-inspired supply chain network.
Node capacities, costs, and margins are calibrated to reflect actual industry order-of-magnitude values.
"""

from src.model.supply_chain import Arc, Node, NodeType, SupplyChainNetwork


def build_base_network() -> SupplyChainNetwork:
    net = SupplyChainNetwork(planning_horizon=30)

    # ---- Production Wells (West Texas / Permian Basin cluster) ----
    wells = [
        Node("W1", "Midland Basin North",  NodeType.WELL, 31.9, -102.1, max_capacity=45_000, operating_cost=18.50),
        Node("W2", "Midland Basin South",  NodeType.WELL, 31.5, -102.4, max_capacity=38_000, operating_cost=19.20),
        Node("W3", "Delaware Basin",        NodeType.WELL, 31.2, -103.1, max_capacity=52_000, operating_cost=17.80),
        Node("W4", "Wolfcamp Shale",        NodeType.WELL, 32.1, -101.9, max_capacity=28_000, operating_cost=22.10),
        Node("W5", "Bone Spring",           NodeType.WELL, 31.7, -103.5, max_capacity=33_000, operating_cost=20.40),
    ]

    # ---- Midstream Storage Terminals ----
    terminals = [
        Node("T1", "Midland Hub",     NodeType.STORAGE, 31.9, -102.0, max_capacity=2_500_000,
             holding_cost=0.08, initial_inventory=400_000, operating_cost=0.50),
        Node("T2", "Odessa Terminal", NodeType.STORAGE, 31.8, -102.3, max_capacity=1_800_000,
             holding_cost=0.10, initial_inventory=250_000, operating_cost=0.45),
        Node("T3", "Crane Facility",  NodeType.STORAGE, 31.4, -102.3, max_capacity=1_200_000,
             holding_cost=0.09, initial_inventory=180_000, operating_cost=0.52),
    ]

    # ---- Refineries (Gulf Coast) ----
    refineries = [
        Node("R1", "Houston Refinery",   NodeType.REFINERY, 29.7, -95.4,
             max_capacity=120_000, min_throughput=40_000,
             operating_cost=8.50, crude_yield=0.92, refinery_margin=14.20),
        Node("R2", "Port Arthur Ref.",   NodeType.REFINERY, 29.9, -93.9,
             max_capacity=95_000, min_throughput=30_000,
             operating_cost=9.10, crude_yield=0.90, refinery_margin=12.80),
        Node("R3", "Beaumont Complex",   NodeType.REFINERY, 30.1, -94.1,
             max_capacity=75_000, min_throughput=25_000,
             operating_cost=8.80, crude_yield=0.91, refinery_margin=13.50),
    ]

    # ---- Distribution Hubs ----
    distribution = [
        Node("D1", "Houston Export Hub",    NodeType.DISTRIBUTION, 29.7, -95.0),
        Node("D2", "Gulf Coast Pipeline HQ", NodeType.DISTRIBUTION, 30.0, -94.5),
    ]

    # ---- Demand Centers ----
    demand_nodes = [
        Node("M1", "Chicago Market",    NodeType.DEMAND, 41.9, -87.6,
             demand=55_000, unmet_penalty=500),
        Node("M2", "New York Market",   NodeType.DEMAND, 40.7, -74.0,
             demand=48_000, unmet_penalty=550),
        Node("M3", "Dallas Market",     NodeType.DEMAND, 32.8, -96.8,
             demand=35_000, unmet_penalty=480),
        Node("M4", "Export Terminal A", NodeType.DEMAND, 29.3, -94.8,
             demand=40_000, unmet_penalty=420),
        Node("M5", "Los Angeles Market",NodeType.DEMAND, 34.0, -118.2,
             demand=42_000, unmet_penalty=510),
    ]

    for n in wells + terminals + refineries + distribution + demand_nodes:
        net.add_node(n)

    # ---- Pipeline Arcs: Wells -> Storage ----
    well_to_storage = [
        Arc("W1", "T1", capacity=45_000, transport_cost=2.10),
        Arc("W1", "T2", capacity=20_000, transport_cost=2.40),
        Arc("W2", "T2", capacity=38_000, transport_cost=1.90),
        Arc("W2", "T3", capacity=15_000, transport_cost=2.60),
        Arc("W3", "T1", capacity=30_000, transport_cost=3.20),
        Arc("W3", "T3", capacity=52_000, transport_cost=2.30),
        Arc("W4", "T1", capacity=28_000, transport_cost=2.80),
        Arc("W4", "T2", capacity=18_000, transport_cost=3.10),
        Arc("W5", "T2", capacity=25_000, transport_cost=2.50),
        Arc("W5", "T3", capacity=33_000, transport_cost=2.20),
    ]

    # ---- Pipeline Arcs: Storage -> Refineries ----
    storage_to_refinery = [
        Arc("T1", "R1", capacity=80_000, transport_cost=3.80),
        Arc("T1", "R2", capacity=60_000, transport_cost=4.20),
        Arc("T1", "R3", capacity=45_000, transport_cost=4.10),
        Arc("T2", "R1", capacity=70_000, transport_cost=3.60),
        Arc("T2", "R2", capacity=55_000, transport_cost=3.90),
        Arc("T3", "R2", capacity=50_000, transport_cost=4.50),
        Arc("T3", "R3", capacity=40_000, transport_cost=4.00),
    ]

    # ---- Refined Product Arcs: Refineries -> Distribution ----
    refinery_to_dist = [
        Arc("R1", "D1", capacity=120_000, transport_cost=1.20),
        Arc("R1", "D2", capacity=80_000,  transport_cost=1.50),
        Arc("R2", "D1", capacity=60_000,  transport_cost=1.80),
        Arc("R2", "D2", capacity=95_000,  transport_cost=1.30),
        Arc("R3", "D2", capacity=75_000,  transport_cost=1.40),
    ]

    # ---- Distribution -> Demand (pipeline + tanker, some with fixed activation costs) ----
    dist_to_demand = [
        Arc("D1", "M1", capacity=70_000, transport_cost=5.50, is_pipeline=False, fixed_cost=0),
        Arc("D1", "M2", capacity=60_000, transport_cost=6.20, is_pipeline=False, fixed_cost=0),
        Arc("D1", "M4", capacity=50_000, transport_cost=2.10),
        Arc("D2", "M1", capacity=55_000, transport_cost=5.80, is_pipeline=False, fixed_cost=0),
        Arc("D2", "M3", capacity=45_000, transport_cost=3.40),
        Arc("D2", "M4", capacity=40_000, transport_cost=2.40),
        Arc("D2", "M5", capacity=55_000, transport_cost=8.20, is_pipeline=False, fixed_cost=12_000),
        Arc("D1", "M5", capacity=50_000, transport_cost=8.80, is_pipeline=False, fixed_cost=14_000),
        Arc("D1", "M3", capacity=40_000, transport_cost=3.80),
    ]

    for a in well_to_storage + storage_to_refinery + refinery_to_dist + dist_to_demand:
        net.add_arc(a)

    return net


def apply_demand_spike(net: SupplyChainNetwork, node_id: str, multiplier: float) -> SupplyChainNetwork:
    """Increase demand at a specific market node by a multiplier."""
    if node_id in net.nodes:
        net.nodes[node_id].demand *= multiplier
    return net


def apply_supply_disruption(net: SupplyChainNetwork, well_id: str, reduction_pct: float) -> SupplyChainNetwork:
    """Reduce well production capacity (e.g., equipment failure, weather event)."""
    if well_id in net.nodes:
        net.nodes[well_id].max_capacity *= (1 - reduction_pct)
    return net


def apply_cost_shock(net: SupplyChainNetwork, transport_increase_pct: float) -> SupplyChainNetwork:
    """Simulate a freight/pipeline cost increase across all arcs."""
    for arc in net.arcs:
        arc.transport_cost *= (1 + transport_increase_pct)
    return net


def apply_refinery_outage(net: SupplyChainNetwork, refinery_id: str) -> SupplyChainNetwork:
    """Take a refinery offline (set capacity to zero)."""
    if refinery_id in net.nodes:
        net.nodes[refinery_id].max_capacity = 0
        net.nodes[refinery_id].min_throughput = 0
    return net
