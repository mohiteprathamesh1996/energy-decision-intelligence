"""
Core data structures for the oil supply chain network.
Nodes and arcs are typed; capacity, cost, and production parameters
are kept explicit to avoid implicit assumptions in the optimizer.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class NodeType(Enum):
    WELL = "well"
    STORAGE = "storage"
    REFINERY = "refinery"
    DISTRIBUTION = "distribution"
    DEMAND = "demand"


@dataclass
class Node:
    id: str
    name: str
    node_type: NodeType
    latitude: float
    longitude: float

    # Type-specific parameters
    max_capacity: float = 0.0       # bbl/day for wells; bbl for storage tanks
    min_throughput: float = 0.0     # minimum operational flow (refinery/storage)
    operating_cost: float = 0.0     # $/bbl

    # Refinery-specific
    crude_yield: float = 1.0        # fraction of crude converted to refined products
    refinery_margin: float = 0.0    # $/bbl margin above crude cost

    # Demand node
    demand: float = 0.0             # bbl/day required
    unmet_penalty: float = 500.0    # $/bbl penalty for unmet demand

    # Storage
    holding_cost: float = 0.0       # $/bbl/day
    initial_inventory: float = 0.0  # bbl


@dataclass
class Arc:
    origin: str
    destination: str
    capacity: float         # bbl/day
    transport_cost: float   # $/bbl
    transit_days: int = 1   # pipeline/vessel transit time

    # Pipeline-specific
    is_pipeline: bool = True
    fixed_cost: float = 0.0  # $/day if arc is activated


@dataclass
class SupplyChainNetwork:
    nodes: Dict[str, Node] = field(default_factory=dict)
    arcs: List[Arc] = field(default_factory=list)
    planning_horizon: int = 30  # days

    def add_node(self, node: Node):
        self.nodes[node.id] = node

    def add_arc(self, arc: Arc):
        self.arcs.append(arc)

    def get_nodes_by_type(self, node_type: NodeType) -> List[Node]:
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def get_outbound_arcs(self, node_id: str) -> List[Arc]:
        return [a for a in self.arcs if a.origin == node_id]

    def get_inbound_arcs(self, node_id: str) -> List[Arc]:
        return [a for a in self.arcs if a.destination == node_id]

    def arc_key(self, arc: Arc) -> Tuple[str, str]:
        return (arc.origin, arc.destination)

    @property
    def arc_index(self) -> List[Tuple[str, str]]:
        return [self.arc_key(a) for a in self.arcs]

    def arc_lookup(self, origin: str, dest: str) -> Optional[Arc]:
        for a in self.arcs:
            if a.origin == origin and a.destination == dest:
                return a
        return None
