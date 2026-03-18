"""
Domain model for the oil supply chain optimization platform.

Key additions over a basic network model:
- Crude grade differentiation (API gravity, sulfur content)
- Refinery crude diet compatibility windows
- Ship-or-pay pipeline contracts with deficit tracking
- Carbon intensity per grade for emissions accounting
- Multi-period planning horizon with date awareness
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Dict, List, Optional, Tuple


class NodeType(Enum):
    WELL = "well"
    STORAGE = "storage"
    REFINERY = "refinery"
    DISTRIBUTION = "distribution"
    DEMAND = "demand"


@dataclass
class CrudeGrade:
    id: str
    name: str
    api_gravity: float       # degrees API (higher = lighter)
    sulfur_content: float    # weight percent (lower = sweeter)
    price_differential: float  # $/bbl vs. WTI benchmark (negative = discount)
    carbon_intensity: float  # kg CO2e/bbl (upstream: production + gathering)


@dataclass
class CrudeDiet:
    """
    Defines the crude processing envelope for a refinery.
    Expressed as API gravity and sulfur bounds. The optimizer enforces
    these as linearized blend-quality constraints.
    """
    api_min: float
    api_max: float
    sulfur_max: float        # hard limit from desulfurization capacity
    optimal_api: float       # for reference; not enforced in model


@dataclass
class ShipOrPayContract:
    """
    Pipeline volume commitment. Producer pays a fixed tariff on committed
    volume regardless of actual throughput. Shortfall triggers a per-barrel
    deficiency charge on top of the fixed tariff.
    """
    arc_origin: str
    arc_dest: str
    min_daily_volume: float   # bbl/day committed
    contract_tariff: float    # $/bbl (typically below spot tariff)
    deficiency_charge: float  # $/bbl below minimum


@dataclass
class Node:
    id: str
    name: str
    node_type: NodeType
    latitude: float
    longitude: float

    # General
    max_capacity: float = 0.0
    min_throughput: float = 0.0
    operating_cost: float = 0.0       # $/bbl processed/produced

    # Refinery
    crude_diet: Optional[CrudeDiet] = None
    refinery_margin: float = 0.0      # base $/bbl crack spread contribution
    crude_yield: float = 1.0          # bbl refined products per bbl crude

    # Demand
    demand: float = 0.0               # bbl/day
    unmet_penalty: float = 500.0      # $/bbl

    # Storage
    holding_cost: float = 0.0         # $/bbl/day
    initial_inventory: float = 0.0    # total initial bbl (split by grade in data layer)
    initial_inv_by_grade: Dict[str, float] = field(default_factory=dict)

    # Well-specific
    primary_grade: Optional[str] = None    # grade ID produced by this well
    decline_rate: float = 0.0             # fractional daily decline (for multi-period)

    # Carbon
    flare_rate: float = 0.0          # kg CO2e/bbl produced (flaring/venting)


@dataclass
class Arc:
    origin: str
    destination: str
    capacity: float          # bbl/day max throughput
    transport_cost: float    # $/bbl variable tariff
    transit_days: int = 1
    is_pipeline: bool = True
    fixed_cost: float = 0.0  # $/day if arc is activated (infrastructure cost)

    # Pipeline batch scheduling (qualitative; affects grade routing logic)
    accepts_sour: bool = True   # False = sweet-only pipeline


@dataclass
class SupplyChainNetwork:
    nodes: Dict[str, Node] = field(default_factory=dict)
    arcs: List[Arc] = field(default_factory=list)
    grades: Dict[str, CrudeGrade] = field(default_factory=dict)
    contracts: List[ShipOrPayContract] = field(default_factory=list)
    planning_horizon: int = 14     # days
    start_date: Optional[date] = None
    carbon_budget_per_day: Optional[float] = None  # tonnes CO2e/day (None = unconstrained)

    def add_node(self, node: Node): self.nodes[node.id] = node
    def add_arc(self, arc: Arc): self.arcs.append(arc)
    def add_grade(self, grade: CrudeGrade): self.grades[grade.id] = grade
    def add_contract(self, c: ShipOrPayContract): self.contracts.append(c)

    def get_nodes_by_type(self, t: NodeType) -> List[Node]:
        return [n for n in self.nodes.values() if n.node_type == t]

    def arc_key(self, arc: Arc) -> Tuple[str, str]:
        return (arc.origin, arc.destination)

    @property
    def arc_index(self) -> List[Tuple[str, str]]:
        return [self.arc_key(a) for a in self.arcs]

    def arc_lookup(self, origin: str, dest: str) -> Optional[Arc]:
        return next((a for a in self.arcs if a.origin == origin and a.destination == dest), None)

    def contract_lookup(self, origin: str, dest: str) -> Optional[ShipOrPayContract]:
        return next((c for c in self.contracts if c.arc_origin == origin and c.arc_dest == dest), None)

    def grade_ids(self) -> List[str]:
        return list(self.grades.keys())

    def well_capacity(self, well_id: str, period: int) -> float:
        """Apply production decline for multi-period planning."""
        n = self.nodes[well_id]
        return n.max_capacity * ((1 - n.decline_rate) ** (period - 1))
