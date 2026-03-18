# Oil Supply Chain Optimization Platform

---

## Executive Summary

Crude oil economics are sensitive to routing decisions that get made under imperfect information. The difference between an optimal and a suboptimal allocation across a 5-well, 3-refinery network can exceed **$800K/day** — not from price swings, but from how flows are scheduled across constrained infrastructure.

This platform solves a Mixed-Integer Linear Program (MILP) over a Permian Basin → Gulf Coast supply chain network. It finds the daily production routing, storage allocation, and refinery loading schedule that maximizes net margin, then lets operators stress-test that plan against supply disruptions, demand spikes, refinery outages, and freight cost shocks.

The model is industry-calibrated: well capacities, pipeline costs, refinery margins, and demand volumes reflect Permian/Gulf Coast order-of-magnitude benchmarks. Output is suitable for operations planning, midstream contract evaluation, and capital allocation reviews.

---

## System Overview

### The Supply Chain

A barrel of West Texas crude follows a predictable but congested path:

1. **Lifted** from a shale well (Midland Basin, Delaware Basin, Wolfcamp, Bone Spring)
2. **Gathered** via pipeline to a midstream storage terminal (Midland Hub, Odessa, Crane)
3. **Shipped** via long-haul pipeline to a Gulf Coast refinery (Houston, Port Arthur, Beaumont)
4. **Refined** and moved to a distribution hub
5. **Delivered** to end markets (Chicago, New York, Dallas, Los Angeles, export terminal)

Each step has capacity constraints, operating costs, and — at the refinery — a margin uplift that converts crude into refined product value. The total throughput across this network on any given day is constrained by the **tightest bottleneck**, not the average.

The decision problem is: given today's production limits, refinery slots, storage levels, and market demand, **how should we route flows to maximize net margin?**

### Modeling Philosophy

The network is modeled as a directed graph where each node has a type (well, storage, refinery, distribution, demand) and each arc has capacity and per-barrel transport cost. Binary variables activate arcs with fixed infrastructure costs. The single-period formulation gives you one optimal plan per solve; wrapping it in a rolling horizon (via the scenario runner) enables multi-period operational planning.

---

## Mathematical Formulation

### Sets

| Symbol | Definition |
|--------|-----------|
| $\mathcal{N}$ | Set of all nodes |
| $\mathcal{W} \subseteq \mathcal{N}$ | Production wells |
| $\mathcal{S} \subseteq \mathcal{N}$ | Storage terminals |
| $\mathcal{R} \subseteq \mathcal{N}$ | Refineries |
| $\mathcal{D} \subseteq \mathcal{N}$ | Demand (market) nodes |
| $\mathcal{A} \subseteq \mathcal{N} \times \mathcal{N}$ | Set of arcs (pipelines / vessels) |
| $\mathcal{A}^F \subseteq \mathcal{A}$ | Arcs with fixed activation costs |

### Parameters

| Symbol | Units | Description |
|--------|-------|-------------|
| $P_i$ | bbl/day | Max production capacity at well $i \in \mathcal{W}$ |
| $Q_r^{\min}, Q_r^{\max}$ | bbl/day | Refinery throughput bounds for $r \in \mathcal{R}$ |
| $C_{ij}$ | bbl/day | Arc capacity |
| $\tau_{ij}$ | \$/bbl | Variable transport cost |
| $f_{ij}$ | \$/day | Fixed cost to activate arc $(i,j) \in \mathcal{A}^F$ |
| $m_r$ | \$/bbl | Refinery margin (product value less crude cost) |
| $o_i$ | \$/bbl | Node operating cost |
| $h_i$ | \$/bbl/day | Holding cost at storage $i \in \mathcal{S}$ |
| $d_k$ | bbl/day | Demand at market node $k \in \mathcal{D}$ |
| $\pi_k$ | \$/bbl | Unmet demand penalty at $k$ |
| $I_i^0$ | bbl | Initial inventory at storage $i$ |
| $I_i^{\max}$ | bbl | Storage capacity ceiling |

### Decision Variables

$$
x_{ij} \geq 0 \quad \forall (i,j) \in \mathcal{A} \qquad \text{(flow, bbl/day)}
$$

$$
s_i \geq 0 \quad \forall i \in \mathcal{S} \qquad \text{(end-of-period inventory, bbl)}
$$

$$
u_k \geq 0 \quad \forall k \in \mathcal{D} \qquad \text{(unmet demand slack, bbl/day)}
$$

$$
y_{ij} \in \{0, 1\} \quad \forall (i,j) \in \mathcal{A}^F \qquad \text{(arc activation binary)}
$$

### Objective

Maximize net daily margin:

$$
\max \quad \underbrace{\sum_{r \in \mathcal{R}} m_r \sum_{(i,r) \in \mathcal{A}} x_{ir}}_{\text{refinery revenue}} \;-\; \underbrace{\sum_{(i,j) \in \mathcal{A}} \tau_{ij} \, x_{ij}}_{\text{transport}} \;-\; \underbrace{\sum_{i \in \mathcal{N}} o_i \sum_{(i,j) \in \mathcal{A}} x_{ij}}_{\text{operating}} \;-\; \underbrace{\sum_{i \in \mathcal{S}} h_i \, s_i}_{\text{holding}} \;-\; \underbrace{\sum_{(i,j) \in \mathcal{A}^F} f_{ij} \, y_{ij}}_{\text{fixed}} \;-\; \underbrace{\sum_{k \in \mathcal{D}} \pi_k \, u_k}_{\text{penalty}}
$$

### Constraints

**Well production capacity:**
$$
\sum_{(i,j) \in \mathcal{A}} x_{ij} \leq P_i \qquad \forall i \in \mathcal{W}
$$

**Refinery throughput bounds:**
$$
Q_r^{\min} \leq \sum_{(i,r) \in \mathcal{A}} x_{ir} \leq Q_r^{\max} \qquad \forall r \in \mathcal{R}
$$

**Storage flow balance:**
$$
\sum_{(i,k) \in \mathcal{A}} x_{ik} + I_k^0 = \sum_{(k,j) \in \mathcal{A}} x_{kj} + s_k \qquad \forall k \in \mathcal{S}
$$

**Storage capacity:**
$$
s_k \leq I_k^{\max} \qquad \forall k \in \mathcal{S}
$$

**Demand satisfaction (with penalty slack):**
$$
\sum_{(i,k) \in \mathcal{A}} x_{ik} + u_k \geq d_k \qquad \forall k \in \mathcal{D}
$$

**Arc capacity:**
$$
x_{ij} \leq C_{ij} \qquad \forall (i,j) \in \mathcal{A}
$$

**Arc activation (big-M linking):**
$$
x_{ij} \leq C_{ij} \cdot y_{ij} \qquad \forall (i,j) \in \mathcal{A}^F
$$

**Distribution node flow conservation:**
$$
\sum_{(i,k) \in \mathcal{A}} x_{ik} = \sum_{(k,j) \in \mathcal{A}} x_{kj} \qquad \forall k \in \mathcal{N} \setminus (\mathcal{W} \cup \mathcal{S} \cup \mathcal{R} \cup \mathcal{D})
$$

---

## System Architecture

```
oil-supply-chain/
├── src/
│   ├── model/
│   │   ├── supply_chain.py   # Domain model: Node, Arc, Network
│   │   └── optimizer.py      # Pyomo MILP formulation + solve
│   ├── analysis/
│   │   └── scenario.py       # Scenario runner, deep-copy isolation
│   └── viz/
│       └── charts.py         # Plotly charts for Streamlit
├── data/
│   └── generate_data.py      # Network builder + scenario modifiers
├── app/
│   └── dashboard.py          # Streamlit UI
├── run.py                    # CLI entry point
├── Dockerfile
└── requirements.txt
```

**Why this structure?**

The optimization model knows nothing about the UI, and the UI knows nothing about Pyomo. `supply_chain.py` defines the domain in pure Python dataclasses — no solver dependency. The optimizer receives a `SupplyChainNetwork` and returns an `OptimizationResult`; it's fully decoupled from how data is generated or displayed.

Scenario analysis deep-copies the base network before applying modifiers. This means scenarios are stateless and safely composable — you can stack a supply disruption on top of a demand spike without mutation side effects.

The dashboard caches the base network with `@st.cache_resource` and reruns the optimizer only when sidebar controls change, keeping interactive response time under a second for typical network sizes.

**Trade-offs made:**
- Single-period MILP rather than multi-period stochastic program. The rolling-horizon wrapper in `ScenarioRunner` gives multi-day insight without the exponential complexity of a full stochastic model.
- CBC over CPLEX/Gurobi. Open-source solver keeps the project self-contained. The solver interface is abstracted — swapping to Gurobi requires one constructor argument.
- No spatial routing (shortest path). Arc costs are pre-computed; optimal path selection within each arc segment is outside scope here.

---

## Implementation Details

### Solver Choice

CBC (COIN-OR Branch and Cut) is used by default. For this network size (~150 variables, ~120 constraints), CBC typically returns an optimal solution in under 2 seconds with a 0.1% optimality gap.

For larger instances (500+ nodes, multi-period horizon), the recommended path is:

1. Relax integrality (LP relaxation) for bounds
2. Run CBC with a tighter time limit and 0.5% gap tolerance
3. Switch to Gurobi for production environments where license is available

Pyomo's solver abstraction makes this a one-line change:

```python
opt = SupplyChainOptimizer(network, solver="gurobi")
```

### Modeling Decisions

**Why MILP and not LP?**

Some arcs have fixed infrastructure costs (vessel chartering, pipeline activation fees). These costs are binary — you either pay the fixed cost or you don't route that arc at all. Relaxing to LP would overestimate the solution by allowing fractional arc activation, underpricing fixed costs.

**Unmet demand as a penalty, not a hard constraint**

Forcing full demand satisfaction can make the model infeasible during disruptions — exactly when you need insight most. The soft penalty formulation always produces a feasible solution. The penalty coefficient ($500/bbl default) is high enough that the solver aggressively avoids unmet demand, but the model won't crash during a 60% supply disruption.

**Refinery minimum throughput**

Refineries have minimum economic operating rates (typically 35–40% of nameplate capacity). Below that, fixed costs dominate and operations become cash-negative. The `min_throughput` parameter captures this.

---

## Scenario Analysis

The scenario runner provides a structured way to quantify **downside risk and opportunity cost** without modifying the base model.

Each scenario is defined by a pure function that takes a network and returns a modified network:

```python
Scenario(
    name="Hurricane Disruption",
    description="W3 Delaware Basin reduced 60%",
    modifier=lambda net: apply_supply_disruption(net, "W3", 0.60)
)
```

Scenarios are composable:

```python
modifier=lambda net: apply_demand_spike(
    apply_supply_disruption(net, "W3", 0.40), "M1", 1.30
)
```

### Standard Scenario Set

| Scenario | Trigger | Expected Impact |
|----------|---------|-----------------|
| Hurricane Disruption | W3 Delaware Basin −60% | Flow reroutes through T1/T2; R2/R3 underloaded |
| Chicago Demand Spike | M1 demand +40% | D1 saturates; unmet demand likely if no reserve |
| Port Arthur Outage | R2 offline | R1 absorbs overflow; T3 inventory builds |
| Freight Cost +25% | All arc costs +25% | Lower-cost arcs preferred; some long routes dropped |
| Dual Disruption | W3 −40% + Chicago +30% | Stress test; unmet demand threshold breached |

**What to look for:**

- If a scenario produces large unmet demand, the bottleneck is supply-side — more well capacity or pipeline capacity is needed.
- If objective drops sharply with a cost shock but unmet demand is low, the network is sensitive to routing cost, not volume. Hedging via fixed-rate contracts would reduce exposure.
- Scenarios where service level stays at 100% despite disruption indicate network redundancy — potentially over-invested in redundant capacity.

---

## Results & Insights

In the base case, the optimizer consistently routes:

1. **W3 Delaware Basin** at full capacity — lowest operating cost ($17.80/bbl) and well-positioned for T3 → R2 routing with competitive combined cost
2. **R1 Houston** loaded near maximum — highest refinery margin ($14.20/bbl) makes it the model's preferred sink for any barrel that can reach it within arc capacity limits
3. **T1 Midland Hub** acts as the primary aggregation point — its connectivity (feeds all three refineries) makes it the natural flow concentrator
4. **M5 Los Angeles** is the last market served — highest combined transport cost pushes it to the margin; it receives flow only when domestic markets are covered

The Chicago scenario reveals the network's structural constraint: **the D1 → M1 arc is the binding bottleneck**. At +40% demand, even with all production at capacity, D1 cannot push enough volume north. The actionable insight is that incremental capacity on D1 → M1 has a higher return than additional production — the well capacity is already underutilized in this scenario.

The freight cost shock scenario shows a counterintuitive result: **total volume through R1 decreases** despite R1 having the highest margin. When all arc costs rise proportionally, the relative advantage of routing to R1 (which sits at the end of the longest pipelines) diminishes. R3 Beaumont, closer to the storage terminals, gains share.

---

## How to Run

### Option 1 — Docker (recommended)

```bash
docker build -t oil-optimizer .
docker run -p 8501:8501 oil-optimizer
# Open http://localhost:8501
```

### Option 2 — Local

```bash
# Install CBC solver first:
# macOS: brew install coin-or-tools
# Ubuntu: apt install coinor-cbc

pip install -r requirements.txt
streamlit run app/dashboard.py
```

### Option 3 — CLI (no UI)

```bash
# Base case only
python run.py

# Base + all scenarios
python run.py --scenario all

# Use GLPK instead of CBC
python run.py --solver glpk
```

### Python API

```python
from data.generate_data import build_base_network, apply_supply_disruption
from src.model.optimizer import SupplyChainOptimizer

net = build_base_network()
apply_supply_disruption(net, "W3", 0.50)

opt = SupplyChainOptimizer(net)
result = opt.solve()

print(f"Net margin: ${result.objective_value:,.0f}/day")
print(f"Service level: {(1 - sum(result.unmet_demand.values()) / 220_000) * 100:.1f}%")
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pyomo >= 6.7` | MILP modeling layer |
| CBC (system) | Branch-and-cut solver |
| `streamlit >= 1.35` | Dashboard |
| `plotly >= 5.22` | Charts and network map |
| `pandas >= 2.2` | Tabular output |

---

*Network data is synthetic but calibrated to reflect Permian Basin / Gulf Coast operational parameters. This is an academic/portfolio model — production deployment would require integration with SCADA, real-time pricing feeds, and a licensed solver.*
