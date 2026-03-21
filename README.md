# Oil Supply Chain Optimization Platform

A decision-support system for crude oil logistics from **Permian Basin wells to Gulf Coast refineries**. Uses mathematical optimization (MILP) to decide how much oil to move, where, and when — accounting for crude quality constraints, pipeline contracts, carbon limits, and price uncertainty.

---

## What This Does

Crude oil from Permian Basin wells travels ~350 miles to Gulf Coast refineries through a network of gathering pipelines, storage terminals, and long-haul interstate pipelines. At each step, there are constraints: pipelines have capacity limits, refineries can only process certain grades of crude, and contracts require minimum shipment volumes.

This platform solves that routing problem mathematically, across a multi-day planning horizon, in under a second. It also answers higher-order questions:

- **What happens if a major well goes offline?** → Scenario analysis
- **How much is a reliable price forecast worth?** → EVPI / VSS from stochastic programming
- **What does adaptive daily re-planning buy us vs. a fixed plan?** → Rolling horizon MPC
- **What is the implied cost of a carbon cap?** → Pareto frontier analysis

---

## Network

| Layer | Nodes | Details |
|-------|-------|---------|
| Production | 5 wells | Midland Basin N/S, Delaware Basin, Wolfcamp Shale, Bone Spring |
| Storage | 3 terminals | Midland Hub, Odessa Terminal, Crane Facility — ~7M bbl combined |
| Refining | 3 refineries | Houston (light sweet), Port Arthur (coking), Beaumont (mid-range) |
| Distribution | 2 hubs | Houston Products Hub, Gulf Coast Pipeline Hub |
| Demand | 5 markets | Chicago, New York, Dallas/Fort Worth, Los Angeles, Freeport Export |
| Transport | 34 arcs | Interstate pipelines + product tankers; 4 ship-or-pay contracts |

**Crude grades:** WTI (API 40.8, low sulfur), WTS (API 34.0, sour), Permian Heavy (API 27.5, high sulfur/carbon)

---

## What the Optimizer Decides

At every time period, for every crude grade, on every pipeline arc: **how many barrels to move**.

Subject to:
- Well production capacity (with exponential decline over horizon)
- Pipeline and vessel capacity
- Refinery crude diet: API gravity and sulfur bounds (linearized blending constraints)
- Storage capacity and grade-segregated inventory balance
- Ship-or-pay minimum volume commitments
- Soft demand satisfaction (unmet demand incurs a penalty cost)
- Optional: hard carbon budget per day

The objective maximizes crack spread revenue minus lifting costs, transport tariffs, holding costs, fixed arc activation costs, ship-or-pay deficiency charges, and unmet demand penalties.

---

## Mathematical Formulation

### Decision Variables

| Variable | Description |
|----------|-------------|
| $x_{ij}^{tg} \geq 0$ | Flow on arc $(i,j)$ in period $t$ for grade $g$ (bbl/day) |
| $s_i^{tg} \geq 0$ | End-of-period inventory at storage node $i$ for grade $g$ (bbl) |
| $u_k^t \geq 0$ | Unmet demand slack at market $k$ in period $t$ (bbl/day) |
| $\delta_{ij}^t \geq 0$ | Ship-or-pay deficiency on arc $(i,j)$ in period $t$ (bbl/day) |
| $y_{ij} \in \{0,1\}$ | Arc activation binary for fixed-cost arcs |

### Objective

$$\max \underbrace{\sum_{t,r,g} m_r^t \sum_{(i,r)} x_{ir}^{tg}}_{\text{crack spread revenue}} - \underbrace{\sum_{t,n} c_n \sum_{(n,j)} x_{nj}^{tg}}_{\text{lifting + opex}} - \underbrace{\sum_{t,(i,j),g} \tau_{ij} x_{ij}^{tg}}_{\text{transport}} - \underbrace{\sum_{t,i,g} h_i s_i^{tg}}_{\text{holding}} - \underbrace{\sum_{(i,j)} f_{ij} y_{ij}}_{\text{fixed}} - \underbrace{\sum_{t,(i,j)} \rho_{ij} \delta_{ij}^t}_{\text{SOP deficit}} - \underbrace{\sum_{t,k} \pi_k u_k^t}_{\text{unmet penalty}}$$

### Key Constraints

**Well production with decline:**
$$\sum_{j} x_{nj}^{tg} \leq P_n \cdot (1-\lambda_n)^{t-1}$$

**Refinery crude diet — API gravity** (linearized blending):
$$\alpha^{\min}_r \sum_{i,g} x_{ir}^{tg} \leq \sum_{i,g} \alpha_g x_{ir}^{tg} \leq \alpha^{\max}_r \sum_{i,g} x_{ir}^{tg}$$

The bilinear weighted-average constraint is linearized by multiplying through by total throughput (valid since refinery minimum throughput is enforced, keeping the denominator positive).

**Sulfur tolerance:**
$$\sum_{i,g} \sigma_g x_{ir}^{tg} \leq \sigma^{\max}_r \sum_{i,g} x_{ir}^{tg}$$

**Ship-or-pay:**
$$\sum_g x_{ij}^{tg} + \delta_{ij}^t \geq V^{\min}_{ij}$$

**Carbon budget (optional hard constraint per period):**
$$\frac{1}{1000} \sum_{(i,j),g} e_g x_{ij}^{tg} \leq E^{\text{budget}}$$

---

## Two-Stage Stochastic Programming

Crack spread volatility (≈22% annualized) means a plan built on expected prices can be suboptimal. The stochastic extension answers: *how much does that cost?*

**Stage 1 (commit before seeing prices):** arc activation binaries $y_{ij}$

**Stage 2 (optimize after observing scenario):** all routing flows $x^{\omega}$

Crack spread scenarios are log-normal GBM paths:
$$m_r^{t,\omega} = m_r^0 \cdot \exp\!\left(-\tfrac{1}{2}\sigma^2 t + \sigma\sqrt{t}\, Z^\omega\right), \quad Z^\omega \sim \mathcal{N}(0,1)$$

**Decision-theory metrics:**

| Metric | Formula | Meaning |
|--------|---------|---------|
| WS | $\sum_\omega p_\omega z^*(\omega)$ | Value with perfect price foresight |
| RP | Extensive form optimum | Value of stochastic optimization |
| EEV | $\sum_\omega p_\omega z(\bar{x}, \omega)$ | Value of deterministic mean-price plan |
| EVPI | WS − RP | Ceiling on what a perfect forecast is worth |
| VSS | RP − EEV | Cost of ignoring uncertainty |

The inequality chain WS ≥ RP ≥ EEV always holds (Jensen's inequality).

---

## Analyses

### Scenario Analysis (9 scenarios)

| Category | Scenario | What it tests |
|----------|----------|---------------|
| Supply | Hurricane – Delaware Basin (−60%) | Well disruption, T3 inventory buffer adequacy |
| Supply | Wolfcamp Decline Acceleration | Shale decline sensitivity |
| Supply | Dual Disruption | W3 −50% + Chicago +35% stress test |
| Demand | Chicago Demand Surge (+40%) | Winter peak demand |
| Infrastructure | Port Arthur Refinery Outage | Sour crude stranding risk |
| Infrastructure | T1–R1 Pipeline Expansion | Capacity investment value |
| Cost | Freight Cost +25% | Tariff shock, refinery share shift |
| Cost | Houston Crack Spread +20% | Margin sensitivity |
| Policy | Carbon Budget (~22% below base) | Implicit carbon cost, grade substitution |

### Sensitivity / Tornado Chart

Parametric ±10% perturbation on crack spreads, well capacities, transport costs, and demand. Results sorted by total objective swing — shows which parameters management should monitor most closely.

### Rolling Horizon (MPC) Simulation

Simulates 30 days of the real planning loop:
1. Solve 7-day lookahead MILP
2. Execute Day 1 only
3. Inject noise: production ±5%, demand ±8%, crack spread daily volatility
4. Update state, re-plan

**Replanning value** = realized rolling-horizon margin − realized static-plan margin. Quantifies what daily re-optimization is worth versus holding the original plan.

### Carbon–Margin Pareto Frontier

Steps the carbon budget from 100% down to 55% of the base-case emissions level. Records the objective penalty at each level and computes the **implicit carbon price** ($/tCO₂e) — the shadow price of the carbon constraint. Useful for evaluating whether a carbon offset purchase or fuel switch is economically justified.

---

## System Architecture

```
energy-decision-intelligence/
├── src/
│   ├── model/
│   │   ├── supply_chain.py      # Domain: Node, Arc, CrudeGrade, ShipOrPayContract
│   │   ├── optimizer.py         # Multi-period MILP (PuLP + bundled CBC)
│   │   └── stochastic.py        # Two-stage stochastic + EVPI/VSS
│   ├── analysis/
│   │   ├── scenario.py          # 9 deterministic what-if scenarios
│   │   ├── sensitivity.py       # Parametric perturbation + bottleneck analysis
│   │   ├── rolling_horizon.py   # MPC-style 30-day simulation
│   │   └── demand_forecast.py   # SARIMA-style demand forecasting
│   └── viz/
│       ├── charts.py            # Plotly figures (network map, waterfall, grade mix, etc.)
│       └── rolling_charts.py    # Rolling horizon charts + forecast fan
├── data/
│   ├── generate_data.py         # Network builder + scenario modifiers
│   └── market_data.py           # EIA-calibrated price and scenario generator
├── app/
│   └── dashboard.py             # Streamlit UI (5 tabs)
├── notebooks/
│   └── analysis.ipynb           # Full analysis walkthrough
├── tests/                       # Unit + integration tests
├── run.py                       # CLI entry point
├── Dockerfile
└── requirements.txt
```

**Design rationale:**

`supply_chain.py` is a pure-Python dataclass layer with zero solver dependency. The optimizer receives a `SupplyChainNetwork` and returns a `MultiPeriodResult` — it does not know where the data came from or how results will be displayed. This lets the same optimizer run from CLI, Streamlit dashboard, notebook, or batch job without modification.

Grade-indexed flows ($x_{ij}^{tg}$) triple the variable count versus a grade-agnostic model, but are necessary to enforce refinery crude diet constraints. The linearized blending constraints (API and sulfur) keep the model as LP/MIP instead of MINLP, which is critical for solve-time performance.

---

## How to Run

### Docker (recommended — no solver install needed)
```bash
docker build -t oil-optimizer .
docker run -p 8501:8501 oil-optimizer
# Open http://localhost:8501
```

### Local
```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```

### CLI
```bash
python run.py                          # 14-day base case
python run.py --mode scenarios         # + all 9 scenarios
python run.py --mode stochastic        # + two-stage stochastic
python run.py --mode full              # everything
python run.py --horizon 30
```

### Python API
```python
from data.generate_data import build_base_network
from src.model.optimizer import MultiPeriodOptimizer
from src.model.stochastic import run_stochastic_analysis

net = build_base_network(horizon=14)
result = MultiPeriodOptimizer(net).solve()
print(f"Net margin: ${result.objective_value:,.0f}")
print(f"Service level: {1 - sum(result.unmet_demand.values()) / (sum(n.demand for n in net.get_nodes_by_type('DEMAND')) * 14):.1%}")

sr = run_stochastic_analysis(net, n_scenarios=10)
print(f"EVPI: ${sr.evpi:,.0f}  |  VSS: ${sr.vss:,.0f}")
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pulp` ≥ 2.8 | MILP modeling; bundles CBC solver (no system install needed) |
| `numpy` ≥ 1.26 | Stochastic scenario generation, risk metrics |
| `pandas` ≥ 2.2 | Tabular output, scenario DataFrames |
| `plotly` ≥ 5.22 | Interactive visualizations |
| `streamlit` ≥ 1.35 | Web dashboard |

To use Gurobi instead of CBC: replace `pulp.PULP_CBC_CMD(...)` with `pulp.GUROBI_CMD(...)` — the rest of the code is unchanged.

---

## Data Calibration

All parameters are calibrated to publicly available industry data:

| Parameter | Source |
|-----------|--------|
| Well capacities and decline rates | EIA Drilling Productivity Reports, Permian Basin, 2023 |
| Pipeline tariffs | FERC Form 6 filings; Longhorn, Sunrise, BridgeTex schedules |
| Refinery crack spreads | EIA Weekly Petroleum Status Report, Gulf Coast 3-2-1, 2020–2024 avg |
| Carbon intensities | ICF International upstream analysis; API OPGEE model, Permian sub-plays |
| Grade differentials | EIA crude marketing survey; CME WTI/WTS futures spread |
| Ship-or-pay structures | Public tariff filings for Permian-to-Gulf interstate pipelines |

*Academic/portfolio model. Production deployment requires SCADA integration, real-time OPIS/Platts feeds, and licensed solvers for large-scale instances.*
