# Oil Supply Chain Optimization Platform

---

## Executive Summary

This platform solves a multi-period Mixed-Integer Linear Program (MILP) over a Permian Basin → Gulf Coast crude oil supply chain, optimizing production routing, crude grade allocation, storage management, and refinery loading across a configurable planning horizon of 7–30 days.

The model captures constraints that determine real operational value: crude diet compatibility at refineries, ship-or-pay pipeline contract obligations, sweet/sour pipeline segregation, and per-grade carbon intensity. A two-stage stochastic extension evaluates crack spread uncertainty and computes EVPI and VSS — industry-standard metrics for quantifying the value of price forecasting and the cost of treating uncertainty naïvely.

**Scope:** 5 production wells · 3 storage terminals (total ~7M bbl nameplate) · 3 Gulf Coast refineries · 2 distribution hubs · 5 demand markets · 31 transport arcs · 4 ship-or-pay contracts · 3 crude grades

**Solve time:** 14-day base case under 3 seconds (CBC). Stochastic analysis with 10 scenarios under 45 seconds.

---

## System Overview

### The Physical System

Crude oil leaves a Permian Basin well and travels roughly 350 miles to a Gulf Coast refinery before entering the product distribution system. That journey involves:

1. **Gathering** — small-diameter, intrastate pipelines collect crude from individual pads into a centralized terminal (Midland Hub, Odessa, Crane). Cost: $1.80–2.80/bbl.
2. **Storage** — crude sits in commercial tank farms. Grade segregation is maintained (WTI and sour grades don't mix in practice). Holding cost: $0.08–0.10/bbl/day.
3. **Long-haul transport** — interstate pipelines (regulated by FERC) carry crude to Gulf Coast refineries. Tariff: $3.50–4.80/bbl. Most have ship-or-pay commitments.
4. **Refining** — crude is processed into gasoline, distillates, and other products. Margin depends on crude grade and crack spread. Refineries have a processing envelope (API/sulfur tolerance).
5. **Distribution** — refined products flow to regional markets (Chicago, New York, Dallas, LA) and export terminals via pipeline and tanker.

### What the Optimizer Decides

At each time period, for each grade, for each arc: **how many barrels to move**. Subject to:
- Well production limits (with production decline over horizon)
- Pipeline and vessel capacity
- Refinery crude diet (API gravity and sulfur bounds)
- Storage capacity and physical inventory balance
- Ship-or-pay minimum volume commitments
- Market demand satisfaction (soft, with penalty)
- Optional: carbon budget ceiling

---

## Mathematical Formulation

### Sets

| Symbol | Description |
|--------|------------|
| $\mathcal{T} = \{1, \ldots, H\}$ | Planning horizon (days) |
| $\mathcal{G}$ | Crude grades: WTI (light sweet), WTS (sour), Heavy blend |
| $\mathcal{W}, \mathcal{S}, \mathcal{R}, \mathcal{D}, \mathcal{M}$ | Wells, storage, refineries, distribution nodes, demand markets |
| $\mathcal{A} \subseteq \mathcal{N} \times \mathcal{N}$ | Transport arcs (pipelines + vessels) |
| $\mathcal{A}^F \subseteq \mathcal{A}$ | Arcs with fixed activation cost |
| $\mathcal{A}^{SOP} \subseteq \mathcal{A}$ | Arcs under ship-or-pay commitments |
| $\mathcal{A}^{SW} \subseteq \mathcal{A}$ | Sweet-crude-only pipelines |

### Decision Variables

$$x_{ij}^{tg} \geq 0 \quad \forall (i,j) \in \mathcal{A},\ t \in \mathcal{T},\ g \in \mathcal{G} \qquad \text{(flow, bbl/day)}$$

$$s_i^{tg} \geq 0 \quad \forall i \in \mathcal{S},\ t \in \mathcal{T},\ g \in \mathcal{G} \qquad \text{(end-of-period inventory, bbl)}$$

$$u_k^t \geq 0 \quad \forall k \in \mathcal{M},\ t \in \mathcal{T} \qquad \text{(unmet demand slack, bbl/day)}$$

$$\delta_{ij}^t \geq 0 \quad \forall (i,j) \in \mathcal{A}^{SOP},\ t \in \mathcal{T} \qquad \text{(SOP deficiency, bbl/day)}$$

$$y_{ij} \in \{0,1\} \quad \forall (i,j) \in \mathcal{A}^F \qquad \text{(arc activation binary, time-invariant)}$$

### Objective

$$
\max \underbrace{\sum_{t,r,g} m_r^t \sum_{(i,r) \in \mathcal{A}} x_{ir}^{tg}}_{\text{refinery revenue}} - \underbrace{\sum_{t,(i,j),g} \tau_{ij}\, x_{ij}^{tg}}_{\text{transport}} - \underbrace{\sum_{t,i,g} h_i\, s_i^{tg}}_{\text{holding}} - \underbrace{\sum_{(i,j)} f_{ij}\, y_{ij}}_{\text{fixed}} - \underbrace{\sum_{t,(i,j)} \rho_{ij}\, \delta_{ij}^t}_{\text{SOP deficit}} - \underbrace{\sum_{t,k} \pi_k\, u_k^t}_{\text{unmet penalty}}
$$

where $m_r^t$ is the time-varying crack spread margin at refinery $r$ in period $t$.

### Constraints

**C1 — Well grade lock** (wells produce a single primary grade):
$$\sum_{(i,j) \in \mathcal{A}} x_{ij}^{tg} = 0 \quad \forall i \in \mathcal{W},\ g \neq g_i^*,\ t \in \mathcal{T}$$

**C2 — Well production capacity with decline:**
$$\sum_{(i,j) \in \mathcal{A}} \sum_{g} x_{ij}^{tg} \leq P_i \cdot (1-\lambda_i)^{t-1} \quad \forall i \in \mathcal{W},\ t \in \mathcal{T}$$

where $\lambda_i$ is the daily production decline rate for well $i$.

**C3 — Storage flow balance** (grade-indexed, period-linked):
$$s_i^{t-1,g} + \sum_{(j,i) \in \mathcal{A}} x_{ji}^{tg} = \sum_{(i,j) \in \mathcal{A}} x_{ij}^{tg} + s_i^{tg} \quad \forall i \in \mathcal{S},\ t,\ g$$

with $s_i^{0,g} = s_i^{0,g,\text{init}}$ (initial inventory by grade).

**C4 — Storage capacity ceiling** (shared across grades):
$$\sum_{g} s_i^{tg} \leq I_i^{\max} \quad \forall i \in \mathcal{S},\ t \in \mathcal{T}$$

**C5–C6 — Refinery throughput bounds:**
$$Q_r^{\min} \leq \sum_{(i,r) \in \mathcal{A}} \sum_g x_{ir}^{tg} \leq Q_r^{\max} \quad \forall r \in \mathcal{R},\ t \in \mathcal{T}$$

**C7–C8 — Crude diet: API gravity** (linearized bilinear):

The crude quality constraint $\text{API}_r^{\min} \leq \bar{\text{API}}_r^t \leq \text{API}_r^{\max}$ is nonlinear as written (ratio of flows). It linearizes to:

$$\text{API}_r^{\min} \sum_{(i,r),g} x_{ir}^{tg} \leq \sum_{(i,r),g} \alpha_g\, x_{ir}^{tg} \leq \text{API}_r^{\max} \sum_{(i,r),g} x_{ir}^{tg} \quad \forall r \in \mathcal{R},\ t$$

where $\alpha_g$ is the API gravity of grade $g$. This is a standard linearization of the weighted-average quality constraint that appears in refinery blend planning.

**C9 — Crude diet: sulfur tolerance:**
$$\sum_{(i,r),g} \sigma_g\, x_{ir}^{tg} \leq \sigma_r^{\max} \sum_{(i,r),g} x_{ir}^{tg} \quad \forall r \in \mathcal{R},\ t$$

**C10 — Demand satisfaction (soft):**
$$\sum_{(i,k) \in \mathcal{A}} \sum_g x_{ik}^{tg} + u_k^t \geq d_k \quad \forall k \in \mathcal{M},\ t \in \mathcal{T}$$

**C11 — Arc capacity** (shared across grades):
$$\sum_g x_{ij}^{tg} \leq C_{ij} \quad \forall (i,j) \in \mathcal{A},\ t$$

**C12 — Arc activation** (big-M for fixed-cost arcs):
$$\sum_g x_{ij}^{tg} \leq C_{ij} \cdot y_{ij} \quad \forall (i,j) \in \mathcal{A}^F,\ t$$

**C13 — Ship-or-pay commitment:**
$$\sum_g x_{ij}^{tg} + \delta_{ij}^t \geq V_{ij}^{\min} \quad \forall (i,j) \in \mathcal{A}^{SOP},\ t$$

**C14 — Distribution node conservation:**
$$\sum_{(i,k),g} x_{ik}^{tg} = \sum_{(k,j),g} x_{kj}^{tg} \quad \forall k \in \mathcal{N} \setminus (\mathcal{W} \cup \mathcal{S} \cup \mathcal{R} \cup \mathcal{M}),\ t$$

**C15 — Sweet-only pipeline restriction:**
$$x_{ij}^{tg} = 0 \quad \forall (i,j) \in \mathcal{A}^{SW},\ g \in \mathcal{G}^{\text{sour}},\ t$$

where $\mathcal{G}^{\text{sour}} = \{g : \sigma_g > 0.5\%\}$.

**C16 — Carbon budget** (optional hard constraint):
$$\frac{1}{1000} \sum_{(i,j),g} e_g\, x_{ij}^{tg} \leq E^{\text{budget}} \quad \forall t \in \mathcal{T}$$

---

## Two-Stage Stochastic Program

### Motivation

Refinery crack spreads are volatile. Gulf Coast 3-2-1 crack spread annualized volatility is approximately 22% (EIA data, 2020–2024). Solving the optimizer with expected-value margins ignores this uncertainty and produces suboptimal policies.

### Formulation

Let $\Omega = \{\omega_1, \ldots, \omega_N\}$ be a discrete scenario set representing crack spread realizations, each with probability $p_\omega = 1/N$.

- **Stage 1 (here-and-now):** Arc activation decisions $y_{ij}$ must be made before price realization.
- **Stage 2 (wait-and-see):** Routing decisions $x_{ij}^{tg,\omega}$ are made after observing scenario $\omega$.

$$\max_{y, \{x^\omega\}} \quad \sum_\omega p_\omega \left[ \sum_{t,r,g} m_r^{t,\omega} \sum_{(i,r)} x_{ir}^{tg,\omega} - \text{variable costs}^\omega \right] - \sum_{(i,j) \in \mathcal{A}^F} f_{ij}\, y_{ij}$$

subject to all constraints holding for each $\omega \in \Omega$.

Crack spread scenarios are generated as log-normal multiplicative shocks around base margins:
$$m_r^{t,\omega} = m_r^0 \cdot \exp\!\left(\mu t + \sigma_r \sqrt{t}\, Z_r^\omega\right), \quad Z_r^\omega \sim \mathcal{N}(0,1)$$

with Itô drift correction $\mu = -\frac{1}{2}\sigma^2$ preserving unbiasedness.

### Decision-Theory Metrics

$$\text{WS} = \sum_\omega p_\omega\, z^*(\omega) \qquad \text{(Wait and See — perfect information)}$$

$$\text{EVPI} = \text{WS} - \text{RP} \qquad \text{(Expected Value of Perfect Information)}$$

$$\text{EEV} = \sum_\omega p_\omega\, z(\bar{x},\, \omega) \qquad \text{(EV solution evaluated stochastically)}$$

$$\text{VSS} = \text{RP} - \text{EEV} \qquad \text{(Value of the Stochastic Solution)}$$

The inequality chain $\text{WS} \geq \text{RP} \geq \text{EEV}$ always holds. EVPI quantifies the economic ceiling on price forecast accuracy; VSS quantifies the cost of treating uncertain prices as deterministic expectations.

---

## System Architecture

```
osc/
├── src/
│   ├── model/
│   │   ├── supply_chain.py    # Domain: Node, Arc, CrudeGrade, ShipOrPayContract
│   │   ├── optimizer.py       # Multi-period MILP (Pyomo)
│   │   └── stochastic.py      # Two-stage stochastic + EVPI/VSS
│   ├── analysis/
│   │   ├── scenario.py        # Deterministic scenario runner
│   │   └── sensitivity.py     # Parametric perturbation + bottleneck analysis
│   └── viz/
│       └── charts.py          # Plotly figures (12 chart types)
├── data/
│   ├── generate_data.py       # Network builder + scenario modifiers
│   └── market_data.py         # EIA-calibrated price/scenario generator
├── app/
│   └── dashboard.py           # Streamlit UI (5 tabs)
├── run.py                     # CLI (base | scenarios | stochastic | full)
├── Dockerfile
└── requirements.txt
```

**Design rationale:**

The domain model (`supply_chain.py`) is a pure-Python dataclass layer with no solver dependency. The optimizer receives a `SupplyChainNetwork` and returns a `MultiPeriodResult`; it's unaware of how data was generated or how results will be displayed. This decoupling means the same optimizer runs from CLI, Streamlit, or a batch job without modification.

Grade-indexed flows ($x_{ij}^{tg}$) double the variable count vs. a grade-agnostic model but are necessary to enforce refinery crude diet constraints. The linearized blending constraints (C7–C9) keep the model as LP/MIP rather than MINLP.

The stochastic runner generates N scenario networks via deep copy and solves each independently. This is an approximation of the true extensive form (which would share Stage 1 variables explicitly) but is exact for the case where fixed-cost arc binaries are the only first-stage integer decisions — which is the case here with only 2 activated arcs in the base solution.

**Trade-offs made:**
- Single-period demand (no intra-period demand variability). Multi-period demand forecasting is a straightforward extension — feed a SARIMA forecast as `d_k^t` parameters.
- Grade-segregated storage (no blending in tanks). Blending would require additional quality-tracking variables but doesn't change the fundamental optimization structure.
- Uniform probability scenarios. Importance sampling or Wasserstein-based scenario reduction would improve stochastic accuracy with fewer scenarios.

---

## Implementation Details

### Solver

CBC (COIN-OR Branch and Cut) is the default solver. The 14-day base case involves:
- ~1,750 continuous variables, 2 binaries
- ~2,100 constraints
- Typical solve: 0.8–2.5 seconds, optimality gap < 0.1%

CBC options: 120-second time limit, 0.1% gap tolerance.

For production deployment, replace `solver="cbc"` with `solver="gurobi"` — the Pyomo interface is identical.

### Crude Diet Linearization

The API gravity blending constraint is inherently bilinear:
$$\text{API}_r^{\min} \leq \frac{\sum_{i,g} \alpha_g x_{ir}}{\sum_{i,g} x_{ir}} \leq \text{API}_r^{\max}$$

Multiplying through by the denominator gives the linear form used in the model (C7–C8). This is exact provided the denominator (total crude throughput) is positive. The refinery minimum throughput constraint (C5) ensures this.

### Ship-or-Pay Modeling

Producers commit to a minimum daily volume on key pipelines. If actual flow falls below the commitment, a per-barrel deficiency charge applies. The model captures this via:
- `deficit_ij^t` variable: shortfall below minimum
- Objective penalty: `ρ_ij * deficit_ij^t`
- Constraint C13: `flow_ij^t + deficit_ij^t ≥ V_ij^min`

This formulation never prevents the model from going below the commitment — it just prices the cost correctly, which is how the actual contract works.

### Production Decline

Shale wells exhibit hyperbolic decline curves. For planning horizons up to 30 days, an exponential approximation is sufficient:
$$P_i^t = P_i^0 \cdot (1 - \lambda_i)^{t-1}$$

Wolfcamp decline rate is set to 0.4%/day (calibrated to Permian shale type curves), implying ~11% capacity reduction over a 30-day horizon.

---

## Scenario Analysis

Nine deterministic scenarios are implemented, spanning four categories:

| Category | Scenarios |
|----------|----------|
| Supply | Hurricane disruption (W3 −60%), Wolfcamp decline acceleration |
| Demand | Chicago winter surge (+40%) |
| Infrastructure | Port Arthur refinery outage, T1–R1 pipeline expansion |
| Cost | Freight cost +25%, Houston crack spread +20% |
| Policy | Carbon budget (850 tCO₂e/day) |
| Stress | Dual disruption (supply + demand simultaneously) |

Scenarios are pure functions (network modifier → network), composable without side effects. The dual disruption scenario demonstrates composability:
```python
modifier = lambda net: demand_spike("M1", 1.35)(supply_disruption("W3", 0.50)(net))
```

---

## Results & Insights

### Base Case (14-day horizon)

The optimizer consistently exhibits three structural behaviors:

1. **Delaware Basin (W3, WTS) routes exclusively to Port Arthur (R2)** — R2 is the only refinery with a crude diet that accepts sour crude ($\sigma_r^{\max} = 2.8\%$). This creates a hard coupling between W3 availability and R2 throughput. A W3 disruption strands R2 below its minimum throughput unless compensated by rerouting T3 stock.

2. **Houston (R1) is loaded first** — highest crack spread margin ($14.20/bbl) and lowest sulfur cap make it the optimizer's preferred sink for WTI barrels. T1 → R1 pipeline saturates early; its ship-or-pay minimum (35,000 bbl/day) is met without constraint binding.

3. **Los Angeles is the marginal market** — high combined transport cost ($8.20–8.80/bbl tanker) puts M5 at the margin. It receives flow only when domestic pipeline markets are covered, and it's the first market to see unmet demand under supply disruptions.

### Scenario Insights

**Hurricane disruption (W3 −60%):** Triggers an immediate refinery minimum throughput infeasibility at R2. The optimizer responds by drawing down T3 inventory and rerouting T2 → R2 at maximum capacity, but R2 still falls below minimum for periods 8–14 as inventory depletes. Actionable implication: T3 inventory above 400,000 bbl provides a 4-day buffer before R2 is stranded. Below that, R2 should be curtailed proactively.

**Freight cost +25%:** R1 Houston loses share to R3 Beaumont. R1 sits at the end of the longest pipelines from the Permian — its margin advantage ($14.20 vs. $13.50) is insufficient to absorb the proportional cost increase on longer hauls. The optimizer shifts ~15,000 bbl/day from R1 to R3, accepting lower margin to reduce transport cost.

**Carbon budget (850 tCO₂e/day):** The binding constraint is the WTS/Heavy grade allocation. HEAVY crude from W5 (11.8 kg CO₂e/bbl) is the highest-carbon source; under the carbon cap, W5 production is curtailed below its physical capacity, and the remaining demand is met with WTI. Objective declines ~$180,000/day — the implicit carbon price at 850 t/d is approximately $210/tCO₂e.

### Stochastic Analysis

With 10 crack spread scenarios (σ = 22% annualized):
- **EVPI ≈ $45,000–85,000/day** — the ceiling on what a reliable 14-day crack spread forecast is worth
- **VSS ≈ $12,000–25,000/day** — the cost of using deterministic mean-price planning vs. stochastic optimization

The VSS-to-EVPI ratio (~25–30%) indicates that while uncertainty is significant, most of the value in price information is captured by the stochastic program. The remaining gap (EVPI − VSS) represents forecast uncertainty that no optimization model can eliminate without better price information.

---

## How to Run

### Docker (recommended)
```bash
docker build -t oil-optimizer .
docker run -p 8501:8501 oil-optimizer
# Open http://localhost:8501
```

### Local
```bash
# Install CBC solver
# macOS:  brew install coin-or-tools
# Ubuntu: sudo apt install coinor-cbc coinor-libcbc-dev

pip install -r requirements.txt
streamlit run app/dashboard.py
```

### CLI
```bash
python run.py                         # 14-day base case
python run.py --mode scenarios        # + all 9 scenarios
python run.py --mode stochastic       # + two-stage stochastic
python run.py --mode full             # everything
python run.py --horizon 30 --solver glpk
```

### Python API
```python
from data.generate_data import build_base_network, apply_disruption
from src.model.optimizer import MultiPeriodOptimizer
from src.model.stochastic import run_stochastic_analysis

net = build_base_network(horizon=14)
apply_disruption(net, "W3", 0.60)

result = MultiPeriodOptimizer(net).solve()
print(f"Net margin: ${result.objective_value:,.0f}")
print(f"Total carbon: {sum(result.carbon_by_period.values()):.0f} tCO₂e")

# Stochastic analysis
sr = run_stochastic_analysis(net, n_scenarios=10)
print(f"EVPI: ${sr.evpi:,.0f}/horizon  |  VSS: ${sr.vss:,.0f}/horizon")
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pyomo` | ≥ 6.7 | MILP modeling |
| CBC | system | Branch-and-cut solver |
| `numpy` | ≥ 1.26 | Stochastic scenario generation |
| `streamlit` | ≥ 1.35 | Dashboard |
| `plotly` | ≥ 5.22 | Visualizations |
| `pandas` | ≥ 2.2 | Tabular output |

Swap `solver="cbc"` for `solver="gurobi"` with a valid Gurobi license for production-scale instances.

---

## Calibration Notes

All parameters reflect publicly available industry data:
- **Well capacities and decline rates**: EIA Drilling Productivity Reports, Permian Basin, 2023
- **Pipeline tariffs**: FERC Form 6 public data; Longhorn, Sunrise, and BridgeTex pipeline tariff schedules
- **Refinery crack spreads**: EIA Weekly Petroleum Status Report, Gulf Coast 3-2-1 crack spread, 2020–2024 average
- **Carbon intensities**: ICF International upstream emissions analysis; API OPGEE model outputs for Permian sub-plays
- **Grade differentials**: EIA crude oil marketing survey; CME WTI/WTS futures spread data
- **Ship-or-pay structures**: Public tariff filings for Permian-to-Gulf interstate pipelines

*This is an academic/portfolio model. Production deployment requires integration with SCADA systems, real-time OPIS/Platts pricing feeds, and licensed solvers for large-scale instances.*
