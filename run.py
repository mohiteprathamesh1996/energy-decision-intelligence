"""
CLI entry point. Runs without a UI — useful for batch jobs, CI validation,
and headless server environments.

Usage:
  python run.py                           # base case only
  python run.py --mode scenarios          # base + all scenarios
  python run.py --mode stochastic         # stochastic analysis
  python run.py --mode full               # everything
  python run.py --solver glpk --horizon 7
"""

import argparse
import copy
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)


def _divider(w=65): print("─" * w)


def run_base(solver, horizon):
    from data.generate_data import build_base_network
    from src.model.optimizer import MultiPeriodOptimizer
    from src.model.supply_chain import NodeType

    net = build_base_network(horizon=horizon)
    opt = MultiPeriodOptimizer(net, solver=solver)
    result = opt.solve()

    T = result.planning_horizon
    total_unmet = sum(result.unmet_demand.values())
    total_demand = sum(n.demand for n in net.get_nodes_by_type(NodeType.DEMAND)) * T
    svc = (1 - total_unmet / max(total_demand, 1)) * 100
    total_carbon = sum(result.carbon_by_period.values())
    grade_ids = net.grade_ids()
    arc_keys = net.arc_index

    print()
    _divider()
    print(f"  BASE CASE  |  {T}-day horizon  |  {solver.upper()}")
    _divider()
    print(f"  Status             : {result.status}")
    print(f"  Net Margin (total) : ${result.objective_value:>13,.0f}")
    print(f"  Gross Revenue      : ${result.revenue:>13,.0f}")
    print(f"  Transport Cost     : ${result.transport_cost:>13,.0f}")
    print(f"  Holding Cost       : ${result.holding_cost:>13,.0f}")
    print(f"  SOP Deficiency     : ${result.sop_cost:>13,.0f}")
    print(f"  Unmet Penalties    : ${result.penalty_cost:>13,.0f}")
    print(f"  Service Level      : {svc:>12.1f}%")
    print(f"  Total Carbon       : {total_carbon:>10.1f} tCO₂e")
    print(f"  Solver Time        : {result.solver_time:>12.2f}s")
    _divider()

    # Grade mix summary
    print("\n  Crude Grade Flow (avg bbl/day):")
    for g in grade_ids:
        total = sum(
            result.flows_by_grade.get((i, j, t, g), 0)
            for (i, j) in arc_keys
            for t in range(1, T + 1)
        ) / T
        print(f"    {g:<8}: {total:>10,.0f}")

    # Top flows
    print("\n  Top 8 Flows (avg bbl/day):")
    avg_flows = {
        (i, j): sum(result.flows_by_period.get((i, j, t), 0) for t in range(1, T + 1)) / T
        for (i, j) in arc_keys
    }
    for (i, j), v in sorted(avg_flows.items(), key=lambda x: -x[1])[:8]:
        if v > 100:
            arc = net.arc_lookup(i, j)
            util = v / arc.capacity * 100
            print(f"    {net.nodes[i].name:<28} → {net.nodes[j].name:<28}  {v:>8,.0f}  ({util:>5.1f}%)")

    # SOP contracts
    print("\n  Ship-or-Pay Contract Status:")
    for c in net.contracts:
        avg_flow = sum(result.flows_by_period.get((c.arc_origin, c.arc_dest, t), 0) for t in range(1, T + 1)) / T
        avg_def = sum(result.sop_deficits.get((c.arc_origin, c.arc_dest, t), 0) for t in range(1, T + 1)) / T
        status = "✓ COMPLIANT" if avg_def < 100 else f"⚠ DEFICIT {avg_def:,.0f} bbl/d"
        print(f"    {net.nodes[c.arc_origin].name} → {net.nodes[c.arc_dest].name}: "
              f"{avg_flow:,.0f}/{c.min_daily_volume:,.0f} bbl/d  {status}")

    return result, net


def run_scenarios(solver, horizon, base_result, base_net):
    from src.analysis.scenario import ScenarioRunner, build_standard_scenarios
    from src.model.supply_chain import NodeType

    runner = ScenarioRunner(base_net, solver=solver)
    runner._base_result = base_result
    scenarios = build_standard_scenarios(base_net)
    results = runner.run_all(scenarios)

    total_demand_base = sum(n.demand for n in base_net.get_nodes_by_type(NodeType.DEMAND))

    print()
    _divider(85)
    print(f"  {'SCENARIO':<35} {'NET MARGIN':>12}  {'Δ BASE':>12}  {'SVC LEVEL':>9}  {'CARBON':>10}")
    _divider(85)
    for sr in results:
        T = sr.result.planning_horizon
        unmet = sum(sr.result.unmet_demand.values())
        svc = (1 - unmet / max(total_demand_base * T, 1)) * 100
        carbon = sum(sr.result.carbon_by_period.values()) / T
        arrow = "▲" if sr.delta_objective >= 0 else "▼"
        print(
            f"  {sr.scenario_name:<35} "
            f"${sr.result.objective_value:>10,.0f}  "
            f"{arrow}${abs(sr.delta_objective):>10,.0f}  "
            f"{svc:>8.1f}%  "
            f"{carbon:>8.0f} t/d"
        )
    _divider(85)


def run_stochastic(solver, horizon):
    from data.generate_data import build_base_network
    from src.model.stochastic import run_stochastic_analysis

    print("\n  Running two-stage stochastic analysis (10 crack spread scenarios)...")
    net = build_base_network(horizon=horizon)
    sr = run_stochastic_analysis(net, n_scenarios=10, solver=solver)

    print()
    _divider()
    print("  STOCHASTIC ANALYSIS RESULTS")
    _divider()
    print(f"  RP   (Recourse Problem)      : ${sr.rp:>12,.0f}")
    print(f"  WS   (Wait and See)          : ${sr.ws:>12,.0f}")
    print(f"  EEV  (Mean-Value Evaluated)  : ${sr.eev:>12,.0f}")
    print(f"  EVPI (Value of Perfect Info) : ${sr.evpi:>12,.0f}")
    print(f"  VSS  (Value of Stoch. Sol.)  : ${sr.vss:>12,.0f}")
    print(f"  Scenarios                    : {sr.num_scenarios}")
    print(f"  Solver Time                  : {sr.solver_time:.1f}s")
    _divider()


def main():
    parser = argparse.ArgumentParser(description="Oil Supply Chain Optimizer CLI")
    parser.add_argument("--mode", choices=["base", "scenarios", "stochastic", "full"], default="base")
    parser.add_argument("--solver", choices=["cbc", "glpk"], default="cbc")
    parser.add_argument("--horizon", type=int, default=14)
    args = parser.parse_args()

    result, net = run_base(args.solver, args.horizon)

    if args.mode in ("scenarios", "full"):
        run_scenarios(args.solver, args.horizon, result, net)

    if args.mode in ("stochastic", "full"):
        run_stochastic(args.solver, args.horizon)


if __name__ == "__main__":
    main()
