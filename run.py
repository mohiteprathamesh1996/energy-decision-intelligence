"""
CLI runner. Useful for batch jobs, CI validation, or headless server execution.
Usage: python run.py [--scenario all|base] [--solver cbc|glpk]
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_base(solver: str):
    from data.generate_data import build_base_network
    from src.model.optimizer import SupplyChainOptimizer

    net = build_base_network()
    opt = SupplyChainOptimizer(net, solver=solver)

    logger.info("Building model...")
    opt.build()
    logger.info("Solving...")
    result = opt.solve(tee=False)

    print("\n" + "=" * 60)
    print("  BASE CASE RESULTS")
    print("=" * 60)
    print(f"  Status:           {result.status}")
    print(f"  Net Margin/Day:   ${result.objective_value:>12,.0f}")
    print(f"  Gross Revenue:    ${result.revenue:>12,.0f}")
    print(f"  Transport Cost:   ${result.transport_cost:>12,.0f}")
    print(f"  Holding Cost:     ${result.holding_cost:>12,.0f}")
    print(f"  Unmet Penalties:  ${result.penalty_cost:>12,.0f}")
    print(f"  Solver Time:       {result.solver_time:.2f}s")
    print("=" * 60)

    total_unmet = sum(result.unmet_demand.values())
    if total_unmet > 0:
        print("\n  Unmet Demand:")
        for node_id, v in result.unmet_demand.items():
            if v > 1:
                print(f"    {node_id}: {v:,.0f} bbl/day")

    print("\n  Top Flows:")
    for (i, j), v in sorted(result.flows.items(), key=lambda x: -x[1])[:8]:
        if v > 100:
            print(f"    {i} → {j}: {v:,.0f} bbl/day")

    return result


def run_all_scenarios(solver: str):
    from data.generate_data import build_base_network
    from src.analysis.scenario import ScenarioRunner, build_standard_scenarios

    net = build_base_network()
    runner = ScenarioRunner(net, solver=solver)
    scenarios = build_standard_scenarios(net)

    base = runner.run_base()
    results = runner.run_all(scenarios)

    print("\n" + "=" * 70)
    print(f"  {'SCENARIO':<30} {'NET MARGIN':>14}  {'Δ BASE':>14}  {'SVC LEVEL':>10}")
    print("=" * 70)
    from src.model.supply_chain import NodeType
    total_demand = sum(n.demand for n in net.get_nodes_by_type(NodeType.DEMAND))
    for sr in results:
        unmet = sum(sr.result.unmet_demand.values())
        svc = (1 - unmet / max(total_demand, 1)) * 100
        arrow = "▲" if sr.delta_objective >= 0 else "▼"
        print(
            f"  {sr.scenario_name:<30} "
            f"${sr.result.objective_value:>12,.0f}  "
            f"{arrow}${abs(sr.delta_objective):>12,.0f}  "
            f"{svc:>9.1f}%"
        )
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Oil Supply Chain Optimizer CLI")
    parser.add_argument("--scenario", choices=["base", "all"], default="base")
    parser.add_argument("--solver", choices=["cbc", "glpk"], default="cbc")
    args = parser.parse_args()

    if args.scenario == "base":
        run_base(args.solver)
    elif args.scenario == "all":
        run_base(args.solver)
        print()
        run_all_scenarios(args.solver)


if __name__ == "__main__":
    main()
