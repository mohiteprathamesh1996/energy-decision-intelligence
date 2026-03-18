# Contributing

## Architecture Principles

**Separation of concerns.** Domain model (`src/model/supply_chain.py`) has no solver dependency. Optimizer receives a `SupplyChainNetwork`, returns a `MultiPeriodResult`. Visualization has no optimizer import. This layering means any layer can be replaced independently.

**No magic numbers.** All calibrated constants live in `config/params.yaml`. Code references `get_config()`, not hardcoded floats.

**Scenarios as pure functions.** Each scenario modifier is a function `SupplyChainNetwork → SupplyChainNetwork` that deep-copies and modifies. No side effects. Composable.

**Tests without solver.** Domain, scenario modifier, and stochastic scenario generation tests all run without CBC. Only integration tests require a solver. This keeps CI fast.

## Development Setup

```bash
# Clone and install
git clone https://github.com/your-username/oil-supply-chain-optimizer
cd oil-supply-chain-optimizer

# Install CBC (required for solve tests)
# macOS
brew install coin-or-tools

# Ubuntu
sudo apt install coinor-cbc coinor-libcbc-dev

# Python dependencies
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/ && mypy src/
```

## Adding a New Scenario

1. Add a modifier function to `src/analysis/scenario.py`:
   ```python
   def my_new_scenario(param: float) -> Callable:
       def _mod(net: SupplyChainNetwork) -> SupplyChainNetwork:
           # modify net (deep copy already done by ScenarioRunner)
           return net
       return _mod
   ```
2. Add an entry to `build_standard_scenarios()`.
3. Add a unit test in `tests/test_optimizer.py` under `TestScenarioModifiers`.

## Adding a New Constraint

1. Add the constraint as a new rule function in `src/model/optimizer.py`, following the existing `c_*` naming convention.
2. Register it: `m.c_my_constraint = pyo.Constraint(m.NODES, m.PERIODS, rule=my_rule)`.
3. Document it with a comment block showing the mathematical form.
4. Add a test in `tests/test_optimizer.py` verifying the constraint is created.
5. Add a scenario that exercises the constraint.
6. Update `README.md` constraint table.

## Upgrading the Solver

The solver string is the only change required:

```python
# In any optimizer call
MultiPeriodOptimizer(net, solver="gurobi")   # or "cplex"

# For stochastic analysis
run_stochastic_analysis(net, solver="gurobi")
```

Pyomo's solver interface is identical across CBC/GLPK/Gurobi/CPLEX. Gurobi is 5–10x faster on large stochastic instances.

## Code Style

- Ruff with line-length=100 (see `pyproject.toml`)
- Type hints on all public functions
- Docstrings: `"""Short summary.\n\nExtended description if needed."""`
- No global state outside `get_config()` (which is `lru_cache`-d)

## Submitting Changes

1. Create a branch: `git checkout -b feature/description`
2. Run `pytest tests/` and `ruff check src/`
3. Update `README.md` if the change affects the mathematical formulation
4. Open a PR with a description of what changed and why
