"""
Configuration loader.

Reads config/params.yaml and provides typed access to all model parameters.
This is the single source of truth for all calibrated constants.

Usage
-----
from src.utils.config import get_config, ModelConfig

cfg = get_config()
print(cfg.planning.default_horizon)    # 14
print(cfg.crack_spreads["R1_houston"]["mean"])  # 14.20
"""

import os
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "params.yaml"


@dataclass
class PlanningConfig:
    default_horizon: int = 14
    max_horizon: int = 30
    default_solver: str = "cbc"
    solver_gap: float = 0.001
    solver_timelimit: int = 300


@dataclass
class StochasticConfig:
    n_scenarios: int = 10
    annual_vol: float = 0.22
    seed: int = 42
    confidence_level: float = 0.95


@dataclass
class SensitivityConfig:
    delta_pct: float = 0.10
    bottleneck_threshold: float = 0.70
    bottleneck_expand_bbl: float = 5_000.0


@dataclass
class CarbonConfig:
    default_budget_tco2_per_day: Optional[float] = None
    intensity_units: str = "kg_CO2e_per_bbl"


@dataclass
class ModelConfig:
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    stochastic: StochasticConfig = field(default_factory=StochasticConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    carbon: CarbonConfig = field(default_factory=CarbonConfig)

    # Raw dicts for structured data that doesn't warrant dedicated dataclasses
    crack_spreads: Dict[str, Any] = field(default_factory=dict)
    crude_grades: Dict[str, Any] = field(default_factory=dict)
    pipeline_tariffs: Dict[str, Any] = field(default_factory=dict)
    ship_or_pay: Dict[str, Any] = field(default_factory=dict)
    wti_benchmark_usd: float = 78.50


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        logger.warning(f"Config file not found at {path}; using defaults")
        return {}
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def get_config(path: Optional[str] = None) -> ModelConfig:
    """
    Load and cache the model configuration.
    Call get_config.cache_clear() to force reload.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH

    # Allow environment variable override (useful in Docker)
    env_path = os.environ.get("OSC_CONFIG_PATH")
    if env_path:
        config_path = Path(env_path)

    raw = _load_yaml(config_path)
    if not raw:
        logger.info("Using default configuration")
        return ModelConfig()

    planning_raw = raw.get("planning", {})
    planning = PlanningConfig(
        default_horizon=planning_raw.get("default_horizon", 14),
        max_horizon=planning_raw.get("max_horizon", 30),
        default_solver=planning_raw.get("default_solver", "cbc"),
        solver_gap=planning_raw.get("solver_gap", 0.001),
        solver_timelimit=planning_raw.get("solver_timelimit", 300),
    )

    stoch_raw = raw.get("stochastic", {})
    stochastic = StochasticConfig(
        n_scenarios=stoch_raw.get("n_scenarios", 10),
        annual_vol=stoch_raw.get("annual_vol", 0.22),
        seed=stoch_raw.get("seed", 42),
        confidence_level=stoch_raw.get("confidence_level", 0.95),
    )

    sens_raw = raw.get("sensitivity", {})
    sensitivity = SensitivityConfig(
        delta_pct=sens_raw.get("delta_pct", 0.10),
        bottleneck_threshold=sens_raw.get("bottleneck_threshold", 0.70),
        bottleneck_expand_bbl=sens_raw.get("bottleneck_expand_bbl", 5_000.0),
    )

    carbon_raw = raw.get("carbon", {})
    carbon = CarbonConfig(
        default_budget_tco2_per_day=carbon_raw.get("default_budget_tco2_per_day"),
        intensity_units=carbon_raw.get("intensity_units", "kg_CO2e_per_bbl"),
    )

    cfg = ModelConfig(
        planning=planning,
        stochastic=stochastic,
        sensitivity=sensitivity,
        carbon=carbon,
        crack_spreads=raw.get("crack_spreads", {}),
        crude_grades=raw.get("crude_grades", {}),
        pipeline_tariffs=raw.get("pipeline_tariffs", {}),
        ship_or_pay=raw.get("ship_or_pay", {}),
        wti_benchmark_usd=raw.get("wti_benchmark_usd", 78.50),
    )
    logger.info(f"Loaded config from {config_path}")
    return cfg
