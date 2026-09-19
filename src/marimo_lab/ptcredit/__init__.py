"""Simulation model for the private-tracker point system.

Three layers, used by the notebooks under
``notebooks/A_Application_math/A01_privite_bitorrent_credit``:

``world``
    the exogenous draw — torrent sizes, node bandwidth, uptime, disk;
``credit``
    the mechanism — health, marginal reward, pessimistic node estimators;
``sim``
    the agent-based run that closes the loop between them.
"""

from __future__ import annotations

from .credit import (
    NodeEstimator,
    RewardParams,
    health,
    health_marginal,
    reward_rate,
    slowdown,
    slowdown_plain,
    softplus,
    target_capacity,
)
from .sim import SimParams, SimResult, build_world, run
from .world import (
    DEFAULT_BANDWIDTH,
    DEFAULT_MIX,
    DEFAULT_SIZE_CLASSES,
    MEGABIT_PER_GIB,
    BandwidthModel,
    Catalog,
    Population,
    SizeClass,
    calibrate_bandwidth,
    hours_to_transfer,
    sample_bandwidth,
    sample_catalog,
    sample_population,
)

__all__ = [
    "DEFAULT_BANDWIDTH",
    "DEFAULT_MIX",
    "DEFAULT_SIZE_CLASSES",
    "MEGABIT_PER_GIB",
    "BandwidthModel",
    "Catalog",
    "NodeEstimator",
    "Population",
    "RewardParams",
    "SimParams",
    "SimResult",
    "SizeClass",
    "build_world",
    "calibrate_bandwidth",
    "health",
    "health_marginal",
    "hours_to_transfer",
    "reward_rate",
    "run",
    "sample_bandwidth",
    "sample_catalog",
    "sample_population",
    "slowdown",
    "slowdown_plain",
    "softplus",
    "target_capacity",
]
