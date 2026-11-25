"""Heuristic baselines for DFL communication decisions.

Provides fixed-rule policies that replace the RL (PPO) actor for
benchmarking purposes. All policies use the same DFL infrastructure.
"""

from src.baselines.heuristic_policies import (
    EmbeddingsOnlyPolicy,
    Fixed7030Policy,
    FixedSwitchPolicy,
    HeuristicPolicy,
    PeriodicPolicy,
    ThresholdPolicy,
    WeightsOnlyPolicy,
    create_heuristic_policy,
)

__all__ = [
    "HeuristicPolicy",
    "WeightsOnlyPolicy",
    "EmbeddingsOnlyPolicy",
    "FixedSwitchPolicy",
    "PeriodicPolicy",
    "ThresholdPolicy",
    "Fixed7030Policy",
    "create_heuristic_policy",
]
