"""Unvalidated v9 candidate: context-aware pending choice and target scoring only.

It inherits the promoted v8 strategy unchanged for recruit, upgrades,
refreshes, selling and positioning.  This isolation makes its validation an
ablation of selection behaviour rather than another broad rule rewrite.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from smart_agent import act_with_config
from smart_agent_v8_upgrade_only import CONFIG as V8_CONFIG


CONFIG = {
    **V8_CONFIG,
    "choice_scoring": True,
    "choice_pair_bonus": 8.0,
    "choice_synergy_weight": 0.5,
    "choice_economy_bonus": 2.0,
    "choice_cost_weight": 0.75,
    "choice_target_weight": 100.0,
    # Hero selection requires long-horizon hero-specific planning. It is
    # deliberately outside this narrowly-scoped candidate.
    "choice_scored_kinds": ("discover", "trinket", "dark_gift"),
}


def act(observation: dict[str, np.ndarray], context: dict[str, Any]) -> int:
    return act_with_config(observation, context, CONFIG)
