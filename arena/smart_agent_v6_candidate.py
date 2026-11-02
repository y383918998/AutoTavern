"""Unvalidated v6: v2 economy plus explicit visible engine/payoff scoring."""
from __future__ import annotations

from typing import Any
import numpy as np

from smart_agent import act_with_config
from smart_agent_v2 import CONFIG as V2_CONFIG


CONFIG = {
    **V2_CONFIG,
    # Strong enough to beat a modest raw-stat difference, but not a large one.
    "engine_battlecry_bonus": 8.0,
    "engine_deathrattle_bonus": 8.0,
    "eternal_knight_bonus": 7.0,
    # /stats is aggregate final-board data, so keep it below one point.
    "public_prior_weight": 0.5,
}


def act(observation: dict[str, np.ndarray], context: dict[str, Any]) -> int:
    return act_with_config(observation, context, CONFIG)
