"""Unvalidated v7: v6 engine scores also apply to full-board replacements."""
from __future__ import annotations

from typing import Any
import numpy as np

from smart_agent import act_with_config
from smart_agent_v6_candidate import CONFIG as V6_CONFIG


CONFIG = {
    **V6_CONFIG,
    "replacement_scoring": True,
    # Protect only an engine with a visible payoff already on board.
    "engine_retention_bonus": 18.0,
}


def act(observation: dict[str, np.ndarray], context: dict[str, Any]) -> int:
    return act_with_config(observation, context, CONFIG)
