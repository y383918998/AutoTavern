"""Unvalidated v8: v7 engine preservation plus low-health tempo protection."""
from __future__ import annotations

from typing import Any
import numpy as np

from smart_agent import act_with_config
from smart_agent_v7 import CONFIG as V7_CONFIG


CONFIG = {
    **V7_CONFIG,
    "low_health_threshold": 15,
    "low_health_stat_weight": 0.5,
    "low_health_no_upgrade": True,
    "low_health_refresh_block": True,
}


def act(observation: dict[str, np.ndarray], context: dict[str, Any]) -> int:
    return act_with_config(observation, context, CONFIG)
