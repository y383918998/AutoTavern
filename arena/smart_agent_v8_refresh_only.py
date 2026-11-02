"""Ablation: v7 plus low-health paid-refresh restraint only."""
from __future__ import annotations
from typing import Any
import numpy as np
from smart_agent import act_with_config
from smart_agent_v7 import CONFIG as V7_CONFIG

CONFIG = {**V7_CONFIG, "low_health_threshold": 15, "low_health_refresh_block": True}

def act(observation: dict[str, np.ndarray], context: dict[str, Any]) -> int:
    return act_with_config(observation, context, CONFIG)
