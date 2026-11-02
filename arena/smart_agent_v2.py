"""Promoted local policy: validated delayed-upgrade variant of smart_agent.

Promotion evidence: two independent 512-game fixed-seed evaluations passed
the improvement gate. The original smart_agent remains available for rollback.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from smart_agent import act_with_config


CONFIG = {
    "upgrade_turn_factor": 2.4,
    "pair_bonus": 15.0,
    "refresh_min_gold": 3,
}


def act(observation: dict[str, np.ndarray], context: dict[str, Any]) -> int:
    return act_with_config(observation, context, CONFIG)
