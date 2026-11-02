"""Stable v8 policy: v7 plus low-health upgrade restraint.

Promoted after a paired 512-game local validation against v7:
mean placement 4.506 -> 4.326, top-4 257 -> 272, and eighth places
59 -> 49.  V7 remains available as an explicit rollback policy.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from smart_agent import act_with_config
from smart_agent_v8_upgrade_only import CONFIG


def act(observation: dict[str, np.ndarray], context: dict[str, Any]) -> int:
    return act_with_config(observation, context, CONFIG)
