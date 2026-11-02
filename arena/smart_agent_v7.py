"""Stable v7 policy: promoted after three independent local validations.

Aggregate evidence (2,048 paired local games versus v2): mean placement
4.613 -> 4.448, top-4 987 -> 1063, and eighth places 253 -> 250.
The v2 module remains available as an explicit rollback policy.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from smart_agent import act_with_config
from smart_agent_v7_candidate import CONFIG


def act(observation: dict[str, np.ndarray], context: dict[str, Any]) -> int:
    return act_with_config(observation, context, CONFIG)
