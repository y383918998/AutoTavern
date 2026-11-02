"""Unvalidated v5 candidate: v2 economy plus conservative tribe commitment."""
from __future__ import annotations
from typing import Any
import numpy as np
from smart_agent import act_with_config
from smart_agent_v2 import CONFIG as V2_CONFIG

# A tribe is only preferred after at least two visible copies already exist.
CONFIG={**V2_CONFIG,"tribe_synergy_bonus":7.0,"tribe_text_bonus":3.0}
def act(observation:dict[str,np.ndarray],context:dict[str,Any])->int:
    return act_with_config(observation,context,CONFIG)
