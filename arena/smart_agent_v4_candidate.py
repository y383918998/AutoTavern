"""Unvalidated opponent-aware positioning candidate built on smart_agent_v2."""
from __future__ import annotations
from typing import Any
import numpy as np
from smart_agent import act_with_config
from smart_agent_v2 import CONFIG as V2_CONFIG

CONFIG={**V2_CONFIG,"positioning":True,"opponent_aware_positioning":True,
        "position_cleave_bonus":15.0,"position_deathrattle_bonus":7.0}
def act(observation:dict[str,np.ndarray],context:dict[str,Any])->int:
    return act_with_config(observation,context,CONFIG)
