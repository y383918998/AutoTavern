"""Run a conservatively promoted smart-policy configuration."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
from smart_agent import act_with_config

path = os.environ.get("TAVERN_SMART_CONFIG", "")
if not path: raise RuntimeError("Set TAVERN_SMART_CONFIG to an approved promoted_config.json")
spec = json.loads(Path(path).read_text(encoding="utf-8"))
if spec.get("version") != "smart_agent_v2" or not isinstance(spec.get("config"), dict):
    raise RuntimeError("Not an approved smart-agent configuration")

def act(observation: dict[str, np.ndarray], context: dict[str, Any]) -> int:
    return act_with_config(observation, context, spec["config"])
