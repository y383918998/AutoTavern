"""Use the learned policy only when it is confident; otherwise use smart_agent."""
from __future__ import annotations
import os
import json
from typing import Any
import numpy as np
import learned_agent
from smart_agent import act as fallback

THRESHOLD = float(os.environ.get("TAVERN_HYBRID_CONFIDENCE", "0.80"))
MIN_VALID_ACCURACY = float(os.environ.get("TAVERN_HYBRID_MIN_VALID_ACCURACY", "0.70"))
METADATA = json.loads(str(learned_agent.model["metadata"]))
MODEL_VALID_ACCURACY = float(METADATA.get("masked_accuracy", 0.0))

def act(observation: dict[str, np.ndarray], context: dict[str, Any]) -> int:
    if MODEL_VALID_ACCURACY < MIN_VALID_ACCURACY:
        return fallback(observation, context)
    mask = np.asarray(observation["action_mask"], dtype=bool)
    values = np.where(mask, learned_agent.logits(observation), -np.inf)
    legal = values[mask]
    peak = float(legal.max())
    confidence = float(np.exp(peak - np.logaddexp.reduce(legal)))
    if confidence >= THRESHOLD:
        return int(values.argmax())
    return fallback(observation, context)
