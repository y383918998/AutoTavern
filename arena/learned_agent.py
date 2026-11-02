"""Arena policy for a locally trained behavior-cloning model."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import numpy as np
from policy_features import combine

path = os.environ.get("TAVERN_POLICY_MODEL", "")
if not path:
    raise RuntimeError("Set TAVERN_POLICY_MODEL to a local policy .npz before starting Arena")
model = np.load(Path(path))
COEF, INTERCEPT, MEAN, SCALE = (model[k] for k in ("coef", "intercept", "mean", "scale"))
if COEF.shape != (245, 4081) or INTERCEPT.shape != (245,) or MEAN.shape != (4081,):
    raise RuntimeError("Unexpected local policy model dimensions; retrain with the current scripts")

def logits(observation: dict[str, np.ndarray]) -> np.ndarray:
    x = combine(observation["features"], observation["cards"])
    return ((x - MEAN) / SCALE) @ COEF.T + INTERCEPT

def act(observation: dict[str, np.ndarray], context: dict[str, Any]) -> int:
    mask = np.asarray(observation["action_mask"], dtype=bool)
    if mask.shape != (245,) or not mask.any(): raise ValueError("Unexpected arena action mask")
    return int(np.where(mask, logits(observation), -np.inf).argmax())
