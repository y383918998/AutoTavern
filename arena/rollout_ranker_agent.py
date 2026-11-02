"""Local-only rollout-ranker policy.  It is not the stable Arena policy."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from policy_features import combine


path = os.environ.get("TAVERN_ROLLOUT_MODEL", "")
if not path:
    raise RuntimeError("Set TAVERN_ROLLOUT_MODEL to a local rollout-ranker .npz")
model = np.load(Path(path))
COEF, INTERCEPT, MEAN, SCALE = (model[key] for key in ("coef", "intercept", "mean", "scale"))
if COEF.shape != (245, 4081) or INTERCEPT.shape != (245,):
    raise RuntimeError("Unexpected rollout-ranker shape")


def scores(observation: dict[str, np.ndarray]) -> np.ndarray:
    x = (combine(observation["features"], observation["cards"]) - MEAN) / SCALE
    return x @ COEF.T + INTERCEPT


def act(observation: dict[str, np.ndarray], context: dict[str, Any]) -> int:
    mask = np.asarray(observation["action_mask"], dtype=bool)
    if mask.shape != (245,) or not mask.any():
        raise ValueError("Unexpected Arena action mask")
    return int(np.where(mask, scores(observation), -np.inf).argmax())
