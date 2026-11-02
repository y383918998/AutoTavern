"""A local, deterministic policy for TavernArena.

It consumes only the authenticated seat's standard observation and returns one
currently legal TavernGym action.  No data, checkpoint, or game state leaves
this computer; the connector transmits only the resulting action integer.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _first_allowed(mask: np.ndarray, start: int, stop: int) -> int | None:
    """Return the first allowed action in the half-open range, if one exists."""
    for action in range(start, stop):
        if bool(mask[action]):
            return action
    return None


def act(observation: dict[str, np.ndarray], context: dict[str, Any]) -> int:
    """Choose a legal action.

    Priority is: resolve a mandatory choice, buy a tavern card, upgrade, use a
    hero power, then end the turn.  The final fallback protects against future
    action-layout changes by selecting an action directly from the legal mask.
    """
    mask = np.asarray(observation["action_mask"], dtype=bool)
    if mask.shape != (245,) or not mask.any():
        raise ValueError("Arena observation has no legal action")

    for start, stop in ((30, 46), (8, 16), (3, 4), (4, 6)):
        action = _first_allowed(mask, start, stop)
        if action is not None:
            return action

    if mask[0]:  # EndTurn
        return 0

    return int(np.flatnonzero(mask)[0])
