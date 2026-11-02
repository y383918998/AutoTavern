"""Deterministic local examples for the v9 pending-choice rule.

Run: python choice_examples.py
"""
from __future__ import annotations

import numpy as np

from smart_agent_v9_choice_candidate import act


def card(card_id: str, atk: int, hp: int, *, tier: int = 1, races=(), text: str = "", cost: int = 0):
    return {"id": card_id, "type": "minion", "atk": atk, "hp": hp, "tier": tier,
            "races": list(races), "text": text, "cost": cost, "keywords": []}


def observation(*actions: int):
    mask = np.zeros(245, dtype=bool)
    mask[list(actions)] = True
    return {"action_mask": mask}


def main() -> None:
    # Discover: the second option completes a visible pair, outweighing a raw-stat lead.
    pair = card("PAIR", 3, 3, tier=2, races=("BEAST",))
    stronger = card("BIG", 5, 5, tier=2)
    view = {"me": {"board": [pair], "hand": [], "shop": [], "health": 30,
                    "pending": {"options": [stronger, pair], "targets": []}}}
    assert act(observation(30, 31), {"player_view": view, "state": {}}) == 31

    # A targeted buff selects the strongest legal visible board target, not action order.
    small, large = card("SMALL", 2, 2), card("LARGE", 8, 8)
    target_view = {"me": {"board": [small, large], "hand": [], "shop": [], "health": 30,
                           "pending": {"options": [], "targets": [
                               {"location": {"zone": "board", "slot": 0}},
                               {"location": {"zone": "board", "slot": 1}},
                           ]}}}
    assert act(observation(30, 31), {"player_view": target_view, "state": {}}) == 31
    # Hero selection is intentionally unchanged: it needs hero-specific planning.
    hero_view = {"me": {"board": [], "hand": [], "shop": [], "health": 30,
                         "pending": {"kind": "hero", "options": [
                             {"id": "HERO_A", "type": "hero", "armor": 0},
                             {"id": "HERO_B", "type": "hero", "armor": 20},
                         ], "targets": []}}}
    assert act(observation(30, 31), {"player_view": hero_view, "state": {}}) == 30
    print("PASS: v9 scores visible pair synergy and legal friendly targets; no hidden state is used.")


if __name__ == "__main__":
    main()
