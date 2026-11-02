"""Deterministic checks for v8 low-health tempo behavior."""
from collections import Counter

from smart_agent import DEFAULT_CONFIG, _buy_score


brann = {"id": "BG_LOE_077", "text": "Your Battlecries trigger twice."}
battlecry = {"id": "bc", "type": "minion", "atk": 2, "hp": 2, "tier": 1, "text": "Battlecry: Get a spell."}
fighter = {"id": "fighter", "type": "minion", "atk": 5, "hp": 5, "tier": 1, "text": ""}
cfg = {**DEFAULT_CONFIG, "engine_battlecry_bonus": 8.0, "low_health_threshold": 15, "low_health_stat_weight": 0.5}
safe = {"health": 30, "board": [brann], "hand": []}
danger = {"health": 12, "board": [brann], "hand": []}
assert _buy_score(battlecry, safe, Counter(), Counter(), 15.0, cfg) > _buy_score(fighter, safe, Counter(), Counter(), 15.0, cfg)
assert _buy_score(fighter, danger, Counter(), Counter(), 15.0, cfg) > _buy_score(battlecry, danger, Counter(), Counter(), 15.0, cfg)
print("PASS: v8 retains engine preference when safe and prioritizes immediate stats at low health.")
