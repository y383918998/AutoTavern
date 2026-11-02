"""Deterministic checks for engine-aware full-board replacement scoring."""
import numpy as np

from smart_agent import DEFAULT_CONFIG, _retention_value, _weakest_sell


brann = {"id": "BG_LOE_077", "atk": 2, "hp": 4, "tier": 5, "text": "Your Battlecries trigger twice."}
payoff = {"id": "payoff", "atk": 4, "hp": 4, "tier": 3, "text": "Battlecry: Get a spell."}
weak = {"id": "weak", "atk": 1, "hp": 1, "tier": 1, "text": ""}
mask = np.zeros(245, dtype=bool)
mask[16:19] = True
cfg = {**DEFAULT_CONFIG, "replacement_scoring": True, "engine_retention_bonus": 18.0}
assert _retention_value(brann, [brann, payoff, weak], cfg) == 18.0
assert _weakest_sell(mask, [brann, payoff, weak], cfg) == 2
assert _retention_value(brann, [brann, weak], cfg) == 0.0
print("PASS: active Brann is retained; the true weak minion is sold first.")
