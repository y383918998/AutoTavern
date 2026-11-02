"""Deterministic sanity checks for v6's visible engine scoring."""
from collections import Counter

from smart_agent import DEFAULT_CONFIG, _engine_value


cfg = {
    **DEFAULT_CONFIG,
    "engine_battlecry_bonus": 8.0,
    "engine_deathrattle_bonus": 8.0,
    "eternal_knight_bonus": 7.0,
    "public_prior_weight": 0.5,
}
brann = {"id": "BG_LOE_077", "text": "Your Battlecries trigger twice."}
titus = {"id": "BG25_354", "text": "Your Deathrattles trigger an extra time."}
assert _engine_value({"id": "x", "text": "Battlecry: Get a spell."}, {"board": [brann], "hand": []}, Counter(), cfg) == 8.0
assert _engine_value({"id": "x", "text": "Deathrattle: Summon a token."}, {"board": [titus], "hand": []}, Counter(), cfg) == 8.0
assert _engine_value({"id": "BG25_008", "text": ""}, {"board": [], "hand": []}, Counter({"BG25_008": 1}), cfg) == 7.3
assert _engine_value({"id": "BG36_620", "text": ""}, {"board": [], "hand": []}, Counter(), cfg) == -0.4
print("PASS: Brann/Titus payoffs, Eternal Knight scaling, and capped public tie-breaker work.")
