"""Local sanity checks for conservative tribe-commitment scoring."""
from smart_agent import _dominant_races, _synergy_value, DEFAULT_CONFIG

me={"board":[{"races":["ELEMENTAL"]},{"races":["ELEMENTAL"]}],"hand":[]}
races=_dominant_races(me)
cfg={**DEFAULT_CONFIG,"tribe_synergy_bonus":7.0,"tribe_text_bonus":3.0}
assert _synergy_value({"races":["ELEMENTAL"],"text":""},races,cfg)==7.0
assert _synergy_value({"races":["DEMON"],"text":""},races,cfg)==0.0
assert _synergy_value({"races":[],"text":"Your Elementals have +1 Attack."},races,cfg)==3.0
print("PASS: two visible Elementals prefer Elemental cards and Elemental enablers; off-tribe cards get no bonus.")
