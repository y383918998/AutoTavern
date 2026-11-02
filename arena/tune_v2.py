"""Conservatively tune v2 against independent local seed splits."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import mean
from local_evaluate import run_game
from smart_agent import act_with_config
from smart_agent_v2 import CONFIG as BASE

GRID = [
    BASE,
    {**BASE, "upgrade_turn_factor": 2.2},
    {**BASE, "upgrade_turn_factor": 2.6},
    {**BASE, "pair_bonus": 24.0},
    {**BASE, "refresh_min_gold": 4},
    {**BASE, "sell_margin": 2.0},
    {**BASE, "sell_margin": 6.0},
]

def evaluate(config, seeds):
    policy = lambda observation, context: act_with_config(observation, context, config)
    places = [run_game(seed, policy) for seed in seeds]
    return {"mean_placement": mean(places), "top2": sum(p <= 2 for p in places),
            "top4": sum(p <= 4 for p in places), "eighth": places.count(8), "games": len(places)}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True); p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--search-games", type=int, default=32); p.add_argument("--valid-games", type=int, default=128)
    p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    rows=[]
    for r in range(a.rounds):
        root=(a.seed + r * 500_000_000) % 1_900_000_000
        winner, search = min(((c, evaluate(c, range(root, root+a.search_games))) for c in GRID),
                             key=lambda x: x[1]["mean_placement"])
        baseline=evaluate(BASE, range(root+100_000_000, root+100_000_000+a.valid_games))
        candidate=evaluate(winner, range(root+100_000_000, root+100_000_000+a.valid_games))
        passed=(candidate["mean_placement"] <= baseline["mean_placement"]-0.15
                and candidate["top4"] >= baseline["top4"] and candidate["eighth"] <= baseline["eighth"])
        rows.append({"round":r+1,"winner":winner,"search":search,"baseline":baseline,"candidate":candidate,"passed":passed})
        print(json.dumps({"round":r+1,"winner":winner,"baseline_mean":round(baseline["mean_placement"],3),"candidate_mean":round(candidate["mean_placement"],3),"passed":passed}),flush=True)
    winner_keys={json.dumps(x["winner"],sort_keys=True) for x in rows}
    approved=len(winner_keys)==1 and all(x["passed"] for x in rows)
    report={"baseline":BASE,"rounds":rows,"approved":approved}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"out":str(a.out),"approved":approved,"winner":rows[0]["winner"] if approved else None}))

if __name__=="__main__": main()
