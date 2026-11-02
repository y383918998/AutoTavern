"""Safely compare interpretable smart-policy rule candidates on local seed splits."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import mean
from local_evaluate import run_game
from smart_agent import DEFAULT_CONFIG, act_with_config

GRID = [
    DEFAULT_CONFIG,
    {**DEFAULT_CONFIG, "upgrade_turn_factor": 1.6},
    {**DEFAULT_CONFIG, "upgrade_turn_factor": 2.4},
    {**DEFAULT_CONFIG, "pair_bonus": 24.0},
    {**DEFAULT_CONFIG, "refresh_min_gold": 4},
]

def evaluate(config, seeds):
    policy = lambda observation, context: act_with_config(observation, context, config)
    places = [run_game(seed, policy) for seed in seeds]
    return {"mean_placement": mean(places), "top2": sum(p <= 2 for p in places), "top4": sum(p <= 4 for p in places),
            "eighth": places.count(8), "placements": places}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--search-games", type=int, default=16)
    p.add_argument("--valid-games", type=int, default=64)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    search_seeds = range(a.seed, a.seed + a.search_games)
    rows = [{"config": c, "search": evaluate(c, search_seeds)} for c in GRID]
    winner = min(rows, key=lambda row: row["search"]["mean_placement"])
    valid_seeds = range(a.seed + 100_000_000, a.seed + 100_000_000 + a.valid_games)
    baseline, candidate = evaluate(DEFAULT_CONFIG, valid_seeds), evaluate(winner["config"], valid_seeds)
    approved = (candidate["mean_placement"] <= baseline["mean_placement"] - 0.20
                and candidate["eighth"] <= baseline["eighth"])
    result = {"seed": a.seed, "search_games": a.search_games, "valid_games": a.valid_games,
              "baseline_config": DEFAULT_CONFIG, "search": rows, "winner": winner["config"],
              "validation": {"baseline": baseline, "candidate": candidate}, "approved": approved}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(a.out), "winner": winner["config"],
                      "baseline_mean": round(baseline["mean_placement"], 3),
                      "candidate_mean": round(candidate["mean_placement"], 3), "approved": approved}))

if __name__ == "__main__": main()
