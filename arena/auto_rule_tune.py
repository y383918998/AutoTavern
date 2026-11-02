"""Multi-round local rule search with conservative, automatic promotion.

Promotion never edits smart_agent.py. A candidate must win every independent
round and pass the aggregate gate before a standalone config is written.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import mean
from local_evaluate import run_game
from rule_search import GRID
from smart_agent import DEFAULT_CONFIG, act_with_config

def key(config):
    return json.dumps(config, sort_keys=True)

def evaluate(config, seeds):
    policy = lambda observation, context: act_with_config(observation, context, config)
    places = [run_game(seed, policy) for seed in seeds]
    return {"mean_placement": mean(places), "top2": sum(p <= 2 for p in places), "top4": sum(p <= 4 for p in places),
            "eighth": places.count(8), "games": len(places)}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--search-games", type=int, default=32)
    p.add_argument("--valid-games", type=int, default=128)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    if a.rounds < 2: p.error("--rounds must be at least 2")
    rounds = []
    for r in range(a.rounds):
        base = (a.seed + r * 500_000_000) % 1_900_000_000
        search = range(base, base + a.search_games)
        scored = [(config, evaluate(config, search)) for config in GRID]
        winner, search_score = min(scored, key=lambda row: row[1]["mean_placement"])
        valid = range(base + 100_000_000, base + 100_000_000 + a.valid_games)
        baseline, candidate = evaluate(DEFAULT_CONFIG, valid), evaluate(winner, valid)
        passed = (candidate["mean_placement"] <= baseline["mean_placement"] - 0.20
                  and candidate["eighth"] <= baseline["eighth"])
        rounds.append({"round": r + 1, "seed": base, "winner": winner, "search": search_score,
                       "validation": {"baseline": baseline, "candidate": candidate}, "passed": passed})
        print(json.dumps({"round": r + 1, "winner": winner, "passed": passed,
                          "baseline_mean": round(baseline["mean_placement"], 3),
                          "candidate_mean": round(candidate["mean_placement"], 3)}), flush=True)
    same = len({key(row["winner"]) for row in rounds}) == 1
    all_passed = all(row["passed"] for row in rounds)
    base_mean = mean(row["validation"]["baseline"]["mean_placement"] for row in rounds)
    cand_mean = mean(row["validation"]["candidate"]["mean_placement"] for row in rounds)
    base_eighth = sum(row["validation"]["baseline"]["eighth"] for row in rounds)
    cand_eighth = sum(row["validation"]["candidate"]["eighth"] for row in rounds)
    approved = same and all_passed and cand_mean <= base_mean - 0.20 and cand_eighth <= base_eighth
    a.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"seed": a.seed, "rounds": rounds, "same_winner": same, "all_passed": all_passed,
              "aggregate": {"baseline_mean": base_mean, "candidate_mean": cand_mean,
                            "baseline_eighth": base_eighth, "candidate_eighth": cand_eighth},
              "approved": approved}
    (a.out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if approved:
        config = {"version": "smart_agent_v2", "source": "auto_rule_tune", "config": rounds[0]["winner"],
                  "aggregate": report["aggregate"]}
        (a.out_dir / "promoted_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(a.out_dir), "approved": approved,
                      "promoted_config": str(a.out_dir / "promoted_config.json") if approved else None}))

if __name__ == "__main__": main()
