"""Validate one fixed, interpretable policy configuration on unseen local seeds."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import mean
from local_evaluate import run_game
from smart_agent import DEFAULT_CONFIG, act_with_config

def evaluate(config, seeds):
    policy = lambda observation, context: act_with_config(observation, context, config)
    places = [run_game(seed, policy) for seed in seeds]
    return {"mean_placement": mean(places), "top2": sum(p <= 2 for p in places), "top4": sum(p <= 4 for p in places),
            "wins": places.count(1), "eighth": places.count(8), "games": len(places)}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--games", type=int, default=512)
    p.add_argument("--config", required=True, help="JSON object with smart-agent parameters")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    candidate = json.loads(a.config)
    seeds = range(a.seed, a.seed + a.games)
    baseline, result = evaluate(DEFAULT_CONFIG, seeds), evaluate(candidate, seeds)
    improvement = baseline["mean_placement"] - result["mean_placement"]
    approved = improvement >= 0.20 and result["eighth"] <= baseline["eighth"]
    report = {"seed": a.seed, "candidate": candidate, "baseline": baseline, "candidate_result": result,
              "mean_placement_improvement": improvement, "approved": approved}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(a.out), "baseline_mean": round(baseline["mean_placement"], 3),
                      "candidate_mean": round(result["mean_placement"], 3),
                      "baseline_top2": baseline["top2"], "candidate_top2": result["top2"],
                      "improvement": round(improvement, 3), "approved": approved}))

if __name__ == "__main__": main()
