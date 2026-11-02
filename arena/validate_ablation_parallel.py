"""Paired local policy validation using independent CPU worker processes.

Each worker receives complete games with distinct seeds.  Baseline and
candidate therefore see exactly the same seed set, just as in the serial
validator.  No network, key, model upload, or shared game state is used.
"""
from __future__ import annotations

import argparse
import importlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

from local_evaluate import run_game


_BASELINE = None
_CANDIDATE = None


def _load(spec: str):
    module, separator, name = spec.partition(":")
    if not separator:
        raise ValueError("policy must use module:function")
    return getattr(importlib.import_module(module), name)


def _init_worker(baseline_spec: str, candidate_spec: str) -> None:
    global _BASELINE, _CANDIDATE
    _BASELINE = _load(baseline_spec)
    _CANDIDATE = _load(candidate_spec)


def _run_pair(seed: int) -> tuple[int, int]:
    if _BASELINE is None or _CANDIDATE is None:
        raise RuntimeError("worker policy was not initialized")
    return run_game(seed, _BASELINE), run_game(seed, _CANDIDATE)


def _stats(placements: list[int]) -> dict[str, int | float]:
    return {
        "mean_placement": mean(placements),
        "wins": placements.count(1),
        "top2": sum(place <= 2 for place in placements),
        "top4": sum(place <= 4 for place in placements),
        "eighth": placements.count(8),
        "games": len(placements),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="smart_agent_v2:act")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--games", default=512, type=int)
    parser.add_argument("--jobs", default=4, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.games < 1 or not 1 <= args.jobs <= 8:
        parser.error("--games must be positive and --jobs must be between 1 and 8")

    seeds = range(args.seed, args.seed + args.games)
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker,
                             initargs=(args.baseline, args.policy)) as pool:
        pairs = list(pool.map(_run_pair, seeds, chunksize=max(1, args.games // (args.jobs * 8))))
    baseline = _stats([pair[0] for pair in pairs])
    candidate = _stats([pair[1] for pair in pairs])
    improvement = baseline["mean_placement"] - candidate["mean_placement"]
    approved = (
        improvement >= 0.15
        and candidate["top4"] >= baseline["top4"]
        and candidate["eighth"] <= baseline["eighth"]
    )
    report = {
        "policy": args.policy,
        "baseline_policy": args.baseline,
        "seed": args.seed,
        "jobs": args.jobs,
        "baseline": baseline,
        "candidate": candidate,
        "mean_placement_improvement": improvement,
        "approved": approved,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(args.out), "policy": args.policy, "jobs": args.jobs,
        "baseline_mean": round(baseline["mean_placement"], 3),
        "candidate_mean": round(candidate["mean_placement"], 3), "approved": approved,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
