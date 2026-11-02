"""Run the local policy against TavernGym's seven scripted bots.

Usage: python local_evaluate.py --games 32 --seed 0
No network connection is made by this program.
"""

from __future__ import annotations

import argparse
import importlib
import json
from statistics import mean

import numpy as np
import taverngym

def load_policy(spec: str):
    module, sep, name = spec.partition(":")
    if not sep:
        raise ValueError("--policy must use module:function")
    return getattr(importlib.import_module(module), name)


def run_game(seed: int, policy) -> int:
    # Mirror the arena's per-turn safety cap so a malformed learned policy
    # cannot inflate local results by taking unbounded reversible actions.
    game = taverngym.Game(seed=seed, config_json=json.dumps({"turn_action_caps": [100]}))
    policy_state: dict[str, object] = {}
    safety_limit = 20_000

    for _ in range(safety_limit):
        if game.is_over():
            return int(list(game.placements())[0])
        for seat in range(game.num_players):
            if not game.needs_action(seat):
                continue
            if seat:
                game.step(seat, game.bot_action(seat))
                continue
            view = json.loads(game.player_view(0))
            raw_features, raw_cards = game.observe2(0)
            action = policy(
                {"features": np.frombuffer(raw_features, dtype="<f4"),
                 "cards": np.frombuffer(raw_cards, dtype="<i4"),
                 "action_mask": np.asarray(view["mask"], dtype=bool)},
                {"player_view": view, "state": policy_state},
            )
            if not view["mask"][action]:
                raise RuntimeError(f"policy emitted illegal action {action}")
            game.step(0, action)
    raise RuntimeError("local game exceeded safety limit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", default="smart_agent:act")
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be positive")

    policy = load_policy(args.policy)
    placements = [run_game(args.seed + offset, policy) for offset in range(args.games)]
    summary = {
        "games": args.games,
        "seed": args.seed,
        "policy": args.policy,
        "mean_placement": round(mean(placements), 3),
        "wins": placements.count(1),
        "top2": sum(place <= 2 for place in placements),
        "top4": sum(place <= 4 for place in placements),
        "placements": placements,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
