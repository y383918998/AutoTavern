"""Collect local-only observe2/action trajectories from TavernGym's scripted bot.

Example: python collect_bot_data.py --games 16 --seed 0 --out artifacts/bot_train.npz
The output contains no server data and is intentionally ignored by Git.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import taverngym


def collect_game(seed: int) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int]]:
    game = taverngym.Game(seed=seed)
    features: list[np.ndarray] = []
    cards: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []
    while not game.is_over():
        for seat in range(game.num_players):
            if not game.needs_action(seat):
                continue
            raw_features, raw_cards = game.observe2(seat)
            mask = np.frombuffer(game.legal_mask(seat), dtype=np.uint8).astype(bool, copy=True)
            action = int(game.bot_action(seat))
            if not mask[action]:
                raise RuntimeError(f"bot emitted illegal action {action}")
            features.append(np.frombuffer(raw_features, dtype="<f4").copy())
            cards.append(np.frombuffer(raw_cards, dtype="<i4").copy())
            masks.append(mask)
            actions.append(action)
            game.step(seat, action)
    return features, cards, masks, actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be positive")

    all_features: list[np.ndarray] = []
    all_cards: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_actions: list[int] = []
    for offset in range(args.games):
        features, cards, masks, actions = collect_game(args.seed + offset)
        all_features.extend(features)
        all_cards.extend(cards)
        all_masks.extend(masks)
        all_actions.extend(actions)

    x = np.asarray(all_features, dtype=np.float32)
    c = np.asarray(all_cards, dtype=np.int32)
    m = np.asarray(all_masks, dtype=bool)
    y = np.asarray(all_actions, dtype=np.int16)
    if x.shape[1:] != (4017,) or c.shape[1:] != (447,) or m.shape[1:] != (245,):
        raise RuntimeError(f"unexpected observation shapes: {x.shape}, {c.shape}, {m.shape}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        features=x,
        cards=c,
        action_mask=m,
        actions=y,
        metadata=json.dumps({
            "games": args.games,
            "first_seed": args.seed,
            "source_hash": taverngym.build_info()["source_hash"],
            "layout": dict(taverngym.layout()),
            "game_version": taverngym.game_version(),
        }),
    )
    print(json.dumps({"out": str(args.out), "samples": len(y), "games": args.games}))


if __name__ == "__main__":
    main()
