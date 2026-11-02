"""Collect audited, local counterfactual labels for non-hero choice decisions.

Only player-visible data are saved.  For every Discover/trinket/Dark Gift or
friendly target choice, each legal option is played from the same snapshot over
the same *reseeded* continuation set.  This estimates the expected placement
of alternatives without leaking the source game's future RNG stream.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import taverngym

import collect_rollout_preferences as rollout


SCHEMA_VERSION = 2
CHOICE_PROTOCOL = "choice-reseeded-counterfactual-v2"
LEGACY_PROTOCOL = (1, "choice-reseeded-counterfactual-v1")
SUPPORTED_PROTOCOLS = {LEGACY_PROTOCOL, (SCHEMA_VERSION, CHOICE_PROTOCOL)}
SCORABLE_KINDS = {"discover", "trinket", "dark_gift", "choose_one"}


def _stable_bucket(value: str, buckets: int) -> int:
    """A process-stable visible-card identity sketch (unlike Python hash())."""
    digest = hashlib.blake2s(value.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") % buckets


def _owned(me: dict[str, Any]) -> Counter[str]:
    return Counter(str(card.get("id", "")) for card in [*me.get("board", ()), *me.get("hand", ())])


def _state_features(me: dict[str, Any], turn: int) -> list[float]:
    board = list(me.get("board", ()))
    return [
        min(float(turn), 20.0) / 20.0, min(float(me.get("health", 0)), 40.0) / 40.0,
        min(float(me.get("gold", 0)), 20.0) / 20.0, min(float(me.get("tier", 0)), 7.0) / 7.0,
        len(board) / 7.0, len(me.get("hand", ())) / 10.0, len(me.get("shop", ())) / 7.0,
        min(sum(float(card.get("atk", 0)) for card in board), 200.0) / 200.0,
        min(sum(float(card.get("hp", 0)) for card in board), 200.0) / 200.0,
    ]


def _candidate_features(me: dict[str, Any], pending: dict[str, Any], action: int, turn: int) -> list[float]:
    """Visible state/option features.  v2 appends card identity and effect sketch."""
    board = list(me.get("board", ()))
    index = action - 30
    options = pending.get("options") or []
    targets = pending.get("targets") or []
    card: dict[str, Any] = options[index] if index < len(options) and isinstance(options[index], dict) else {}
    is_target = index < len(targets) and isinstance(targets[index], dict)
    if is_target:
        location = targets[index].get("location") or {}
        if location.get("zone") == "board" and 0 <= location.get("slot", -1) < len(board):
            card = board[location["slot"]]
    text = str(card.get("text", "")).upper()
    keys = set(card.get("keywords", ()))
    races = Counter(race for other in board for race in other.get("races", ()) if race)
    dominant = {race for race, count in races.items() if count == max(races.values(), default=0)}
    copies = _owned(me)[str(card.get("id", ""))]
    kind = str(pending.get("kind", "")).lower()
    base = _state_features(me, turn) + [
        min(float(card.get("atk", 0)), 100.0) / 100.0,
        min(float(card.get("hp", 0)), 100.0) / 100.0,
        float(card.get("tier", 0)) / 7.0, min(float(card.get("cost", 0)), 10.0) / 10.0,
        float(bool(card.get("golden"))), min(float(copies), 3.0) / 3.0,
        float(bool(dominant & set(card.get("races", ())))), float(is_target),
        float("BATTLECRY" in text or "BATTLECRY" in keys), float("DEATHRATTLE" in text or "DEATHRATTLE" in keys),
        float("DIVINE_SHIELD" in keys), float("REBORN" in keys), float("VENOMOUS" in keys),
        float("GOLD" in text or "COIN" in text), float("REFRESH" in text), float("DISCOVER" in text),
        float(kind == "discover"), float(kind == "trinket"), float(kind == "dark_gift"),
        float(kind == "choose_one"), float(kind == "target"),
    ]
    # Keep columns 0..29 unchanged: older family-specific tools rely on them.
    # A hashed ID sketch distinguishes otherwise identical-looking Discover and
    # trinket options without persisting hidden game state or raw snapshots.
    identity = [0.0] * 12
    card_id = str(card.get("id", ""))
    if card_id:
        identity[_stable_bucket(card_id, len(identity))] = 1.0
    effect_terms = (
        "BATTLECRY", "DEATHRATTLE", "AVENGE", "START OF COMBAT",
        "END OF TURN", "PERMANENT", "NEXT TURN", "DISCOUNT",
        "TAVERN", "SPELL", "REROLL", "SUMMON",
    )
    effects = [float(term in text) for term in effect_terms]
    # Explicit low-cardinality type features plus a few interactions.  These
    # let a linear listwise ranker express "economy trinket while rich" and
    # "pair completion later in the game" without an opaque lookup table.
    card_type = str(card.get("type", "")).lower()
    type_flags = [
        float("trinket" in card_type or kind == "trinket"),
        float("spell" in card_type or "SPELL" in text),
        float(bool(card.get("races", ()))),
    ]
    state = _state_features(me, turn)
    interactions = [
        base[14] * state[0],  # owned copies x turn
        base[22] * state[2],  # economy wording x gold
        base[11] * state[4],  # same-race fit x board size
        base[25] * state[3],  # Discover x tavern tier
    ]
    return base + identity + effects + type_flags + interactions


def _eligible_choices(observation: dict[str, np.ndarray], view: dict[str, Any], families: set[str]) -> list[int]:
    pending = (view.get("me") or {}).get("pending") or {}
    kind = str(pending.get("kind", "")).lower()
    targets = pending.get("targets") or []
    board_target = any(isinstance(target, dict) and (target.get("location") or {}).get("zone") == "board"
                       for target in targets)
    if kind not in SCORABLE_KINDS and not board_target:
        return []
    family = kind if kind in SCORABLE_KINDS else "target"
    if families and family not in families:
        return []
    return [int(action) for action in np.flatnonzero(observation["action_mask"][30:46]) + 30]


def collect_game(seed: int, samples_per_game: int, continuations: int,
                 families: set[str] | None = None) -> list[tuple[np.ndarray, list[int], list[list[int]]]]:
    if rollout._POLICY is None:
        raise RuntimeError("worker policy was not initialized")
    game = taverngym.Game(seed=seed, config_json=json.dumps({"turn_action_caps": [100]}))
    teacher_state: dict[str, Any] = {}
    samples: list[tuple[np.ndarray, list[int], list[list[int]]]] = []
    decisions = 0
    for _ in range(20_000):
        if game.is_over() or len(samples) >= samples_per_game:
            return samples
        progressed = False
        for seat in range(game.num_players):
            if not game.needs_action(seat):
                continue
            progressed = True
            if seat:
                game.step(seat, game.bot_action(seat))
                continue
            observation, view = rollout._observation(game)
            decisions += 1
            candidates = _eligible_choices(observation, view, families or set())
            if len(candidates) >= 2 and len(samples) < samples_per_game:
                me = view.get("me", {})
                pending = me.get("pending") or {}
                vectors = np.asarray([_candidate_features(me, pending, action, int(view.get("turn", 0)))
                                      for action in candidates], dtype=np.float32)
                snapshot = json.loads(game.state_json())
                placements: list[list[int]] = [[] for _ in candidates]
                for continuation in range(continuations):
                    branch_seed = rollout.continuation_seed(seed, decisions, continuation)
                    for index, action in enumerate(candidates):
                        branch = rollout._branch_from_snapshot(snapshot, branch_seed)
                        branch.step(0, action)
                        placements[index].append(rollout._finish(branch, rollout._POLICY, copy.deepcopy(teacher_state)))
                samples.append((vectors, candidates, placements))
            action = int(rollout._POLICY(observation, {"player_view": view, "state": teacher_state}))
            if not observation["action_mask"][action]:
                raise RuntimeError(f"teacher emitted illegal action {action}")
            game.step(0, action)
        if not progressed:
            raise RuntimeError("game made no progress")
    raise RuntimeError("source game exceeded safety limit")


def _worker(task: tuple[int, int, int, tuple[str, ...]]):
    return collect_game(*task)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="smart_agent_v8:act")
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples-per-game", type=int, default=4)
    parser.add_argument("--continuations", type=int, default=6)
    parser.add_argument("--families", default="all",
                        help="comma-separated choice kinds; e.g. discover,trinket, or all")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.games < 1 or args.samples_per_game < 1 or not 2 <= args.continuations <= 16 or not 1 <= args.jobs <= 8:
        parser.error("invalid collection limits")
    families = set() if args.families == "all" else {value.strip().lower() for value in args.families.split(",") if value.strip()}
    allowed = SCORABLE_KINDS | {"target"}
    if not families <= allowed:
        parser.error(f"--families must be all or a subset of {sorted(allowed)}")
    tasks = [(args.seed + offset, args.samples_per_game, args.continuations, tuple(sorted(families)))
             for offset in range(args.games)]
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=rollout._init_worker, initargs=(args.policy,)) as pool:
        nested = list(pool.map(_worker, tasks))
    records = [record for group in nested for record in group]
    if not records:
        raise RuntimeError("no eligible choice states; increase --games")
    width = max(len(record[1]) for record in records)
    dim = records[0][0].shape[1]
    features = np.zeros((len(records), width, dim), dtype=np.float32)
    actions = np.full((len(records), width), -1, dtype=np.int16)
    placements = np.zeros((len(records), width, args.continuations), dtype=np.int8)
    for row, (vectors, choices, results) in enumerate(records):
        features[row, :len(choices)] = vectors
        actions[row, :len(choices)] = choices
        placements[row, :len(choices)] = np.asarray(results, dtype=np.int8)
    means = placements.sum(axis=2, dtype=np.float32) / args.continuations
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, candidate_features=features, candidate_actions=actions,
                        candidate_placements=placements, candidate_mean_placements=means,
                        metadata=json.dumps({"schema_version": SCHEMA_VERSION, "choice_protocol": CHOICE_PROTOCOL,
                                             "rng_protocol": rollout.RNG_PROTOCOL, "visible_observation_only": True,
                                             "policy": args.policy, "games": args.games, "first_seed": args.seed,
                                             "continuations": args.continuations, "families": sorted(families) or ["all"],
                                             "feature_schema": "visible_choice_v2",
                                             "feature_dim": int(dim),
                                             "source_hash": taverngym.build_info()["source_hash"]}))
    best = np.min(np.where(actions >= 0, means, 9), axis=1)
    print(json.dumps({"out": str(args.out), "states": len(records), "candidate_actions": int(np.count_nonzero(actions >= 0)),
                      "rollout_evaluations": int(np.count_nonzero(actions >= 0) * args.continuations),
                      "best_action_mean_placement": round(float(best.mean()), 3)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
