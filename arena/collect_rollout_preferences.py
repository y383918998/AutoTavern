"""Build local counterfactual action-preference data from TavernGym.

For sampled decisions made by a visible-state teacher policy, this tool clones
the *local* game, applies several legal first actions, then lets the same
teacher finish each branch.  It stores only owner-visible observations, legal
actions, and local final placements.  It never contacts Arena or saves a game
snapshot, replay, key, or hidden state in the output.

Each branch is explicitly re-seeded after it is cloned.  Thus a rollout is an
independent counterfactual continuation, not a copy that accidentally exposes
the source game's future random stream.  Across many states these outcomes
form preference labels.  The saved dataset contains a provenance record so
that the audit tool can reject data collected with an older protocol.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
import taverngym


Policy = Callable[[dict[str, np.ndarray], dict[str, Any]], int]
_POLICY: Policy | None = None
SCHEMA_VERSION = 2
RNG_PROTOCOL = "reseeded-counterfactual-v1"


def load_policy(spec: str) -> Policy:
    module, separator, name = spec.partition(":")
    if not separator:
        raise ValueError("policy must use module:function")
    return getattr(importlib.import_module(module), name)


def _init_worker(policy_spec: str) -> None:
    global _POLICY
    _POLICY = load_policy(policy_spec)


def _observation(game: taverngym.Game) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    view = json.loads(game.player_view(0))
    raw_features, raw_cards = game.observe2(0)
    return ({
        "features": np.frombuffer(raw_features, dtype="<f4").copy(),
        "cards": np.frombuffer(raw_cards, dtype="<i4").copy(),
        "action_mask": np.asarray(view["mask"], dtype=bool),
    }, view)


def _finish(game: taverngym.Game, policy: Policy, policy_state: dict[str, Any]) -> int:
    """Run one branch to its local terminal placement with legal visible actions."""
    for _ in range(20_000):
        if game.is_over():
            return int(list(game.placements())[0])
        for seat in range(game.num_players):
            if not game.needs_action(seat):
                continue
            if seat:
                game.step(seat, game.bot_action(seat))
                continue
            observation, view = _observation(game)
            action = int(policy(observation, {"player_view": view, "state": policy_state}))
            if not observation["action_mask"][action]:
                raise RuntimeError(f"rollout policy emitted illegal action {action}")
            game.step(0, action)
    raise RuntimeError("counterfactual rollout exceeded safety limit")


def _candidate_actions(mask: np.ndarray, teacher_action: int, max_actions: int) -> list[int]:
    """Cover strategic alternatives first while retaining only legal actions."""
    legal = set(map(int, np.flatnonzero(mask)))
    groups = (
        [teacher_action],
        list(range(8, 16)),       # buys
        [3, 1, 0, 4, 5],          # tier, refresh, end, powers
        list(range(30, 46)),      # Discover/choose
        list(range(16, 30)),      # sell/activate
        list(range(46, 196)),     # play/cast/magnetize
        list(range(196, 245)),    # positioning
    )
    result: list[int] = []
    for group in groups:
        for action in group:
            if action in legal and action not in result:
                result.append(action)
                if len(result) >= max_actions:
                    return result
    return result


def continuation_seed(source_seed: int, decision: int, index: int) -> int:
    """Derive a non-zero local RNG seed without retaining the source RNG state."""
    payload = f"{RNG_PROTOCOL}:{source_seed}:{decision}:{index}".encode("ascii")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return value or 1


def _branch_from_snapshot(snapshot: dict[str, Any], branch_seed: int) -> taverngym.Game:
    """Create a local branch with a shared, reproducible future RNG stream."""
    state = copy.deepcopy(snapshot)
    rng = random.Random(branch_seed)
    state["seed"] = branch_seed
    # This state is never persisted.  All candidate actions use the same stream
    # for a continuation, which reduces comparison noise without exposing it to
    # the policy.
    state["rng"] = {"s": [rng.getrandbits(64) for _ in range(4)]}
    if not any(state["rng"]["s"]):
        state["rng"]["s"][0] = 1
    return taverngym.Game.from_json(json.dumps(state))


def collect_game(seed: int, samples_per_game: int, every: int, max_actions: int, continuations: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int, list[int], list[list[int]]]]:
    if _POLICY is None:
        raise RuntimeError("worker policy was not initialized")
    game = taverngym.Game(seed=seed, config_json=json.dumps({"turn_action_caps": [100]}))
    teacher_state: dict[str, Any] = {}
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int, list[int], list[list[int]]]] = []
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
            observation, view = _observation(game)
            teacher_action = int(_POLICY(observation, {"player_view": view, "state": teacher_state}))
            if not observation["action_mask"][teacher_action]:
                raise RuntimeError(f"teacher emitted illegal action {teacher_action}")
            decisions += 1
            candidates = _candidate_actions(observation["action_mask"], teacher_action, max_actions)
            # Do not spend branch rollouts on states with no decision alternative.
            if len(samples) < samples_per_game and len(candidates) >= 2 and decisions % every == 0:
                snapshot = json.loads(game.state_json())
                placements: list[list[int]] = [[] for _ in candidates]
                continuation_seeds = [continuation_seed(seed, decisions, index)
                                      for index in range(continuations)]
                for branch_seed in continuation_seeds:
                    for index, action in enumerate(candidates):
                        branch = _branch_from_snapshot(snapshot, branch_seed)
                        branch_state = copy.deepcopy(teacher_state)
                        branch.step(0, action)
                        placements[index].append(_finish(branch, _POLICY, branch_state))
                samples.append((observation["features"], observation["cards"], observation["action_mask"],
                                int(view.get("turn", 0)), teacher_action, candidates, placements))
            game.step(0, teacher_action)
        if not progressed:
            raise RuntimeError("game made no progress")
    raise RuntimeError("source game exceeded safety limit")


def _collect_worker(task: tuple[int, int, int, int, int]):
    return collect_game(*task)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="smart_agent_v8:act")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples-per-game", type=int, default=4)
    parser.add_argument("--every", type=int, default=20, help="sample every N owner decisions")
    parser.add_argument("--max-actions", type=int, default=8)
    parser.add_argument("--continuations", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.games < 1 or args.samples_per_game < 1 or args.every < 1 or not 2 <= args.max_actions <= 16 or not 1 <= args.jobs <= 8 or not 2 <= args.continuations <= 16:
        parser.error("invalid collection limits")

    tasks = [(args.seed + offset, args.samples_per_game, args.every, args.max_actions, args.continuations) for offset in range(args.games)]
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker, initargs=(args.policy,)) as pool:
        nested = list(pool.map(_collect_worker, tasks))
    records = [record for group in nested for record in group]
    if not records:
        raise RuntimeError("no preference states sampled; reduce --every")

    width = args.max_actions
    features = np.asarray([record[0] for record in records], dtype=np.float32)
    cards = np.asarray([record[1] for record in records], dtype=np.int32)
    masks = np.asarray([record[2] for record in records], dtype=bool)
    turns = np.asarray([record[3] for record in records], dtype=np.int16)
    teacher_actions = np.asarray([record[4] for record in records], dtype=np.int16)
    actions = np.full((len(records), width), -1, dtype=np.int16)
    placements = np.full((len(records), width, args.continuations), 0, dtype=np.int8)
    for row, record in enumerate(records):
        choices, results = record[5], record[6]
        actions[row, :len(choices)] = choices
        placements[row, :len(results), :] = np.asarray(results, dtype=np.int8)
    counts = np.count_nonzero(placements > 0, axis=2)
    sums = np.where(placements > 0, placements, 0).sum(axis=2, dtype=np.float32)
    mean_placements = np.zeros((len(records), width), dtype=np.float32)
    np.divide(sums, counts, out=mean_placements, where=counts > 0)
    best_actions = actions[np.arange(len(records)), np.nanargmin(np.where(actions >= 0, mean_placements, np.nan), axis=1)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, features=features, cards=cards, action_mask=masks, turns=turns,
        teacher_actions=teacher_actions, candidate_actions=actions,
        candidate_placements=placements, candidate_mean_placements=mean_placements.astype(np.float32), best_actions=best_actions,
        metadata=json.dumps({"schema_version": SCHEMA_VERSION, "rng_protocol": RNG_PROTOCOL,
                             "visible_observation_only": True,
                             "policy": args.policy, "games": args.games, "first_seed": args.seed,
                             "samples_per_game": args.samples_per_game, "every": args.every,
                             "max_actions": args.max_actions, "continuations": args.continuations, "jobs": args.jobs,
                             "source_hash": taverngym.build_info()["source_hash"]}),
    )
    teacher_index = np.argmax(actions == teacher_actions[:, None], axis=1)
    teacher_placements = mean_placements[np.arange(len(records)), teacher_index]
    best_placements = np.min(np.where(actions >= 0, mean_placements, 9), axis=1)
    print(json.dumps({"out": str(args.out), "states": len(records), "candidate_actions": int(np.count_nonzero(actions >= 0)),
                      "rollout_evaluations": int(np.count_nonzero(actions >= 0) * args.continuations),
                      "teacher_mean_placement": round(float(teacher_placements.mean()), 3),
                      "best_action_mean_placement": round(float(best_placements.mean()), 3)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
