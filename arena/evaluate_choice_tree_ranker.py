"""Cross-batch, non-linear baseline for visible choice-rollout labels.

This is deliberately an evaluator, not a policy loader.  It tests whether the
choice signal is non-linear before adding any learned behaviour to the agent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from train_choice_ranker import _confident, _family, _load


def _rows(features: np.ndarray, actions: np.ndarray, placements: np.ndarray):
    x, y, groups = [], [], []
    for state, (vectors, action_row, placement_row) in enumerate(zip(features, actions, placements)):
        legal = action_row >= 0
        count = int(legal.sum())
        x.append(vectors[legal])
        y.append(placement_row[legal])
        groups.extend([state] * count)
    return np.concatenate(x), np.concatenate(y), np.asarray(groups)


def _score(model, features: np.ndarray, actions: np.ndarray, placements: np.ndarray) -> dict[str, float]:
    chosen, random, oracle = [], [], []
    for vectors, action_row, placement_row in zip(features, actions, placements):
        legal = action_row >= 0
        values = placement_row[legal]
        pick = int(np.argmin(model.predict(vectors[legal])))
        chosen.append(float(values[pick]))
        random.append(float(values.mean()))
        oracle.append(float(values.min()))
    return {"chosen_mean_placement": float(np.mean(chosen)),
            "random_mean_placement": float(np.mean(random)),
            "oracle_mean_placement": float(np.mean(oracle)),
            "improvement_vs_random": float(np.mean(random) - np.mean(chosen))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--valid", required=True, type=Path)
    parser.add_argument("--family", default="discover", choices=("all", "discover", "trinket", "dark_gift", "choose_one", "target"))
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--min-leaf", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train_features, train_actions, train_placements, train_meta = _load(args.train)
    valid_features, valid_actions, valid_placements, valid_meta = _load(args.valid)
    if (train_meta.get("schema_version"), train_meta.get("choice_protocol"), train_meta.get("source_hash")) != (valid_meta.get("schema_version"), valid_meta.get("choice_protocol"), valid_meta.get("source_hash")):
        raise ValueError("train and valid must share feature protocol and simulator build")
    train_features, train_actions, train_placements = _family(train_features, train_actions, train_placements, args.family)
    valid_features, valid_actions, valid_placements = _family(valid_features, valid_actions, valid_placements, args.family)
    if len(train_features) < 128 or len(valid_features) < 128:
        raise ValueError("need 128 choice states in each independent batch")
    x, y, groups = _rows(train_features, train_actions, train_placements)
    # Equal total influence for each decision, regardless of option count.
    weights = np.bincount(groups)[groups].astype(np.float64)
    model = ExtraTreesRegressor(n_estimators=args.trees, max_depth=10,
                                min_samples_leaf=args.min_leaf, max_features=0.8,
                                n_jobs=-1, random_state=args.seed)
    model.fit(x, y, sample_weight=1.0 / weights)
    report = {"family": args.family, "train_states": int(len(train_features)),
              "valid_states": int(len(valid_features)), "trees": args.trees,
              "min_leaf": args.min_leaf, "valid": _score(model, valid_features, valid_actions, valid_placements)}
    # A 0.10 placement gain is deliberately below the eventual policy gate but
    # high enough to distinguish real signal from a random option preference.
    report["worth_policy_integration"] = report["valid"]["improvement_vs_random"] >= 0.10
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
