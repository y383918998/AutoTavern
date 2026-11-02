"""Train a compact shared listwise ranker for audited choice rollout data.

This trains one small weight vector over visible state-option features; it is
not a 245-by-observation lookup table and cannot be deployed automatically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from collect_choice_preferences import SUPPORTED_PROTOCOLS
from collect_rollout_preferences import RNG_PROTOCOL

# Candidate feature columns 25..29 encode the visible pending-menu family.
# They are part of the persisted collector contract, not inferred from actions.
FAMILY_COLUMNS = {"discover": 25, "trinket": 26, "dark_gift": 27,
                  "choose_one": 28, "target": 29}

def _load(path: Path):
    data = np.load(path)
    metadata = json.loads(str(data["metadata"]))
    if (metadata.get("schema_version"), metadata.get("choice_protocol")) not in SUPPORTED_PROTOCOLS or metadata.get("rng_protocol") != RNG_PROTOCOL:
        raise ValueError("unrecognized choice dataset protocol")
    return (data["candidate_features"].astype(np.float32), data["candidate_actions"].astype(np.int16),
            data["candidate_mean_placements"].astype(np.float32), metadata)


def _metrics(features, actions, labels, weights):
    correct = 0
    random_accuracy = []
    for row in range(len(features)):
        legal = actions[row] >= 0
        scores = features[row, legal] @ weights
        chosen = int(np.argmax(scores))
        correct += int(labels[row, legal][chosen] > 0)
        random_accuracy.append(float(labels[row, legal].sum()) / int(legal.sum()))
    return correct / len(features), float(np.mean(random_accuracy))


def _confident(features, actions, placements, min_gap: float):
    safe = np.where(actions >= 0, placements, 9.0)
    ordered = np.sort(safe, axis=1)
    keep = (ordered[:, 1] - ordered[:, 0]) >= min_gap
    features, actions, placements = features[keep], actions[keep], placements[keep]
    best = np.min(np.where(actions >= 0, placements, 9.0), axis=1, keepdims=True)
    labels = ((actions >= 0) & (placements == best)).astype(np.float32)
    labels /= labels.sum(axis=1, keepdims=True)
    return features, actions, labels


def _family(features, actions, placements, family: str):
    """Keep whole decision states of one visible choice family."""
    if family == "all":
        return features, actions, placements
    column = FAMILY_COLUMNS[family]
    first_legal = np.argmax(actions >= 0, axis=1)
    rows = np.arange(len(actions))
    keep = features[rows, first_legal, column] > 0.5
    return features[keep], actions[keep], placements[keep]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--valid", type=Path,
                        help="independent audited dataset; never used for fitting or normalization")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.02)
    parser.add_argument("--min-gap", type=float, default=0.25)
    parser.add_argument("--min-states", type=int, default=256)
    parser.add_argument("--family", choices=("all", *FAMILY_COLUMNS), default="all",
                        help="train only one pending-menu family")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    features, actions, placements, train_metadata = _load(args.train)
    features, actions, placements = _family(features, actions, placements, args.family)
    features, actions, labels = _confident(features, actions, placements, args.min_gap)
    if len(features) < args.min_states:
        raise ValueError(f"only {len(features)} confident choice states; need {args.min_states}")
    rng = np.random.default_rng(args.seed)
    if args.valid:
        valid_features, valid_actions, valid_placements, valid_metadata = _load(args.valid)
        if (train_metadata.get("schema_version"), train_metadata.get("choice_protocol"), train_metadata.get("source_hash")) != (valid_metadata.get("schema_version"), valid_metadata.get("choice_protocol"), valid_metadata.get("source_hash")):
            raise ValueError("train and external validation must use the same feature protocol and simulator build")
        valid_features, valid_actions, valid_placements = _family(valid_features, valid_actions, valid_placements, args.family)
        valid_features, valid_actions, valid_labels = _confident(valid_features, valid_actions, valid_placements, args.min_gap)
        if len(valid_features) < 64:
            raise ValueError(f"only {len(valid_features)} confident external validation states; need 64")
        train_idx = np.arange(len(features))
        valid_idx = None
    else:
        order = rng.permutation(len(features))
        split = max(1, int(len(features) * 0.8))
        train_idx, valid_idx = order[:split], order[split:]
        valid_features, valid_actions, valid_labels = features[valid_idx], actions[valid_idx], labels[valid_idx]
    train_rows = features[train_idx][actions[train_idx] >= 0]
    mean, scale = train_rows.mean(axis=0), train_rows.std(axis=0)
    scale[scale < 1e-4] = 1.0
    features = (features - mean) / scale
    valid_features = (valid_features - mean) / scale
    weights = np.zeros(features.shape[2], dtype=np.float32)
    for epoch in range(args.epochs):
        rate = args.learning_rate / (1.0 + 0.03 * epoch)
        for row in rng.permutation(train_idx):
            legal = actions[row] >= 0
            x, target = features[row, legal], labels[row, legal]
            logits = x @ weights
            logits -= logits.max()
            probs = np.exp(logits) / np.exp(logits).sum()
            weights -= rate * (x.T @ (probs - target) + args.l2 * weights)
    train_acc, train_random = _metrics(features[train_idx], actions[train_idx], labels[train_idx], weights)
    valid_acc, valid_random = _metrics(valid_features, valid_actions, valid_labels, weights)
    accepted = valid_acc >= valid_random + 0.10 and valid_acc >= 0.45
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, weights=weights, mean=mean, scale=scale,
                        metadata=json.dumps({"source": str(args.train), "valid_source": str(args.valid) if args.valid else None,
                                             "family": args.family,
                                             "confident_states": len(features), "train_states": len(train_idx),
                                             "valid_states": len(valid_features),
                                             "train_accuracy": train_acc, "valid_accuracy": valid_acc,
                                             "valid_random_accuracy": valid_random,
                                             "accepted_for_policy_evaluation": accepted}))
    print(json.dumps({"out": str(args.out), "family": args.family,
                      "train_states": len(train_idx), "valid_states": len(valid_features),
                      "train_accuracy": round(train_acc, 3), "valid_accuracy": round(valid_acc, 3),
                      "valid_random_accuracy": round(valid_random, 3), "accepted_for_policy_evaluation": accepted}, ensure_ascii=False))


if __name__ == "__main__":
    main()
