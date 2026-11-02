"""Train a local masked action ranker from rollout-preference datasets.

This is supervised policy improvement, not an automatically deployed policy.
Each training label is the legal candidate action with the best local
counterfactual placement.  The model is later gated behind the v8 fallback.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from policy_features import combine


SCHEMA_VERSION = 2
RNG_PROTOCOL = "reseeded-counterfactual-v1"


def load(path: Path):
    data = np.load(path)
    required = {"features", "cards", "candidate_actions", "candidate_placements"}
    if not required <= set(data.files):
        raise ValueError(f"{path} is not a rollout preference dataset")
    if "metadata" not in data.files:
        raise ValueError(f"{path} has no rollout provenance metadata; recollect it")
    metadata = json.loads(str(data["metadata"]))
    if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("rng_protocol") != RNG_PROTOCOL:
        raise ValueError(f"{path} uses an unaudited rollout protocol; recollect it")
    if metadata.get("visible_observation_only") is not True:
        raise ValueError(f"{path} does not attest visible-observation-only features")
    x = combine(data["features"], data["cards"]).astype(np.float32)
    actions = data["candidate_actions"].astype(np.int16)
    placements = data["candidate_mean_placements"] if "candidate_mean_placements" in data.files else data["candidate_placements"]
    if placements.ndim == 3:
        counts = np.count_nonzero(placements > 0, axis=2)
        sums = np.where(placements > 0, placements, 0).sum(axis=2, dtype=np.float32)
        placements = np.zeros(counts.shape, dtype=np.float32)
        np.divide(sums, counts, out=placements, where=counts > 0)
    placements = placements.astype(np.float32)
    return x, actions, placements


def labels(actions: np.ndarray, placements: np.ndarray) -> np.ndarray:
    valid = placements > 0
    safe = np.where(valid, placements, 9)
    best = safe.min(axis=1, keepdims=True)
    # Uniformly share probability between ties, rather than picking by action id.
    y = (valid & (safe == best)).astype(np.float32)
    return y / y.sum(axis=1, keepdims=True)


def candidate_metrics(x: np.ndarray, actions: np.ndarray, y: np.ndarray,
                      coef: np.ndarray, intercept: np.ndarray) -> tuple[float, float]:
    correct = 0
    margins: list[float] = []
    for row, vector in enumerate(x):
        legal = actions[row] >= 0
        choices = actions[row, legal]
        score = coef[choices] @ vector + intercept[choices]
        order = np.argsort(score)
        winner = int(choices[order[-1]])
        target = choices[y[row, legal] > 0]
        correct += int(winner in target)
        margins.append(float(score[order[-1]] - score[order[-2]]) if len(order) > 1 else 0.0)
    return correct / len(x), float(np.mean(margins))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--min-gap", type=float, default=0.25, help="minimum best-vs-next placement gap to label")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1 or not 0 < args.valid_fraction < 0.5 or args.learning_rate <= 0 or args.min_gap < 0:
        parser.error("invalid training options")

    x, actions, placements = load(args.train)
    safe = np.where(placements > 0, placements, 9)
    ordered = np.sort(safe, axis=1)
    keep = (ordered[:, 1] - ordered[:, 0]) >= args.min_gap
    x, actions, placements = x[keep], actions[keep], placements[keep]
    if len(x) < 64:
        raise ValueError(f"only {len(x)} confident preference states; collect or merge until at least 64")
    y = labels(actions, placements)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(x))
    split = max(1, int(len(x) * (1 - args.valid_fraction)))
    train_idx, valid_idx = order[:split], order[split:]
    mean = x[train_idx].mean(axis=0)
    scale = x[train_idx].std(axis=0)
    scale[scale < 1e-4] = 1.0
    x = (x - mean) / scale
    coef = np.zeros((245, x.shape[1]), dtype=np.float32)
    intercept = np.zeros(245, dtype=np.float32)

    for epoch in range(args.epochs):
        for row in rng.permutation(train_idx):
            legal = actions[row] >= 0
            choices = actions[row, legal]
            target = y[row, legal]
            logits = coef[choices] @ x[row] + intercept[choices]
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            gradient = probs - target
            rate = args.learning_rate / (1.0 + epoch * 0.08)
            coef[choices] -= rate * gradient[:, None] * x[row][None, :]
            intercept[choices] -= rate * gradient
    train_accuracy, train_margin = candidate_metrics(x[train_idx], actions[train_idx], y[train_idx], coef, intercept)
    valid_accuracy, valid_margin = candidate_metrics(x[valid_idx], actions[valid_idx], y[valid_idx], coef, intercept)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, coef=coef, intercept=intercept, mean=mean.astype(np.float32),
                        scale=scale.astype(np.float32), metadata=json.dumps({
                            "source": str(args.train), "epochs": args.epochs,
                            "confident_states": len(x), "train_states": len(train_idx), "valid_states": len(valid_idx),
                            "train_candidate_accuracy": train_accuracy,
                            "valid_candidate_accuracy": valid_accuracy,
                            "valid_mean_margin": valid_margin,
                        }))
    print(json.dumps({"out": str(args.out), "train_states": len(train_idx), "valid_states": len(valid_idx),
                      "train_candidate_accuracy": round(train_accuracy, 3),
                      "valid_candidate_accuracy": round(valid_accuracy, 3),
                      "valid_mean_margin": round(valid_margin, 3)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
