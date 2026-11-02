"""Train a local masked-action behavior-cloning policy from collector output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from policy_features import combine


def batches(n: int, size: int, rng: np.random.Generator):
    order = rng.permutation(n)
    for start in range(0, n, size):
        yield order[start:start + size]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    from sklearn.linear_model import SGDClassifier

    train, valid = np.load(args.train), np.load(args.valid)
    for data in (train, valid):
        if not {"features", "cards", "action_mask", "actions"} <= set(data.files):
            raise ValueError("Collect a fresh dataset with cards using collect_bot_data.py")
    xf, xc, y = train["features"], train["cards"], train["actions"]
    vf, vc, vy, vm = valid["features"], valid["cards"], valid["actions"], valid["action_mask"]
    mean = combine(xf[: min(len(xf), 4096)], xc[: min(len(xc), 4096)]).mean(axis=0)
    scale = combine(xf[: min(len(xf), 4096)], xc[: min(len(xc), 4096)]).std(axis=0)
    scale[scale < 1e-4] = 1.0
    counts = np.bincount(y, minlength=245)
    weights = (counts[y].clip(1) ** -0.5)
    weights /= weights.mean()
    clf = SGDClassifier(loss="log_loss", alpha=2e-5, max_iter=1, tol=None, random_state=0)
    rng = np.random.default_rng(0)
    classes = np.arange(245, dtype=np.int16)
    for _ in range(args.epochs):
        for idx in batches(len(y), args.batch_size, rng):
            clf.partial_fit((combine(xf[idx], xc[idx]) - mean) / scale, y[idx],
                            classes=classes, sample_weight=weights[idx])

    raw_ok = masked_ok = total = 0
    for idx in batches(len(vy), args.batch_size, np.random.default_rng(1)):
        logits = ((combine(vf[idx], vc[idx]) - mean) / scale) @ clf.coef_.T + clf.intercept_
        raw_ok += int((logits.argmax(axis=1) == vy[idx]).sum())
        masked_ok += int((np.where(vm[idx], logits, -np.inf).argmax(axis=1) == vy[idx]).sum())
        total += len(idx)
    raw_acc, masked_acc = raw_ok / total, masked_ok / total
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, coef=clf.coef_.astype(np.float32), intercept=clf.intercept_.astype(np.float32),
                        mean=mean.astype(np.float32), scale=scale.astype(np.float32),
                        metadata=json.dumps({"train": str(args.train), "valid": str(args.valid),
                            "epochs": args.epochs, "raw_accuracy": raw_acc, "masked_accuracy": masked_acc,
                            "card_hash_dim": 64}))
    print(json.dumps({"out": str(args.out), "raw_accuracy": raw_acc,
                      "masked_accuracy": masked_acc, "valid_samples": total}))


if __name__ == "__main__":
    main()
