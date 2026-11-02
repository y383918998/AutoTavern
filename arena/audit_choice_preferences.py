"""Audit specialised choice-rollout data before ranker training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from collect_choice_preferences import SUPPORTED_PROTOCOLS
from collect_rollout_preferences import RNG_PROTOCOL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    data = np.load(args.input)
    required = {"candidate_features", "candidate_actions", "candidate_placements", "candidate_mean_placements", "metadata"}
    if missing := required - set(data.files):
        raise ValueError(f"missing required arrays: {sorted(missing)}")
    metadata = json.loads(str(data["metadata"]))
    if (metadata.get("schema_version"), metadata.get("choice_protocol")) not in SUPPORTED_PROTOCOLS or metadata.get("rng_protocol") != RNG_PROTOCOL:
        raise ValueError("unrecognized choice dataset protocol")
    features, actions, placements = data["candidate_features"], data["candidate_actions"], data["candidate_placements"]
    if features.ndim != 3 or actions.shape != features.shape[:2] or placements.shape[:2] != actions.shape:
        raise ValueError("candidate array shapes are inconsistent")
    if metadata.get("feature_dim") not in (None, int(features.shape[2])):
        raise ValueError("metadata feature_dim disagrees with stored feature array")
    if np.any((actions < -1) | (actions >= 245)) or np.any((placements < 0) | (placements > 8)):
        raise ValueError("invalid action or placement in choice data")
    print(json.dumps({"input": str(args.input), "states": int(actions.shape[0]), "feature_dim": int(features.shape[2]),
                      "source_hash": metadata.get("source_hash"), "audit": "PASS"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
