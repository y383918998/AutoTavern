"""Merge compatible local rollout-preference datasets without changing labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARRAYS = ("features", "cards", "action_mask", "turns", "teacher_actions",
          "candidate_actions", "candidate_placements", "candidate_mean_placements", "best_actions")
SCHEMA_VERSION = 2
RNG_PROTOCOL = "reseeded-counterfactual-v1"


def _provenance(data: np.lib.npyio.NpzFile, path: Path) -> dict:
    if "metadata" not in data.files:
        raise ValueError(f"{path} has no rollout provenance metadata; recollect it")
    metadata = json.loads(str(data["metadata"]))
    if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("rng_protocol") != RNG_PROTOCOL:
        raise ValueError(f"{path} uses an unaudited rollout protocol; recollect it")
    if metadata.get("visible_observation_only") is not True:
        raise ValueError(f"{path} does not attest visible-observation-only features")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    loaded = [np.load(path) for path in args.inputs]
    if any(not set(ARRAYS) <= set(data.files) for data in loaded):
        raise ValueError("all inputs must be current rollout-preference datasets")
    provenance = [_provenance(data, path) for data, path in zip(loaded, args.inputs)]
    source_hashes = {item.get("source_hash") for item in provenance}
    if len(source_hashes) != 1:
        raise ValueError(f"cannot merge simulator builds: {source_hashes}")
    for key in ARRAYS:
        shapes = {data[key].shape[1:] for data in loaded}
        if len(shapes) != 1:
            raise ValueError(f"incompatible {key} shapes: {shapes}")
    merged = {key: np.concatenate([data[key] for data in loaded], axis=0) for key in ARRAYS}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **merged, metadata=json.dumps({
        "schema_version": SCHEMA_VERSION, "rng_protocol": RNG_PROTOCOL,
        "visible_observation_only": True, "source_hash": source_hashes.pop(),
        "sources": [str(path) for path in args.inputs], "states": len(merged["features"]),
    }))
    print(json.dumps({"out": str(args.out), "states": len(merged["features"]), "inputs": len(args.inputs)}))


if __name__ == "__main__":
    main()
