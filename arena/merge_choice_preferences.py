"""Merge audited choice-rollout datasets from the same TavernGym build."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from collect_choice_preferences import SUPPORTED_PROTOCOLS
from collect_rollout_preferences import RNG_PROTOCOL


ARRAYS = ("candidate_features", "candidate_actions", "candidate_placements", "candidate_mean_placements")


def _metadata(data: np.lib.npyio.NpzFile, path: Path) -> dict:
    if "metadata" not in data.files:
        raise ValueError(f"{path} has no provenance metadata")
    metadata = json.loads(str(data["metadata"]))
    if (metadata.get("schema_version"), metadata.get("choice_protocol")) not in SUPPORTED_PROTOCOLS or metadata.get("rng_protocol") != RNG_PROTOCOL:
        raise ValueError(f"{path} is not an audited choice dataset")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    loaded = [np.load(path) for path in args.inputs]
    if any(not set(ARRAYS) <= set(data.files) for data in loaded):
        raise ValueError("all inputs must be current choice-rollout datasets")
    metadata = [_metadata(data, path) for data, path in zip(loaded, args.inputs)]
    hashes = {item.get("source_hash") for item in metadata}
    protocols = {(item.get("schema_version"), item.get("choice_protocol"), item.get("rng_protocol")) for item in metadata}
    if len(hashes) != 1:
        raise ValueError(f"cannot merge simulator builds: {hashes}")
    if len(protocols) != 1:
        raise ValueError("cannot merge choice datasets with different feature protocols")
    for key in ARRAYS:
        shapes = {data[key].shape[1:] for data in loaded}
        if len(shapes) != 1:
            raise ValueError(f"incompatible {key} shapes: {shapes}")
    merged = {key: np.concatenate([data[key] for data in loaded], axis=0) for key in ARRAYS}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **merged, metadata=json.dumps({
        "schema_version": metadata[0]["schema_version"], "choice_protocol": metadata[0]["choice_protocol"],
        "rng_protocol": metadata[0]["rng_protocol"], "visible_observation_only": True,
        "source_hash": hashes.pop(), "sources": [str(path) for path in args.inputs],
        "states": int(merged["candidate_actions"].shape[0]),
        "feature_dim": int(merged["candidate_features"].shape[2]),
    }))
    print(json.dumps({"out": str(args.out), "states": int(merged["candidate_actions"].shape[0]),
                      "inputs": len(args.inputs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
