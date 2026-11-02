"""Shared, local-only featurization for imitation-policy training and inference."""

from __future__ import annotations

import numpy as np


CARD_HASH_DIM = 64


def card_sketch(cards: np.ndarray) -> np.ndarray:
    """Encode card identities and their fixed observation slots into 64 numeric features."""
    cards = np.asarray(cards, dtype=np.int32)
    one = cards.ndim == 1
    matrix = cards.reshape(1, -1) if one else cards
    out = np.zeros((len(matrix), CARD_HASH_DIM), dtype=np.float32)
    rows, cols = np.nonzero(matrix)
    values = matrix[rows, cols].astype(np.int64)
    buckets = (values * 1_103_515_245 + cols * 12_345) % CARD_HASH_DIM
    np.add.at(out, (rows, buckets), 1.0)
    return out[0] if one else out


def combine(features: np.ndarray, cards: np.ndarray) -> np.ndarray:
    """Join numeric observe2 features with a compact card-identity sketch."""
    features = np.asarray(features, dtype=np.float32)
    if features.shape[-1] != 4017 or np.asarray(cards).shape[-1] != 447:
        raise ValueError("Expected observe2 features[4017] and cards[447]")
    return np.concatenate((features, card_sketch(cards)), axis=-1)
