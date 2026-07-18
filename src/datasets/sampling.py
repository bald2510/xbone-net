"""Deterministic sampling helpers shared by dataset robustness controls."""

from __future__ import annotations

import numpy as np


def deranged_donor_indices(size: int, seed: int) -> np.ndarray:
    """Return a one-to-one donor map with no fixed points (Sattolo cycle)."""
    if size < 2:
        raise ValueError("A report derangement requires at least two samples.")
    rng = np.random.default_rng(seed)
    donors = np.arange(size, dtype=np.int64)
    for index in range(size - 1, 0, -1):
        swap_index = int(rng.integers(0, index))
        donors[index], donors[swap_index] = donors[swap_index], donors[index]
    if np.any(donors == np.arange(size)) or len(np.unique(donors)) != size:
        raise RuntimeError("Failed to construct a one-to-one derangement.")
    return donors


def cross_class_donor_indices(labels: np.ndarray, seed: int) -> np.ndarray:
    """Return a one-to-one donor map with no same-class report assignments."""
    labels = np.asarray(labels).reshape(-1)
    if labels.size < 2:
        raise ValueError("Cross-class mismatch requires at least two samples.")
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    largest_class = int(counts.max())
    if largest_class * 2 > labels.size:
        majority = classes[int(np.argmax(counts))]
        raise ValueError(
            "A complete cross-class derangement is impossible because class "
            f"{majority!r} contains {largest_class}/{labels.size} samples."
        )

    ordered = np.concatenate(
        [rng.permutation(np.flatnonzero(labels == class_id)) for class_id in classes]
    )
    shifted = np.roll(ordered, -largest_class)
    donors = np.empty(labels.size, dtype=np.int64)
    donors[ordered] = shifted
    recipients = np.arange(labels.size, dtype=np.int64)
    if np.any(recipients == donors):
        raise RuntimeError("Cross-class derangement contains a fixed point.")
    if np.any(labels[recipients] == labels[donors]):
        raise RuntimeError("Cross-class derangement retained a recipient class.")
    if len(np.unique(donors)) != labels.size:
        raise RuntimeError("Cross-class donor assignments are not one-to-one.")
    return donors
