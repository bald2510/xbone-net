"""Loading and diagnostic helpers for empirical-centroid visualisation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA


def _find_state_tensor(
    state_dict: dict[str, Any],
    suffix: str,
    required: bool = True,
) -> torch.Tensor | None:
    exact = state_dict.get(suffix)
    if isinstance(exact, torch.Tensor):
        return exact
    matches = [
        value
        for key, value in state_dict.items()
        if key.endswith(f".{suffix}") and isinstance(value, torch.Tensor)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Checkpoint contains multiple tensors ending in '{suffix}'.")
    if required:
        raise KeyError(
            f"Checkpoint does not contain '{suffix}'. Ensure this is a trained "
            "empirical-centroid Phase-2 checkpoint."
        )
    return None


def load_centroids_from_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load centroid vectors and train counts without rebuilding the model."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch.
        checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint must contain a state dictionary.")

    centroids = _find_state_tensor(state_dict, "head.centroids")
    counts = _find_state_tensor(state_dict, "head.centroid_counts")
    initialized = _find_state_tensor(
        state_dict, "head.centroids_initialized", required=False
    )
    if initialized is not None and not bool(initialized.item()):
        raise ValueError("Checkpoint centroids are marked as uninitialized.")

    centroids_np = centroids.detach().float().cpu().numpy()
    counts_np = counts.detach().long().cpu().numpy()
    if centroids_np.ndim != 2:
        raise ValueError(f"Centroids must be [C,D], got {centroids_np.shape}.")
    if counts_np.shape != (centroids_np.shape[0],):
        raise ValueError(
            f"Centroid counts must have shape {(centroids_np.shape[0],)}, "
            f"got {counts_np.shape}."
        )
    if not np.isfinite(centroids_np).all():
        raise ValueError("Centroids contain NaN or infinite values.")
    if np.any(counts_np <= 0):
        missing = np.flatnonzero(counts_np <= 0).tolist()
        raise ValueError(f"Centroid counts are non-positive for classes: {missing}")
    return centroids_np, counts_np


def load_fused_embeddings(
    embeddings_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load fused embeddings and labels exported by evaluate.py."""
    path = Path(embeddings_path)
    if not path.is_file():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        key = "fused_embeddings" if "fused_embeddings" in archive else "image_embeddings"
        if key not in archive or "labels" not in archive:
            raise KeyError(
                "Embeddings NPZ must contain fused_embeddings (or image_embeddings) "
                "and labels. Generate it with evaluate.py --save-embeddings."
            )
        embeddings = np.asarray(archive[key], dtype=np.float32)
        labels = np.asarray(archive["labels"])

    if labels.ndim == 2:
        labels = labels.argmax(axis=1)
    labels = labels.reshape(-1).astype(np.int64, copy=False)
    if embeddings.ndim != 2 or embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            "Embeddings and labels must have shapes [N,D] and [N]; got "
            f"{embeddings.shape} and {labels.shape}."
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values.")
    return embeddings, labels, key


def l2_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize rows to match the cosine classifier geometry."""
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    zero_rows = np.flatnonzero(norms.reshape(-1) <= eps).tolist()
    if zero_rows:
        raise ValueError(f"Cannot normalize zero-norm rows: {zero_rows[:20]}")
    return values / np.maximum(norms, eps)


def stratified_subsample(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    max_points_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Limit plot density while preserving all observed classes."""
    if max_points_per_class < 1:
        raise ValueError("max_points_per_class must be positive.")
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in range(num_classes):
        indices = np.flatnonzero(labels == class_id)
        if len(indices) > max_points_per_class:
            indices = rng.choice(indices, size=max_points_per_class, replace=False)
        selected.extend(indices.tolist())
    selected_array = np.asarray(sorted(selected), dtype=np.int64)
    return embeddings[selected_array], labels[selected_array]


def project_cosine_space(
    centroids: np.ndarray,
    embeddings: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Fit a reproducible 2-D PCA projection on normalized vectors."""
    normalized_centroids = l2_normalize(centroids)
    normalized_embeddings = l2_normalize(embeddings) if embeddings is not None else None
    fit_values = (
        np.concatenate([normalized_embeddings, normalized_centroids], axis=0)
        if normalized_embeddings is not None
        else normalized_centroids
    )
    if min(fit_values.shape) < 2:
        raise ValueError("At least two samples and two feature dimensions are required.")
    pca = PCA(n_components=2, svd_solver="randomized", random_state=seed)
    projected = pca.fit_transform(fit_values)
    if normalized_embeddings is None:
        return projected, None, pca.explained_variance_ratio_
    split = normalized_embeddings.shape[0]
    return projected[split:], projected[:split], pca.explained_variance_ratio_


def compute_centroid_diagnostics(
    centroids: np.ndarray,
    embeddings: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute centroid separation and optional sample-to-centroid margins."""
    normalized_centroids = l2_normalize(centroids)
    similarity = normalized_centroids @ normalized_centroids.T
    other_similarity = similarity.copy()
    np.fill_diagonal(other_similarity, -np.inf)
    nearest_ids = other_similarity.argmax(axis=1)
    nearest_similarity = other_similarity[np.arange(len(centroids)), nearest_ids]
    angular_distance = np.degrees(
        np.arccos(np.clip(nearest_similarity, -1.0, 1.0))
    )

    diagnostics: dict[str, Any] = {
        "cosine_similarity_matrix": similarity,
        "nearest_centroid_ids": nearest_ids,
        "nearest_centroid_similarity": nearest_similarity,
        "nearest_centroid_angular_distance_degrees": angular_distance,
        "centroid_separation": 1.0 - nearest_similarity,
    }
    if embeddings is None:
        return diagnostics
    if labels is None:
        raise ValueError("labels are required when embeddings are provided.")
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Embeddings and labels have incompatible lengths.")
    if np.any(labels < 0) or np.any(labels >= len(centroids)):
        invalid = np.unique(labels[(labels < 0) | (labels >= len(centroids))]).tolist()
        raise ValueError(f"Embedding labels are outside the centroid range: {invalid}")

    normalized_embeddings = l2_normalize(embeddings)
    sample_similarity = normalized_embeddings @ normalized_centroids.T
    own_similarity = sample_similarity[np.arange(len(labels)), labels]
    competing = sample_similarity.copy()
    competing[np.arange(len(labels)), labels] = -np.inf
    best_other = competing.max(axis=1)
    margins = own_similarity - best_other
    predictions = sample_similarity.argmax(axis=1)

    per_class = []
    for class_id in range(len(centroids)):
        mask = labels == class_id
        if not np.any(mask):
            per_class.append(
                {
                    "sample_count": 0,
                    "mean_own_similarity": None,
                    "mean_cosine_margin": None,
                    "std_cosine_margin": None,
                    "nearest_centroid_accuracy": None,
                }
            )
            continue
        class_margins = margins[mask]
        per_class.append(
            {
                "sample_count": int(mask.sum()),
                "mean_own_similarity": float(own_similarity[mask].mean()),
                "mean_cosine_margin": float(class_margins.mean()),
                "std_cosine_margin": float(class_margins.std()),
                "nearest_centroid_accuracy": float((predictions[mask] == class_id).mean()),
            }
        )
    diagnostics["sample_margins"] = margins
    diagnostics["sample_predictions"] = predictions
    diagnostics["per_class"] = per_class
    diagnostics["overall_nearest_centroid_accuracy"] = float(
        (predictions == labels).mean()
    )
    diagnostics["overall_mean_cosine_margin"] = float(margins.mean())
    return diagnostics
