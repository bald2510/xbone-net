"""Post-hoc out-of-distribution detection utilities for XBone-Net."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D embedding matrix, got {array.shape}.")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


class OODDetector:
    """OOD detector supporting Mahalanobis, cosine k-NN, energy, and text anchors."""

    def __init__(self):
        self.class_means: dict[int, np.ndarray] = {}
        self.shared_cov_inv: Optional[np.ndarray] = None
        self.ref_embeddings: Optional[np.ndarray] = None
        self.ref_labels: Optional[np.ndarray] = None
        self._fitted = False

    def fit(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        prototypes: Optional[np.ndarray] = None,
    ):
        embeddings = _l2_normalize(embeddings)
        labels = np.asarray(labels)
        if labels.ndim == 2:
            labels = np.argmax(labels, axis=1)
        labels = labels.astype(np.int64).reshape(-1)
        if len(labels) != len(embeddings):
            raise ValueError("Embedding and label counts differ.")
        if len(embeddings) < 2:
            raise ValueError("At least two reference samples are required.")

        self.ref_embeddings = embeddings
        self.ref_labels = labels
        unique_classes = np.unique(labels)
        if unique_classes.size < 1:
            raise ValueError("Reference labels contain no classes.")

        normalized_prototypes = None
        if prototypes is not None:
            normalized_prototypes = _l2_normalize(prototypes)

        self.class_means = {}
        centered_parts = []
        for class_id in unique_classes:
            class_embeddings = embeddings[labels == class_id]
            if normalized_prototypes is not None and 0 <= class_id < len(normalized_prototypes):
                center = normalized_prototypes[class_id]
            else:
                center = class_embeddings.mean(axis=0)
                center /= max(np.linalg.norm(center), 1e-12)
            self.class_means[int(class_id)] = center
            centered_parts.append(class_embeddings - center)

        # Learned prototypes can provide centers for classes absent from a small
        # calibration subset, while covariance is estimated only from observed data.
        if normalized_prototypes is not None:
            for class_id, center in enumerate(normalized_prototypes):
                self.class_means.setdefault(int(class_id), center)

        centered = np.vstack(centered_parts)
        covariance = LedoitWolf().fit(centered).covariance_
        self.shared_cov_inv = np.linalg.pinv(covariance, hermitian=True)
        self._fitted = True
        return self

    def score_mahalanobis(self, test_embeddings: np.ndarray) -> np.ndarray:
        if not self._fitted or self.shared_cov_inv is None:
            raise RuntimeError("Call fit() before Mahalanobis scoring.")
        test_embeddings = _l2_normalize(test_embeddings)
        scores = np.full(test_embeddings.shape[0], np.inf, dtype=np.float64)
        for center in self.class_means.values():
            diff = test_embeddings - center
            distances = np.einsum("ni,ij,nj->n", diff, self.shared_cov_inv, diff)
            scores = np.minimum(scores, distances)
        return scores

    def score_knn(
        self,
        test_embeddings: np.ndarray,
        k: int = 5,
        exclude_self: bool = False,
    ) -> np.ndarray:
        if not self._fitted or self.ref_embeddings is None:
            raise RuntimeError("Call fit() before k-NN scoring.")
        test_embeddings = _l2_normalize(test_embeddings)
        similarities = test_embeddings @ self.ref_embeddings.T
        distances = 1.0 - similarities

        if exclude_self and distances.shape[0] == distances.shape[1]:
            np.fill_diagonal(distances, np.inf)

        n_available = distances.shape[1] - (1 if exclude_self and distances.shape[0] == distances.shape[1] else 0)
        if n_available < 1:
            raise ValueError("No reference neighbor is available for k-NN scoring.")
        k = max(1, min(int(k), n_available))
        nearest = np.partition(distances, kth=k - 1, axis=1)[:, :k]
        return nearest.mean(axis=1)

    def score_energy(self, logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Return energy where larger values indicate stronger OOD evidence."""
        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim != 2:
            raise ValueError("logits must have shape [N,C].")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        scaled = logits / temperature
        row_max = scaled.max(axis=1, keepdims=True)
        logsumexp = row_max[:, 0] + np.log(np.exp(scaled - row_max).sum(axis=1))
        return -temperature * logsumexp

    def score_text_anchor(
        self,
        test_img_embeddings: np.ndarray,
        anchor_text_embeddings: np.ndarray,
    ) -> np.ndarray:
        test_img_embeddings = _l2_normalize(test_img_embeddings)
        anchor_text_embeddings = _l2_normalize(anchor_text_embeddings)
        return 1.0 - (test_img_embeddings @ anchor_text_embeddings.T).max(axis=1)

    def score(self, test_embeddings: np.ndarray, method: str = "mahalanobis", **kwargs) -> np.ndarray:
        if method == "mahalanobis":
            return self.score_mahalanobis(test_embeddings)
        if method == "knn":
            return self.score_knn(
                test_embeddings,
                k=kwargs.get("k", 5),
                exclude_self=kwargs.get("exclude_self", False),
            )
        if method == "energy":
            return self.score_energy(kwargs["logits"], kwargs.get("temperature", 1.0))
        if method == "text_anchor":
            return self.score_text_anchor(test_embeddings, kwargs["anchor_text_embeddings"])
        raise ValueError(
            f"Unknown OOD method: {method}. Choose mahalanobis, knn, energy, or text_anchor."
        )


def evaluate_ood(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict:
    """Evaluate scores under the convention ``higher = more OOD``."""
    id_scores = np.asarray(id_scores, dtype=np.float64).reshape(-1)
    ood_scores = np.asarray(ood_scores, dtype=np.float64).reshape(-1)
    if id_scores.size == 0 or ood_scores.size == 0:
        raise ValueError("Both ID and OOD score arrays must be non-empty.")
    if not np.isfinite(id_scores).all() or not np.isfinite(ood_scores).all():
        raise ValueError("OOD scores contain NaN or infinity.")

    labels = np.concatenate(
        [np.zeros(id_scores.size, dtype=np.int64), np.ones(ood_scores.size, dtype=np.int64)]
    )
    scores = np.concatenate([id_scores, ood_scores])
    auroc = float(roc_auc_score(labels, scores))
    aupr_out = float(average_precision_score(labels, scores))

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eligible = np.flatnonzero(tpr >= 0.95)
    if eligible.size:
        best = eligible[np.argmin(fpr[eligible])]
        fpr_at_95 = float(fpr[best])
        threshold_at_95 = float(thresholds[best])
    else:
        fpr_at_95 = 1.0
        threshold_at_95 = float("nan")

    return {
        "auroc": auroc,
        "aupr_out": aupr_out,
        "fpr_at_95tpr": fpr_at_95,
        "threshold_at_95tpr": threshold_at_95,
        "n_id": int(id_scores.size),
        "n_ood": int(ood_scores.size),
    }