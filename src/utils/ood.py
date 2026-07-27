"""Post-hoc out-of-distribution detection utilities for XBone-Net.

All score functions in this module follow one convention: larger values mean
stronger OOD evidence.  Detector fitting, ID-only threshold calibration, and
final ID/OOD evaluation are deliberately separate operations so test samples
are never used to choose a deployment threshold.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


OOD_PROTOCOL_VERSION = 4
MAHALANOBIS_SCORE_DEFINITION = (
    "minimum_class_conditional_mahalanobis_distance_to_empirical_centroid"
)
COSINE_CENTROIDS_SCORE_DEFINITION = (
    "one_minus_maximum_cosine_similarity_to_empirical_centroid"
)


def _embedding_matrix(array: np.ndarray, name: str = "embeddings") -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D {name} matrix, got {values.shape}.")
    if values.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one feature.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    return values


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    array = _embedding_matrix(array)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def _finite_scores(scores: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    return values


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("logits must have shape [N,C] with C >= 2.")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.maximum(exponentials.sum(axis=1, keepdims=True), 1e-12)


class OODDetector:
    """OOD detector supporting density-, confidence-, and text-based scores."""

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
        # Mahalanobis statistics are estimated in the feature archive's native
        # coordinate system.  Do not normalize inside the detector: doing so
        # changes both the empirical mean and covariance away from the
        # classical definition.  A separately normalized copy is retained for
        # cosine k-NN scoring.
        embeddings = _embedding_matrix(embeddings)
        labels = np.asarray(labels)
        if labels.ndim == 2:
            labels = np.argmax(labels, axis=1)
        labels = labels.astype(np.int64).reshape(-1)
        if len(labels) != len(embeddings):
            raise ValueError("Embedding and label counts differ.")
        if len(embeddings) < 2:
            raise ValueError("At least two reference samples are required.")
        if np.any(labels < 0):
            raise ValueError("Reference labels must be non-negative class IDs.")

        self.ref_embeddings = _l2_normalize(embeddings)
        self.ref_labels = labels
        unique_classes = np.unique(labels)
        if unique_classes.size < 1:
            raise ValueError("Reference labels contain no classes.")

        prototype_centers = None
        if prototypes is not None:
            prototype_centers = _embedding_matrix(prototypes, "prototype")
            if prototype_centers.shape[1] != embeddings.shape[1]:
                raise ValueError(
                    "Prototype and reference embedding dimensions differ."
                )

        self.class_means = {}
        centered_parts = []
        for class_id in unique_classes:
            class_embeddings = embeddings[labels == class_id]
            if (
                prototype_centers is not None
                and 0 <= class_id < len(prototype_centers)
            ):
                center = prototype_centers[class_id]
            else:
                center = class_embeddings.mean(axis=0)
            self.class_means[int(class_id)] = center
            centered_parts.append(class_embeddings - center)

        # Centroids may cover a class absent from a deliberately small reference
        # subset; covariance is still estimated from observed samples only.
        if prototype_centers is not None:
            for class_id, center in enumerate(prototype_centers):
                self.class_means.setdefault(int(class_id), center)

        centered = np.vstack(centered_parts)
        covariance = LedoitWolf().fit(centered).covariance_
        self.shared_cov_inv = np.linalg.pinv(covariance, hermitian=True)
        self._fitted = True
        return self

    def score_mahalanobis_centroid(
        self, test_embeddings: np.ndarray
    ) -> np.ndarray:
        """Return the classical Mahalanobis distance to the nearest ID class.

        For each class this computes ``sqrt((x-mu)^T Sigma^-1 (x-mu))``
        using empirical class means and one shared Ledoit-Wolf covariance.
        The minimum class-conditional distance is the OOD score.
        """
        if not self._fitted or self.shared_cov_inv is None:
            raise RuntimeError("Call fit() before Mahalanobis scoring.")
        test_embeddings = _embedding_matrix(test_embeddings, "test embedding")
        if test_embeddings.shape[1] != self.shared_cov_inv.shape[0]:
            raise ValueError(
                "Test and reference embedding dimensions differ."
            )
        squared_scores = np.full(
            test_embeddings.shape[0], np.inf, dtype=np.float64
        )
        for center in self.class_means.values():
            diff = test_embeddings - center
            squared_distances = np.einsum(
                "ni,ij,nj->n", diff, self.shared_cov_inv, diff
            )
            squared_scores = np.minimum(squared_scores, squared_distances)
        # A symmetric pseudo-inverse can still produce tiny negative values
        # from floating-point roundoff.  Zero-clipping preserves the classical
        # non-negative distance before taking the square root.
        return np.sqrt(np.maximum(squared_scores, 0.0))

    def score_mahalanobis(self, test_embeddings: np.ndarray) -> np.ndarray:
        """Backward-compatible alias for ``score_mahalanobis_centroid``."""
        return self.score_mahalanobis_centroid(test_embeddings)

    def score_cosine_centroids(
        self, test_embeddings: np.ndarray
    ) -> np.ndarray:
        """Return cosine distance to the nearest empirical class centroid.

        Both samples and class means are L2-normalized so the score is
        ``1 - max_c cosine(z, mu_c)``. Larger values indicate stronger OOD
        evidence and match the geometry used by the empirical-centroid head.
        """
        if not self._fitted or not self.class_means:
            raise RuntimeError("Call fit() before cosine-centroid scoring.")
        test_embeddings = _l2_normalize(test_embeddings)
        centers = _l2_normalize(np.stack(list(self.class_means.values()), axis=0))
        if test_embeddings.shape[1] != centers.shape[1]:
            raise ValueError(
                "Test and empirical-centroid embedding dimensions differ."
            )
        return 1.0 - (test_embeddings @ centers.T).max(axis=1)

    def score_knn(
        self,
        test_embeddings: np.ndarray,
        k: int = 5,
        exclude_self: bool = False,
    ) -> np.ndarray:
        if not self._fitted or self.ref_embeddings is None:
            raise RuntimeError("Call fit() before k-NN scoring.")
        test_embeddings = _l2_normalize(test_embeddings)
        distances = 1.0 - test_embeddings @ self.ref_embeddings.T

        if exclude_self and distances.shape[0] == distances.shape[1]:
            np.fill_diagonal(distances, np.inf)

        excluded = int(exclude_self and distances.shape[0] == distances.shape[1])
        n_available = distances.shape[1] - excluded
        if n_available < 1:
            raise ValueError("No reference neighbor is available for k-NN scoring.")
        k = max(1, min(int(k), n_available))
        nearest = np.partition(distances, kth=k - 1, axis=1)[:, :k]
        return nearest.mean(axis=1)

    @staticmethod
    def score_energy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
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

    @staticmethod
    def score_msp(logits: np.ndarray) -> np.ndarray:
        """Return ``1 - max softmax probability``."""
        return 1.0 - _softmax(logits).max(axis=1)

    @staticmethod
    def score_entropy(logits: np.ndarray) -> np.ndarray:
        """Return predictive entropy normalized to [0, 1]."""
        probabilities = _softmax(logits)
        entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=1)
        return entropy / np.log(probabilities.shape[1])

    @staticmethod
    def score_max_logit(logits: np.ndarray) -> np.ndarray:
        """Return negative maximum logit so larger values mean more OOD."""
        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim != 2:
            raise ValueError("logits must have shape [N,C].")
        return -logits.max(axis=1)

    @staticmethod
    def score_text_anchor(
        test_img_embeddings: np.ndarray,
        anchor_text_embeddings: np.ndarray,
    ) -> np.ndarray:
        test_img_embeddings = _l2_normalize(test_img_embeddings)
        anchor_text_embeddings = _l2_normalize(anchor_text_embeddings)
        return 1.0 - (test_img_embeddings @ anchor_text_embeddings.T).max(axis=1)

    def score(
        self,
        test_embeddings: np.ndarray,
        method: str = "mahalanobis_centroid",
        **kwargs,
    ) -> np.ndarray:
        if method == "mahalanobis_centroid":
            return self.score_mahalanobis_centroid(test_embeddings)
        if method == "cosine_centroids":
            return self.score_cosine_centroids(test_embeddings)
        if method == "knn":
            return self.score_knn(
                test_embeddings,
                k=kwargs.get("k", 5),
                exclude_self=kwargs.get("exclude_self", False),
            )
        if method == "entropy":
            return self.score_entropy(kwargs["logits"])
        raise ValueError(
            "Unknown OOD method: "
            f"{method}. Choose cosine_centroids, mahalanobis_centroid, "
            "knn, or entropy."
        )


def calibrate_ood_threshold(
    calibration_id_scores: np.ndarray,
    target_id_fpr: float = 0.05,
) -> float:
    """Choose an ID-only threshold with approximately ``target_id_fpr`` rejects."""
    scores = _finite_scores(calibration_id_scores, "calibration_id_scores")
    if not 0.0 < target_id_fpr < 1.0:
        raise ValueError("target_id_fpr must be between 0 and 1.")
    quantile = 1.0 - float(target_id_fpr)
    try:
        return float(np.quantile(scores, quantile, method="higher"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(scores, quantile, interpolation="higher"))


def evaluate_ood(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> dict:
    """Return the three report metrics under ``higher = more OOD``.

    AUROC-OOD and AUPR-Out designate OOD as the positive class. FPR@95%TPR
    follows the established OOD convention: ID is positive for this operating
    point, so FPR is the fraction of OOD samples incorrectly accepted when 95%
    of ID samples are accepted.
    """
    id_scores = _finite_scores(id_scores, "id_scores")
    ood_scores = _finite_scores(ood_scores, "ood_scores")

    labels = np.concatenate([
        np.zeros(id_scores.size, dtype=np.int64),
        np.ones(ood_scores.size, dtype=np.int64),
    ])
    scores = np.concatenate([id_scores, ood_scores])
    auroc_ood = float(roc_auc_score(labels, scores))
    aupr_out = float(average_precision_score(labels, scores))

    # Literature-standard FPR95 treats ID acceptance as the positive decision:
    # at 95% ID TPR, FPR is the fraction of OOD samples incorrectly accepted as
    # ID. Since this module stores higher-is-OOD scores, ID confidence is -score.
    fpr_id, tpr_id, _ = roc_curve(1 - labels, -scores, pos_label=1)
    eligible = np.flatnonzero(tpr_id >= 0.95)
    if eligible.size:
        best = eligible[np.argmin(fpr_id[eligible])]
        fpr_at_95 = float(fpr_id[best])
    else:
        fpr_at_95 = 1.0

    return {
        "auroc_ood": auroc_ood,
        "aupr_out": aupr_out,
        "fpr_at_95tpr": fpr_at_95,
    }


def bootstrap_ood_metrics(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    n_bootstrap: int = 2_000,
    seed: int = 42,
    alpha: float = 0.05,
    paired: bool = False,
    id_groups: Optional[np.ndarray] = None,
    ood_groups: Optional[np.ndarray] = None,
) -> dict[str, list[float]]:
    """Percentile bootstrap confidence intervals.

    ``paired=True`` is intended for report-mismatch experiments where each OOD
    score is generated from the same image as the corresponding ID score.
    """
    id_scores = _finite_scores(id_scores, "id_scores")
    ood_scores = _finite_scores(ood_scores, "ood_scores")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    if paired and len(id_scores) != len(ood_scores):
        raise ValueError("Paired bootstrap requires equal ID and OOD sample counts.")

    def validate_groups(groups, expected, name):
        if groups is None:
            return None
        values = np.asarray(groups).astype(str).reshape(-1)
        if len(values) != expected:
            raise ValueError(f"{name} must contain one group per score.")
        if np.any(values == ""):
            return None
        return values

    id_groups = validate_groups(id_groups, len(id_scores), "id_groups")
    ood_groups = validate_groups(ood_groups, len(ood_scores), "ood_groups")
    if paired and (id_groups is not None or ood_groups is not None):
        raise ValueError("Paired and clustered bootstrap modes cannot be combined.")

    rng = np.random.default_rng(seed)

    def draw_indices(length: int, groups: Optional[np.ndarray]) -> np.ndarray:
        if groups is None:
            return rng.integers(0, length, size=length)
        unique = np.unique(groups)
        sampled_groups = rng.choice(unique, size=len(unique), replace=True)
        return np.concatenate(
            [np.flatnonzero(groups == group) for group in sampled_groups]
        )

    samples: dict[str, list[float]] = {}
    for _ in range(int(n_bootstrap)):
        id_indices = draw_indices(len(id_scores), id_groups)
        ood_indices = (
            id_indices
            if paired
            else draw_indices(len(ood_scores), ood_groups)
        )
        metrics = evaluate_ood(
            id_scores[id_indices],
            ood_scores[ood_indices],
        )
        for key, value in metrics.items():
            if isinstance(value, (float, int)) and np.isfinite(value):
                samples.setdefault(key, []).append(float(value))

    lower = 100.0 * alpha / 2.0
    upper = 100.0 * (1.0 - alpha / 2.0)
    return {
        key: [float(np.percentile(values, lower)), float(np.percentile(values, upper))]
        for key, values in samples.items()
        if values
    }
