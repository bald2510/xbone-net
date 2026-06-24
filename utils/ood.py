"""
OOD Detection Module for VLM Embedding Space
=============================================
Provides 4 OOD scoring methods that operate on pre-extracted embeddings:
  1. Mahalanobis Distance
  2. Cosine-KNN (k-nearest neighbors)
  3. Energy Score
  4. Text-Anchor Distance (VLM-specific, zero-reference)

Usage:
    detector = OODDetector()
    detector.fit(id_embeddings, id_labels)
    scores = detector.score(ood_embeddings, method="mahalanobis")
"""

import numpy as np
from sklearn.covariance import EmpiricalCovariance, ShrunkCovariance
from sklearn.metrics import roc_auc_score


class OODDetector:
    """OOD detection using VLM embedding space features."""

    def __init__(self):
        self.class_means = {}
        self.shared_cov_inv = None
        self.ref_embeddings = None
        self.ref_labels = None
        self._fitted = False

    def fit(self, embeddings: np.ndarray, labels: np.ndarray):
        """
        Fit the ID distribution from reference embeddings.

        Args:
            embeddings: (N, D) L2-normalized embedding vectors
            labels: (N,) integer class labels or (N, C) multi-label binary matrix
        """
        self.ref_embeddings = embeddings
        self.ref_labels = labels

        # Handle multi-label: use argmax of label vector as proxy class
        if labels.ndim == 2:
            proxy_labels = np.argmax(labels, axis=1)
        else:
            proxy_labels = labels

        # Compute per-class means
        unique_classes = np.unique(proxy_labels)
        self.class_means = {}
        all_centered = []

        for c in unique_classes:
            mask = proxy_labels == c
            class_embeds = embeddings[mask]
            self.class_means[c] = class_embeds.mean(axis=0)
            all_centered.append(class_embeds - self.class_means[c])

        # Shared covariance (tied across classes)
        centered = np.vstack(all_centered)
        try:
            cov_estimator = EmpiricalCovariance().fit(centered)
            self.shared_cov_inv = np.linalg.inv(cov_estimator.covariance_)
        except np.linalg.LinAlgError:
            # Fallback to shrunk covariance if singular
            cov_estimator = ShrunkCovariance().fit(centered)
            self.shared_cov_inv = np.linalg.inv(cov_estimator.covariance_)

        self._fitted = True

    def score_mahalanobis(self, test_embeddings: np.ndarray) -> np.ndarray:
        """
        Mahalanobis distance score. Higher = more OOD.

        Args:
            test_embeddings: (M, D) test embedding vectors

        Returns:
            (M,) OOD scores (higher = more likely OOD)
        """
        assert self._fitted, "Call fit() first with ID reference embeddings."

        scores = np.full(len(test_embeddings), np.inf)
        for c, mean in self.class_means.items():
            diff = test_embeddings - mean  # (M, D)
            # Mahalanobis: sqrt(diff @ Sigma_inv @ diff.T)
            maha = np.sum(diff @ self.shared_cov_inv * diff, axis=1)
            scores = np.minimum(scores, maha)  # min across classes

        return scores

    def score_knn(self, test_embeddings: np.ndarray, k: int = 5) -> np.ndarray:
        """
        Cosine-KNN distance score. Higher = more OOD.

        Args:
            test_embeddings: (M, D) test embedding vectors
            k: number of nearest neighbors

        Returns:
            (M,) OOD scores
        """
        assert self._fitted, "Call fit() first with ID reference embeddings."

        # Cosine similarity → distance
        # Embeddings are assumed L2-normalized
        sim_matrix = test_embeddings @ self.ref_embeddings.T  # (M, N)
        # Convert similarity to distance
        dist_matrix = 1.0 - sim_matrix

        # Top-k smallest distances (nearest neighbors)
        k = min(k, dist_matrix.shape[1])
        topk_dists = np.partition(dist_matrix, k, axis=1)[:, :k]
        scores = topk_dists.mean(axis=1)

        return scores

    def score_energy(self, logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Energy score. Lower energy = more OOD (negate for consistency).

        Args:
            logits: (M, C) classification logits (pre-softmax)
            temperature: temperature scaling

        Returns:
            (M,) OOD scores (higher = more likely OOD)
        """
        # Energy = -T * log(sum(exp(logits/T)))
        scaled = logits / temperature
        energy = -temperature * np.log(np.sum(np.exp(scaled - scaled.max(axis=1, keepdims=True)), axis=1) 
                                        + np.exp(-scaled.max(axis=1)))
        # Negate so higher = more OOD
        return -energy

    def score_text_anchor(
        self,
        test_img_embeddings: np.ndarray,
        anchor_text_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Text-anchor distance. Uses text prompts as ID anchors.
        Higher distance from all anchors = more OOD.

        This method requires ZERO reference images — only text prompts.

        Args:
            test_img_embeddings: (M, D) image embeddings (L2-normalized)
            anchor_text_embeddings: (K, D) text prompt embeddings for K ID classes

        Returns:
            (M,) OOD scores (higher = more likely OOD)
        """
        # Max cosine similarity to any anchor text
        sim_matrix = test_img_embeddings @ anchor_text_embeddings.T  # (M, K)
        max_sim = sim_matrix.max(axis=1)  # Best match to any ID class

        # OOD score = 1 - max_similarity (higher = more OOD)
        return 1.0 - max_sim

    def score(self, test_embeddings: np.ndarray, method: str = "mahalanobis", **kwargs) -> np.ndarray:
        """
        Unified scoring interface.

        Args:
            test_embeddings: (M, D) test embeddings
            method: one of "mahalanobis", "knn", "energy", "text_anchor"
            **kwargs: method-specific args (k for knn, logits for energy, etc.)
        """
        if method == "mahalanobis":
            return self.score_mahalanobis(test_embeddings)
        elif method == "knn":
            return self.score_knn(test_embeddings, k=kwargs.get("k", 5))
        elif method == "energy":
            return self.score_energy(kwargs["logits"], kwargs.get("temperature", 1.0))
        elif method == "text_anchor":
            return self.score_text_anchor(test_embeddings, kwargs["anchor_text_embeddings"])
        else:
            raise ValueError(f"Unknown OOD method: {method}. Choose from: mahalanobis, knn, energy, text_anchor")


def evaluate_ood(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> dict:
    """
    Evaluate OOD detection quality.

    Args:
        id_scores: (N,) OOD scores for in-distribution samples
        ood_scores: (M,) OOD scores for out-of-distribution samples

    Returns:
        dict with AUROC and FPR@95TPR
    """
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(labels, scores)

    # FPR@95TPR: False positive rate when true positive rate = 95%
    # Sort by score descending
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]

    n_ood = int(labels.sum())
    n_id = len(labels) - n_ood

    # Find threshold where 95% of OOD samples are detected
    tp_target = int(0.95 * n_ood)
    tp_count = 0
    fp_count = 0

    for i in range(len(sorted_labels)):
        if sorted_labels[i] == 1:
            tp_count += 1
        else:
            fp_count += 1
        if tp_count >= tp_target:
            break

    fpr_at_95 = fp_count / n_id if n_id > 0 else 0.0

    return {
        "auroc": float(auroc),
        "fpr_at_95tpr": float(fpr_at_95),
        "n_id": n_id,
        "n_ood": n_ood,
    }
