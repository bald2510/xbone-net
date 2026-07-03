"""
Out-of-Distribution (OOD) Detection for XBone-Net Embeddings.
===============================================================================
Provides post-hoc OOD scoring methods operating on XBone-Net feature embeddings:
  - Mahalanobis distance: Class-conditional Gaussian with shared covariance
  - Cosine k-NN: Average cosine distance to k-nearest in-distribution neighbors
  - Energy score: LogSumExp-based energy derived from classification logits
  - Text-anchor: Zero-shot cosine similarity to VLM text prompts

Evaluates detection performance via AUROC and FPR@95TPR metrics.
"""

import numpy as np
from sklearn.covariance import EmpiricalCovariance, ShrunkCovariance
from sklearn.metrics import roc_auc_score


# ============================================================
# Post-Hoc OOD Detector Class
# ============================================================

class OODDetector:
    """Post-hoc OOD detector for XBone-Net feature embeddings.

    Supports multiple scoring strategies operating on latent embeddings,
    logits, or VLM text anchors to identify out-of-distribution samples.

    Attributes:
        class_means (dict): Mapping from class index to centroid vector.
        shared_cov_inv (np.ndarray): Inverse shared covariance matrix for Mahalanobis.
        ref_embeddings (np.ndarray): In-distribution reference embeddings matrix.
        ref_labels (np.ndarray): Class labels corresponding to ref_embeddings.

    Example:
        detector = OODDetector()
        detector.fit(id_embeddings, id_labels)
        scores = detector.score(test_embeddings, method="mahalanobis")
    """

    def __init__(self):
        """Initialize empty OODDetector instance."""
        self.class_means = {}
        self.shared_cov_inv = None
        self.ref_embeddings = None
        self.ref_labels = None
        self._fitted = False

    def fit(self, embeddings: np.ndarray, labels: np.ndarray, prototypes: np.ndarray = None):
        """Fit class-conditional Gaussian model on in-distribution embeddings.

        Args:
            embeddings: ID reference embeddings of shape (N, D).
            labels: Ground-truth integer labels (N,) or one-hot matrix (N, C).
            prototypes: Optional learned class prototypes of shape (C, D).
        """
        self.ref_embeddings = embeddings
        self.ref_labels = labels

        # --- Convert one-hot labels to integer class indices ---
        if labels.ndim == 2:
            proxy_labels = np.argmax(labels, axis=1)
        else:
            proxy_labels = labels

        unique_classes = np.unique(proxy_labels)
        self.class_means = {}
        all_centered = []

        # --- L2-normalize prototypes if available ---
        normed_prototypes = None
        if prototypes is not None:
            norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
            normed_prototypes = prototypes / (norms + 1e-8)

        # --- Compute class centroids and centered data ---
        for c in unique_classes:
            mask = proxy_labels == c
            class_embeds = embeddings[mask]
            if normed_prototypes is not None and c < len(normed_prototypes):
                self.class_means[c] = normed_prototypes[c]
            else:
                self.class_means[c] = class_embeds.mean(axis=0)
            all_centered.append(class_embeds - self.class_means[c])

        centered = np.vstack(all_centered)

        # --- Estimate shared covariance matrix ---
        try:
            cov_estimator = EmpiricalCovariance().fit(centered)
            self.shared_cov_inv = np.linalg.inv(cov_estimator.covariance_)
        except np.linalg.LinAlgError:
            cov_estimator = ShrunkCovariance().fit(centered)
            self.shared_cov_inv = np.linalg.inv(cov_estimator.covariance_)

        self._fitted = True

    def score_mahalanobis(self, test_embeddings: np.ndarray) -> np.ndarray:
        """Compute minimum Mahalanobis distance across ID class centroids.

        Formula:
            d(x) = min_c (x - mu_c)^T Sigma^{-1} (x - mu_c)

        Args:
            test_embeddings: Test embeddings of shape (M, D).

        Returns:
            OOD scores of shape (M,) - higher values indicate OOD.
        """
        assert self._fitted, "Call fit() first with ID reference embeddings."

        # --- Distance to nearest class centroid ---
        scores = np.full(len(test_embeddings), np.inf)
        for c, mean in self.class_means.items():
            diff = test_embeddings - mean
            maha = np.sum(diff @ self.shared_cov_inv * diff, axis=1)
            scores = np.minimum(scores, maha)

        return scores

    def score_knn(self, test_embeddings: np.ndarray, k: int = 5) -> np.ndarray:
        """Compute average cosine distance to the k nearest ID neighbors.

        Formula:
            d(x) = 1/k sum_{i=1}^k (1 - cos(x, r_i))

        Args:
            test_embeddings: L2-normalized test embeddings of shape (M, D).
            k: Number of nearest neighbors to average.

        Returns:
            OOD scores of shape (M,) - higher values indicate OOD.
        """
        assert self._fitted, "Call fit() first with ID reference embeddings."

        # --- Compute cosine distance to reference embeddings ---
        sim_matrix = test_embeddings @ self.ref_embeddings.T
        dist_matrix = 1.0 - sim_matrix

        k = min(k, dist_matrix.shape[1])
        topk_dists = np.partition(dist_matrix, k, axis=1)[:, :k]
        scores = topk_dists.mean(axis=1)

        return scores

    def score_energy(self, logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Compute energy-based OOD score from classification logits.

        Formula:
            E(x) = -T * log sum_c exp(f_c(x) / T)
            score(x) = -E(x)

        Args:
            logits: Classification logits of shape (M, C).
            temperature: Temperature scaling factor.

        Returns:
            OOD scores of shape (M,) - higher values indicate OOD.
        """
        # --- Compute log-sum-exp energy ---
        scaled = logits / temperature
        energy = -temperature * np.log(np.sum(np.exp(scaled - scaled.max(axis=1, keepdims=True)), axis=1) 
                                        + np.exp(-scaled.max(axis=1)))
        return -energy

    def score_text_anchor(
        self,
        test_img_embeddings: np.ndarray,
        anchor_text_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Compute zero-shot OOD score via text-anchor cosine similarity.

        Formula:
            score(x) = 1 - max_c cos(x, t_c)

        Args:
            test_img_embeddings: L2-normalized image embeddings of shape (M, D).
            anchor_text_embeddings: Text anchor embeddings of shape (A, D).

        Returns:
            OOD scores of shape (M,) - higher values indicate OOD.
        """
        # --- Cosine distance to nearest text anchor ---
        sim_matrix = test_img_embeddings @ anchor_text_embeddings.T
        max_sim = sim_matrix.max(axis=1)
        return 1.0 - max_sim

    def score(self, test_embeddings: np.ndarray, method: str = "mahalanobis", **kwargs) -> np.ndarray:
        """Unified dispatch for all supported OOD scoring methods.

        Args:
            test_embeddings: Test embeddings of shape (M, D).
            method: Scoring strategy ('mahalanobis', 'knn', 'energy', 'text_anchor').
            **kwargs: Method-specific arguments.

        Returns:
            OOD scores of shape (M,) - higher values indicate OOD.

        Raises:
            ValueError: If method is unrecognised.
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


# ============================================================
# OOD Evaluation Metrics
# ============================================================

def evaluate_ood(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> dict:
    """Evaluate OOD detection performance using AUROC and FPR@95TPR.

    Formula:
        FPR@95TPR = FP / N_id at threshold where TP / N_ood >= 0.95

    Args:
        id_scores: In-distribution OOD scores of shape (N_id,).
        ood_scores: Out-of-distribution scores of shape (N_ood,).

    Returns:
        Dictionary containing auroc, fpr_at_95tpr, n_id, and n_ood.
    """
    # --- Prepare labels and scores ---
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(labels, scores)

    # --- Sweep threshold for FPR at 95% TPR ---
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]

    n_ood = int(labels.sum())
    n_id = len(labels) - n_ood

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


