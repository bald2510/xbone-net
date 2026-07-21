"""Non-parametric classifier based on empirical class centroids."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmpiricalCentroidHead(nn.Module):
    """Classify embeddings by cosine similarity to train-set class means.

    The centroids are train-set buffers rather than optimizer parameters.  An
    optional learnable class bias can calibrate argmax boundaries without
    discarding the centroid geometry or changing within-class score rankings.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_classes: int = 14,
        scale: float = 15.0,
        use_class_bias: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or num_classes < 2:
            raise ValueError("feature_dim must be positive and num_classes must be >= 2.")
        if scale <= 0:
            raise ValueError("scale must be positive.")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.scale = float(scale)
        self.use_class_bias = bool(use_class_bias)
        self.eps = float(eps)

        if self.use_class_bias:
            self.class_bias = nn.Parameter(torch.zeros(self.num_classes))
        else:
            self.register_parameter("class_bias", None)

        self.register_buffer(
            "centroids", torch.zeros(self.num_classes, self.feature_dim)
        )
        self.register_buffer(
            "centroid_counts", torch.zeros(self.num_classes, dtype=torch.long)
        )
        self.register_buffer(
            "centroids_initialized", torch.tensor(False, dtype=torch.bool)
        )

    @property
    def prototypes(self) -> torch.Tensor:
        """Backward-compatible alias used by the existing OOD pipeline."""
        return self.centroids

    @torch.no_grad()
    def set_centroids(
        self,
        centroids: torch.Tensor,
        counts: torch.Tensor,
    ) -> None:
        """Install empirical centroids computed from a labelled training set."""
        expected_shape = (self.num_classes, self.feature_dim)
        if tuple(centroids.shape) != expected_shape:
            raise ValueError(
                f"Expected centroids with shape {expected_shape}, "
                f"got {tuple(centroids.shape)}."
            )
        if tuple(counts.shape) != (self.num_classes,):
            raise ValueError(
                f"Expected counts with shape {(self.num_classes,)}, "
                f"got {tuple(counts.shape)}."
            )
        missing = torch.nonzero(counts <= 0, as_tuple=False).flatten().tolist()
        if missing:
            raise ValueError(
                "Empirical centroid estimation requires every configured class "
                f"to occur in the training subset. Missing classes: {missing}"
            )
        if not torch.isfinite(centroids).all():
            raise ValueError("Centroids contain NaN or infinite values.")
        zero_norm = torch.nonzero(
            centroids.norm(dim=-1) <= self.eps, as_tuple=False
        ).flatten().tolist()
        if zero_norm:
            raise ValueError(f"Empirical centroids have zero norm for classes: {zero_norm}")

        self.centroids.copy_(centroids.to(self.centroids))
        self.centroid_counts.copy_(counts.to(self.centroid_counts))
        self.centroids_initialized.fill_(True)

    def forward(self, features: torch.Tensor, return_features: bool = False):
        if features.ndim != 2:
            raise ValueError(
                "EmpiricalCentroidHead expects [B,D] features, "
                f"got {tuple(features.shape)}."
            )
        if features.size(-1) != self.feature_dim:
            raise ValueError(
                f"Expected feature dimension {self.feature_dim}, "
                f"got {features.size(-1)}."
            )
        if not bool(self.centroids_initialized.item()):
            raise RuntimeError(
                "Empirical centroids are not initialized. Compute them from the "
                "labelled training set before classification."
            )

        normed_features = F.normalize(features, dim=-1, eps=self.eps)
        normed_centroids = F.normalize(self.centroids, dim=-1, eps=self.eps)
        logits = self.scale * (normed_features @ normed_centroids.T)
        if self.class_bias is not None:
            logits = logits + self.class_bias
        if return_features:
            return logits, features
        return logits
