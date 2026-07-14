"""
Loss Functions for XBone-Net Two-Phase Pipeline.
===============================================================================
Implements streamlined objective functions used across both training phases of XBone-Net:

Phase 1 (Multimodal Contrastive Alignment):
  - SoftTargetSemanticMatchingLoss: Soft-target cross-entropy encoding same-class similarity

Phase 2 (Supervised Classification):
  - Validated class-weighted cross-entropy for empirical-centroid or linear logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Phase 1 Helper Functions
# ============================================================

def get_clip_logit_params(clip_model, require_bias: bool = False):
    """Retrieve learnable temperature and bias parameters from an OpenCLIP model.

    The logit_scale parameter represents log(1/tau) where tau is temperature:
        logits = exp(logit_scale) * sim(image, text)

    Args:
        clip_model: OpenCLIP model instance with logit_scale parameter.
        require_bias: If True and model lacks logit_bias, registers a bias parameter.

    Returns:
        Tuple of (logit_scale, logit_bias) nn.Parameter objects.

    Raises:
        AttributeError: If clip_model does not have a logit_scale parameter.
    """
    if not hasattr(clip_model, "logit_scale"):
        # Fallback for non-OpenCLIP backbones (e.g., MedCLIP): create a learnable logit_scale
        # initialized to ln(1/0.07) ≈ 2.66, matching standard CLIP temperature
        import math
        logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
        setattr(clip_model, "logit_scale", logit_scale)
        print("  [Loss] Created learnable logit_scale for non-OpenCLIP backbone.")

    logit_scale = clip_model.logit_scale
    logit_bias = getattr(clip_model, "logit_bias", None)

    if logit_bias is None and require_bias:
        if hasattr(clip_model, "logit_bias"):
            logit_bias = clip_model.logit_bias
        else:
            logit_bias = nn.Parameter(torch.full([], -10.0, device=logit_scale.device))
            clip_model.register_parameter("logit_bias", logit_bias)

    return logit_scale, logit_bias


def _pairwise_logits(image_features, text_features, logit_scale, logit_bias=None):
    """Compute scaled pairwise cosine similarity logits between image and text.

    Formula:
        logits = exp(scale) * (I_norm @ T_norm^T) + bias

    Args:
        image_features: Image embeddings of shape (B, D).
        text_features: Text embeddings of shape (B, D).
        logit_scale: Learnable log-temperature parameter (scalar nn.Parameter).
        logit_bias: Optional learnable bias parameter (scalar nn.Parameter).

    Returns:
        Logits tensor of shape (B, B) with pairwise similarity scores.
    """
    image_features = F.normalize(image_features, dim=-1, p=2)
    text_features = F.normalize(text_features, dim=-1, p=2)
    logits = logit_scale.exp() * image_features @ text_features.T
    if logit_bias is not None:
        logits = logits + logit_bias
    return logits


# ============================================================
# Phase 1: Contrastive Loss Function
# ============================================================

class SoftTargetSemanticMatchingLoss(nn.Module):
    """Soft-target semantic matching contrastive loss for Phase 1.

    Constructs soft target matrices encoding intra-class similarity rather than
    one-hot matching. Pairs sharing disease labels receive target 0.95, while
    exact diagonal pairs receive 1.0.

    Formula:
        L = -0.5 * (mean_i sum_j y_v2t[i,j] * log_softmax(logits_v2t)[i,j]
                  + mean_j sum_i y_t2v[i,j] * log_softmax(logits_t2i)[i,j])

    Attributes:
        logit_scale (nn.Parameter): Learnable log-temperature parameter.
        target_similarity (float): Soft target probability assigned to same-class pairs.
    """

    def __init__(self, clip_model, target_similarity: float = 0.95):
        """Initialize SoftTargetSemanticMatchingLoss.

        Args:
            clip_model: OpenCLIP model providing logit scale.
            target_similarity: Soft target value for same-class non-diagonal pairs.
        """
        super().__init__()
        if not 0.0 <= float(target_similarity) <= 1.0:
            raise ValueError("target_similarity must be in [0, 1].")
        self.logit_scale, _ = get_clip_logit_params(clip_model)
        self.target_similarity = float(target_similarity)

    def forward(self, image_features, text_features, disease_labels=None):
        """Compute soft-target symmetric contrastive loss.

        Args:
            image_features: Image embeddings of shape (B, D).
            text_features: Text embeddings of shape (B, D).
            disease_labels: Ground-truth class labels of shape (B,).

        Returns:
            Scalar loss tensor.
        """
        if image_features.ndim != 2 or text_features.ndim != 2:
            raise ValueError(
                "Semantic matching expects [B,D] image and text features, got "
                f"{tuple(image_features.shape)} and {tuple(text_features.shape)}."
            )
        if image_features.shape != text_features.shape:
            raise ValueError("Image and text features must have identical [B,D] shapes.")
        batch_size = image_features.size(0)
        device = image_features.device
        logits_v2t = _pairwise_logits(image_features, text_features, self.logit_scale)
        logits_t2v = logits_v2t.T

        # --- Construct soft target matrix ---
        targets_v2t = torch.eye(batch_size, device=device)
        if disease_labels is not None:
            labels_col = disease_labels.view(-1, 1)
            labels_row = disease_labels.view(1, -1)
            same_class_mask = (labels_col == labels_row) & (~torch.eye(batch_size, dtype=torch.bool, device=device))
            targets_v2t[same_class_mask] = self.target_similarity

        targets_v2t = targets_v2t / targets_v2t.sum(dim=1, keepdim=True)
        targets_t2v = targets_v2t.T / targets_v2t.T.sum(dim=1, keepdim=True)

        loss_v2t = -torch.sum(targets_v2t * F.log_softmax(logits_v2t, dim=1), dim=1).mean()
        loss_t2v = -torch.sum(targets_t2v * F.log_softmax(logits_t2v, dim=1), dim=1).mean()
        return 0.5 * (loss_v2t + loss_t2v)





LOSS_REGISTRY = {
    "semantic_matching": SoftTargetSemanticMatchingLoss,
}


def build_loss(loss_type: str, clip_model=None, **kwargs):
    """Build Phase 1 contrastive loss module.

    Args:
        loss_type: Loss identifier ('semantic_matching').
        clip_model: OpenCLIP model providing logit scale/bias parameters.
        **kwargs: Additional arguments passed to loss constructor.

    Returns:
        nn.Module loss instance.
    """
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"Loss type '{loss_type}' not supported. Choose from {list(LOSS_REGISTRY.keys())}")
    if clip_model is None:
        raise ValueError("build_loss requires `clip_model`.")
    kwargs.pop("temperature", None)
    return LOSS_REGISTRY[loss_type](clip_model=clip_model, **kwargs)


def resolve_phase2_loss_type(loss_type: str | None, classifier_type: str) -> str:
    """Resolve and validate the supervised Phase-2 objective.

    Empirical centroids are fixed train-set statistics, so their classifier is
    optimized through cross-entropy over scaled cosine-similarity logits. A
    distinct name is retained in configuration to make that coupling explicit
    and to reject accidentally pairing an empirical-centroid objective with a
    trainable linear head (or vice versa).
    """
    classifier_type = str(classifier_type).strip().lower()
    supported_classifiers = {"empirical_centroid", "linear"}
    if classifier_type not in supported_classifiers:
        raise ValueError(
            f"Unsupported Phase-2 classifier_type='{classifier_type}'. "
            f"Choose from {sorted(supported_classifiers)}."
        )
    expected = (
        "empirical_centroid_ce"
        if classifier_type == "empirical_centroid"
        else "ce"
    )
    resolved = expected if loss_type is None else str(loss_type).strip().lower()
    supported = {"ce", "empirical_centroid_ce"}
    if resolved not in supported:
        raise ValueError(
            f"Unsupported Phase-2 loss_type='{resolved}'. "
            f"Choose from {sorted(supported)}."
        )
    if resolved != expected:
        raise ValueError(
            f"Phase-2 loss_type='{resolved}' is incompatible with "
            f"classifier_type='{classifier_type}'. Expected '{expected}'."
        )
    return resolved


def build_phase2_loss(
    loss_type: str | None,
    classifier_type: str,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """Build the validated Phase-2 classification loss."""
    resolve_phase2_loss_type(loss_type, classifier_type)
    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(label_smoothing),
    )
