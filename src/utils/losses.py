"""
Loss Functions for XBone-Net Two-Stage Pipeline.
===============================================================================
Implements objective functions used across both training phases of XBone-Net,
focusing on multi-class ASL + Prototype loss architecture:

Phase 1 (Contrastive Learning):
  - SoftTargetSemanticMatchingLoss: Soft-target cross-entropy encoding same-class similarity

Phase 2 (Multi-Class Classification & Prototype Learning):
  - AsymmetricLossMulticlass: ASL adapted for multi-class classification with hard negative suppression
  - PrototypeLossMulticlass: Multi-class metric learning cosine distance objective to prototypes
  - OODMarginLoss: Regularization pushing out-of-distribution embeddings away from class prototypes
  - CombinedPhase2LossMulticlass: Composite multi-class loss combining ASL/CE + Prototype + OOD margin:
    L_total = L_ASL + lambda_proto * L_proto + lambda_ood * L_ood
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Helper Functions & Parameter Extraction
# ============================================================

def get_clip_logit_params(clip_model, require_bias: bool = False):
    """Retrieve learnable temperature and bias parameters from an OpenCLIP model.

    The logit_scale parameter represents log(1/tau) where tau is temperature:
        logits = exp(logit_scale) * sim(image, text)

    Args:
        clip_model: OpenCLIP model instance with logit_scale parameter.
        require_bias: If True and model lacks logit_bias, creates a registered
            parameter initialized to -10.0.

    Returns:
        Tuple of (logit_scale, logit_bias) nn.Parameter objects.

    Raises:
        AttributeError: If clip_model does not have a logit_scale attribute.
    """
    if not hasattr(clip_model, "logit_scale"):
        raise AttributeError("Expected OpenCLIP model with a `logit_scale` parameter.")

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
# Phase 1: Contrastive Loss Functions
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
        self.logit_scale, _ = get_clip_logit_params(clip_model)
        self.target_similarity = target_similarity

    def forward(self, image_features, text_features, disease_labels=None):
        """Compute soft-target symmetric contrastive loss.

        Args:
            image_features: Image embeddings of shape (B, D).
            text_features: Text embeddings of shape (B, D).
            disease_labels: Ground-truth class labels of shape (B,).

        Returns:
            Scalar loss tensor.
        """
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


# ============================================================
# Phase 2: Multi-Class ASL + Prototype Loss Modules
# ============================================================

class AsymmetricLossMulticlass(nn.Module):
    """Asymmetric Loss (ASL) adapted for multi-class classification.

    Applies asymmetric focal penalties and probability margin clipping to
    one-hot encoded targets, suppressing negative class gradients and focusing
    learning capacity on hard positive and minority classes.

    Formula:
        L_pos = - y_onehot * (1 - p)^gamma_pos * log(p)
        L_neg = - (1 - y_onehot) * max(p - clip, 0)^gamma_neg * log(1 - max(p - clip, 0))

    Attributes:
        gamma_pos (float): Positive focal exponent.
        gamma_neg (float): Negative focal exponent for hard negative down-weighting.
        clip (float): Asymmetric margin clipping factor.
        class_weights (torch.Tensor): Optional per-class weight vector.
    """

    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05, class_weights=None):
        """Initialize Multi-Class Asymmetric Loss.

        Args:
            gamma_pos: Positive focusing exponent (default: 0.0).
            gamma_neg: Negative focusing exponent for suppressing easy negatives (default: 4.0).
            clip: Probability margin threshold for hard negative suppression (default: 0.05).
            class_weights: Optional per-class weight vector for handling class imbalance.
        """
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Multi-Class ASL over predicted logits and target indices.

        Args:
            logits: Unnormalized network outputs of shape (B, C).
            targets: Target tensor of shape (B,) containing class indices or (B, C) one-hot.

        Returns:
            Scalar mean loss value.
        """
        if targets.ndim == 1:
            num_classes = logits.shape[-1]
            one_hot = F.one_hot(targets, num_classes=num_classes).float()
        else:
            one_hot = targets.float()

        p = torch.sigmoid(logits)
        eps = 1e-8

        # Positive loss for target class
        pos_loss = one_hot * (1 - p).pow(self.gamma_pos) * torch.log(p.clamp(min=eps))
        
        # Asymmetric negative loss for non-target classes with margin clipping
        p_m = (p - self.clip).clamp(min=0)
        neg_loss = (1 - one_hot) * p_m.pow(self.gamma_neg) * torch.log((1 - p_m).clamp(min=eps))

        asl_loss = -(pos_loss + neg_loss)
        
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            asl_loss = asl_loss * weights.view(1, -1)

        return asl_loss.sum(dim=-1).mean()


class AsymmetricLoss(AsymmetricLossMulticlass):
    """Alias for multi-label backward compatibility."""
    pass


class PrototypeLossMulticlass(nn.Module):
    """Multi-class prototype distance objective for metric learning.

    Pulls sample features towards their target class prototype while pushing them
    away from non-target class prototypes using cosine distance.

    Formula:
        L_pull = mean(1 - cos(f_i, p_target))
        L_push = mean(relu(margin - (1 - cos(f_i, p_non_target))))

    Attributes:
        margin (float): Safety distance margin for non-target prototypes.
    """

    def __init__(self, margin: float = 0.5):
        """Initialize multi-class prototype loss.

        Args:
            margin: Safety distance margin for non-target prototypes.
        """
        super().__init__()
        self.margin = margin

    def forward(self, features: torch.Tensor, prototypes: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute multi-class pull-push prototype loss.

        Args:
            features: Extracted image features of shape (B, D).
            prototypes: Class prototypes matrix of shape (C, D).
            targets: Ground-truth class index tensor of shape (B,).

        Returns:
            Scalar loss value combining positive pull and negative push.
        """
        normed_f = F.normalize(features, dim=-1)
        normed_p = F.normalize(prototypes, dim=-1)
        cosine_sim = normed_f @ normed_p.T
        dist = 1.0 - cosine_sim

        # Pull positive features towards target prototype
        pos_dist = dist[torch.arange(len(targets), device=targets.device), targets]
        pull_loss = pos_dist.mean()

        # Push features away from non-target prototypes
        mask = torch.ones_like(dist, dtype=torch.bool)
        mask[torch.arange(len(targets), device=targets.device), targets] = False
        neg_dist = dist[mask].view(len(targets), -1)
        push_loss = F.relu(self.margin - neg_dist).mean()
        return pull_loss + push_loss


class PrototypeLoss(nn.Module):
    """Alias for backward compatibility."""
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.proto_multi = PrototypeLossMulticlass(margin)

    def forward(self, features, prototypes, targets):
        if targets.ndim > 1:
            targets = targets.argmax(dim=-1)
        return self.proto_multi(features, prototypes, targets)


class OODMarginLoss(nn.Module):
    """Margin loss pushing out-of-distribution (OOD) features away from prototypes.

    Formula:
        L = mean(relu(margin - min_k dist(f_ood, p_k)))

    Attributes:
        margin (float): Minimum desired distance to nearest class prototype.
    """

    def __init__(self, margin: float = 1.0):
        """Initialize OOD margin loss.

        Args:
            margin: Desired minimum distance threshold.
        """
        super().__init__()
        self.margin = margin

    def forward(self, ood_features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """Compute OOD separation margin loss.

        Args:
            ood_features: Features from OOD samples of shape (N, D).
            prototypes: Class prototype vectors of shape (C, D).

        Returns:
            Scalar loss tensor.
        """
        if ood_features is None or len(ood_features) == 0:
            return torch.tensor(0.0, device=prototypes.device)

        normed_ood = F.normalize(ood_features, dim=-1)
        normed_proto = F.normalize(prototypes, dim=-1)
        cosine_sim = normed_ood @ normed_proto.T
        dist = 1.0 - cosine_sim
        min_dist, _ = dist.min(dim=1)
        loss = F.relu(self.margin - min_dist).mean()
        return loss


class CombinedPhase2LossMulticlass(nn.Module):
    """Combined Weighted Cross-Entropy and Prototype loss for multi-class classification.

    Formula:
        L_total = (1 - lambda_proto) * L_weighted_ce + lambda_proto * L_proto

    Attributes:
        ce (nn.CrossEntropyLoss): Class-weighted cross-entropy classification loss.
        proto_loss (PrototypeLossMulticlass): Multi-class prototype distance objective.
        lambda_proto (float): Convex weight factor for prototype loss (default: 0.3).
    """

    def __init__(
        self,
        class_weights=None,
        proto_margin: float = 0.5,
        lambda_proto: float = 0.3,
        label_smoothing: float = 0.1,
        **kwargs,
    ):
        """Initialize multi-class Phase 2 combined Weighted CE + Proto loss module.

        Args:
            class_weights: Class-balanced weight tensor for handling class imbalance.
            proto_margin: Distance margin for prototype loss (default: 0.5).
            lambda_proto: Convex weight factor for prototype loss (default: 0.3).
            label_smoothing: Label smoothing factor for cross-entropy (default: 0.1).
            **kwargs: Unused extra arguments for backward compatibility.
        """
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.proto_loss = PrototypeLossMulticlass(proto_margin)
        self.lambda_proto = lambda_proto

    def forward(self, logits, targets, features=None, prototypes=None, **kwargs):
        """Compute composite multi-class training loss.

        Args:
            logits: Output classification logits of shape (B, C).
            targets: Ground-truth class index tensor of shape (B,).
            features: Latent feature vectors of shape (B, D).
            prototypes: Class prototypes matrix of shape (C, D).
            **kwargs: Unused extra arguments.

        Returns:
            Scalar combined loss tensor.
        """
        l_ce = self.ce(logits, targets)

        if features is not None and prototypes is not None:
            l_proto = self.proto_loss(features, prototypes, targets)
            loss = l_ce + self.lambda_proto * l_proto
        else:
            loss = l_ce

        return loss


class CombinedPhase2Loss(CombinedPhase2LossMulticlass):
    """Alias for backward compatibility."""
    pass
