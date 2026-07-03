"""
Loss Functions for XBone-Net Two-Stage Pipeline.
===============================================================================
Implements objective functions used across both training phases of XBone-Net:

Phase 1 (Contrastive Learning):
  - SoftTargetSemanticMatchingLoss: Soft-target cross-entropy encoding same-class similarity
  - InfoNCELoss: Standard symmetric CLIP contrastive loss
  - SigCLIPLoss: Sigmoid-based contrastive loss (SigLIP)

Phase 2 (Classification):
  - AsymmetricLoss: ASL for multi-label pathology classification with hard negative mining
  - PrototypeLoss / PrototypeLossMulticlass: Metric learning distance objective to class prototypes
  - OODMarginLoss: Regularization pushing out-of-distribution embeddings away from prototypes
  - CombinedPhase2Loss / CombinedPhase2LossMulticlass: Composite losses uniting classification,
    prototype, and OOD regularization: L_total = L_cls + lambda_proto * L_proto + lambda_ood * L_ood
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip.loss import ClipLoss, SigLipLoss


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

    # --- Register bias parameter if required and missing ---
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
    # --- Normalize embeddings and compute scaled similarity matrix ---
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
        logit_bias (nn.Parameter): Optional learnable logit bias parameter.
    """

    def __init__(self, clip_model):
        """Initialize with temperature parameters from OpenCLIP model.

        Args:
            clip_model: OpenCLIP model instance.
        """
        super().__init__()
        self.logit_scale, self.logit_bias = get_clip_logit_params(clip_model)

    def forward(self, image_features, text_features, disease_labels):
        """Compute soft-target semantic matching loss.

        Args:
            image_features: Image embeddings of shape (B, D).
            text_features: Text embeddings of shape (B, D).
            disease_labels: Ground-truth labels of shape (B,) or (B, C).

        Returns:
            Scalar loss tensor.
        """
        # --- Pairwise logits calculation ---
        logits_per_image = _pairwise_logits(image_features, text_features, self.logit_scale, self.logit_bias)
        logits_per_text = logits_per_image.T
        batch_size = image_features.shape[0]

        # --- Build pairwise equality mask ---
        if disease_labels.ndim == 1:
            eq_mask = (disease_labels.unsqueeze(1) == disease_labels.unsqueeze(0))
        else:
            eq_mask = (disease_labels.unsqueeze(1) == disease_labels.unsqueeze(0)).all(dim=-1)

        # --- Construct soft targets ---
        targets = torch.zeros((batch_size, batch_size), device=image_features.device)
        targets[eq_mask] = 0.95  # Soft target for same-class pairs
        targets[torch.arange(batch_size), torch.arange(batch_size)] = 1.0  # Exact match

        # --- Row and column normalization ---
        row_sums = targets.sum(dim=1, keepdim=True).clamp(min=1e-8)
        col_sums = targets.sum(dim=0, keepdim=True).clamp(min=1e-8)
        y_v2t = targets / row_sums
        y_t2v = targets / col_sums

        log_preds_v2t = F.log_softmax(logits_per_image, dim=1)
        log_preds_t2v = F.log_softmax(logits_per_text, dim=1)

        # --- Symmetric soft cross-entropy ---
        loss_i = -(y_v2t * log_preds_v2t).sum(dim=1).mean()
        loss_t = -(y_t2v * log_preds_t2v).sum(dim=1).mean()

        return (loss_i + loss_t) / 2


class InfoNCELoss(nn.Module):
    """Standard InfoNCE symmetric CLIP contrastive loss.

    Formula:
        L = 0.5 * (CE(logits_i2t, targets) + CE(logits_t2i, targets))

    Attributes:
        logit_scale (nn.Parameter): Learnable log-temperature parameter.
        clip_loss (ClipLoss): OpenCLIP loss wrapper module.
    """

    def __init__(self, clip_model):
        """Initialize InfoNCE loss wrapper.

        Args:
            clip_model: OpenCLIP model instance.
        """
        super().__init__()
        self.logit_scale, _ = get_clip_logit_params(clip_model)
        self.clip_loss = ClipLoss()

    def forward(self, image_features, text_features, disease_labels=None):
        """Compute InfoNCE contrastive loss.

        Args:
            image_features: Image embeddings of shape (B, D).
            text_features: Text embeddings of shape (B, D).
            disease_labels: Unused, kept for interface compatibility.

        Returns:
            Scalar loss tensor.
        """
        image_features = F.normalize(image_features, dim=-1, p=2)
        text_features = F.normalize(text_features, dim=-1, p=2)
        return self.clip_loss(
            image_features,
            text_features,
            self.logit_scale.exp(),
        )


class SigCLIPLoss(nn.Module):
    """Sigmoid contrastive loss (SigLIP).

    Formula:
        L = -sum_{i,j} log sigmoid(z_{i,j} * (exp(scale) * sim(i,j) + bias))

    Attributes:
        logit_scale (nn.Parameter): Learnable log-temperature parameter.
        logit_bias (nn.Parameter): Learnable logit bias parameter.
        siglip_loss (SigLipLoss): OpenCLIP SigLIP loss wrapper module.
    """

    def __init__(self, clip_model):
        """Initialize SigLIP loss module.

        Args:
            clip_model: OpenCLIP model instance.
        """
        super().__init__()
        self.logit_scale, self.logit_bias = get_clip_logit_params(clip_model, require_bias=True)
        self.siglip_loss = SigLipLoss()

    def forward(self, image_features, text_features, disease_labels=None):
        """Compute SigLIP contrastive loss.

        Args:
            image_features: Image embeddings of shape (B, D).
            text_features: Text embeddings of shape (B, D).
            disease_labels: Unused, kept for interface compatibility.

        Returns:
            Scalar loss tensor.
        """
        image_features = F.normalize(image_features, dim=-1, p=2)
        text_features = F.normalize(text_features, dim=-1, p=2)
        return self.siglip_loss(
            image_features,
            text_features,
            self.logit_scale.exp(),
            self.logit_bias,
        )


LOSS_REGISTRY = {
    "semantic_matching": SoftTargetSemanticMatchingLoss,
    "infonce": InfoNCELoss,
    "sigclip": SigCLIPLoss,
}


def build_loss(loss_type: str, clip_model=None, **kwargs):
    """Build Phase 1 contrastive loss module from configuration name.

    Args:
        loss_type: Loss identifier ('semantic_matching', 'infonce', or 'sigclip').
        clip_model: OpenCLIP model providing logit scale/bias parameters.
        **kwargs: Additional arguments passed to loss constructor.

    Returns:
        nn.Module loss instance.

    Raises:
        ValueError: If loss_type is unknown or clip_model is None.
    """
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"Loss type '{loss_type}' not supported. Choose from {list(LOSS_REGISTRY.keys())}")
    if clip_model is None:
        raise ValueError("build_loss requires `clip_model` (OpenCLIP module).")
    kwargs.pop("temperature", None)
    return LOSS_REGISTRY[loss_type](clip_model=clip_model, **kwargs)


# ============================================================
# Phase 2: Multi-Label Loss Modules
# ============================================================

class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (ASL) for multi-label classification.

    Formula:
        L = - [ y * (1-p)^gamma_pos * log(p) + (1-y) * p_m^gamma_neg * log(1-p_m) ]
        where p = sigmoid(logits), p_m = max(p - clip, 0).

    Attributes:
        gamma_pos (float): Positive focal exponent.
        gamma_neg (float): Negative focal exponent for hard negative down-weighting.
        clip (float): Asymmetric margin clipping factor.
    """

    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05):
        """Initialize Asymmetric Loss parameters.

        Args:
            gamma_pos: Positive focusing hyperparameter.
            gamma_neg: Negative focusing hyperparameter.
            clip: Probability margin threshold for hard negative suppression.
        """
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute ASL over predicted logits and binary targets.

        Args:
            logits: Unnormalized network outputs of shape (B, C).
            targets: Binary ground-truth labels of shape (B, C).

        Returns:
            Scalar mean loss value.
        """
        # --- Sigmoid probabilities ---
        p = torch.sigmoid(logits)
        eps = 1e-8

        # --- Positive and negative loss components ---
        pos_loss = targets * (1 - p).pow(self.gamma_pos) * torch.log(p.clamp(min=eps))
        p_m = (p - self.clip).clamp(min=0)
        neg_loss = (1 - targets) * p_m.pow(self.gamma_neg) * torch.log((1 - p_m).clamp(min=eps))
        loss = -(pos_loss + neg_loss)
        return loss.mean()


class PrototypeLoss(nn.Module):
    """Prototype distance loss for multi-label metric learning.

    Formula:
        L = y * dist(f, p) + (1 - y) * relu(margin - dist(f, p))
        where dist(f, p) = 1 - cos_sim(f, p).

    Attributes:
        margin (float): Distance margin threshold for negative prototype push.
    """

    def __init__(self, margin: float = 0.5):
        """Initialize Prototype Loss module.

        Args:
            margin: Distance margin for negative classes.
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        features: torch.Tensor,
        prototypes: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute prototype distance loss.

        Args:
            features: Sample representations of shape (B, D).
            prototypes: Class prototype vectors of shape (C, D).
            targets: Binary target indicator tensor of shape (B, C).

        Returns:
            Scalar mean loss value.
        """
        normed_features = F.normalize(features, dim=-1)
        normed_prototypes = F.normalize(prototypes, dim=-1)
        cosine_sim = normed_features @ normed_prototypes.T
        dist = 1.0 - cosine_sim
        pos_loss = targets * dist
        neg_loss = (1 - targets) * F.relu(self.margin - dist)
        loss = pos_loss + neg_loss
        return loss.mean()


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
            margin: Safety margin distance between OOD samples and prototypes.
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        features: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute OOD separation margin loss.

        Args:
            features: Out-of-distribution feature embeddings of shape (N_ood, D).
            prototypes: Class prototypes tensor of shape (C, D).

        Returns:
            Scalar loss tensor.
        """
        if features is None or features.shape[0] == 0:
            return torch.tensor(0.0, device=prototypes.device)
        normed_features = F.normalize(features, dim=-1)
        normed_prototypes = F.normalize(prototypes, dim=-1)
        cosine_sim = normed_features @ normed_prototypes.T
        dist = 1.0 - cosine_sim
        min_dist, _ = dist.min(dim=-1)
        loss = F.relu(self.margin - min_dist)
        return loss.mean()


class CombinedPhase2Loss(nn.Module):
    """Combined ASL, Prototype, and OOD loss for multi-label classification.

    Formula:
        L_total = L_asl + lambda_proto * L_proto + lambda_ood * L_ood

    Attributes:
        asl (AsymmetricLoss): Primary multi-label classification loss.
        proto_loss (PrototypeLoss): Distance-based prototype objective.
        ood_loss (OODMarginLoss): Out-of-distribution distance objective.
        lambda_proto (float): Prototype loss weighting factor.
        lambda_ood (float): OOD loss weighting factor.
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        asl_clip: float = 0.05,
        proto_margin: float = 0.5,
        ood_margin: float = 1.0,
        lambda_proto: float = 0.5,
        lambda_ood: float = 0.1,
    ):
        """Initialize multi-label Phase 2 loss module.

        Args:
            gamma_pos: Positive focusing parameter for ASL.
            gamma_neg: Negative focusing parameter for ASL.
            asl_clip: ASL probability threshold clip.
            proto_margin: Prototype distance margin.
            ood_margin: OOD safety margin.
            lambda_proto: Weight hyperparameter for prototype loss.
            lambda_ood: Weight hyperparameter for OOD loss.
        """
        super().__init__()
        self.asl = AsymmetricLoss(gamma_pos, gamma_neg, asl_clip)
        self.proto_loss = PrototypeLoss(proto_margin)
        self.ood_loss = OODMarginLoss(ood_margin)
        self.lambda_proto = lambda_proto
        self.lambda_ood = lambda_ood

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        features: torch.Tensor = None,
        prototypes: torch.Tensor = None,
        ood_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute composite multi-label training loss.

        Args:
            logits: Output classification logits of shape (B, C).
            targets: Target binary labels of shape (B, C).
            features: Latent feature vectors of shape (B, D).
            prototypes: Class prototype vectors of shape (C, D).
            ood_features: Synthetic/real OOD feature vectors of shape (N_ood, D).

        Returns:
            Scalar combined loss tensor.
        """
        # --- Primary ASL classification loss ---
        loss = self.asl(logits, targets)

        # --- Optional prototype distance regularization ---
        if features is not None and prototypes is not None:
            loss = loss + self.lambda_proto * self.proto_loss(features, prototypes, targets)

        # --- Optional OOD margin regularization ---
        if ood_features is not None and prototypes is not None:
            loss = loss + self.lambda_ood * self.ood_loss(ood_features, prototypes)
        return loss


# ============================================================
# Phase 2: Multi-Class Loss Modules
# ============================================================

class PrototypeLossMulticlass(nn.Module):
    """Prototype loss for multi-class classification.

    Formula:
        L = dist(f, p_target) + relu(margin - dist(f, p_non_target))

    Attributes:
        margin (float): Minimum distance margin for non-target class prototypes.
    """

    def __init__(self, margin: float = 0.5):
        """Initialize multi-class prototype loss.

        Args:
            margin: Safety distance margin for non-target prototypes.
        """
        super().__init__()
        self.margin = margin

    def forward(self, features, prototypes, targets):
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

        # --- Pull positive features towards target prototype ---
        pos_dist = dist[torch.arange(len(targets), device=targets.device), targets]
        pull_loss = pos_dist.mean()

        # --- Push features away from non-target prototypes ---
        mask = torch.ones_like(dist, dtype=torch.bool)
        mask[torch.arange(len(targets), device=targets.device), targets] = False
        neg_dist = dist[mask].view(len(targets), -1)
        push_loss = F.relu(self.margin - neg_dist).mean()
        return pull_loss + push_loss


class CombinedPhase2LossMulticlass(nn.Module):
    """Combined Cross-Entropy, Prototype, and OOD loss for multi-class classification.

    Formula:
        L_total = L_ce + lambda_proto * L_proto + lambda_ood * L_ood

    Attributes:
        ce (nn.CrossEntropyLoss): Cross-entropy classification loss with optional weights.
        proto_loss (PrototypeLossMulticlass): Multi-class prototype distance objective.
        ood_loss (OODMarginLoss): OOD distance margin objective.
        lambda_proto (float): Weight for prototype loss component.
        lambda_ood (float): Weight for OOD regularization component.
    """

    def __init__(self, class_weights=None, proto_margin=0.5,
                 ood_margin=1.0, lambda_proto=0.5, lambda_ood=0.0,
                 label_smoothing=0.1):
        """Initialize multi-class Phase 2 combined loss module.

        Args:
            class_weights: Optional class weighting tensor for imbalanced data.
            proto_margin: Distance margin for prototype loss.
            ood_margin: Distance margin for OOD separation.
            lambda_proto: Weight factor for prototype loss.
            lambda_ood: Weight factor for OOD loss.
            label_smoothing: Label smoothing factor for cross-entropy.
        """
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.proto_loss = PrototypeLossMulticlass(proto_margin)
        self.ood_loss = OODMarginLoss(ood_margin)
        self.lambda_proto = lambda_proto
        self.lambda_ood = lambda_ood

    def forward(self, logits, targets, features=None, prototypes=None, ood_features=None):
        """Compute composite multi-class training loss.

        Args:
            logits: Output classification logits of shape (B, C).
            targets: Ground-truth class index tensor of shape (B,).
            features: Latent feature vectors of shape (B, D).
            prototypes: Class prototype vectors of shape (C, D).
            ood_features: OOD feature vectors of shape (N_ood, D).

        Returns:
            Scalar combined loss tensor.
        """
        # --- Primary Cross-Entropy loss ---
        loss = self.ce(logits, targets)

        # --- Optional prototype regularization ---
        if features is not None and prototypes is not None:
            loss = loss + self.lambda_proto * self.proto_loss(features, prototypes, targets)

        # --- Optional OOD regularization ---
        if ood_features is not None and prototypes is not None:
            loss = loss + self.lambda_ood * self.ood_loss(ood_features, prototypes)
        return loss


