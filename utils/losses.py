import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip.loss import ClipLoss, SigLipLoss


def get_clip_logit_params(clip_model, require_bias=False):
    """
    Return BioMedCLIP/OpenCLIP temperature parameters (logit_scale is stored in log-space).
    Optionally register logit_bias for SigLIP-style losses when the checkpoint has none.
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
    """L2-normalized cosine logits with the model's learned temperature."""
    image_features = F.normalize(image_features, dim=-1, p=2)
    text_features = F.normalize(text_features, dim=-1, p=2)
    logits = logit_scale.exp() * image_features @ text_features.T
    if logit_bias is not None:
        logits = logits + logit_bias
    return logits


# =========================================================================
# Phase 1 Contrastive Losses
# =========================================================================

class SoftTargetSemanticMatchingLoss(nn.Module):
    """
    Semantic Matching Loss that creates a soft target matrix based on sharing classes.
    Uses BioMedCLIP's own logit_scale so logits are not stuck at the random baseline ln(batch_size).
    
    GPU-accelerated and fully vectorized (compatible with 1D and 2D label inputs).
    """

    def __init__(self, clip_model):
        super().__init__()
        self.logit_scale, self.logit_bias = get_clip_logit_params(clip_model)

    def forward(self, image_features, text_features, disease_labels):
        logits_per_image = _pairwise_logits(image_features, text_features, self.logit_scale, self.logit_bias)
        logits_per_text = logits_per_image.T

        batch_size = image_features.shape[0]
        
        # Fully vectorized target matrix generation
        if disease_labels.ndim == 1:
            eq_mask = (disease_labels.unsqueeze(1) == disease_labels.unsqueeze(0))
        else:
            eq_mask = (disease_labels.unsqueeze(1) == disease_labels.unsqueeze(0)).all(dim=-1)

        targets = torch.zeros((batch_size, batch_size), device=image_features.device)
        targets[eq_mask] = 0.95
        targets[torch.arange(batch_size), torch.arange(batch_size)] = 1.0

        row_sums = targets.sum(dim=1, keepdim=True).clamp(min=1e-8)
        col_sums = targets.sum(dim=0, keepdim=True).clamp(min=1e-8)
        y_v2t = targets / row_sums
        y_t2v = targets / col_sums

        log_preds_v2t = F.log_softmax(logits_per_image, dim=1)
        log_preds_t2v = F.log_softmax(logits_per_text, dim=1)

        loss_i = -(y_v2t * log_preds_v2t).sum(dim=1).mean()
        loss_t = -(y_t2v * log_preds_t2v).sum(dim=1).mean()

        return (loss_i + loss_t) / 2


class InfoNCELoss(nn.Module):
    """
    Standard symmetric CLIP contrastive loss (OpenCLIP ClipLoss).
    Reuses the pretrained logit_scale from BioMedCLIP instead of re-initializing it.
    """

    def __init__(self, clip_model):
        super().__init__()
        self.logit_scale, _ = get_clip_logit_params(clip_model)
        self.clip_loss = ClipLoss()

    def forward(self, image_features, text_features, disease_labels=None):
        image_features = F.normalize(image_features, dim=-1, p=2)
        text_features = F.normalize(text_features, dim=-1, p=2)
        return self.clip_loss(
            image_features,
            text_features,
            self.logit_scale.exp(),
        )


class SigCLIPLoss(nn.Module):
    """
    Sigmoid contrastive loss (OpenCLIP SigLipLoss).
    Batch-size independent — does not exhibit the ln(batch_size) softmax baseline.
    """

    def __init__(self, clip_model):
        super().__init__()
        self.logit_scale, self.logit_bias = get_clip_logit_params(clip_model, require_bias=True)
        self.siglip_loss = SigLipLoss()

    def forward(self, image_features, text_features, disease_labels=None):
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


def build_loss(loss_type, clip_model=None, **kwargs):
    """
    Factory function to instantiate Phase 1 loss modules.
    """
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"Loss type '{loss_type}' not supported. Choose from {list(LOSS_REGISTRY.keys())}")
    if clip_model is None:
        raise ValueError(
            "build_loss requires `clip_model` (OpenCLIP module). "
            "Pass model.backbone.model from XBoneMultiModalModel."
        )
    kwargs.pop("temperature", None)
    return LOSS_REGISTRY[loss_type](clip_model=clip_model, **kwargs)


# =========================================================================
# Phase 2 Classification Losses
# =========================================================================

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    Uses different focusing parameters for positive (γ+) and negative (γ-) samples,
    and probability shifting (margin) for hard negative suppression.
    """

    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        eps = 1e-8
        pos_loss = targets * (1 - p).pow(self.gamma_pos) * torch.log(p.clamp(min=eps))
        p_m = (p - self.clip).clamp(min=0)
        neg_loss = (1 - targets) * p_m.pow(self.gamma_neg) * torch.log((1 - p_m).clamp(min=eps))
        loss = -(pos_loss + neg_loss)
        return loss.mean()


class PrototypeLoss(nn.Module):
    """
    Prototype-based contrastive loss for multi-label classification.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        features: torch.Tensor,
        prototypes: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        normed_features = F.normalize(features, dim=-1)
        normed_prototypes = F.normalize(prototypes, dim=-1)
        cosine_sim = normed_features @ normed_prototypes.T
        dist = 1.0 - cosine_sim
        pos_loss = targets * dist
        neg_loss = (1 - targets) * F.relu(self.margin - dist)
        loss = pos_loss + neg_loss
        return loss.mean()


class OODMarginLoss(nn.Module):
    """
    OOD Margin Loss for out-of-distribution detection.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        features: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
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
    """
    Combined Phase 2 loss:  L = L_ASL + λ1 * L_proto + λ2 * L_OOD
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
        loss = self.asl(logits, targets)
        if features is not None and prototypes is not None:
            loss = loss + self.lambda_proto * self.proto_loss(features, prototypes, targets)
        if ood_features is not None and prototypes is not None:
            loss = loss + self.lambda_ood * self.ood_loss(ood_features, prototypes)
        return loss


class PrototypeLossMulticlass(nn.Module):
    """Prototype loss for multi-class: pull toward correct prototype only."""
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, features, prototypes, targets):
        normed_f = F.normalize(features, dim=-1)
        normed_p = F.normalize(prototypes, dim=-1)
        cosine_sim = normed_f @ normed_p.T
        dist = 1.0 - cosine_sim
        pos_dist = dist[torch.arange(len(targets), device=targets.device), targets]
        pull_loss = pos_dist.mean()
        mask = torch.ones_like(dist, dtype=torch.bool)
        mask[torch.arange(len(targets), device=targets.device), targets] = False
        neg_dist = dist[mask].view(len(targets), -1)
        push_loss = F.relu(self.margin - neg_dist).mean()
        return pull_loss + push_loss


class CombinedPhase2LossMulticlass(nn.Module):
    """CE + Prototype loss for multi-class classification."""
    def __init__(self, class_weights=None, proto_margin=0.5,
                 ood_margin=1.0, lambda_proto=0.5, lambda_ood=0.0,
                 label_smoothing=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.proto_loss = PrototypeLossMulticlass(proto_margin)
        self.ood_loss = OODMarginLoss(ood_margin)
        self.lambda_proto = lambda_proto
        self.lambda_ood = lambda_ood

    def forward(self, logits, targets, features=None, prototypes=None, ood_features=None):
        loss = self.ce(logits, targets)
        if features is not None and prototypes is not None:
            loss = loss + self.lambda_proto * self.proto_loss(features, prototypes, targets)
        if ood_features is not None and prototypes is not None:
            loss = loss + self.lambda_ood * self.ood_loss(ood_features, prototypes)
        return loss
