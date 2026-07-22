import torch
import torch.nn as nn

from src.models.composer import XBoneMultiModalModel
from src.models.drl import DRLAuxiliaryBranch, drl_ood_score


class _DummyBackbone(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.image = nn.Linear(feature_dim, feature_dim)
        self.text = nn.Linear(feature_dim, feature_dim)
        self.last_image_key_padding_mask = None

    def forward(self, images, input_ids, **kwargs):
        image_tokens = self.image(images)
        text_tokens = self.text(input_ids)
        self.last_image_key_padding_mask = torch.zeros(
            image_tokens.shape[:2], dtype=torch.bool, device=image_tokens.device
        )
        return image_tokens, text_tokens


class _DummyFusion(nn.Module):
    supports_padding_mask = True

    def forward(self, image, text, **kwargs):
        return 0.5 * (image.mean(dim=1) + text.mean(dim=1))


def test_drl_auxiliary_shapes_and_weights():
    torch.manual_seed(1)
    branch = DRLAuxiliaryBranch(
        feature_dim=8,
        num_classes=3,
        hidden_dim=12,
        dropout=0.0,
        epsilon=1e-3,
    )
    image = torch.randn(4, 5, 8)
    text = torch.randn(4, 6, 8)
    label_representation = torch.randn(4, 8)
    mask = torch.tensor(
        [[False, False, False, False], [False, True, True, True]] * 2
    )

    logits, features, details = branch(
        image,
        text,
        label_representation,
        image_local_padding_mask=mask,
        return_details=True,
    )

    assert logits.shape == (4, 3)
    assert features.shape == (4, 8)
    assert details["components"].shape == (4, 4, 8)
    assert torch.allclose(
        details["component_weights"].sum(dim=1),
        torch.ones(4),
        atol=1e-6,
    )
    assert torch.isfinite(logits).all()


def test_drl_score_uses_larger_is_ood_convention():
    confident = drl_ood_score(
        torch.tensor([[8.0, -8.0]]),
        torch.tensor([[7.0, -7.0]]),
    )
    uncertain = drl_ood_score(
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[0.0, 0.0]]),
    )
    assert confident.item() < uncertain.item()


def test_forward_drl_does_not_change_primary_prediction_path():
    torch.manual_seed(2)
    dimension = 8
    model = XBoneMultiModalModel(
        _DummyBackbone(dimension),
        _DummyFusion(),
        nn.Linear(dimension, 3),
    )
    model.drl_auxiliary = DRLAuxiliaryBranch(
        feature_dim=dimension,
        num_classes=3,
        hidden_dim=12,
        dropout=0.0,
    )
    model.eval()
    images = torch.randn(2, 3, dimension)
    text = torch.randn(2, 4, dimension)

    primary = model(images, text)
    output = model.forward_drl(images, text)

    assert torch.allclose(primary, output["primary_logits"], atol=1e-6)
    assert output["distribution_features"].shape == (2, dimension)
    assert output["drl_ood_score"].shape == (2,)


def test_phase3_train_mode_keeps_primary_path_in_eval_mode():
    model = XBoneMultiModalModel(
        _DummyBackbone(8),
        _DummyFusion(),
        nn.Linear(8, 3),
    )
    model.drl_auxiliary = DRLAuxiliaryBranch(8, 3, hidden_dim=8)
    model.phase3_mode = True
    model.train()

    assert not model.backbone.training
    assert not model.fusion.training
    assert not model.head.training
    assert model.drl_auxiliary.training
