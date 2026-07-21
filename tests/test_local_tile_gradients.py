import unittest

import torch

from src.models.backbone.biomedclip import (
    BiomedCLIPFoundation,
    GlobalGuidedAttentionPool,
    SpatialTokenResampler,
)


class _DummyTileTrunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter_scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward_features(self, images):
        patches = images.flatten(2).transpose(1, 2) * self.adapter_scale
        cls_token = torch.zeros(
            images.size(0),
            1,
            images.size(1),
            dtype=images.dtype,
            device=images.device,
        )
        return torch.cat([cls_token, patches], dim=1)


class _DummyVisual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = _DummyTileTrunk()
        self.head = torch.nn.Identity()


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _DummyVisual()


def _make_backbone(use_checkpointing=True):
    backbone = BiomedCLIPFoundation.__new__(BiomedCLIPFoundation)
    torch.nn.Module.__init__(backbone)
    backbone.model = _DummyModel()
    backbone.local_pool_grid = 2
    backbone.include_local_cls_token = False
    backbone.tile_encode_chunk_size = 1
    backbone.local_tile_gradient_checkpointing = use_checkpointing
    return backbone


class LocalTileGradientTests(unittest.TestCase):
    def test_attention_pool_starts_as_masked_mean(self):
        pooler = GlobalGuidedAttentionPool(dim=3, hidden_dim=2)
        global_feature = torch.tensor([[1.0, 0.0, 0.0]])
        local_features = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [9.0, 9.0, 9.0]]]
        )
        valid = torch.tensor([[True, True, False]])

        output = pooler(global_feature, local_features, valid)

        torch.testing.assert_close(
            output, local_features[:, :2].mean(dim=1)
        )
        torch.testing.assert_close(
            pooler.last_attention, torch.tensor([[0.5, 0.5, 0.0]])
        )

    def test_pooling_can_preserve_cls_with_one_patch_summary(self):
        backbone = _make_backbone(use_checkpointing=False)
        backbone.local_pool_grid = 1
        backbone.include_local_cls_token = True
        projected = torch.arange(5 * 3, dtype=torch.float32).reshape(1, 5, 3)

        output = backbone._pool_patch_tokens(projected)

        self.assertEqual(tuple(output.shape), (1, 2, 3))
        torch.testing.assert_close(output[:, 0], projected[:, 0])
        torch.testing.assert_close(
            output[:, 1], projected[:, 1:].mean(dim=1)
        )

    def test_contrastive_pooling_keeps_global_as_anchor(self):
        backbone = _make_backbone(use_checkpointing=False)
        backbone.contrastive_local_weight = 0.0
        backbone.last_image_key_padding_mask = torch.tensor(
            [[False, False, True]]
        )
        image_features = torch.randn(1, 3, 3)

        pooled = backbone.pool_contrastive_image_features(image_features)

        torch.testing.assert_close(
            pooled,
            torch.nn.functional.normalize(image_features[:, 0], dim=-1),
        )

    def test_checkpointed_tile_path_updates_trainable_visual_parameter(self):
        backbone = _make_backbone(use_checkpointing=True)
        tiles = torch.ones(2, 1, 2, 2)

        output = backbone._encode_valid_local_tiles(tiles, track_gradients=True)
        output.sum().backward()

        gradient = backbone.model.visual.trunk.adapter_scale.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs()), 0.0)

    def test_frozen_tile_path_does_not_build_autograd_graph(self):
        backbone = _make_backbone(use_checkpointing=True)
        tiles = torch.ones(2, 1, 2, 2)

        output = backbone._encode_valid_local_tiles(tiles, track_gradients=False)

        self.assertFalse(output.requires_grad)
        self.assertIsNone(backbone.model.visual.trunk.adapter_scale.grad)

    def test_single_view_ablation_matches_nine_token_budget(self):
        backbone = _make_backbone(use_checkpointing=False)
        backbone.single_view_pool_shape = (2, 4)
        backbone.num_visual_tokens = 8
        backbone.explain_mode = False
        backbone.visual_resampler = SpatialTokenResampler(
            dim=3,
            num_tokens=8,
            num_heads=1,
            depth=1,
            dropout=0.0,
            use_spatial_coordinates=False,
            aggregation="passthrough",
        )

        output = backbone._encode_single_view_image(
            torch.arange(3 * 14 * 14, dtype=torch.float32).reshape(1, 3, 14, 14)
        )

        self.assertEqual(tuple(output.shape), (1, 9, 3))
        self.assertEqual(tuple(backbone.last_image_key_padding_mask.shape), (1, 9))
        self.assertFalse(bool(backbone.last_image_key_padding_mask.any()))


if __name__ == "__main__":
    unittest.main()
