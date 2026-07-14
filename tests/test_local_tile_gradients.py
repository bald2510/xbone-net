import unittest

import torch

from src.models.backbone.biomedclip import BiomedCLIPFoundation


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
    backbone.tile_encode_chunk_size = 1
    backbone.local_tile_gradient_checkpointing = use_checkpointing
    return backbone


class LocalTileGradientTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
