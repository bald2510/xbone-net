import unittest

import torch

from src.models.backbone.biomedclip import SpatialTokenResampler


class SpatialTokenResamplerTests(unittest.TestCase):
    def test_mean_pool_ignores_masked_tokens(self):
        module = SpatialTokenResampler(
            dim=8,
            num_tokens=2,
            num_heads=2,
            depth=1,
            dropout=0.0,
            use_spatial_coordinates=False,
            aggregation="mean_pool",
        )
        global_feature = torch.zeros(1, 8)
        local_tokens = torch.stack(
            [torch.ones(8), torch.full((8,), 3.0), torch.full((8,), 100.0)]
        ).unsqueeze(0)
        local_mask = torch.tensor([[True, True, False]])
        local_boxes = torch.zeros(1, 3, 4)

        output = module(global_feature, local_tokens, local_mask, local_boxes)

        self.assertEqual(tuple(output.shape), (1, 2, 8))
        torch.testing.assert_close(output[:, 1], torch.full((1, 8), 2.0))
        torch.testing.assert_close(
            module.last_attention,
            torch.tensor([[[0.5, 0.5, 0.0]]]),
        )

    def test_coordinate_projection_is_inactive_when_disabled(self):
        module = SpatialTokenResampler(
            dim=8,
            num_tokens=2,
            num_heads=2,
            depth=1,
            dropout=0.0,
            use_spatial_coordinates=False,
        )
        output = module(
            torch.randn(1, 8),
            torch.randn(1, 3, 8),
            torch.ones(1, 3, dtype=torch.bool),
            torch.rand(1, 3, 4),
        )
        output.sum().backward()

        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in module.spatial_projection.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
