import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from src.models.fusion import build_fusion_module
from src.models.fusion.gated_cross_attention import GatedCrossAttentionFusion
from src.utils.explainability import fusion_from_tokens


class GatedCrossAttentionFusionTests(unittest.TestCase):
    def _build(self) -> GatedCrossAttentionFusion:
        return GatedCrossAttentionFusion(
            img_dim=8,
            text_dim=8,
            embed_dim=8,
            num_heads=2,
            dropout=0.0,
            gate_hidden_dim=4,
            gate_bias_init=-2.0,
        ).eval()

    def test_registry_builds_gated_fusion(self):
        fusion = build_fusion_module(
            {
                "type": "gated_cross_attention",
                "params": {
                    "img_dim": 8,
                    "text_dim": 8,
                    "embed_dim": 8,
                    "num_heads": 2,
                },
            }
        )
        self.assertIsInstance(fusion, GatedCrossAttentionFusion)

    def test_initial_gate_starts_near_concat(self):
        torch.manual_seed(7)
        fusion = self._build()
        output, info = fusion(
            torch.randn(3, 5, 8),
            torch.randn(3, 7, 8),
            return_attn=True,
        )
        expected_gate = torch.full((3, 1), torch.sigmoid(torch.tensor(-2.0)))
        torch.testing.assert_close(info["fusion_gate"], expected_gate)
        expected_output = info["concat_branch"] + expected_gate * (
            info["cross_attention_branch"] - info["concat_branch"]
        )
        torch.testing.assert_close(output, expected_output)

    def test_masked_image_tokens_do_not_change_output(self):
        torch.manual_seed(11)
        fusion = self._build()
        image_a = torch.randn(2, 5, 8)
        image_b = image_a.clone()
        image_b[:, -2:] = torch.randn_like(image_b[:, -2:]) * 1000.0
        text = torch.randn(2, 7, 8)
        image_mask = torch.tensor(
            [[False, False, True, True]] * 2,
            dtype=torch.bool,
        )
        output_a = fusion(image_a, text, img_key_padding_mask=image_mask)
        output_b = fusion(image_b, text, img_key_padding_mask=image_mask)
        torch.testing.assert_close(output_a, output_b, rtol=1e-5, atol=1e-5)

    def test_attention_metadata_exposes_gate_and_both_paths(self):
        fusion = self._build()
        output, info = fusion(
            torch.randn(2, 4, 8),
            torch.randn(2, 6, 8),
            return_attn=True,
        )
        self.assertEqual(tuple(output.shape), (2, 8))
        self.assertEqual(tuple(info["fusion_gate"].shape), (2, 1))
        self.assertEqual(tuple(info["concat_branch"].shape), (2, 8))
        self.assertEqual(tuple(info["cross_attention_branch"].shape), (2, 8))
        self.assertTrue(torch.all(info["fusion_gate"] > 0))
        self.assertTrue(torch.all(info["fusion_gate"] < 1))

    def test_gate_bias_validation_target(self):
        fusion = self._build()
        observed = float(
            torch.sigmoid(fusion.gate_network[-1].bias.detach())[0]
        )
        self.assertAlmostEqual(observed, 1.0 / (1.0 + math.exp(2.0)), places=6)

    def test_explainability_path_uses_exact_gated_fusion(self):
        torch.manual_seed(23)
        fusion = self._build()
        image = torch.randn(2, 5, 8)
        text = torch.randn(2, 7, 8)
        image_full_mask = torch.tensor(
            [[False, False, False, True, True]] * 2,
            dtype=torch.bool,
        )
        text_attention_mask = torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0]] * 2,
            dtype=torch.long,
        )
        model = SimpleNamespace(
            fusion=fusion,
            head=nn.Identity(),
            backbone=SimpleNamespace(
                last_image_key_padding_mask=image_full_mask
            ),
        )
        expected = fusion(
            image,
            text,
            img_key_padding_mask=image_full_mask[:, 1:],
            txt_key_padding_mask=text_attention_mask[:, 1:] == 0,
        )
        observed, _ = fusion_from_tokens(
            model,
            image,
            text,
            text_attention_mask,
        )
        torch.testing.assert_close(observed, expected)


if __name__ == "__main__":
    unittest.main()
