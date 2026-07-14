import math
import unittest

import torch

from src.models.fusion.cross_attention import (
    CrossAttentionFusion,
    reduce_attention_to_keys,
)


class CrossAttentionFusionTests(unittest.TestCase):
    def test_all_fusion_components_have_comparable_norms(self):
        torch.manual_seed(7)
        fusion = CrossAttentionFusion(
            img_dim=8,
            text_dim=8,
            embed_dim=8,
            num_heads=2,
            dropout=0.0,
        ).eval()
        image = torch.randn(3, 5, 8)
        text = torch.randn(3, 7, 8)
        image[:, 0] *= 0.01
        image[:, 1:] *= 100.0
        text[:, 0] *= 0.1
        text[:, 1:] *= 10.0

        _, info = fusion(image, text, return_attn=True)
        target_norm = math.sqrt(8)
        for name in (
            "img_global",
            "txt_global",
            "txt_context_for_image",
            "img_context_for_text",
        ):
            observed = info[name].norm(dim=-1)
            torch.testing.assert_close(
                observed,
                torch.full_like(observed, target_norm),
                rtol=2e-3,
                atol=2e-3,
            )

    def test_masked_text_tokens_cannot_change_fused_output(self):
        torch.manual_seed(11)
        fusion = CrossAttentionFusion(
            img_dim=8,
            text_dim=8,
            embed_dim=8,
            num_heads=2,
            dropout=0.0,
        ).eval()
        image = torch.randn(2, 4, 8)
        text_a = torch.randn(2, 6, 8)
        text_b = text_a.clone()
        text_b[:, -2:] = torch.randn_like(text_b[:, -2:]) * 1000.0
        text_mask = torch.tensor(
            [[False, False, False, True, True]] * 2,
            dtype=torch.bool,
        )

        output_a = fusion(image, text_a, txt_key_padding_mask=text_mask)
        output_b = fusion(image, text_b, txt_key_padding_mask=text_mask)
        torch.testing.assert_close(output_a, output_b, rtol=1e-5, atol=1e-5)

    def test_attention_reduction_preserves_keys_and_removes_padding(self):
        attention = torch.tensor(
            [[[[0.1, 0.2, 0.7]], [[0.3, 0.3, 0.4]]]],
            dtype=torch.float32,
        )
        mask = torch.tensor([[False, False, True]])

        reduced = reduce_attention_to_keys(attention, mask)

        self.assertEqual(tuple(reduced.shape), (1, 3))
        torch.testing.assert_close(reduced.sum(dim=-1), torch.ones(1))
        torch.testing.assert_close(reduced[:, 2], torch.zeros(1))
        self.assertGreater(float(reduced[0, 1]), float(reduced[0, 0]))

    def test_fusion_mlp_consumes_only_two_enhanced_branches(self):
        fusion = CrossAttentionFusion(
            img_dim=8,
            text_dim=8,
            embed_dim=8,
            num_heads=2,
            dropout=0.0,
        ).eval()
        captured = {}

        def capture_input(module, args):
            captured["shape"] = tuple(args[0].shape)

        handle = fusion.fusion_mlp[0].register_forward_pre_hook(capture_input)
        try:
            output = fusion(torch.randn(2, 4, 8), torch.randn(2, 6, 8))
        finally:
            handle.remove()

        self.assertEqual(captured["shape"], (2, 16))
        self.assertEqual(fusion.fusion_mlp[0].in_features, 16)
        self.assertEqual(tuple(output.shape), (2, 8))
        self.assertFalse(hasattr(fusion, "skip_projection"))


if __name__ == "__main__":
    unittest.main()
