import unittest

import torch
import torch.nn as nn

from src.utils.efficiency import (
    batch_metadata,
    count_supported_flops,
    parameter_summary,
    summarize_measurements,
)


class EfficiencyUtilitiesTests(unittest.TestCase):
    def test_parameter_summary_includes_frozen_and_module_breakdown(self):
        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2, bias=False))
        model[0].weight.requires_grad = False

        summary = parameter_summary(model)

        self.assertEqual(summary["total"], 21)
        self.assertEqual(summary["trainable"], 9)
        self.assertEqual(summary["frozen"], 12)
        self.assertEqual(summary["by_module"]["0"]["total"], 15)

    def test_measurement_summary_uses_population_statistics(self):
        summary = summarize_measurements([1.0, 2.0, 3.0])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["mean"], 2.0)
        self.assertEqual(summary["p50"], 2.0)
        self.assertAlmostEqual(summary["p95"], 2.9)

    def test_batch_metadata_records_dynamic_tiles_and_text(self):
        batch = {
            "pixel_values": torch.zeros(2, 3, 224, 224),
            "tile_values": torch.zeros(2, 4, 3, 224, 224),
            "tile_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]),
        }
        attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

        metadata = batch_metadata(batch, attention_mask)

        self.assertEqual(metadata["valid_tiles_per_sample"], [2, 3])
        self.assertEqual(metadata["valid_text_tokens_per_sample"], [2, 3])
        self.assertEqual(metadata["global_image_shape"], [3, 224, 224])

    def test_flop_counter_counts_linear_matrix_multiplication(self):
        layer = nn.Linear(4, 3, bias=False)
        inputs = torch.zeros(2, 4)

        flops = count_supported_flops(lambda: layer(inputs), layer)

        # PyTorch counts a multiply-add as two FLOPs: 2 * B * in * out.
        self.assertEqual(flops, 48)


if __name__ == "__main__":
    unittest.main()
