import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from inference import (
    load_model_checkpoint,
    project_visual_attention_to_source,
    rasterize_spatial_attention,
)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2)


class InferenceCheckpointTests(unittest.TestCase):
    def test_missing_explicit_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            load_model_checkpoint(
                _TinyModel(),
                {"params": {}},
                torch.device("cpu"),
                custom_checkpoint_path="missing_phase2_checkpoint.pth",
            )

    def test_inference_without_any_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, "random weights is disabled"):
            load_model_checkpoint(
                _TinyModel(), {"params": {}}, torch.device("cpu")
            )

    def test_valid_phase2_checkpoint_is_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_phase2.pth"
            source = _TinyModel()
            with torch.no_grad():
                source.projection.weight.fill_(3.0)
            torch.save(source.state_dict(), checkpoint)

            target = _TinyModel()
            loaded = load_model_checkpoint(
                target,
                {"params": {"model_dir": directory}},
                torch.device("cpu"),
            )
            self.assertEqual(Path(loaded), checkpoint)
            self.assertTrue(
                torch.equal(target.projection.weight, source.projection.weight)
            )


class InferenceAttentionProjectionTests(unittest.TestCase):
    def test_visual_attention_projects_through_resampler_and_respects_mask(self):
        resampler_attention = torch.tensor(
            [[[0.8, 0.2, 0.0], [0.1, 0.3, 0.6]]]
        )
        local_boxes = torch.tensor(
            [[[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 1.0, 1.0], [0, 0, 0, 0]]]
        )
        backbone = SimpleNamespace(
            visual_resampler=SimpleNamespace(last_attention=resampler_attention),
            last_local_token_boxes=local_boxes,
            last_local_mask=torch.tensor([[True, True, False]]),
        )
        model = SimpleNamespace(backbone=backbone)

        boxes, scores = project_visual_attention_to_source(
            model, torch.tensor([[0.75, 0.25]])
        )
        self.assertIs(boxes, local_boxes)
        self.assertTrue(
            torch.allclose(scores, torch.tensor([[0.625, 0.225, 0.0]]))
        )

    def test_rasterization_preserves_original_aspect_ratio_and_coordinates(self):
        heatmap = rasterize_spatial_attention(
            boxes=np.array(
                [[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 1.0, 1.0]],
                dtype=np.float32,
            ),
            scores=np.array([1.0, 0.2], dtype=np.float32),
            image_size=(200, 100),
        )
        self.assertEqual(heatmap.shape, (100, 200))
        self.assertGreater(float(heatmap[:, :100].mean()), float(heatmap[:, 100:].mean()))


if __name__ == "__main__":
    unittest.main()
