import argparse
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from src.datasets.high_resolution import build_sparse_focal_views
from tools.visualize_preprocessing import (
    DEFAULT_EXPERIMENT_CONFIG,
    apply_config_overrides,
    compute_annotation_coverage,
    load_high_resolution_config,
    render_preprocessing,
)


class PreprocessingVisualizationTests(unittest.TestCase):
    @staticmethod
    def _synthetic_xray() -> Image.Image:
        image = Image.new("L", (600, 320), color=8)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((45, 30, 555, 290), radius=45, fill=100)
        draw.line((90, 245, 510, 75), fill=235, width=20)
        draw.ellipse((365, 95, 475, 205), outline=250, width=12)
        return image.convert("RGB")

    @staticmethod
    def _five_tile_config() -> dict:
        return {
            "strategy": "sparse_focal",
            "global_size": 96,
            "tile_size": 96,
            "canonical_long_side": 384,
            "allow_upsample": False,
            "candidate_stride": 72,
            "num_tiles": 5,
            "coverage_tiles": 2,
            "min_foreground_ratio": 0.60,
            "nms_iou": 0.30,
            "foreground": {"margin": 0.04},
        }

    def test_loads_canonical_config_and_applies_only_explicit_overrides(self):
        original = load_high_resolution_config(DEFAULT_EXPERIMENT_CONFIG)
        args = argparse.Namespace(num_tiles=5, foreground_margin=0.08)

        resolved = apply_config_overrides(original, args)

        self.assertEqual(original["num_tiles"], 4)
        self.assertEqual(original["foreground"]["margin"], 0.04)
        self.assertEqual(resolved["num_tiles"], 5)
        self.assertEqual(resolved["foreground"]["margin"], 0.08)
        self.assertEqual(resolved["candidate_stride"], original["candidate_stride"])

    def test_annotation_coverage_reports_partial_hit_without_affecting_selection(self):
        annotations = [
            {
                "label": "lesion",
                "shape_type": "rectangle",
                "points": [[20, 20], [39, 39]],
            }
        ]

        metrics = compute_annotation_coverage(
            annotations,
            image_size=(100, 100),
            foreground_box=(0, 0, 100, 100),
            tile_boxes=((20, 20, 30, 40), (60, 60, 80, 80)),
        )

        self.assertTrue(metrics["available"])
        self.assertEqual(metrics["foreground_coverage_ratio"], 1.0)
        self.assertAlmostEqual(metrics["local_union_coverage_ratio"], 0.5)
        self.assertEqual(metrics["tile_hit_indices"], [1])
        self.assertAlmostEqual(metrics["per_tile_coverage_ratios"][0], 0.5)
        self.assertEqual(metrics["per_tile_coverage_ratios"][1], 0.0)

    def test_dynamic_layout_renders_more_than_four_tiles(self):
        source = self._synthetic_xray()
        views = build_sparse_focal_views(source, self._five_tile_config())
        self.assertEqual(len(views.tiles), 5)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "five_tiles.png"
            render_preprocessing(source, views, [], output, dpi=72)

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
