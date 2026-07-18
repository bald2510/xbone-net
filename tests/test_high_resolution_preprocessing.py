import unittest

import torch
from PIL import Image, ImageDraw

from src.datasets.high_resolution import (
    build_sparse_focal_views,
    find_foreground_box,
    prepare_high_resolution_inputs,
)


class SparseFocalPreprocessingTests(unittest.TestCase):
    @staticmethod
    def _synthetic_xray() -> Image.Image:
        image = Image.new("L", (1200, 600), color=8)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((120, 70, 1080, 530), radius=80, fill=90)
        draw.line((180, 420, 1000, 160), fill=235, width=34)
        draw.ellipse((720, 180, 900, 360), outline=250, width=18)
        return image.convert("RGB")

    @staticmethod
    def _config() -> dict:
        return {
            "strategy": "sparse_focal",
            "global_size": 224,
            "tile_size": 224,
            "canonical_long_side": 896,
            "allow_upsample": False,
            "candidate_stride": 168,
            "num_tiles": 4,
            "coverage_tiles": 2,
            "min_foreground_ratio": 0.60,
            "nms_iou": 0.30,
            "foreground": {"margin": 0.04},
        }

    def test_foreground_box_removes_scanner_border_conservatively(self):
        box = find_foreground_box(self._synthetic_xray(), margin=0.04)

        self.assertGreater(box[0], 0)
        self.assertGreater(box[1], 0)
        self.assertLess(box[2], 1200)
        self.assertLess(box[3], 600)
        self.assertLessEqual(box[0], 120)
        self.assertLessEqual(box[1], 70)
        self.assertGreaterEqual(box[2], 1080)
        self.assertGreaterEqual(box[3], 530)

    def test_foreground_box_falls_back_when_crop_discards_contrast_energy(self):
        image = Image.new("L", (400, 400), color=0)
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 20, 280, 280), fill=255)
        draw.rectangle((80, 300, 380, 390), fill=40)

        box = find_foreground_box(
            image.convert("RGB"),
            min_contrast=0.20,
            min_energy_retained=0.95,
        )

        self.assertEqual(box, (0, 0, 400, 400))

    def test_sparse_focal_views_are_fixed_budget_and_deterministic(self):
        image = self._synthetic_xray()
        first = build_sparse_focal_views(image, self._config())
        second = build_sparse_focal_views(image, self._config())

        self.assertEqual(first.global_image.size, (224, 224))
        self.assertEqual(len(first.tiles), 4)
        self.assertTrue(all(tile.size == (224, 224) for tile in first.tiles))
        self.assertEqual(first.tile_boxes, second.tile_boxes)
        self.assertEqual(first.tile_roles, second.tile_roles)
        self.assertEqual(first.tile_roles.count("coverage"), 2)
        for left, top, right, bottom in first.tile_boxes:
            self.assertGreaterEqual(left, 0)
            self.assertGreaterEqual(top, 0)
            self.assertLessEqual(right, image.width)
            self.assertLessEqual(bottom, image.height)
            self.assertGreater(right, left)
            self.assertGreater(bottom, top)

    def test_model_input_contract_remains_224_square(self):
        def to_tensor(image: Image.Image) -> torch.Tensor:
            array = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            return array.reshape(image.height, image.width, 3).permute(2, 0, 1).float()

        fields = prepare_high_resolution_inputs(
            self._synthetic_xray(),
            to_tensor,
            self._config(),
        )

        self.assertEqual(tuple(fields["pixel_values"].shape), (3, 224, 224))
        self.assertEqual(tuple(fields["tile_values"].shape), (4, 3, 224, 224))
        self.assertEqual(tuple(fields["tile_boxes"].shape), (4, 4))
        self.assertTrue(torch.all(fields["tile_boxes"] >= 0))
        self.assertTrue(torch.all(fields["tile_boxes"] <= 1))

    def test_cached_selection_reproduces_the_same_views(self):
        def to_tensor(image: Image.Image) -> torch.Tensor:
            array = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            return array.reshape(image.height, image.width, 3).permute(2, 0, 1)

        image = self._synthetic_xray()
        first, selection = prepare_high_resolution_inputs(
            image,
            to_tensor,
            self._config(),
            return_selection=True,
        )
        cached = prepare_high_resolution_inputs(
            image,
            to_tensor,
            self._config(),
            selection=selection,
        )

        torch.testing.assert_close(first["pixel_values"], cached["pixel_values"])
        torch.testing.assert_close(first["tile_values"], cached["tile_values"])
        torch.testing.assert_close(first["tile_boxes"], cached["tile_boxes"])

    def test_legacy_uniform_grid_strategy_is_rejected(self):
        config = self._config()
        config["strategy"] = "uniform_grid"
        with self.assertRaisesRegex(ValueError, "legacy uniform grid has been removed"):
            build_sparse_focal_views(self._synthetic_xray(), config)


if __name__ == "__main__":
    unittest.main()
