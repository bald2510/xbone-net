import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.datasets.analysis import (
    CTCHOODDataset,
    cross_class_derangement,
    same_class_derangement,
)
from src.datasets.fracatlas import (
    FRACATLAS_IMAGE_RESOLVER_VERSION,
    FracAtlasDataset,
)


class AnalysisDatasetTests(unittest.TestCase):
    def test_cross_class_derangement_has_no_fixed_or_same_class_pairs(self):
        labels = np.asarray([0, 0, 0, 1, 1, 2, 2, 3])
        recipients, donors = cross_class_derangement(labels, seed=9)
        self.assertEqual(len(recipients), len(labels))
        self.assertEqual(len(np.unique(donors)), len(labels))
        self.assertFalse(np.any(recipients == donors))
        self.assertFalse(np.any(labels[recipients] == labels[donors]))

    def test_impossible_cross_class_derangement_is_rejected(self):
        with self.assertRaises(ValueError):
            cross_class_derangement(np.asarray([0, 0, 0, 1]), seed=1)

    def test_same_class_derangement_excludes_singletons(self):
        labels = np.asarray([0, 0, 1, 2, 2, 2])
        recipients, donors = same_class_derangement(labels, seed=5)
        self.assertNotIn(2, recipients.tolist())
        self.assertFalse(np.any(recipients == donors))
        np.testing.assert_array_equal(labels[recipients], labels[donors])

    def test_ctch_ood_fails_closed_on_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            xray = root / "xray"
            clinical = root / "clinical"
            for path in (images, xray, clinical):
                path.mkdir()
            manifest = root / "ood.csv"
            pd.DataFrame(
                {
                    "image_id": ["complete.jpg", "missing.jpg"],
                    "patient": [1, 2],
                    "group": ["a", "b"],
                    "OOD": [1, 1],
                }
            ).to_csv(manifest, index=False)
            Image.new("RGB", (8, 8)).save(images / "complete.jpg")
            (xray / "complete.txt").write_text("xray", encoding="utf-8")
            (clinical / "complete.txt").write_text("clinical", encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                CTCHOODDataset(
                    str(images), str(xray), str(clinical), str(manifest)
                )
            exploratory = CTCHOODDataset(
                str(images),
                str(xray),
                str(clinical),
                str(manifest),
                allow_missing=True,
            )
            self.assertEqual(len(exploratory), 1)
            self.assertEqual(exploratory.coverage["missing_rows"], 1)

    def test_fracatlas_resolves_released_class_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            reports = root / "reports"
            (images / "Fractured").mkdir(parents=True)
            (images / "Non_fractured").mkdir(parents=True)
            reports.mkdir()
            Image.new("RGB", (8, 8), color="white").save(
                images / "Fractured" / "fractured.png"
            )
            Image.new("RGB", (8, 8), color="black").save(
                images / "Non_fractured" / "normal.png"
            )
            for stem in ("fractured", "normal"):
                (reports / f"{stem}.txt").write_text(stem, encoding="utf-8")

            split = root / "split.csv"
            labels = root / "labels.csv"
            pd.DataFrame(
                {
                    "image_id": ["fractured.png", "normal.png"],
                    "split": ["test", "test"],
                }
            ).to_csv(split, index=False)
            pd.DataFrame(
                {
                    "image_id": ["fractured.png", "normal.png"],
                    "fractured": [1, 0],
                }
            ).to_csv(labels, index=False)

            dataset = FracAtlasDataset(
                str(images),
                str(reports),
                str(split),
                str(labels),
                split="test",
                strict_files=True,
            )

            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.coverage["resolved_images"], 2)
            self.assertEqual(dataset.coverage["missing_images"], 0)
            self.assertEqual(dataset.coverage["missing_reports"], 0)
            self.assertEqual(
                dataset.coverage["resolver_version"],
                FRACATLAS_IMAGE_RESOLVER_VERSION,
            )
            self.assertEqual(dataset.coverage["image_layout"], "class_directory")
            for index in range(len(dataset)):
                image, report, label = dataset[index]
                self.assertIsInstance(image, Image.Image)
                self.assertIn(report, {"fractured", "normal"})
                expected_pixel = 255 if int(label) == 1 else 0
                self.assertEqual(image.getpixel((0, 0))[0], expected_pixel)

    def test_fracatlas_fails_closed_when_image_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            reports = root / "reports"
            images.mkdir()
            reports.mkdir()
            (reports / "missing.txt").write_text("report", encoding="utf-8")
            split = root / "split.csv"
            labels = root / "labels.csv"
            pd.DataFrame(
                {"image_id": ["missing.png"], "split": ["test"]}
            ).to_csv(split, index=False)
            pd.DataFrame(
                {"image_id": ["missing.png"], "fractured": [1]}
            ).to_csv(labels, index=False)

            with self.assertRaises(FileNotFoundError):
                FracAtlasDataset(
                    str(images),
                    str(reports),
                    str(split),
                    str(labels),
                    split="test",
                    strict_files=True,
                )

    def test_fracatlas_records_controlled_truncated_jpeg_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images" / "Non_fractured"
            reports = root / "reports"
            images.mkdir(parents=True)
            reports.mkdir()
            image_path = images / "truncated.jpg"
            pixels = np.random.default_rng(7).integers(
                0, 256, size=(64, 64, 3), dtype=np.uint8
            )
            Image.fromarray(pixels).save(image_path, quality=90)
            image_path.write_bytes(image_path.read_bytes()[:-20])
            (reports / "truncated.txt").write_text("report", encoding="utf-8")
            split = root / "split.csv"
            labels = root / "labels.csv"
            pd.DataFrame(
                {"image_id": ["truncated.jpg"], "split": ["test"]}
            ).to_csv(split, index=False)
            pd.DataFrame(
                {"image_id": ["truncated.jpg"], "fractured": [0]}
            ).to_csv(labels, index=False)

            with self.assertRaises(OSError):
                FracAtlasDataset(
                    str(root / "images"),
                    str(reports),
                    str(split),
                    str(labels),
                    split="test",
                    strict_files=True,
                )

            dataset = FracAtlasDataset(
                str(root / "images"),
                str(reports),
                str(split),
                str(labels),
                split="test",
                strict_files=True,
                allow_truncated_images=True,
            )
            self.assertEqual(dataset.coverage["truncated_image_count"], 1)
            self.assertEqual(dataset.coverage["decode_failure_count"], 0)
            image, _, _ = dataset[0]
            self.assertEqual(image.size, (64, 64))


if __name__ == "__main__":
    unittest.main()
