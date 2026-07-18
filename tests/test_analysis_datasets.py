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


if __name__ == "__main__":
    unittest.main()

