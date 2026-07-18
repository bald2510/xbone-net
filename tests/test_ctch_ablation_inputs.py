import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from src.datasets.ctch import CTCHDataset


class CTCHAblationInputTests(unittest.TestCase):
    def _make_dataset(self, root: Path, split: str, **kwargs) -> CTCHDataset:
        image_ids = [f"case_{index}.png" for index in range(3)]
        pd.DataFrame(
            {"image_id": image_ids, "split": [split] * len(image_ids)}
        ).to_csv(root / "split.csv", index=False)
        pd.DataFrame(
            {"image_id": image_ids, "class_id": [0, 1, 0]}
        ).to_csv(root / "labels.csv", index=False)

        xray_dir = root / "xray"
        clinical_dir = root / "clinical"
        xray_dir.mkdir()
        clinical_dir.mkdir()
        for index, image_id in enumerate(image_ids):
            stem = Path(image_id).stem
            (xray_dir / f"{stem}.txt").write_text(
                f"xray report {index}", encoding="utf-8"
            )
            (clinical_dir / f"{stem}.txt").write_text(
                f"clinical report {index}", encoding="utf-8"
            )

        return CTCHDataset(
            img_dir=str(root / "images"),
            csv_split_path=str(root / "split.csv"),
            csv_labels_path=str(root / "labels.csv"),
            classes=["normal", "fracture"],
            num_classes=2,
            split=split,
            xray_report_dir=str(xray_dir),
            clinical_report_dir=str(clinical_dir),
            seed=0,
            **kwargs,
        )

    def test_report_shuffle_can_be_restricted_to_test_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = self._make_dataset(
                root,
                "test",
                shuffle_report_splits=["test"],
            )

            self.assertTrue(dataset.shuffle_reports)
            _, xray_text, clinical_text, _ = dataset[0]
            shuffled_index = int(dataset.shuffled_report_indices[0])
            self.assertEqual(xray_text, f"xray report {shuffled_index}")
            self.assertEqual(clinical_text, f"clinical report {shuffled_index}")

    def test_text_only_replaces_image_before_preprocessing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = self._make_dataset(
                Path(temp_dir),
                "train",
                text_only=True,
            )

            image, _, _, _ = dataset[0]
            self.assertEqual(image.size, (224, 224))
            self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))

    def test_letterbox_preprocessing_preserves_aspect_ratio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir = root / "images"
            images_dir.mkdir()
            for index in range(3):
                Image.new("RGB", (80, 40), color="white").save(
                    images_dir / f"case_{index}.png"
                )

            dataset = self._make_dataset(
                root,
                "train",
                preprocess={
                    "strategy": "letterbox",
                    "target_size": 224,
                    "pad_value": "black",
                },
            )

            image, _, _, _ = dataset[0]
            self.assertEqual(image.size, (224, 224))
            self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))
            self.assertEqual(image.getpixel((112, 112)), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
