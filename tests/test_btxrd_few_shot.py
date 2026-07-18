import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.datasets.btxrd import BTXRDDataset


class BTXRDFewShotTests(unittest.TestCase):
    def _build_dataset(self, root: Path, *, k_shot=1, seed=42):
        classes = ["normal", "fracture", "tumor"]
        rows = []
        labels = []
        for class_id, class_name in enumerate(classes):
            for sample_id in range(3):
                image_id = f"class_{class_id}_{sample_id}.png"
                rows.append({"image_id": image_id, "split": "train"})
                label = {"image_id": image_id, **{name: 0 for name in classes}}
                label[class_name] = 1
                labels.append(label)

        split_path = root / "split.csv"
        label_path = root / "labels.csv"
        pd.DataFrame(rows).to_csv(split_path, index=False)
        pd.DataFrame(labels).to_csv(label_path, index=False)

        return BTXRDDataset(
            img_dir=str(root / "images"),
            report_dir=str(root / "reports"),
            csv_split_path=str(split_path),
            csv_labels_path=str(label_path),
            classes=classes,
            task_type="multiclass",
            split="train",
            k_shot=k_shot,
            seed=seed,
        )

    def test_k_shot_preserves_class_id_with_current_pandas(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._build_dataset(Path(directory), k_shot=1)
            self.assertIn("class_id", dataset.df.columns)
            self.assertEqual(len(dataset), 3)
            self.assertEqual(
                dataset.df["class_id"].value_counts().sort_index().to_dict(),
                {0: 1, 1: 1, 2: 1},
            )
            for _, row in dataset.df.iterrows():
                self.assertEqual(int(dataset._label(row)), int(row["class_id"]))

    def test_k_shot_sampling_is_deterministic_for_a_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._build_dataset(root, k_shot=2, seed=123)
            second = self._build_dataset(root, k_shot=2, seed=123)
            self.assertEqual(first.df["image_id"].tolist(), second.df["image_id"].tolist())

    def test_invalid_k_shot_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "positive integer"):
                self._build_dataset(Path(directory), k_shot=0)

    def test_specific_osteochondroma_subtype_overrides_parent_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            classes = ["osteochondroma", "synovial osteochondroma"]
            pd.DataFrame(
                [{"image_id": "ambiguous.png", "split": "train"}]
            ).to_csv(root / "split.csv", index=False)
            pd.DataFrame(
                [{
                    "image_id": "ambiguous.png",
                    "osteochondroma": 1,
                    "synovial osteochondroma": 1,
                }]
            ).to_csv(root / "labels.csv", index=False)

            dataset = BTXRDDataset(
                img_dir=str(root / "images"),
                report_dir=str(root / "reports"),
                csv_split_path=str(root / "split.csv"),
                csv_labels_path=str(root / "labels.csv"),
                classes=classes,
                task_type="multiclass",
                split="train",
            )
            self.assertEqual(int(dataset.df.iloc[0]["class_id"]), 1)


if __name__ == "__main__":
    unittest.main()
