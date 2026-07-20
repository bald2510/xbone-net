"""Regression tests for OOD-only CTCH artifact repair."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "data" / "CTCH" / "preprocess_ctch.py"
SPEC = importlib.util.spec_from_file_location("preprocess_ctch", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not load {MODULE_PATH}")
PREPROCESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPROCESS)


class FakeTranslator:
    def translate(self, text: str) -> str:
        return f"EN::{text}"


class CTCHOODPreprocessTests(unittest.TestCase):
    def _fixture(self, root: Path, *, id_image_id: str = "id.jpg"):
        paths = PREPROCESS.output_paths(root)
        pd.DataFrame(
            {
                "image_id": [id_image_id],
                "split": ["train"],
                "patient_key": ["patient-id"],
            }
        ).to_csv(paths["splits"], index=False)
        pd.DataFrame(
            {"image_id": [id_image_id], "class_id": [0]}
        ).to_csv(paths["labels"], index=False)
        pd.DataFrame({"image_id": ["ood.jpg"]}).to_csv(
            paths["ood"], index=False
        )

        source_images = root / "source_images"
        source_images.mkdir()
        (source_images / "ood.jpg").write_bytes(b"synthetic-image")
        source_ood = pd.DataFrame(
            [
                {
                    "image_id": "ood.jpg",
                    PREPROCESS.CLINICAL_FIELDS[0][0]: "clinical source",
                    PREPROCESS.XRAY_FIELDS[0][0]: "xray source",
                }
            ]
        )
        return paths, source_images, source_ood

    def test_repair_materializes_only_ood_and_preserves_locked_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, source_images, source_ood = self._fixture(root)

            paths["clinical_vi"].mkdir(parents=True)
            paths["clinical_en"].mkdir(parents=True)
            locked_ood_report = paths["clinical_vi"] / "ood.txt"
            locked_ood_report.write_text(
                "Reason for admission: locked OOD report", encoding="utf-8"
            )
            id_report = paths["clinical_en"] / "id.txt"
            id_report.write_text("locked ID report", encoding="utf-8")

            locked_bytes = {
                "splits": paths["splits"].read_bytes(),
                "labels": paths["labels"].read_bytes(),
                "ood": paths["ood"].read_bytes(),
                "id_report": id_report.read_bytes(),
                "ood_report": locked_ood_report.read_bytes(),
            }

            summary = PREPROCESS.repair_ood_artifacts(
                source_ood,
                paths,
                source_images,
                image_mode="copy",
                workers=1,
                translator=FakeTranslator(),
            )

            self.assertEqual(paths["splits"].read_bytes(), locked_bytes["splits"])
            self.assertEqual(paths["labels"].read_bytes(), locked_bytes["labels"])
            self.assertEqual(paths["ood"].read_bytes(), locked_bytes["ood"])
            self.assertEqual(id_report.read_bytes(), locked_bytes["id_report"])
            self.assertEqual(
                locked_ood_report.read_bytes(), locked_bytes["ood_report"]
            )
            self.assertTrue((paths["images"] / "ood.jpg").is_file())
            self.assertTrue((paths["xray_vi"] / "ood.txt").is_file())
            self.assertTrue((paths["clinical_en"] / "ood.txt").is_file())
            self.assertTrue((paths["xray_en"] / "ood.txt").is_file())
            self.assertTrue(all(not missing for missing in summary["after"].values()))

    def test_repair_refuses_id_ood_overlap_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, source_images, source_ood = self._fixture(
                root, id_image_id="ood.jpg"
            )
            with self.assertRaisesRegex(ValueError, "manifests overlap"):
                PREPROCESS.repair_ood_artifacts(
                    source_ood,
                    paths,
                    source_images,
                    image_mode="copy",
                    workers=1,
                    translator=FakeTranslator(),
                )
            self.assertFalse(paths["images"].exists())
            self.assertFalse((paths["clinical_vi"] / "ood.txt").exists())


if __name__ == "__main__":
    unittest.main()
