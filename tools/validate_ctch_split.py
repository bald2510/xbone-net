"""Validate the materialized CTCH split without modifying images or reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> tuple[str, str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return path.name, digest.hexdigest()


def _cross_split_hash_duplicates(
    image_paths: list[Path],
    split_by_image: dict[str, str],
    workers: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for image_id, digest in executor.map(_sha256, image_paths):
            grouped[digest].append(image_id)
    duplicates = []
    for digest, image_ids in grouped.items():
        splits = sorted({split_by_image[image_id] for image_id in image_ids})
        if len(splits) > 1:
            duplicates.append(
                {
                    "sha256": digest,
                    "image_ids": sorted(image_ids),
                    "splits": splits,
                }
            )
    return duplicates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "data" / "CTCH"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "data_validation" / "ctch_split_validation.json",
    )
    parser.add_argument(
        "--check-content-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--require-patient-key", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    split_path = data_root / "ctch-split.csv"
    labels_path = data_root / "ctch-labels.csv"
    splits = pd.read_csv(split_path)
    labels = pd.read_csv(labels_path)
    errors: list[str] = []
    warnings: list[str] = []

    required_split_columns = {"image_id", "split"}
    if not required_split_columns.issubset(splits.columns):
        errors.append(
            f"Split manifest lacks {sorted(required_split_columns - set(splits.columns))}."
        )
    if "image_id" not in labels.columns:
        errors.append("Label manifest lacks image_id.")
    if errors:
        raise RuntimeError("Cannot validate manifests:\n- " + "\n- ".join(errors))

    splits["image_id"] = splits["image_id"].astype(str)
    labels["image_id"] = labels["image_id"].astype(str)
    if splits["image_id"].duplicated().any():
        errors.append("Duplicate image_id rows exist in ctch-split.csv.")
    if labels["image_id"].duplicated().any():
        errors.append("Duplicate image_id rows exist in ctch-labels.csv.")

    allowed_splits = {"train", "validate", "test"}
    unexpected = sorted(set(splits["split"].astype(str)) - allowed_splits)
    if unexpected:
        errors.append(f"Unexpected split labels: {unexpected}.")

    split_ids = set(splits["image_id"])
    label_ids = set(labels["image_id"])
    if split_ids != label_ids:
        errors.append(
            "Split and label manifests differ: "
            f"split_only={len(split_ids - label_ids)}, "
            f"label_only={len(label_ids - split_ids)}."
        )

    merged = splits.merge(labels, on="image_id", how="inner", validate="one_to_one")
    class_counts: dict[str, dict[str, int]] = {}
    if "class_id" not in merged.columns:
        errors.append("Label manifest lacks class_id.")
    else:
        for split_name in ("train", "validate", "test"):
            counts = (
                merged.loc[merged["split"] == split_name, "class_id"]
                .value_counts()
                .sort_index()
            )
            class_counts[split_name] = {
                str(int(class_id)): int(count)
                for class_id, count in counts.items()
            }

    images_dir = data_root / "images"
    xray_dir = data_root / "reports" / "xray"
    clinical_dir = data_root / "reports" / "clinical"
    missing_images = []
    missing_xray = []
    missing_clinical = []
    image_paths = []
    for image_id in splits["image_id"]:
        image_path = images_dir / image_id
        stem = Path(image_id).stem
        if image_path.is_file():
            image_paths.append(image_path)
        else:
            missing_images.append(image_id)
        if not (xray_dir / f"{stem}.txt").is_file():
            missing_xray.append(image_id)
        if not (clinical_dir / f"{stem}.txt").is_file():
            missing_clinical.append(image_id)
    if missing_images:
        errors.append(f"Missing {len(missing_images)} ID images.")
    if missing_xray:
        errors.append(f"Missing {len(missing_xray)} English X-ray reports.")
    if missing_clinical:
        errors.append(f"Missing {len(missing_clinical)} English clinical reports.")

    patient_validation: dict[str, Any]
    if "patient_key" in splits.columns:
        patient_keys = splits["patient_key"].astype("string")
        missing_patient = int(patient_keys.fillna("").eq("").sum())
        patient_sets = {
            split_name: set(
                patient_keys[splits["split"] == split_name].dropna().astype(str)
            )
            for split_name in ("train", "validate", "test")
        }
        overlap = sorted(
            (patient_sets["train"] & patient_sets["validate"])
            | (patient_sets["train"] & patient_sets["test"])
            | (patient_sets["validate"] & patient_sets["test"])
        )
        patient_validation = {
            "status": "verified" if not overlap and missing_patient == 0 else "failed",
            "missing_patient_keys": missing_patient,
            "overlap_count": len(overlap),
            "overlap_examples": overlap[:20],
            "patients_per_split": {
                key: len(value) for key, value in patient_sets.items()
            },
        }
        if overlap:
            errors.append(f"Patient leakage across splits: {len(overlap)} patient keys.")
        if missing_patient:
            errors.append(f"Missing patient_key for {missing_patient} split rows.")
    else:
        message = (
            "ctch-split.csv has no patient_key; patient-level disjointness cannot "
            "be certified from the materialized dataset."
        )
        warnings.append(message)
        patient_validation = {"status": "unverifiable", "reason": message}
        if args.require_patient_key:
            errors.append(message)

    hash_duplicates: list[dict[str, Any]] = []
    if args.check_content_hashes and not missing_images:
        split_by_image = dict(zip(splits["image_id"], splits["split"].astype(str)))
        hash_duplicates = _cross_split_hash_duplicates(
            image_paths, split_by_image, args.workers
        )
        if hash_duplicates:
            errors.append(
                f"Found {len(hash_duplicates)} exact image-content duplicates across splits."
            )

    result = {
        "type": "ctch_split_validation",
        "data_root": str(data_root),
        "status": (
            "failed"
            if errors
            else ("passed_with_warnings" if warnings else "passed")
        ),
        "scope": "Read-only validation; no images, reports, labels, or split rows modified.",
        "rows": {
            "split": int(len(splits)),
            "labels": int(len(labels)),
            "merged": int(len(merged)),
        },
        "split_counts": {
            str(key): int(value)
            for key, value in splits["split"].value_counts().items()
        },
        "class_counts": class_counts,
        "file_coverage": {
            "missing_images": len(missing_images),
            "missing_xray_reports": len(missing_xray),
            "missing_clinical_reports": len(missing_clinical),
        },
        "patient_level": patient_validation,
        "exact_cross_split_image_duplicates": hash_duplicates,
        "errors": errors,
        "warnings": warnings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
