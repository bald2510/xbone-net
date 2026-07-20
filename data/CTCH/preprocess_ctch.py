"""Prepare the cleaned CTCH workbook for XBone-Net.

The cleaned release contains one selected image per encounter. Source image
files are named ``{ID}_{SelectedImage}``, for example
``2_img-06080-00001.jpg``. Implant cases are excluded. Rows explicitly marked
OOD are kept in a separate manifest and are not mixed into ID classification.

Examples:
    python data/CTCH/preprocess_ctch.py --step prepare
    python data/CTCH/preprocess_ctch.py --step prepare --dry-run
    python data/CTCH/preprocess_ctch.py --step repair-ood
    python data/CTCH/preprocess_ctch.py --step validate
    python data/CTCH/preprocess_ctch.py --step translate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_XLSX = DATA_DIR / "data_labeled_full_selected_cleaned.xlsx"
DEFAULT_IMAGES_SRC = DATA_DIR / "images_no_implants"
DEFAULT_LABELS_TXT = DATA_DIR / "labels.txt"

LABEL_COL = "Label"
PATIENT_COL = "Mã bệnh nhân"
VISIT_COL = "Mã ca khám"
DISEASE_GROUP_COL = "Nhóm bệnh"
ROW_ID_COL = "ID"
SELECTED_IMAGE_COL = "SelectedImage"
DELETED_COL = "Deleted"
IMPLANTS_COL = "Implants"
OOD_COL = "OOD"

CLINICAL_FIELDS = [
    ("LyDoVaoVien_BenhAN", "Reason for admission"),
    ("QuaTrinhBenhLi", "Disease history"),
    ("TienSuBenh_BanThan_LucVaoVien", "Personal medical history"),
]

# Confirmed diagnosis and Label are deliberately excluded to reduce label leakage.
XRAY_FIELDS = [
    ("KhamXet_TomTatCanLamSang", "Imaging and examination summary"),
    ("TongKetBenhAn_TomTatKetQuaXetNghiem", "Investigation summary"),
]


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFC", str(value)).strip()
    return " ".join(text.split())


def normalize_key(value) -> str:
    return normalize_text(value).casefold()


def format_identifier(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return normalize_text(value)


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest without modifying ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_class_labels(labels_path: Path) -> list[str]:
    """Load the ordered CTCH class vocabulary from labels.txt."""
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    labels = [
        normalize_text(line)
        for line in labels_path.read_text(encoding="utf-8-sig").splitlines()
        if normalize_text(line)
    ]
    if not labels:
        raise ValueError(f"No class labels found in: {labels_path}")

    normalized = [normalize_key(label) for label in labels]
    duplicates = [
        labels[index]
        for index, key in enumerate(normalized)
        if key in normalized[:index]
    ]
    if duplicates:
        raise ValueError(f"Duplicate labels in {labels_path.name}: {duplicates}")

    print(f"Loaded {len(labels):,} ordered classes from {labels_path.name}")
    return labels


def output_paths(output_root: Path) -> dict[str, Path]:
    reports = output_root / "reports"
    return {
        "images": output_root / "images",
        "labels": output_root / "ctch-labels.csv",
        "splits": output_root / "ctch-split.csv",
        "ood": output_root / "ctch-ood.csv",
        "clinical_vi": reports / "clinical_vi",
        "xray_vi": reports / "xray_vi",
        "clinical_en": reports / "clinical",
        "xray_en": reports / "xray",
        "translation_cache": output_root / "translation_cache.json",
    }


def load_data(xlsx_path: Path, include_ood_in_other: bool = False):
    if not xlsx_path.is_file():
        raise FileNotFoundError(f"Workbook not found: {xlsx_path}")
    df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
    print(f"Loaded {len(df):,} records from {xlsx_path.name}")

    required = [
        PATIENT_COL,
        VISIT_COL,
        DISEASE_GROUP_COL,
        ROW_ID_COL,
        LABEL_COL,
        SELECTED_IMAGE_COL,
        DELETED_COL,
        IMPLANTS_COL,
        OOD_COL,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required workbook columns: {missing}")

    deleted = pd.to_numeric(df[DELETED_COL], errors="coerce").fillna(0).astype(int)
    implants = pd.to_numeric(df[IMPLANTS_COL], errors="coerce").fillna(0).astype(int)
    selected = df[SELECTED_IMAGE_COL].map(normalize_text).ne("")
    keep = deleted.eq(0) & implants.eq(0) & selected
    filtered = df.loc[keep].copy()
    print(
        f"  Removed {(deleted.ne(0)).sum():,} deleted, "
        f"{(implants.ne(0)).sum():,} implant, and {(~selected).sum():,} unselected rows"
    )

    ood_mask = pd.to_numeric(filtered[OOD_COL], errors="coerce").fillna(0).astype(int).eq(1)
    ood_df = filtered.loc[ood_mask].copy()
    if include_ood_in_other:
        id_df = filtered
        print(
            f"  Keeping {len(ood_df):,} OOD rows in ID classification "
            "with their labels.txt classes by request"
        )
    else:
        id_df = filtered.loc[~ood_mask].copy()
        print(f"  Reserved {len(ood_df):,} OOD rows outside ID classification")
    return id_df.reset_index(drop=True), ood_df.reset_index(drop=True)


def attach_image_ids(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    row_ids = result[ROW_ID_COL].map(format_identifier)
    selected = result[SELECTED_IMAGE_COL].map(
        lambda value: Path(normalize_text(value)).name
    )
    result["image_id"] = row_ids + "_" + selected
    if result["image_id"].duplicated().any():
        duplicates = result.loc[result["image_id"].duplicated(False), "image_id"].tolist()
        raise ValueError(f"Duplicate generated image IDs: {duplicates[:10]}")
    return result


def filter_existing_images(df: pd.DataFrame, images_src: Path, strict: bool = True):
    if not images_src.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_src}")
    exists = df["image_id"].map(lambda name: (images_src / name).is_file())
    missing = df.loc[~exists, "image_id"].tolist()
    print(f"  Matched images: {int(exists.sum()):,}/{len(df):,}")
    if missing and strict:
        raise FileNotFoundError(
            f"{len(missing)} selected images are missing. Examples: {missing[:10]}"
        )
    return df.loc[exists].copy().reset_index(drop=True), missing


def assign_multiclass(
    df: pd.DataFrame,
    class_labels: list[str],
) -> pd.DataFrame:
    """Map workbook labels to class IDs using labels.txt as the authority."""
    result = df.copy()
    label_to_id = {
        normalize_key(label): class_id
        for class_id, label in enumerate(class_labels)
    }
    normalized_labels = result[LABEL_COL].map(normalize_key)
    result["class_id"] = normalized_labels.map(label_to_id)
    missing_mask = result["class_id"].isna()
    if missing_mask.any():
        missing = sorted(
            set(result.loc[missing_mask, LABEL_COL].map(normalize_text))
        )
        raise ValueError(
            f"{int(missing_mask.sum())} rows contain labels absent from labels.txt: "
            f"{missing[:10]}"
        )

    result["class_id"] = result["class_id"].astype(np.int64)
    result["mapped_class"] = result["class_id"].map(class_labels.__getitem__)
    for class_id, class_name in enumerate(class_labels):
        result[class_name] = (result["class_id"] == class_id).astype(np.int8)

    row_sum = result[class_labels].sum(axis=1)
    if not row_sum.eq(1).all():
        raise RuntimeError("Multiclass mapping must assign exactly one class per row")

    print("  Class distribution:")
    for class_id, class_name in enumerate(class_labels):
        print(
            f"    [{class_id:02d}] {class_name:<38} "
            f"{int(result[class_name].sum()):5d}"
        )
    return result


def _primary_class(series: pd.Series) -> str:
    counts = series.value_counts()
    return str(counts.index[0])


def create_patient_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Require train_ratio > 0, val_ratio >= 0, and train+val < 1")

    result = df.copy()
    patient_keys = result[PATIENT_COL].map(format_identifier).astype("string")
    missing_patient = patient_keys.eq("")
    fallback_visits = (
        result.loc[missing_patient, VISIT_COL]
        .map(format_identifier)
        .astype("string")
        .radd("visit_")
    )
    patient_keys.loc[missing_patient] = fallback_visits
    result["patient_key"] = patient_keys

    patient_info = result.groupby("patient_key").agg(
        primary_class=("mapped_class", _primary_class),
        sample_count=("image_id", "size"),
        class_count=("mapped_class", "nunique"),
    )
    conflicting = int(patient_info["class_count"].gt(1).sum())
    if conflicting:
        print(f"  [Warning] {conflicting} patients contain multiple mapped classes")

    rng = np.random.default_rng(seed)
    assignments: dict[str, str] = {}
    for _, class_patients in patient_info.groupby("primary_class", sort=True):
        patient_ids = class_patients.index.to_numpy(copy=True)
        rng.shuffle(patient_ids)
        count = len(patient_ids)
        n_train = int(np.floor(count * train_ratio))
        n_val = int(np.floor(count * val_ratio))
        if count >= 3:
            n_train = min(max(n_train, 1), count - 2)
            n_val = min(max(n_val, 1), count - n_train - 1)
        assignments.update({patient: "train" for patient in patient_ids[:n_train]})
        assignments.update(
            {patient: "validate" for patient in patient_ids[n_train:n_train + n_val]}
        )
        assignments.update({patient: "test" for patient in patient_ids[n_train + n_val:]})

    result["split"] = result["patient_key"].map(assignments)
    if result["split"].isna().any():
        raise RuntimeError("Some patients were not assigned to a split")
    print("  Split distribution:")
    for split in ("train", "validate", "test"):
        subset = result[result["split"] == split]
        print(
            f"    {split:<8} {len(subset):5d} images, "
            f"{subset['patient_key'].nunique():5d} patients"
        )
    return result


def build_report(row: pd.Series, fields, fallback: str) -> str:
    parts = []
    for field, label in fields:
        value = normalize_text(row.get(field))
        if value:
            parts.append(f"{label}: {value}")
    return "\n".join(parts) if parts else fallback


def _safe_clear_directory(path: Path, output_root: Path):
    resolved = path.resolve()
    root = output_root.resolve()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f"Refusing to clear path outside output root: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def write_reports(
    df: pd.DataFrame,
    paths: dict[str, Path],
    *,
    overwrite: bool = True,
):
    paths["clinical_vi"].mkdir(parents=True, exist_ok=True)
    paths["xray_vi"].mkdir(parents=True, exist_ok=True)
    written = Counter()
    for _, row in df.iterrows():
        stem = Path(row["image_id"]).stem
        clinical = build_report(
            row, CLINICAL_FIELDS, "No clinical information available."
        )
        xray = build_report(row, XRAY_FIELDS, "No imaging description available.")
        for report_type, directory, content in (
            ("clinical", paths["clinical_vi"], clinical),
            ("xray", paths["xray_vi"], xray),
        ):
            destination = directory / f"{stem}.txt"
            if destination.exists() and not overwrite:
                written[f"{report_type}_existing"] += 1
                continue
            destination.write_text(content, encoding="utf-8")
            written[f"{report_type}_written"] += 1
    print(
        "  Reports: "
        f"clinical written/existing={written['clinical_written']:,}/"
        f"{written['clinical_existing']:,}, "
        f"X-ray written/existing={written['xray_written']:,}/"
        f"{written['xray_existing']:,}"
    )


def materialize_images(
    df: pd.DataFrame,
    images_src: Path,
    images_dst: Path,
    mode: str,
    workers: int,
):
    if mode == "none":
        print("  Image materialization skipped (--image-mode none)")
        return
    images_dst.mkdir(parents=True, exist_ok=True)

    def materialize(image_id: str):
        source = images_src / image_id
        destination = images_dst / image_id
        if destination.exists():
            return "existing"
        if mode == "hardlink":
            try:
                os.link(source, destination)
                return "linked"
            except OSError:
                shutil.copy2(source, destination)
                return "copied"
        shutil.copy2(source, destination)
        return "copied"

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        statuses = list(executor.map(materialize, df["image_id"].tolist()))
    counts = Counter(statuses)
    print(
        f"  Images: {counts['linked']:,} hard-linked, {counts['copied']:,} copied, "
        f"{counts['existing']:,} already present"
    )


def prepare_dataframe(
    xlsx_path: Path,
    images_src: Path,
    labels_path: Path,
    include_ood_in_other: bool,
    train_ratio: float,
    val_ratio: float,
    seed: int,
):
    class_labels = load_class_labels(labels_path)
    id_df, ood_df = load_data(xlsx_path, include_ood_in_other)
    id_df = attach_image_ids(id_df)
    ood_df = attach_image_ids(ood_df)
    id_df, _ = filter_existing_images(id_df, images_src, strict=True)
    ood_df, _ = filter_existing_images(ood_df, images_src, strict=True)
    id_patients = {
        format_identifier(value)
        for value in id_df[PATIENT_COL]
        if format_identifier(value)
    }
    ood_patients = {
        format_identifier(value)
        for value in ood_df[PATIENT_COL]
        if format_identifier(value)
    }
    patient_overlap = id_patients & ood_patients
    if patient_overlap:
        raise ValueError(
            "CTCH ID and semantic-OOD cohorts are not patient-disjoint: "
            f"{len(patient_overlap)} overlapping patients. Resolve cohort membership "
            "before producing official OOD results."
        )
    id_df = assign_multiclass(id_df, class_labels)
    id_df = create_patient_splits(
        id_df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    return id_df, ood_df, class_labels


def step_prepare(args):
    print("=" * 72)
    print("PREPARE CLEANED CTCH DATASET")
    print("=" * 72)
    output_root = args.output_root.resolve()
    paths = output_paths(output_root)
    id_df, ood_df, class_labels = prepare_dataframe(
        args.input_xlsx.resolve(),
        args.images_src.resolve(),
        args.labels_txt.resolve(),
        args.include_ood_in_other,
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )
    if args.dry_run:
        print("\nDry run complete: no files were written.")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for key in ("images", "clinical_vi", "xray_vi", "clinical_en", "xray_en"):
            _safe_clear_directory(paths[key], output_root)

    labels = id_df[["image_id", "class_id", "mapped_class"] + class_labels]
    # Retain an anonymized grouping key so confidence intervals can resample
    # patients rather than treating repeated radiographs as independent.
    splits = id_df[["image_id", "split", "patient_key"]]
    labels.to_csv(paths["labels"], index=False)
    splits.to_csv(paths["splits"], index=False)
    ood_columns = [
        "image_id", PATIENT_COL, VISIT_COL, LABEL_COL, DISEASE_GROUP_COL, OOD_COL
    ]
    ood_df[ood_columns].to_csv(paths["ood"], index=False)
    print(f"  Saved {len(labels):,} ID rows to {paths['labels'].name}")
    print(f"  Saved {len(ood_df):,} OOD rows to {paths['ood'].name}")
    # OOD evaluation consumes the same image/report schema as ID inference.
    # Materialize both populations; manifests remain separate so no OOD sample
    # can enter a train/validation/test split.
    analysis_df = pd.concat([id_df, ood_df], ignore_index=True)
    analysis_df = analysis_df.drop_duplicates(subset=["image_id"], keep="first")
    write_reports(analysis_df, paths)
    materialize_images(
        analysis_df,
        args.images_src.resolve(),
        paths["images"],
        args.image_mode,
        args.workers,
    )
    print("PREPARE COMPLETE")


def locked_manifest_hashes(paths: dict[str, Path]) -> dict[str, str]:
    """Fingerprint the authoritative ID splits/labels and locked OOD cohort."""
    hashes = {}
    for key in ("labels", "splits", "ood"):
        path = paths[key]
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing locked ID manifest required for OOD repair: {path}"
            )
        hashes[key] = file_sha256(path)
    return hashes


def ood_artifact_coverage(
    image_ids: list[str],
    paths: dict[str, Path],
) -> dict[str, list[str]]:
    """Return missing OOD artifact IDs for every materialized modality."""
    checks = {
        "images": lambda image_id: paths["images"] / image_id,
        "clinical_vi": lambda image_id: (
            paths["clinical_vi"] / f"{Path(image_id).stem}.txt"
        ),
        "xray_vi": lambda image_id: (
            paths["xray_vi"] / f"{Path(image_id).stem}.txt"
        ),
        "clinical_en": lambda image_id: (
            paths["clinical_en"] / f"{Path(image_id).stem}.txt"
        ),
        "xray_en": lambda image_id: (
            paths["xray_en"] / f"{Path(image_id).stem}.txt"
        ),
    }
    return {
        artifact: [
            image_id
            for image_id in image_ids
            if not resolve_path(image_id).is_file()
        ]
        for artifact, resolve_path in checks.items()
    }


def _load_locked_ood_repair_rows(
    source_ood_df: pd.DataFrame,
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, list[str]]:
    """Align workbook metadata to the existing OOD manifest without rewriting it."""
    if not paths["ood"].is_file():
        raise FileNotFoundError(
            f"Missing OOD manifest: {paths['ood']}. Run --step prepare once first."
        )
    manifest = pd.read_csv(paths["ood"])
    if "image_id" not in manifest.columns:
        raise ValueError("CTCH OOD manifest must contain image_id.")
    manifest_ids = manifest["image_id"].astype("string")
    if manifest_ids.isna().any() or manifest_ids.str.strip().eq("").any():
        raise ValueError("CTCH OOD manifest contains an empty image_id.")
    manifest_ids = manifest_ids.astype(str).tolist()
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError("CTCH OOD manifest contains duplicate image_id rows.")

    split_manifest = pd.read_csv(paths["splits"], usecols=["image_id"])
    id_image_ids = set(split_manifest["image_id"].astype(str))
    overlap = id_image_ids & set(manifest_ids)
    if overlap:
        raise ValueError(
            "Refusing OOD repair because ID and OOD manifests overlap by "
            f"{len(overlap)} image IDs."
        )

    if "image_id" not in source_ood_df.columns:
        raise ValueError("Source OOD dataframe must contain image_id.")
    source = source_ood_df.copy()
    source["image_id"] = source["image_id"].astype(str)
    if source["image_id"].duplicated().any():
        duplicates = source.loc[
            source["image_id"].duplicated(False), "image_id"
        ].tolist()
        raise ValueError(f"Duplicate source OOD image IDs: {duplicates[:10]}")

    source_by_id = source.set_index("image_id", drop=False)
    missing_metadata = [
        image_id for image_id in manifest_ids if image_id not in source_by_id.index
    ]
    if missing_metadata:
        raise ValueError(
            f"{len(missing_metadata)} locked OOD rows are absent from the source "
            f"workbook. Examples: {missing_metadata[:10]}"
        )

    extra_source = set(source_by_id.index) - set(manifest_ids)
    if extra_source:
        print(
            f"  [Info] Ignoring {len(extra_source):,} additional workbook OOD rows; "
            "repair-ood keeps ctch-ood.csv unchanged."
        )
    repair_df = source_by_id.loc[manifest_ids].reset_index(drop=True)
    return repair_df, manifest_ids


def translate_report_files(
    paths: dict[str, Path],
    *,
    selected_names: set[str] | None = None,
    translator=None,
    strict: bool = False,
) -> dict[str, int]:
    """Translate missing report files, optionally restricted to exact filenames."""
    cache = {}
    if paths["translation_cache"].exists():
        cache = json.loads(paths["translation_cache"].read_text(encoding="utf-8"))

    translator_instance = translator
    counts = Counter()

    def translate_text(text: str):
        nonlocal translator_instance
        key = text.strip()
        if not key:
            return ""
        if key in cache:
            counts["cache_hits"] += 1
            return cache[key]
        try:
            if translator_instance is None:
                from deep_translator import GoogleTranslator

                translator_instance = GoogleTranslator(source="vi", target="en")
            translated = translator_instance.translate(key)
        except Exception as error:
            if strict:
                raise RuntimeError(
                    "Translation failed in strict OOD repair mode; no fallback "
                    f"report was written. Cause: {error}"
                ) from error
            print(f"  [Warning] Translation failed: {str(error)[:100]}")
            counts["fallbacks"] += 1
            return key
        if translated is None or not str(translated).strip():
            raise RuntimeError("Translator returned empty report text.")
        translated = str(translated).strip()
        cache[key] = translated
        counts["translated_segments"] += 1
        return translated

    try:
        for source_key, target_key, label in (
            ("clinical_vi", "clinical_en", "clinical"),
            ("xray_vi", "xray_en", "X-ray"),
        ):
            source_dir, target_dir = paths[source_key], paths[target_key]
            if not source_dir.is_dir():
                raise FileNotFoundError(f"Run --step prepare first: {source_dir}")
            target_dir.mkdir(parents=True, exist_ok=True)
            files = sorted(source_dir.glob("*.txt"))
            if selected_names is not None:
                files = [file for file in files if file.name in selected_names]
                absent_sources = selected_names - {file.name for file in files}
                if absent_sources:
                    raise FileNotFoundError(
                        f"Missing {len(absent_sources)} {label} source reports. "
                        f"Examples: {sorted(absent_sources)[:10]}"
                    )
            todo = [file for file in files if not (target_dir / file.name).exists()]
            print(f"  {label}: {len(files):,} selected, {len(todo):,} to translate")
            for index, source_file in enumerate(todo, start=1):
                output_lines = []
                for line in source_file.read_text(encoding="utf-8").splitlines():
                    if ": " in line:
                        prefix, content = line.split(": ", 1)
                        output_lines.append(f"{prefix}: {translate_text(content)}")
                    else:
                        output_lines.append(translate_text(line))
                destination = target_dir / source_file.name
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                temporary.write_text("\n".join(output_lines), encoding="utf-8")
                os.replace(temporary, destination)
                counts[f"{label.lower()}_files"] += 1
                if index % 100 == 0:
                    paths["translation_cache"].write_text(
                        json.dumps(cache, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    print(f"    {index:,}/{len(todo):,}")
                    time.sleep(0.25)
    finally:
        paths["translation_cache"].write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return dict(counts)


def repair_ood_artifacts(
    source_ood_df: pd.DataFrame,
    paths: dict[str, Path],
    images_src: Path,
    *,
    image_mode: str,
    workers: int,
    dry_run: bool = False,
    translator=None,
) -> dict[str, object]:
    """Repair only locked CTCH-OOD artifacts and preserve all ID manifests."""
    before_hashes = locked_manifest_hashes(paths)
    repair_df, image_ids = _load_locked_ood_repair_rows(source_ood_df, paths)
    before = ood_artifact_coverage(image_ids, paths)
    print(
        "  Missing OOD artifacts before repair: "
        + ", ".join(f"{key}={len(value)}" for key, value in before.items())
    )

    missing_output_images = set(before["images"])
    missing_source_images = [
        image_id
        for image_id in missing_output_images
        if not (images_src / image_id).is_file()
    ]
    if missing_source_images:
        raise FileNotFoundError(
            f"Cannot restore {len(missing_source_images)} OOD images because the "
            f"source files are absent from {images_src}. Examples: "
            f"{missing_source_images[:10]}"
        )

    if dry_run:
        print("  Dry run complete: OOD artifacts and ID manifests were not modified.")
        return {
            "rows": len(image_ids),
            "before": before,
            "after": before,
            "id_manifest_hashes": before_hashes,
        }

    if missing_output_images:
        if image_mode == "none":
            raise ValueError(
                "OOD images are missing but --image-mode none forbids restoration."
            )
        image_rows = repair_df[
            repair_df["image_id"].isin(missing_output_images)
        ].copy()
        materialize_images(
            image_rows,
            images_src,
            paths["images"],
            image_mode,
            workers,
        )

    # OOD and ID manifests are disjoint, and overwrite=False prevents this
    # repair path from changing any report that already exists.
    write_reports(repair_df, paths, overwrite=False)
    selected_names = {f"{Path(image_id).stem}.txt" for image_id in image_ids}
    translate_report_files(
        paths,
        selected_names=selected_names,
        translator=translator,
        strict=True,
    )

    after_hashes = locked_manifest_hashes(paths)
    if after_hashes != before_hashes:
        raise RuntimeError(
            "A locked CTCH label/split/OOD manifest changed during OOD repair."
        )
    after = ood_artifact_coverage(image_ids, paths)
    incomplete = {key: value for key, value in after.items() if value}
    if incomplete:
        details = ", ".join(f"{key}={len(value)}" for key, value in incomplete.items())
        raise RuntimeError(f"OOD repair finished with incomplete artifacts: {details}")

    print(
        f"  OOD REPAIR PASSED: {len(image_ids):,}/{len(image_ids):,} images and "
        "dual reports are complete; ID manifest SHA-256 values are unchanged."
    )
    return {
        "rows": len(image_ids),
        "before": before,
        "after": after,
        "id_manifest_hashes": after_hashes,
    }


def step_repair_ood(args):
    print("=" * 72)
    print("REPAIR LOCKED CTCH OOD ARTIFACTS ONLY")
    print("=" * 72)
    paths = output_paths(args.output_root.resolve())
    _, source_ood_df = load_data(args.input_xlsx.resolve(), False)
    source_ood_df = attach_image_ids(source_ood_df)
    repair_ood_artifacts(
        source_ood_df,
        paths,
        args.images_src.resolve(),
        image_mode=args.image_mode,
        workers=args.workers,
        dry_run=args.dry_run,
    )


def step_translate(args):
    paths = output_paths(args.output_root.resolve())
    translate_report_files(paths)


def step_validate(args):
    paths = output_paths(args.output_root.resolve())
    class_labels = load_class_labels(args.labels_txt.resolve())
    labels = pd.read_csv(paths["labels"])
    splits = pd.read_csv(paths["splits"])
    ood = pd.read_csv(paths["ood"])
    merged = pd.merge(splits, labels, on="image_id", how="inner", validate="one_to_one")
    errors = []
    if len(labels) != len(splits) or len(merged) != len(labels):
        errors.append("Labels and split manifests have different image sets")
    if labels["image_id"].duplicated().any():
        errors.append("Duplicate image_id in labels manifest")
    missing_class_columns = [
        class_name for class_name in class_labels
        if class_name not in labels.columns
    ]
    if missing_class_columns:
        errors.append(
            f"Missing {len(missing_class_columns)} labels.txt class columns"
        )
    elif not labels[class_labels].sum(axis=1).eq(1).all():
        errors.append("Some label rows are not one-hot multiclass")
    if "class_id" not in labels.columns:
        errors.append("Missing class_id in labels manifest")
    elif not labels["class_id"].between(0, len(class_labels) - 1).all():
        errors.append("class_id is outside the labels.txt range")
    elif not missing_class_columns:
        one_hot_ids = labels[class_labels].to_numpy().argmax(axis=1)
        if not np.array_equal(one_hot_ids, labels["class_id"].to_numpy()):
            errors.append("class_id does not match the one-hot class columns")
    if "mapped_class" not in labels.columns:
        errors.append("Missing mapped_class in labels manifest")
    elif (
        "class_id" in labels.columns
        and labels["class_id"].between(0, len(class_labels) - 1).all()
    ):
        expected_names = labels["class_id"].map(class_labels.__getitem__)
        if not expected_names.eq(labels["mapped_class"]).all():
            errors.append("mapped_class does not match class_id/labels.txt")
    if not set(splits["split"]).issubset({"train", "validate", "test"}):
        errors.append("Unexpected split name")

    image_root = paths["images"] if paths["images"].is_dir() else args.images_src.resolve()
    missing_images = [name for name in labels["image_id"] if not (image_root / name).is_file()]
    missing_clinical = [
        name for name in labels["image_id"]
        if not (paths["clinical_vi"] / f"{Path(name).stem}.txt").is_file()
    ]
    missing_xray = [
        name for name in labels["image_id"]
        if not (paths["xray_vi"] / f"{Path(name).stem}.txt").is_file()
    ]
    missing_clinical_en = [
        name for name in labels["image_id"]
        if not (paths["clinical_en"] / f"{Path(name).stem}.txt").is_file()
    ]
    missing_xray_en = [
        name for name in labels["image_id"]
        if not (paths["xray_en"] / f"{Path(name).stem}.txt").is_file()
    ]
    if missing_images:
        errors.append(f"Missing {len(missing_images)} images")
    if missing_clinical:
        errors.append(f"Missing {len(missing_clinical)} clinical reports")
    if missing_xray:
        errors.append(f"Missing {len(missing_xray)} X-ray reports")
    if missing_clinical_en:
        errors.append(f"Missing {len(missing_clinical_en)} English clinical reports")
    if missing_xray_en:
        errors.append(f"Missing {len(missing_xray_en)} English X-ray reports")

    if "image_id" not in ood.columns:
        errors.append("OOD manifest is missing image_id")
        ood_names = []
    else:
        ood_names = ood["image_id"].astype(str).tolist()
        overlap = set(ood_names) & set(labels["image_id"].astype(str))
        if overlap:
            errors.append(f"ID/OOD manifests overlap by {len(overlap)} image IDs")
    missing_ood_images = [
        name for name in ood_names if not (image_root / name).is_file()
    ]
    missing_ood_clinical = [
        name for name in ood_names
        if not (paths["clinical_vi"] / f"{Path(name).stem}.txt").is_file()
    ]
    missing_ood_xray = [
        name for name in ood_names
        if not (paths["xray_vi"] / f"{Path(name).stem}.txt").is_file()
    ]
    missing_ood_clinical_en = [
        name for name in ood_names
        if not (paths["clinical_en"] / f"{Path(name).stem}.txt").is_file()
    ]
    missing_ood_xray_en = [
        name for name in ood_names
        if not (paths["xray_en"] / f"{Path(name).stem}.txt").is_file()
    ]
    if missing_ood_images:
        errors.append(f"Missing {len(missing_ood_images)} OOD images")
    if missing_ood_clinical:
        errors.append(f"Missing {len(missing_ood_clinical)} OOD clinical reports")
    if missing_ood_xray:
        errors.append(f"Missing {len(missing_ood_xray)} OOD X-ray reports")
    if missing_ood_clinical_en:
        errors.append(
            f"Missing {len(missing_ood_clinical_en)} OOD English clinical reports"
        )
    if missing_ood_xray_en:
        errors.append(
            f"Missing {len(missing_ood_xray_en)} OOD English X-ray reports"
        )

    patient_overlap = None
    source_df, _ = load_data(args.input_xlsx.resolve(), args.include_ood_in_other)
    source_df = attach_image_ids(source_df)
    source_df = source_df[source_df["image_id"].isin(merged["image_id"])].copy()
    source_df = source_df.merge(splits, on="image_id", how="left")
    split_sets = {
        split: set(source_df.loc[source_df["split"] == split, PATIENT_COL].map(format_identifier))
        for split in ("train", "validate", "test")
    }
    patient_overlap = (
        len(split_sets["train"] & split_sets["validate"])
        + len(split_sets["train"] & split_sets["test"])
        + len(split_sets["validate"] & split_sets["test"])
    )
    if patient_overlap:
        errors.append(f"Patient leakage across splits: {patient_overlap} overlaps")

    print(f"Validated {len(labels):,} ID samples")
    print(f"  Splits: {dict(splits['split'].value_counts())}")
    print(
        "  Missing ID images/reports vi/en: "
        f"{len(missing_images)}/{len(missing_clinical)}/{len(missing_xray)}/"
        f"{len(missing_clinical_en)}/{len(missing_xray_en)}"
    )
    print(
        "  OOD coverage images/reports vi/en: "
        f"{len(ood_names) - len(missing_ood_images)}/"
        f"{len(ood_names) - len(missing_ood_clinical)}/"
        f"{len(ood_names) - len(missing_ood_xray)}/"
        f"{len(ood_names) - len(missing_ood_clinical_en)}/"
        f"{len(ood_names) - len(missing_ood_xray_en)} of {len(ood_names)}"
    )
    print(f"  Patient overlap across splits: {patient_overlap}")
    if errors:
        raise RuntimeError("Validation failed:\n- " + "\n- ".join(errors))
    print("VALIDATION PASSED")


def step_analyze(args):
    paths = output_paths(args.output_root.resolve())
    class_labels = load_class_labels(args.labels_txt.resolve())
    labels = pd.read_csv(paths["labels"])
    splits = pd.read_csv(paths["splits"])
    merged = pd.merge(splits, labels, on="image_id", how="inner")
    print(f"Total ID images: {len(merged):,}")
    print(f"Splits: {dict(merged['split'].value_counts())}")
    print("Class distribution:")
    for class_id, class_name in enumerate(class_labels):
        split_counts = {
            split: int(merged.loc[merged["split"] == split, class_name].sum())
            for split in ("train", "validate", "test")
        }
        print(
            f"  [{class_id:02d}] {class_name:<38} "
            f"{int(merged[class_name].sum()):5d} "
            f"{split_counts}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        required=True,
        choices=[
            "prepare",
            "repair-ood",
            "translate",
            "validate",
            "analyze",
            "all",
        ],
    )
    parser.add_argument("--input-xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--images-src", type=Path, default=DEFAULT_IMAGES_SRC)
    parser.add_argument(
        "--labels-txt",
        type=Path,
        default=DEFAULT_LABELS_TXT,
        help="Ordered class vocabulary; line number is the zero-based class ID.",
    )
    parser.add_argument("--output-root", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--image-mode",
        choices=["hardlink", "copy", "none"],
        default="hardlink",
        help="Materialize data/CTCH/images using hardlinks, copies, or not at all.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--include-ood-in-other",
        action="store_true",
        help=(
            "Legacy flag: include rows marked OOD in ID classification, using "
            "their labels.txt class instead of reserving them for ctch-ood.csv."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.step in ("prepare", "all"):
        step_prepare(args)
    if args.step == "repair-ood":
        step_repair_ood(args)
    if args.step in ("translate", "all"):
        step_translate(args)
    if args.step in ("validate", "all"):
        step_validate(args)
    if args.step in ("analyze", "all"):
        step_analyze(args)


if __name__ == "__main__":
    main()
