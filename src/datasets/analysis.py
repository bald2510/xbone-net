"""Datasets used only by the locked CTCH post-hoc analysis pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.analysis import MetadataDataset

from .high_resolution import prepare_global_image, prepare_high_resolution_inputs


def _file_issue(path: Path) -> Optional[str]:
    """Return why a required file cannot be read, or ``None`` when usable."""
    if not path.is_file():
        return "missing"
    try:
        with path.open("rb") as handle:
            handle.read(1)
    except OSError as error:
        return f"unreadable:{type(error).__name__}"
    return None


class CTCHOODDataset(Dataset):
    """CTCH semantic-OOD manifest with strict file-coverage validation.

    OOD labels are intentionally ``-1`` because the 22-way ID classifier has no
    valid target class for these samples.  By default every manifest row must
    have an image plus both report files.  ``allow_missing`` is an explicit
    exploratory mode that filters incomplete rows and exposes coverage metadata.
    """

    def __init__(
        self,
        img_dir: str,
        xray_report_dir: str,
        clinical_report_dir: str,
        csv_manifest_path: str,
        split: str = "test",
        transform=None,
        tokenizer=None,
        high_res: Optional[dict] = None,
        preprocess: Optional[dict] = None,
        allow_missing: bool = False,
        **_: Any,
    ) -> None:
        if split not in {"test", "ood"}:
            raise ValueError("CTCHOODDataset supports only split='test' or 'ood'.")
        self.img_dir = str(img_dir)
        self.xray_report_dir = str(xray_report_dir)
        self.clinical_report_dir = str(clinical_report_dir)
        self.transform = transform
        self.tokenizer = tokenizer
        self.high_res_cfg = high_res or {}
        self.preprocess_cfg = preprocess or {}
        self.use_high_res = bool(self.high_res_cfg.get("enabled", False))
        self.cache_selection = bool(self.high_res_cfg.get("cache_selection", True))
        self._selection_cache: dict[str, Any] = {}

        manifest = Path(csv_manifest_path)
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing CTCH OOD manifest: {manifest}")
        df = pd.read_csv(manifest)
        if "image_id" not in df.columns:
            raise ValueError("CTCH OOD manifest must contain image_id.")
        df["image_id"] = df["image_id"].astype(str)

        complete = []
        missing: list[dict[str, Any]] = []
        for index, row in df.iterrows():
            image_id = str(row["image_id"])
            stem = Path(image_id).stem
            absent = []
            required_files = {
                "image": Path(self.img_dir) / image_id,
                "xray_report": Path(self.xray_report_dir) / f"{stem}.txt",
                "clinical_report": (
                    Path(self.clinical_report_dir) / f"{stem}.txt"
                ),
            }
            for role, path in required_files.items():
                issue = _file_issue(path)
                if issue is not None:
                    absent.append(f"{role}:{issue}")
            complete.append(not absent)
            if absent:
                missing.append({"image_id": image_id, "missing": absent})

        self.coverage = {
            "manifest_rows": int(len(df)),
            "complete_rows": int(sum(complete)),
            "missing_rows": int(len(missing)),
            "coverage_fraction": float(sum(complete) / max(len(df), 1)),
            "allow_missing": bool(allow_missing),
            "missing": missing,
        }
        if missing and not allow_missing:
            preview = ", ".join(item["image_id"] for item in missing[:5])
            raise FileNotFoundError(
                "CTCH semantic-OOD is incomplete: "
                f"{len(missing)}/{len(df)} manifest rows lack an image or report "
                f"({preview}). Re-run data/CTCH/preprocess_ctch.py, or pass "
                "--allow-incomplete-ood only for an exploratory run."
            )
        self.df = df.loc[np.asarray(complete, dtype=bool)].reset_index(drop=True)
        if self.df.empty:
            raise RuntimeError("No complete CTCH OOD samples are available.")

    def __len__(self) -> int:
        return len(self.df)

    def _report(self, directory: str, image_id: str) -> torch.Tensor | str:
        path = Path(directory) / f"{Path(image_id).stem}.txt"
        text = path.read_text(encoding="utf-8").strip().lower()
        text = text or "no clinical information available."
        return self.tokenizer([text]).squeeze(0) if self.tokenizer else text

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        image_id = str(row["image_id"])
        image = Image.open(Path(self.img_dir) / image_id).convert("RGB")
        if self.use_high_res:
            selection = self._selection_cache.get(image_id)
            image_fields, selected = prepare_high_resolution_inputs(
                image,
                self.transform,
                self.high_res_cfg,
                selection=selection,
                return_selection=True,
            )
            if self.cache_selection and selection is None:
                self._selection_cache[image_id] = selected
        else:
            image_fields = {
                "pixel_values": prepare_global_image(
                    image, self.transform, self.preprocess_cfg
                )
            }

        columns = list(self.df.columns)
        patient_value = row[columns[1]] if len(columns) > 1 else ""
        group_value = row[columns[-2]] if len(columns) > 2 else "semantic_ood"
        return {
            **image_fields,
            "xray_input_ids": self._report(self.xray_report_dir, image_id),
            "clinical_input_ids": self._report(self.clinical_report_dir, image_id),
            "labels": torch.tensor(-1, dtype=torch.long),
            "image_id": image_id,
            "group": str(group_value),
            "patient_id": str(patient_value),
            "report_source_id": image_id,
            "scenario": "semantic_ood",
        }


def cross_class_derangement(
    labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic one-to-one report permutation across classes."""
    labels = np.asarray(labels).reshape(-1)
    if labels.size < 2:
        raise ValueError("Cross-class mismatch requires at least two samples.")
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    maximum = int(counts.max())
    if maximum * 2 > labels.size:
        majority = classes[int(np.argmax(counts))]
        raise ValueError(
            "A complete cross-class derangement is impossible because class "
            f"{majority!r} contains {maximum}/{labels.size} samples."
        )
    ordered_parts = []
    for class_id in classes:
        indices = np.flatnonzero(labels == class_id)
        ordered_parts.append(rng.permutation(indices))
    recipients = np.concatenate(ordered_parts)
    donors = np.roll(recipients, -maximum)
    if np.any(labels[recipients] == labels[donors]) or np.any(recipients == donors):
        raise RuntimeError("Failed to construct a valid cross-class derangement.")
    return recipients.astype(np.int64), donors.astype(np.int64)


def same_class_derangement(
    labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Derange reports within class, excluding mathematically impossible singletons."""
    labels = np.asarray(labels).reshape(-1)
    rng = np.random.default_rng(seed)
    recipients, donors = [], []
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        if len(indices) < 2:
            continue
        indices = rng.permutation(indices)
        recipients.extend(indices.tolist())
        donors.extend(np.roll(indices, -1).tolist())
    recipient_array = np.asarray(recipients, dtype=np.int64)
    donor_array = np.asarray(donors, dtype=np.int64)
    if recipient_array.size == 0:
        raise ValueError("No class contains at least two samples.")
    if np.any(recipient_array == donor_array):
        raise RuntimeError("Same-class derangement contains a fixed point.")
    if np.any(labels[recipient_array] != labels[donor_array]):
        raise RuntimeError("Same-class derangement crossed a class boundary.")
    return recipient_array, donor_array


class ReportMismatchDataset(Dataset):
    """Pair each CTCH test image with a deterministic donor report."""

    def __init__(
        self,
        dataset: Dataset,
        mode: str,
        seed: int,
    ) -> None:
        if not hasattr(dataset, "df"):
            raise TypeError("ReportMismatchDataset requires a dataset with a DataFrame.")
        self.dataset = dataset
        self.df = dataset.df
        labels = self.df["class_id"].to_numpy(dtype=np.int64)
        if mode == "cross_class":
            self.recipient_indices, self.donor_indices = cross_class_derangement(
                labels, seed
            )
        elif mode == "same_class":
            self.recipient_indices, self.donor_indices = same_class_derangement(
                labels, seed
            )
        else:
            raise ValueError("mode must be 'cross_class' or 'same_class'.")
        self.mode = mode

    def __len__(self) -> int:
        return len(self.recipient_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        recipient = int(self.recipient_indices[index])
        donor = int(self.donor_indices[index])
        sample = MetadataDataset._as_dict(self.dataset[recipient])
        recipient_row = self.df.iloc[recipient]
        donor_row = self.df.iloc[donor]
        recipient_id = str(recipient_row["image_id"])
        donor_id = str(donor_row["image_id"])

        xray_text = self.dataset._load_report(self.dataset.xray_report_dir, donor_id)
        clinical_text = self.dataset._load_report(
            self.dataset.clinical_report_dir, donor_id
        )
        sample["xray_input_ids"] = self.dataset._tokenize(xray_text)
        sample["clinical_input_ids"] = self.dataset._tokenize(clinical_text)
        sample.update(
            {
                "image_id": recipient_id,
                "group": str(int(recipient_row["class_id"])),
                "patient_id": "",
                "report_source_id": donor_id,
                "scenario": f"report_mismatch_{self.mode}",
            }
        )
        return sample
