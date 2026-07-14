"""BTXRD dataset with deterministic high-resolution grid tiling."""

from __future__ import annotations

import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .high_resolution import (
    letterbox_square,
    make_uniform_grid_tiles,
    normalize_tile_boxes,
    prepare_high_resolution_inputs,
)


class BTXRDDataset(Dataset):
    """Pair BTXRD radiographs with reports and multiclass labels."""

    def __init__(
        self,
        img_dir: str,
        report_dir: str,
        csv_split_path: str,
        csv_labels_path: str,
        pathologies: list | None = None,
        classes: list | None = None,
        task_type: str = "multiclass",
        num_classes: int | None = None,
        split: str = "train",
        train_ratio: float = 1.0,
        transform=None,
        tokenizer=None,
        max_text_len: int = 256,
        clinical_subdir: str = "clinical_v2",
        k_shot: int | None = None,
        seed: int = 42,
        high_res: dict | None = None,
        **kwargs,
    ):
        del num_classes
        self.img_dir = img_dir
        self.report_dir = report_dir
        self.classes = classes or pathologies or []
        self.task_type = task_type
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

        split_frame = pd.read_csv(csv_split_path)
        label_frame = pd.read_csv(csv_labels_path)
        merged = pd.merge(
            split_frame,
            label_frame,
            on=["image_id"],
            how="inner",
            validate="one_to_one",
        )

        if self.task_type == "multiclass" and "class_id" not in merged.columns:
            if not self.classes:
                raise ValueError(
                    "BTXRD multiclass loading requires dataset.params.classes when "
                    "the label manifest has no class_id column."
                )
            missing_class_columns = [
                name for name in self.classes if name not in merged.columns
            ]
            if missing_class_columns:
                raise ValueError(
                    "BTXRD label manifest does not match dataset classes. "
                    f"Missing columns: {missing_class_columns}."
                )
            label_matrix = merged[self.classes].apply(
                pd.to_numeric, errors="coerce"
            )
            # BTXRD contains hierarchical indicators. A specific subtype is the
            # multiclass target when both it and its generic parent are active.
            hierarchical_children = {
                "osteochondroma": (
                    "multiple osteochondromas",
                    "synovial osteochondroma",
                ),
            }
            for parent, children in hierarchical_children.items():
                present_children = [
                    child for child in children if child in label_matrix.columns
                ]
                if parent in label_matrix.columns and present_children:
                    child_is_active = label_matrix[present_children].eq(1).any(axis=1)
                    label_matrix.loc[child_is_active, parent] = 0
            active_counts = label_matrix.eq(1).sum(axis=1)
            invalid_rows = label_matrix.isna().any(axis=1) | active_counts.ne(1)
            if invalid_rows.any():
                raise ValueError(
                    "BTXRD multiclass labels must contain exactly one active class "
                    f"per sample; invalid rows={int(invalid_rows.sum())}."
                )
            merged["class_id"] = label_matrix.to_numpy().argmax(axis=1)

        if self.task_type == "multiclass":
            class_ids = pd.to_numeric(merged["class_id"], errors="coerce")
            expected_classes = len(self.classes)
            invalid_ids = class_ids.isna() | class_ids.lt(0)
            if expected_classes:
                invalid_ids |= class_ids.ge(expected_classes)
            if invalid_ids.any():
                raise ValueError(
                    "BTXRD label manifest contains "
                    f"{int(invalid_ids.sum())} invalid class_id values."
                )
            merged["class_id"] = class_ids.astype(int)

        current_split = "validate" if split == "val" else split
        filtered = merged[merged["split"] == current_split].reset_index(drop=True)
        if current_split == "train" and k_shot is not None:
            if self.task_type != "multiclass" or "class_id" not in filtered.columns:
                raise ValueError(
                    "BTXRD k-shot sampling requires multiclass labels in class_id."
                )
            if isinstance(k_shot, bool) or not isinstance(k_shot, int) or k_shot < 1:
                raise ValueError("k_shot must be a positive integer or null.")
            sampled_groups = [
                group.sample(n=min(len(group), k_shot), random_state=seed)
                for _, group in filtered.groupby("class_id", sort=True)
            ]
            if not sampled_groups:
                raise ValueError("BTXRD training split is empty; cannot apply k-shot sampling.")
            filtered = pd.concat(sampled_groups, ignore_index=True)
            print(
                f"[Dataset] BTXRD few-shot learning: requested={k_shot}-shot, "
                f"classes={filtered['class_id'].nunique()}, "
                f"samples={len(filtered)}, seed={seed}."
            )
        elif current_split == "train" and train_ratio < 1.0:
            filtered = filtered.sample(frac=train_ratio, random_state=seed).reset_index(
                drop=True
            )
            print(f"[Dataset] Training subset: {train_ratio * 100:.1f}%.")
        self.df = filtered

        self.xray_dir = os.path.join(report_dir, "xray")
        self.clinical_dir = os.path.join(report_dir, clinical_subdir)
        self.has_dual_reports = os.path.isdir(self.xray_dir) and os.path.isdir(
            self.clinical_dir
        )

        self.high_res_cfg = high_res or {}
        self.use_high_res = bool(self.high_res_cfg.get("enabled", False))
        self.text_only = bool(kwargs.get("text_only", False))
        self.shuffle_reports = bool(kwargs.get("shuffle_reports", False))
        if self.shuffle_reports:
            import numpy as np
            self.shuffled_report_indices = np.random.default_rng(seed).permutation(
                len(self.df)
            )
        else:
            self.shuffled_report_indices = None

        print(
            f"[Dataset] Initialized '{split.upper()}' with {len(self.df)} samples. "
            f"Dual reports: {self.has_dual_reports}. Uniform high-res grid: "
            f"{self.use_high_res}."
        )

    def __len__(self):
        return len(self.df)

    def _load_and_tokenize(self, path: str, default_text: str):
        raw_text = ""
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as report_file:
                raw_text = report_file.read()
        text = raw_text.strip().lower() or default_text
        return self.tokenizer([text]).squeeze(0) if self.tokenizer else text

    def _label(self, row) -> torch.Tensor:
        if self.task_type == "multiclass":
            return torch.tensor(int(row["class_id"]), dtype=torch.long)
        return torch.tensor(
            [float(row.get(name, 0.0)) for name in self.classes],
            dtype=torch.float32,
        )

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_id = str(row["image_id"])
        stem = os.path.splitext(image_id)[0]
        image_path = os.path.join(self.img_dir, image_id)

        if self.text_only:
            image = Image.new("RGB", (224, 224), color="black")
        else:
            try:
                image = Image.open(image_path).convert("RGB")
            except FileNotFoundError:
                image = Image.new("RGB", (224, 224), color="black")

        report_stem = stem
        if self.shuffled_report_indices is not None:
            shuffled_id = str(
                self.df.iloc[self.shuffled_report_indices[index]]["image_id"]
            )
            report_stem = os.path.splitext(shuffled_id)[0]

        labels = self._label(row)
        high_res_fields = {}
        if self.use_high_res:
            high_res_fields = prepare_high_resolution_inputs(
                image, self.transform, self.high_res_cfg
            )
        else:
            image = self.transform(image) if self.transform else image

        if self.has_dual_reports:
            xray_ids = self._load_and_tokenize(
                os.path.join(self.xray_dir, f"{report_stem}.txt"),
                "no clear bone abnormalities or fracture identified.",
            )
            clinical_ids = self._load_and_tokenize(
                os.path.join(self.clinical_dir, f"{report_stem}.txt"),
                "no clinical information available.",
            )
            if self.use_high_res:
                return {
                    **high_res_fields,
                    "xray_input_ids": xray_ids,
                    "clinical_input_ids": clinical_ids,
                    "labels": labels,
                }
            return image, xray_ids, clinical_ids, labels

        input_ids = self._load_and_tokenize(
            os.path.join(self.report_dir, f"{report_stem}.txt"),
            "no clear bone abnormalities or fracture identified.",
        )
        if self.use_high_res:
            return {**high_res_fields, "input_ids": input_ids, "labels": labels}
        return image, input_ids, labels
