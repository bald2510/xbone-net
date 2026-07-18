"""
CTCH bone pathology dataset loader for XBone-Net architecture.
============================================================
Provides a PyTorch Dataset for X-ray images paired with text reports.
Used in XBone-Net's dual-stage learning framework:
  - Stage 1: Contrastive image-text alignment using X-ray findings reports
  - Stage 2: Cross-attention multimodal fusion combining X-ray findings + clinical history
  - Supports both multi-class (single integer label) and multi-label classification

Supports dual-report loading (xray_report_dir and clinical_report_dir) with fallback
to a single report directory for backward compatibility.
"""

import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .high_resolution import prepare_global_image, prepare_high_resolution_inputs
from .sampling import cross_class_donor_indices, deranged_donor_indices


# ============================================================
# CTCH Dataset Loader
# ============================================================

class CTCHDataset(Dataset):
    """Dataset loader for the CTCH bone pathology dataset.

    Always operates in dual-report mode, returning separate imaging findings
    and clinical history texts per sample for multimodal fusion in XBone-Net.

    Supports both multi-class (via class_id or one-hot → argmax) and
    multi-label classification (binary vector), matching BTXRD format.

    Attributes:
        img_dir: Root directory containing X-ray image files.
        classes: List of target class / pathology names.
        task_type: Classification mode ('multiclass' or 'multilabel').
        xray_report_dir: Directory containing X-ray findings reports (.txt).
        clinical_report_dir: Directory containing clinical history reports (.txt).
        transform: torchvision.transforms pipeline applied to images.
        tokenizer: Text tokenizer callable returning token-ID tensors.
        max_text_len: Maximum token sequence length.
        df: Filtered pandas.DataFrame for the active data split.

    Example:
        ds = CTCHDataset(
            img_dir="data/CTCH/images",
            csv_split_path="data/CTCH/ctch-split.csv",
            csv_labels_path="data/CTCH/ctch-labels.csv",
            classes=["Bình thường", "Gãy xương quay", ...],
            task_type="multiclass",
            xray_report_dir="data/CTCH/reports/xray",
            clinical_report_dir="data/CTCH/reports/clinical",
            split="train",
        )
        image, xray_ids, clinical_ids, labels = ds[0]
    """

    def __init__(
        self,
        img_dir: str,
        csv_split_path: str,
        csv_labels_path: str,
        pathologies: list = None,
        classes: list = None,
        task_type: str = "multiclass",
        num_classes: int = None,
        split: str = "train",
        train_ratio: float = 1.0,
        k_shot: int = None,
        seed: int = 42,
        transform=None,
        tokenizer=None,
        max_text_len=256,
        # Dual report support
        xray_report_dir: str = None,
        clinical_report_dir: str = None,
        # Backward compatibility: single report_dir
        report_dir: str = None,
        high_res: dict = None,
        preprocess: dict = None,
        **kwargs,
    ):
        """Initialize the CTCH dataset.

        Args:
            img_dir: Path to directory containing image files.
            csv_split_path: Path to CSV with image_id and split columns.
            csv_labels_path: Path to CSV with image_id and class/pathology label columns.
            pathologies: List of pathology column names (alias for classes).
            classes: List of target class / pathology names (takes precedence).
            task_type: 'multiclass' (single integer label) or 'multilabel' (binary vector).
            num_classes: Optional explicit class count for external reference.
            split: Data split ('train', 'val', or 'test').
            train_ratio: Fraction of training samples to keep (0.0 to 1.0).
            k_shot: Maximum number of training samples retained per class.
            seed: Random seed used for deterministic per-class sampling.
            transform: torchvision.transforms pipeline for image preprocessing.
            tokenizer: Text tokenizer callable returning token-ID tensors.
            max_text_len: Maximum token sequence length.
            xray_report_dir: Path to directory containing X-ray findings reports.
            clinical_report_dir: Path to directory containing clinical history reports.
            report_dir: Single report directory for both report types (backward compatibility).
            high_res: Fixed-budget sparse focal preprocessing configuration.
            preprocess: Global-image preprocessing used when high-resolution mode is disabled.
            **kwargs: Extra unused arguments for backward compatibility.
        """
        self.img_dir = img_dir
        self.classes = classes or pathologies or []
        self.task_type = task_type
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.high_res_cfg = high_res or {}
        self.use_high_res = bool(self.high_res_cfg.get("enabled", False))
        self.cache_high_res_selection = bool(
            self.high_res_cfg.get("cache_selection", True)
        )
        self._high_res_selection_cache = {}
        self.preprocess_cfg = preprocess or {}
        self.text_only = bool(kwargs.get("text_only", False))
        self.shuffle_reports_all_splits = bool(kwargs.get("shuffle_reports", False))
        self.shuffle_report_mode = str(
            kwargs.get("shuffle_report_mode", "derangement")
        ).lower()
        configured_shuffle_splits = kwargs.get("shuffle_report_splits", []) or []
        self.shuffle_report_splits = {
            "validate" if str(name).lower() == "val" else str(name).lower()
            for name in configured_shuffle_splits
        }

        # --- Resolve report directories ---
        if report_dir and not xray_report_dir and not clinical_report_dir:
            # Backward compatible: single report_dir -> use for both
            self.xray_report_dir = report_dir
            self.clinical_report_dir = report_dir
        else:
            self.xray_report_dir = xray_report_dir
            self.clinical_report_dir = clinical_report_dir

        # --- Load and merge metadata ---
        df_split = pd.read_csv(csv_split_path)
        df_labels = pd.read_csv(csv_labels_path)
        df_merged = pd.merge(df_split, df_labels, on=["image_id"], how="inner")

        # --- Pre-compute class_id for multi-class indexing if missing ---
        if task_type == "multiclass" and "class_id" not in df_merged.columns and self.classes:
            missing_class_columns = [
                class_name
                for class_name in self.classes
                if class_name not in df_merged.columns
            ]
            if missing_class_columns:
                raise ValueError(
                    "CTCH label manifest does not match dataset classes. "
                    f"Missing {len(missing_class_columns)} columns, including "
                    f"{missing_class_columns[:3]}. Regenerate ctch-labels.csv with "
                    "data/CTCH/preprocess_ctch.py."
                )

            def _get_class_id(row):
                for idx_cls, cls_name in enumerate(self.classes):
                    if row.get(cls_name, 0) == 1:
                        return idx_cls
                raise ValueError("A CTCH sample has no active multiclass label")
            df_merged["class_id"] = df_merged.apply(_get_class_id, axis=1)

        if task_type == "multiclass" and "class_id" in df_merged.columns:
            class_ids = pd.to_numeric(df_merged["class_id"], errors="coerce")
            expected_classes = num_classes or len(self.classes)
            invalid = class_ids.isna() | class_ids.lt(0)
            if expected_classes:
                invalid |= class_ids.ge(expected_classes)
            if invalid.any():
                raise ValueError(
                    f"CTCH label manifest contains {int(invalid.sum())} invalid class_id values"
                )
            df_merged["class_id"] = class_ids.astype(int)

        # --- Filter split and apply subsampling ---
        current_split = "validate" if split == "val" else split
        filtered_df = df_merged[df_merged["split"] == current_split].reset_index(drop=True)

        if current_split == "train" and k_shot is not None:
            if task_type != "multiclass" or "class_id" not in filtered_df.columns:
                raise ValueError(
                    "CTCH k-shot sampling requires multiclass labels in class_id."
                )
            if isinstance(k_shot, bool) or not isinstance(k_shot, int) or k_shot < 1:
                raise ValueError("k_shot must be a positive integer or null.")

            sampled_groups = [
                group.sample(n=min(len(group), k_shot), random_state=seed)
                for _, group in filtered_df.groupby("class_id", sort=True)
            ]
            filtered_df = pd.concat(sampled_groups, ignore_index=True)
            print(
                f"[Dataset] CTCH few-shot learning: requested={k_shot}-shot, "
                f"classes={filtered_df['class_id'].nunique()}, "
                f"samples={len(filtered_df)}, seed={seed}."
            )
        elif current_split == "train" and train_ratio < 1.0:
            filtered_df = filtered_df.sample(
                frac=train_ratio, random_state=seed
            ).reset_index(drop=True)
            print(
                "[Dataset] Subsampling enabled: using "
                f"{train_ratio * 100:.1f}% of training data (seed={seed})."
            )

        self.df = filtered_df
        self.shuffle_reports = (
            self.shuffle_reports_all_splits
            or current_split in self.shuffle_report_splits
        )
        if self.shuffle_reports:
            if self.shuffle_report_mode == "cross_class":
                if "class_id" not in self.df.columns:
                    raise ValueError(
                        "shuffle_report_mode='cross_class' requires class_id."
                    )
                self.shuffled_report_indices = cross_class_donor_indices(
                    self.df["class_id"].to_numpy(), seed
                )
            elif self.shuffle_report_mode == "derangement":
                self.shuffled_report_indices = deranged_donor_indices(
                    len(self.df), seed
                )
            else:
                raise ValueError(
                    "shuffle_report_mode must be 'cross_class' or 'derangement'."
                )
        else:
            self.shuffled_report_indices = None

        # --- Detect dual report availability ---
        has_dual = bool(self.xray_report_dir and self.clinical_report_dir)
        print(
            f"[Dataset] CTCH '{split.upper()}' initialized with {len(self.df)} samples. "
            f"Task: {task_type}, Dual reports: {has_dual}, Sparse high-res views: "
            f"{self.use_high_res}, Text-only: {self.text_only}, Reports shuffled: "
            f"{self.shuffle_reports}."
        )

    def __len__(self):
        """Return the total number of samples in the current split."""
        return len(self.df)

    def _load_report(self, report_dir: str, image_id: str) -> str:
        """Load a text report file, clean it, and return normalized content.

        Args:
            report_dir: Directory containing .txt report files.
            image_id: Image filename (with extension) to locate report file.

        Returns:
            Cleaned lowercased report text, or empty string if file missing.
        """
        if report_dir is None:
            return ""
        file_name = os.path.splitext(image_id)[0]
        report_path = os.path.join(report_dir, f"{file_name}.txt")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                return f.read().strip().lower()
        return ""

    def _tokenize(self, text: str):
        """Tokenize text string with default fallback for missing content.

        Args:
            text: Cleaned report string.

        Returns:
            torch.Tensor of token IDs if tokenizer is present, else text string.
        """
        if not text:
            text = "no clinical information available."
        if self.tokenizer:
            return self.tokenizer([text]).squeeze(0)
        return text

    def __getitem__(self, idx: int):
        """Retrieve a sample by index.

        Args:
            idx: Integer index into dataset.

        Returns:
            A dictionary with global image, local tiles, normalized tile boxes,
            dual-report token IDs, and labels when high-resolution mode is
            enabled. Otherwise returns the backward-compatible tuple
            ``(image, xray_ids, clinical_ids, labels)``.
        """
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])

        # --- Load image ---
        img_path = os.path.join(self.img_dir, image_id)
        if self.text_only:
            image = Image.new("RGB", (224, 224), color="black")
        else:
            try:
                image = Image.open(img_path).convert("RGB")
            except FileNotFoundError:
                image = Image.new("RGB", (224, 224), color="black")

        if self.use_high_res:
            cache_key = "__text_only__" if self.text_only else image_id
            cached_selection = self._high_res_selection_cache.get(cache_key)
            high_res_fields, selection = prepare_high_resolution_inputs(
                image,
                self.transform,
                self.high_res_cfg,
                selection=cached_selection,
                return_selection=True,
            )
            if self.cache_high_res_selection and cached_selection is None:
                self._high_res_selection_cache[cache_key] = selection
        else:
            high_res_fields = {}
            image = prepare_global_image(
                image,
                self.transform,
                self.preprocess_cfg,
            )

        # --- Load and tokenize dual text reports ---
        report_image_id = image_id
        if self.shuffled_report_indices is not None:
            report_image_id = str(
                self.df.iloc[self.shuffled_report_indices[idx]]["image_id"]
            )

        xray_text = self._load_report(self.xray_report_dir, report_image_id)
        clinical_text = self._load_report(
            self.clinical_report_dir, report_image_id
        )

        xray_ids = self._tokenize(xray_text)
        clinical_ids = self._tokenize(clinical_text)

        # --- Encode labels ---
        if self.task_type == "multiclass":
            if "class_id" in row:
                labels = torch.tensor(int(row["class_id"]), dtype=torch.long)
            else:
                class_vals = [int(row.get(c, 0)) for c in self.classes]
                class_id = class_vals.index(1) if 1 in class_vals else 0
                labels = torch.tensor(class_id, dtype=torch.long)
        else:
            # Multi-label: binary vector
            label_vals = []
            for path in self.classes:
                val = row.get(path, 0)
                label_vals.append(0.0 if pd.isna(val) else float(val))
            labels = torch.tensor(label_vals, dtype=torch.float32)

        if self.use_high_res:
            return {
                **high_res_fields,
                "xray_input_ids": xray_ids,
                "clinical_input_ids": clinical_ids,
                "labels": labels,
            }
        return image, xray_ids, clinical_ids, labels
