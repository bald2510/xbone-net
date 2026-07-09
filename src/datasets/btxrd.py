"""
BTXRD (Bone Tumor X-Ray Dataset) loader for XBone-Net.
===============================================================================
Provides a PyTorch Dataset for pairing bone tumor X-ray images with text reports:
  - Dual-report mode: separate X-ray findings (xray/) and clinical history (clinical/)
  - Single-report mode: a single report per image file
  - Multi-class classification: mutually exclusive integer labels via class_id
  - Multi-label classification: binary vector over target pathology list

Reads split assignments and labels from CSV files merged on image_id.
"""

import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


# ============================================================
# BTXRD Dataset Loader
# ============================================================

class BTXRDDataset(Dataset):
    """Dataset loader for BTXRD bone tumor X-ray dataset.

    Supports both single report and dual report (xray + clinical) formats,
    as well as multi-class and multi-label classification.

    Dual-report mode is auto-detected: if report_dir contains both xray/ and
    clinical/ subdirectories, each sample returns separately tokenized texts
    for the two report types.

    Attributes:
        img_dir: Root directory containing X-ray image files.
        report_dir: Root directory containing text report files.
        classes: List of target pathology / class names.
        task_type: Classification mode ('multiclass' or 'multilabel').
        has_dual_reports: True if both xray/ and clinical/ subdirectories exist.
        df: Filtered pandas.DataFrame for the active data split.

    Example:
        ds = BTXRDDataset(
            img_dir="data/btxrd/images",
            report_dir="data/btxrd/reports",
            csv_split_path="data/btxrd/splits.csv",
            csv_labels_path="data/btxrd/labels.csv",
            classes=["benign", "malignant"],
            task_type="multiclass",
            split="train",
        )
        image, input_ids, labels = ds[0]
    """

    def __init__(
        self, 
        img_dir: str, 
        report_dir: str,
        csv_split_path: str,
        csv_labels_path: str,
        pathologies: list = None,
        classes: list = None,
        task_type: str = "multiclass",
        num_classes: int = None,
        split: str = "train",
        train_ratio: float = 1.0, 
        transform=None,
        tokenizer=None,
        max_text_len=256,
        clinical_subdir: str = "clinical_v2",
        k_shot: int = None,
        seed: int = 42,
        **kwargs,
    ):
        """Initialize the BTXRD dataset.

        Args:
            img_dir: Path to directory containing X-ray image files.
            report_dir: Path to report root directory. If it contains xray/ and
                clinical/ subdirectories, operates in dual-report mode.
            clinical_subdir: Name of the clinical report subdirectory under
                report_dir. Defaults to 'clinical_v2' (sanitized reports).
                Use 'clinical' for original synthetic reports (ablation only).
            csv_split_path: Path to CSV with image_id and split columns.
            csv_labels_path: Path to CSV with image_id and label columns (class_id
                for multi-class, or pathology columns for multi-label).
            pathologies: List of pathology columns for multi-label (alias for classes).
            classes: List of target class / pathology names (takes precedence).
            task_type: 'multiclass' (single integer label) or 'multilabel' (binary vector).
            num_classes: Optional explicit class count for external reference.
            split: Data split ('train', 'val', or 'test').
            train_ratio: Fraction of training samples to keep (0.0 to 1.0).
            transform: torchvision.transforms pipeline for image preprocessing.
            tokenizer: Text tokenizer callable returning token-ID tensors.
            max_text_len: Maximum token sequence length.
            **kwargs: Extra unused arguments for backward compatibility.
        """
        self.img_dir = img_dir
        self.report_dir = report_dir
        self.classes = classes or pathologies or []
        self.task_type = task_type
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

        # --- Load and merge metadata ---
        df_split = pd.read_csv(csv_split_path)
        df_labels = pd.read_csv(csv_labels_path)
        df_merged = pd.merge(df_split, df_labels, on=["image_id"], how="inner")

        # --- Pre-compute class_id for multi-class indexing if missing ---
        if "class_id" not in df_merged.columns and self.classes:
            def _get_class_id(row):
                for idx_cls, cls_name in enumerate(self.classes):
                    if row.get(cls_name, 0) == 1:
                        return idx_cls
                return 0
            df_merged["class_id"] = df_merged.apply(_get_class_id, axis=1)

        # --- Filter split and apply subsampling ---
        current_split = "validate" if split == "val" else split
        filtered_df = df_merged[df_merged["split"] == current_split].reset_index(drop=True)

        if current_split == "train" and k_shot is not None:
            if "class_id" in filtered_df.columns:
                filtered_df = filtered_df.groupby("class_id", group_keys=False).apply(
                    lambda x: x.sample(n=min(len(x), k_shot), random_state=seed)
                ).reset_index(drop=True)
                print(f"[Dataset] Few-shot learning enabled: {k_shot}-shot sampling (seed={seed}).")
            else:
                print(f"[Dataset] Warning: k_shot={k_shot} requested but 'class_id' not found. Ignored.")
        elif current_split == "train" and train_ratio < 1.0:
            filtered_df = filtered_df.sample(
                frac=train_ratio, 
                random_state=seed
            ).reset_index(drop=True)
            print(f"[Dataset] Subsampling enabled: using {train_ratio * 100:.1f}% of training data (seed={seed}).")

        self.df = filtered_df

        # --- Detect dual report directory structure ---
        self.xray_dir = os.path.join(self.report_dir, "xray")
        self.clinical_dir = os.path.join(self.report_dir, clinical_subdir)
        self.has_dual_reports = os.path.isdir(self.xray_dir) and os.path.isdir(self.clinical_dir)

        print(f"[Dataset] Initialized '{split.upper()}' with {len(self.df)} samples. Dual reports: {self.has_dual_reports}")

    def __len__(self):
        """Return the total number of samples in the current split."""
        return len(self.df)

    def _clean_report(self, text: str) -> str:
        """Clean and normalize raw report text by stripping whitespace and lowercasing."""
        return text.strip().lower()

    def _load_and_tokenize(self, path: str, default_text: str):
        """Load a text report file, clean it, and tokenize if tokenizer is present."""
        raw_text = ""
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw_text = f.read()

        cleaned_text = self._clean_report(raw_text) or default_text
        if self.tokenizer:
            return self.tokenizer([cleaned_text]).squeeze(0)
        return cleaned_text

    def __getitem__(self, idx: int):
        """Retrieve a sample by index.

        Args:
            idx: Integer index into dataset.

        Returns:
            tuple: (image, xray_ids, clinical_ids, labels) in dual-report mode, or
            Single-report mode: (image, input_ids, labels)
        """
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        img_path = os.path.join(self.img_dir, image_id)

        file_name_without_ext = os.path.splitext(image_id)[0]
        report_path = os.path.join(self.report_dir, f"{file_name_without_ext}.txt")

        # --- Load image ---
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            image = Image.new("RGB", (224, 224), color="black")

        if self.transform:
            image = self.transform(image)

        # --- Encode labels ---
        if self.task_type == "multiclass":
            if "class_id" in row:
                labels = torch.tensor(int(row["class_id"]), dtype=torch.long)
            else:
                class_vals = [int(row.get(c, 0)) for c in self.classes]
                class_id = class_vals.index(1) if 1 in class_vals else 0
                labels = torch.tensor(class_id, dtype=torch.long)
        else:
            labels = torch.tensor([float(row.get(path, 0.0)) for path in self.classes], dtype=torch.float32)

        # --- Load text reports ---
        if self.has_dual_reports:
            xray_path = os.path.join(self.xray_dir, f"{file_name_without_ext}.txt")
            clinical_path = os.path.join(self.clinical_dir, f"{file_name_without_ext}.txt")

            xray_ids = self._load_and_tokenize(xray_path, "no clear bone abnormalities or fracture identified.")
            clinical_ids = self._load_and_tokenize(clinical_path, "no clinical information available.")

            return image, xray_ids, clinical_ids, labels
        else:
            input_ids = self._load_and_tokenize(report_path, "no clear bone abnormalities or fracture identified.")
            return image, input_ids, labels