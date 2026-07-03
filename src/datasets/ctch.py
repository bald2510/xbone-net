"""
CTCH pediatric bone fracture dataset loader for XBone-Net architecture.
============================================================
Provides a PyTorch Dataset for pediatric X-ray images paired with text reports.
Used in XBone-Net's dual-stage learning framework:
  - Stage 1: Contrastive image-text alignment using X-ray findings reports
  - Stage 2: Cross-attention multimodal fusion combining X-ray findings + clinical history
  - Multi-label classification: binary target vector over pediatric fracture pathologies

Supports dual-report loading (xray_report_dir and clinical_report_dir) with fallback
to a single report directory for backward compatibility.
"""

import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


# ============================================================
# CTCH Dataset Loader
# ============================================================

class CTCHDataset(Dataset):
    """Dataset loader for the CTCH pediatric bone fracture dataset.

    Always operates in dual-report mode, returning separate imaging findings
    and clinical history texts per sample for multimodal fusion in XBone-Net.

    Attributes:
        img_dir: Root directory containing pediatric X-ray image files.
        pathologies: List of target pathology / class names for multi-label binary vectors.
        xray_report_dir: Directory containing X-ray findings reports (.txt).
        clinical_report_dir: Directory containing clinical history reports (.txt).
        transform: torchvision.transforms pipeline applied to images.
        tokenizer: Text tokenizer callable returning token-ID tensors.
        max_text_len: Maximum token sequence length.
        df: Filtered pandas.DataFrame for the active data split.

    Example:
        ds = CTCHDataset(
            img_dir="data/CTCH/images",
            csv_split_path="data/CTCH/splits.csv",
            csv_labels_path="data/CTCH/labels.csv",
            pathologies=["fracture", "periosteal_reaction"],
            xray_report_dir="data/CTCH/xray_reports",
            clinical_report_dir="data/CTCH/clinical_reports",
            split="train",
        )
        image, xray_ids, clinical_ids, labels = ds[0]
    """

    def __init__(
        self,
        img_dir: str,
        csv_split_path: str,
        csv_labels_path: str,
        pathologies: list,
        split: str = "train",
        train_ratio: float = 1.0,
        transform=None,
        tokenizer=None,
        max_text_len=256,
        # Dual report support
        xray_report_dir: str = None,
        clinical_report_dir: str = None,
        # Backward compatibility: single report_dir
        report_dir: str = None,
        **kwargs,
    ):
        """Initialize the CTCH dataset.

        Args:
            img_dir: Path to directory containing image files.
            csv_split_path: Path to CSV with image_id and split columns.
            csv_labels_path: Path to CSV with image_id and pathology label columns.
            pathologies: List of pathology column names in labels CSV.
            split: Data split ('train', 'val', or 'test').
            train_ratio: Fraction of training samples to keep (0.0 to 1.0).
            transform: torchvision.transforms pipeline for image preprocessing.
            tokenizer: Text tokenizer callable returning token-ID tensors.
            max_text_len: Maximum token sequence length.
            xray_report_dir: Path to directory containing X-ray findings reports.
            clinical_report_dir: Path to directory containing clinical history reports.
            report_dir: Single report directory for both report types (backward compatibility).
            **kwargs: Extra unused arguments for backward compatibility.
        """
        self.img_dir = img_dir
        self.pathologies = pathologies
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

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

        # --- Filter split and apply subsampling ---
        current_split = "validate" if split == "val" else split
        filtered_df = df_merged[df_merged["split"] == current_split].reset_index(drop=True)

        if current_split == "train" and train_ratio < 1.0:
            filtered_df = filtered_df.sample(
                frac=train_ratio, random_state=42
            ).reset_index(drop=True)

        self.df = filtered_df
        print(f"[Dataset] CTCH '{split.upper()}' initialized with {len(self.df)} samples.")

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
            tuple: (image, xray_ids, clinical_ids, labels) where:
                - image: Transformed image tensor
                - xray_ids: Tokenized X-ray findings report tensor
                - clinical_ids: Tokenized clinical history report tensor
                - labels: torch.FloatTensor multi-label binary vector
        """
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])

        # --- Load image ---
        img_path = os.path.join(self.img_dir, image_id)
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            image = Image.new("RGB", (224, 224), color="black")

        if self.transform:
            image = self.transform(image)

        # --- Load and tokenize dual text reports ---
        xray_text = self._load_report(self.xray_report_dir, image_id)
        clinical_text = self._load_report(self.clinical_report_dir, image_id)

        xray_ids = self._tokenize(xray_text)
        clinical_ids = self._tokenize(clinical_text)

        # --- Encode multi-label binary vector ---
        labels = []
        for path in self.pathologies:
            val = row.get(path, 0)
            labels.append(0.0 if pd.isna(val) else float(val))
        labels = torch.tensor(labels, dtype=torch.float32)

        return image, xray_ids, clinical_ids, labels


