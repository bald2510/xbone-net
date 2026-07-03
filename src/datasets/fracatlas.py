"""
FracAtlas dataset loader for XBone-Net architecture.
============================================================
Provides a PyTorch Dataset for the FracAtlas musculoskeletal X-ray dataset.
Used in the XBone-Net evaluation framework:
  - Out-of-Distribution (OOD) evaluation: tests generalization of BTXRD-trained VLM models
  - OOD detection via Mahalanobis distance feature statistics
  - Single-report contrastive text pairing
  - Multi-class (binary integer label) and multi-label target encoding
"""

import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


# ============================================================
# FracAtlas Dataset Loader
# ============================================================

class FracAtlasDataset(Dataset):
    """Dataset loader for the FracAtlas fracture dataset.

    Returns (image, text_report, label) per sample. Operates in single-report mode
    for contrastive evaluation and OOD detection in XBone-Net.

    Attributes:
        img_dir: Root directory containing X-ray image files.
        report_dir: Directory containing text report files (.txt).
        classes: List of target class names (defaults to ['fractured']).
        task_type: Classification mode ('multiclass' or 'multilabel').
        df: Filtered pandas.DataFrame for the active data split.

    Example:
        ds = FracAtlasDataset(
            img_dir="data/fracatlas/images",
            report_dir="data/fracatlas/reports",
            csv_split_path="data/fracatlas/splits.csv",
            csv_labels_path="data/fracatlas/labels.csv",
            split="test",
        )
        image, input_ids, labels = ds[0]
    """

    def __init__(
        self, 
        img_dir: str, 
        report_dir: str,
        csv_split_path: str,
        csv_labels_path: str,
        classes: list = None,
        task_type: str = "multiclass",
        split: str = "train",
        transform=None,
        tokenizer=None,
        max_text_len=256,
        **kwargs,
    ):
        """Initialize the FracAtlas dataset.

        Args:
            img_dir: Path to directory containing X-ray image files.
            report_dir: Path to directory containing report .txt files.
            csv_split_path: Path to CSV with image_id and split columns.
            csv_labels_path: Path to CSV with image_id and fractured label columns.
            classes: List of target class names. Defaults to ['fractured'].
            task_type: 'multiclass' (scalar integer label) or 'multilabel' (float vector).
            split: Data split ('train', 'val', or 'test').
            transform: torchvision.transforms pipeline for image preprocessing.
            tokenizer: Text tokenizer callable returning token-ID tensors.
            max_text_len: Maximum token sequence length.
            **kwargs: Extra unused arguments for backward compatibility.
        """
        self.img_dir = img_dir
        self.report_dir = report_dir
        self.classes = classes or ['fractured']
        self.task_type = task_type
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

        # --- Load and merge metadata ---
        df_split = pd.read_csv(csv_split_path)
        df_labels = pd.read_csv(csv_labels_path)
        df_merged = pd.merge(df_split, df_labels, on=["image_id"], how="inner")
        
        # --- Filter split ---
        current_split = 'validate' if split == 'val' else split
        self.df = df_merged[df_merged['split'] == current_split].reset_index(drop=True)
        
        print(f"[FracAtlasDataset] Loaded '{split.upper()}' split with {len(self.df)} samples.")

    def __len__(self):
        """Return the total number of samples in the current split."""
        return len(self.df)
        
    def _clean_report(self, text: str) -> str:
        """Clean and normalize raw report text by stripping whitespace and lowercasing."""
        return text.strip().lower()

    def _load_and_tokenize(self, path: str, default_text: str):
        """Load a text report from disk, clean it, and tokenize if tokenizer is set.

        Args:
            path: Absolute path to the .txt report file.
            default_text: Fallback string used when report is missing or empty.

        Returns:
            torch.Tensor of token IDs if tokenizer is present, else text string.
        """
        raw_text = ""
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
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
            tuple: (image, input_ids, labels) where:
                - image: Transformed image tensor
                - input_ids: Tokenized report text tensor or raw string
                - labels: torch.long scalar (multiclass) or float vector (multilabel)
        """
        row = self.df.iloc[idx]
        image_id = str(row['image_id'])
        img_path = os.path.join(self.img_dir, image_id)
        
        file_name_without_ext = os.path.splitext(image_id)[0]
        report_path = os.path.join(self.report_dir, f"{file_name_without_ext}.txt")
        
        # --- Load image ---
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), color='black')
            
        if self.transform:
            image = self.transform(image)
            
        # --- Encode label ---
        if self.task_type == "multiclass":
            labels = torch.tensor(int(row['fractured']), dtype=torch.long)
        else:
            labels = torch.tensor([float(row.get('fractured', 0.0))], dtype=torch.float32)

        # --- Load text report ---
        input_ids = self._load_and_tokenize(report_path, "no fracture identified.")
        return image, input_ids, labels


