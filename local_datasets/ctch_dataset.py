import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CTCHDataset(Dataset):
    """
    CTCH dataset with dual report support (X-ray + Clinical).

    Returns (image, xray_text, clinical_text, labels) per sample.
    - xray_text: imaging findings report (for Phase 1 contrastive learning)
    - clinical_text: patient history report (for Phase 2 classification)

    If only one report_dir is provided (via `report_dir`), both text outputs
    will contain the same text (backward compatible with BTXRD-style datasets).
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
        self.img_dir = img_dir
        self.pathologies = pathologies
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

        # Handle report directories
        if report_dir and not xray_report_dir and not clinical_report_dir:
            # Backward compatible: single report_dir → use for both
            self.xray_report_dir = report_dir
            self.clinical_report_dir = report_dir
        else:
            self.xray_report_dir = xray_report_dir
            self.clinical_report_dir = clinical_report_dir

        # Load and merge labels + splits
        df_split = pd.read_csv(csv_split_path)
        df_labels = pd.read_csv(csv_labels_path)
        df_merged = pd.merge(df_split, df_labels, on=["image_id"], how="inner")

        # Normalize split name
        current_split = "validate" if split == "val" else split
        filtered_df = df_merged[df_merged["split"] == current_split].reset_index(drop=True)

        # Optional subsampling for train
        if current_split == "train" and train_ratio < 1.0:
            filtered_df = filtered_df.sample(
                frac=train_ratio, random_state=42
            ).reset_index(drop=True)

        self.df = filtered_df
        print(f"[Dataset] CTCH '{split.upper()}' initialized with {len(self.df)} samples.")

    def __len__(self):
        return len(self.df)

    def _load_report(self, report_dir, image_id):
        """Load a text report from a directory."""
        if report_dir is None:
            return ""
        file_name = os.path.splitext(image_id)[0]
        report_path = os.path.join(report_dir, f"{file_name}.txt")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                return f.read().strip().lower()
        return ""

    def _tokenize(self, text):
        """Tokenize text, with fallback for empty strings."""
        if not text:
            text = "no clinical information available."
        if self.tokenizer:
            return self.tokenizer([text]).squeeze(0)
        return text

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])

        # 1. Load image
        img_path = os.path.join(self.img_dir, image_id)
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            image = Image.new("RGB", (224, 224), color="black")

        if self.transform:
            image = self.transform(image)

        # 2. Load and tokenize both report types
        xray_text = self._load_report(self.xray_report_dir, image_id)
        clinical_text = self._load_report(self.clinical_report_dir, image_id)

        xray_ids = self._tokenize(xray_text)
        clinical_ids = self._tokenize(clinical_text)

        # 3. Labels (multi-label)
        labels = []
        for path in self.pathologies:
            val = row.get(path, 0)
            labels.append(0.0 if pd.isna(val) else float(val))
        labels = torch.tensor(labels, dtype=torch.float32)

        return image, xray_ids, clinical_ids, labels
