import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class FracAtlasDataset(Dataset):
    """
    FracAtlas dataset wrapper.
    Returns (image, text_report, label) for OOD evaluation.
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
        self.img_dir = img_dir
        self.report_dir = report_dir
        self.classes = classes or ['fractured']
        self.task_type = task_type
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

        # Read original CSVs
        df_split = pd.read_csv(csv_split_path)
        df_labels = pd.read_csv(csv_labels_path)
        
        # Merge on image_id
        df_merged = pd.merge(df_split, df_labels, on=["image_id"], how="inner")
        
        # Split names: train, validate, test
        current_split = 'validate' if split == 'val' else split
        self.df = df_merged[df_merged['split'] == current_split].reset_index(drop=True)
        
        print(f"[FracAtlasDataset] Loaded '{split.upper()}' split with {len(self.df)} samples.")

    def __len__(self):
        return len(self.df)
        
    def _clean_report(self, text: str) -> str:
        return text.strip().lower()

    def _load_and_tokenize(self, path, default_text):
        raw_text = ""
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        cleaned_text = self._clean_report(raw_text) or default_text
        if self.tokenizer:
            return self.tokenizer([cleaned_text]).squeeze(0)
        return cleaned_text

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row['image_id'])
        img_path = os.path.join(self.img_dir, image_id)
        
        file_name_without_ext = os.path.splitext(image_id)[0]
        report_path = os.path.join(self.report_dir, f"{file_name_without_ext}.txt")
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), color='black')
            
        if self.transform:
            image = self.transform(image)
            
        # Standard labels formatting
        if self.task_type == "multiclass":
            labels = torch.tensor(int(row['fractured']), dtype=torch.long)
        else:
            labels = torch.tensor([float(row.get('fractured', 0.0))], dtype=torch.float32)

        # Retrieve tokenized reports or raw string
        input_ids = self._load_and_tokenize(report_path, "no fracture identified.")
        return image, input_ids, labels
