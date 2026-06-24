import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class BTXRDDataset(Dataset):
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
        **kwargs,  # absorb extra config keys (class_to_parent, etc.)
    ):
        self.img_dir = img_dir
        self.report_dir = report_dir
        self.classes = classes or pathologies or []
        self.task_type = task_type
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

        # Đọc dữ liệu gốc
        df_split = pd.read_csv(csv_split_path)
        df_labels = pd.read_csv(csv_labels_path)
        
        # SỬA: Hợp nhất dựa trên 'image_id' thay vì subject_id/study_id
        df_merged = pd.merge(df_split, df_labels, on=["image_id"], how="inner")

        # Chuẩn hóa tên tập validate
        current_split = 'validate' if split == 'val' else split
        
        # Lọc theo tập dữ liệu hiện tại (train/validate/test)
        filtered_df = df_merged[df_merged['split'] == current_split].reset_index(drop=True)

        # =========================================================
        # TIẾN HÀNH LẤY MẪU CUỐN CHIẾU (CHỈ ÁP DỤNG CHO TẬP TRAIN)
        # =========================================================
        if current_split == 'train' and train_ratio < 1.0:
            filtered_df = filtered_df.sample(
                frac=train_ratio, 
                random_state=42
            ).reset_index(drop=True)
            print(f"[Dataset Warning] Đã kích hoạt chế độ Subsampling! Chỉ sử dụng {train_ratio*100}% tập Train.")

        self.df = filtered_df
        
        # Check for dual reports subdirectories (xray and clinical)
        self.xray_dir = os.path.join(self.report_dir, "xray")
        self.clinical_dir = os.path.join(self.report_dir, "clinical")
        self.has_dual_reports = os.path.isdir(self.xray_dir) and os.path.isdir(self.clinical_dir)
        
        print(f"[Dataset] Khởi tạo thành công tập '{split.upper()}' với {len(self.df)} mẫu. Dual reports: {self.has_dual_reports}")

    def __len__(self):
        return len(self.df)
        
    def _clean_report(self, text: str) -> str:
        """
        Hàm tiền xử lý text cơ bản.
        SỬA: Xóa bỏ khoảng trắng thừa và chuyển về chữ thường.
        """
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
        
        # Load and transform image
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), color='black')
            
        if self.transform:
            image = self.transform(image)
            
        # 1. XỬ LÝ NHÃN (LABELS)
        if self.task_type == "multiclass":
            labels = torch.tensor(int(row['class_id']), dtype=torch.long)
        else:
            labels = torch.tensor([float(row.get(path, 0.0)) for path in self.classes], dtype=torch.float32)

        # 2. LOAD VÀ TOKENIZE VĂN BẢN (TEXT)
        if self.has_dual_reports:
            xray_path = os.path.join(self.xray_dir, f"{file_name_without_ext}.txt")
            clinical_path = os.path.join(self.clinical_dir, f"{file_name_without_ext}.txt")
            
            xray_ids = self._load_and_tokenize(xray_path, "no clear bone abnormalities or fracture identified.")
            clinical_ids = self._load_and_tokenize(clinical_path, "no clinical information available.")
            
            return image, xray_ids, clinical_ids, labels
        else:
            input_ids = self._load_and_tokenize(report_path, "no clear bone abnormalities or fracture identified.")
            return image, input_ids, labels