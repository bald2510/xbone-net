import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class FracAtlasDataset(Dataset):
    def __init__(
        self, 
        img_dir: str, 
        report_dir: str,
        csv_split_path: str,
        csv_labels_path: str,
        pathologies: list,
        split: str = "train",
        train_ratio: float = 1.0, 
        transform=None,
        tokenizer=None,
        max_text_len=256
    ):
        self.img_dir = img_dir
        self.report_dir = report_dir
        self.pathologies = pathologies
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
        print(f"[Dataset] Khởi tạo thành công tập '{split.upper()}' với {len(self.df)} mẫu.")

    def __len__(self):
        return len(self.df)
        
    def _clean_report(self, text: str) -> str:
        """
        Hàm tiền xử lý text cơ bản.
        SỬA: Xóa bỏ khoảng trắng thừa và chuyển về chữ thường.
        """
        return text.strip().lower()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. XỬ LÝ ĐƯỜNG DẪN DỮ LIỆU
        # SỬA: Trực tiếp lấy image_id thay vì phân tách subject/study/dicom
        image_id = str(row['image_id'])
        
        img_path = os.path.join(self.img_dir, image_id)
        
        # Tạo tên file báo cáo bằng cách bỏ đuôi ảnh (vd: .jpeg) và thay bằng .txt
        file_name_without_ext = os.path.splitext(image_id)[0]
        report_path = os.path.join(self.report_dir, f"{file_name_without_ext}.txt")
        
        # 2. LOAD VÀ BIẾN ĐỔI ẢNH (IMAGE)
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), color='black')
            
        if self.transform:
            image = self.transform(image)
            
        # ==========================================
        # 3. LOAD VÀ TOKENIZE VĂN BẢN (TEXT)
        # ==========================================
        raw_text = ""
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
                
        cleaned_text = self._clean_report(raw_text)
    
        # Nếu chưa có text, điền câu giả định
        if not cleaned_text:
            cleaned_text = "no clear bone abnormalities or fracture identified."
            
        if self.tokenizer:
            text_inputs = self.tokenizer([cleaned_text])
            # Hàm squeeze(0) để loại bỏ chiều batch_size, đưa tensor từ [1, L] về [L]
            input_ids = text_inputs.squeeze(0)
        else:
            input_ids = cleaned_text
            
        # 4. XỬ LÝ NHÃN (LABELS)
        labels = []
        for path in self.pathologies:
            val = row[path]
            # SỬA: Xóa bỏ logic xét nhãn -1.0 của CheXpert, vì BTXRD chỉ có 0 và 1
            if pd.isna(val):
                labels.append(0.0)
            else:
                labels.append(float(val))
                
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return image, input_ids, labels