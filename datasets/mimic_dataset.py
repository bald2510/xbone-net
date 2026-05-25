import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class MimicCxrDataset(Dataset):
    def __init__(
        self, 
        img_dir: str, 
        report_dir: str,
        csv_split_path: str,
        csv_chexpert_path: str,
        pathologies: list,
        split: str = "train",
        train_ratio: float = 1.0,  # THÊM MỚI: Nhận train_ratio từ config
        transform=None,
        tokenizer=None,
        uncertainty_strategy="U-Ones",
        max_text_len=256
    ):
        self.img_dir = img_dir
        self.report_dir = report_dir
        self.pathologies = pathologies
        self.transform = transform
        self.tokenizer = tokenizer
        self.uncertainty_strategy = uncertainty_strategy
        self.max_text_len = max_text_len

        # Đọc dữ liệu gốc
        df_split = pd.read_csv(csv_split_path)
        df_chexpert = pd.read_csv(csv_chexpert_path)
        df_merged = pd.merge(df_split, df_chexpert, on=["subject_id", "study_id"], how="left")

        # Chuẩn hóa tên tập validate
        current_split = 'validate' if split == 'val' else split
        
        # Lọc theo tập dữ liệu hiện tại (train/validate/test)
        filtered_df = df_merged[df_merged['split'] == current_split].reset_index(drop=True)

        # =========================================================
        # TIẾN HÀNH LẤY MẪU CUỐN CHIẾU (CHỈ ÁP DỤNG CHO TẬP TRAIN)
        # =========================================================
        if current_split == 'train' and train_ratio < 1.0:
            # random_state=42 giúp cố định 10% này ở mọi lần bạn chạy lại code
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
        """
        text = text.upper()
        idx_findings = text.find("FINDINGS:")
        idx_impression = text.find("IMPRESSION:")
        
        extracted_text = ""
        if idx_findings != -1:
            extracted_text += text[idx_findings:] + " "
        elif idx_impression != -1:
            extracted_text += text[idx_impression:]
        else:
            lines = text.split('\n')
            extracted_text = " ".join(lines[-(len(lines)//3):])
            
        return extracted_text.strip().lower()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. XỬ LÝ ĐƯỜNG DẪN DỮ LIỆU
        subject_id = str(row['subject_id'])
        study_id = str(row['study_id'])
        dicom_id = str(row['dicom_id'])
        
        p_folder = f"p{subject_id[:2]}"
        img_path = os.path.join(self.img_dir, p_folder, f"p{subject_id}", f"s{study_id}", f"{dicom_id}.jpg")
        report_path = os.path.join(self.report_dir, p_folder, f"p{subject_id}", f"s{study_id}.txt")
        
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
            with open(report_path, 'r') as f:
                raw_text = f.read()
                
        cleaned_text = self._clean_report(raw_text)
        
        # Nếu chưa có text, điền câu giả định
        if not cleaned_text:
            cleaned_text = "no clear findings or impression."
            
        if self.tokenizer:
            # SỬA LỖI TẠI ĐÂY:
            # Tokenizer của open_clip chỉ cần nhận một string hoặc list các strings.
            # Nó sẽ tự động padding và truncate về độ dài mặc định (thường là 256 cho BioMedCLIP)
            # Đầu ra mặc định đã là torch.Tensor kích thước [1, context_length]
            text_inputs = self.tokenizer([cleaned_text])
            
            # Hàm squeeze(0) để loại bỏ chiều batch_size, đưa tensor từ [1, L] về [L]
            input_ids = text_inputs.squeeze(0)
        else:
            input_ids = cleaned_text
            
        # 4. XỬ LÝ NHÃN (LABELS)
        labels = []
        for path in self.pathologies:
            val = row[path]
            if pd.isna(val):
                labels.append(0.0)
            elif val == -1.0:
                labels.append(1.0 if self.uncertainty_strategy == "U-Ones" else 0.0)
            else:
                labels.append(float(val))
                
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return image, input_ids, labels