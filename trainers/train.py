import os
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
from pathlib import Path
import sys
import glob
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)

# Import model của bạn
from src.models.build_model import XBoneMultiModalModel

# Cấu hình Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================================================================
# DATASET CHO MIMIC-CXR
# =========================================================================
class MIMICCXRDataset(Dataset):
    def __init__(self, df, image_dir, prompt_dir, preprocess_fn, tokenizer_fn, is_train=True):
        """
        df: DataFrame chứa các cột ['subject_id', 'study_id'] và các cột Labels.
        image_dir: Thư mục gốc chứa ảnh (vd: data/images)
        prompt_dir: Thư mục chứa các file txt prompt đã sinh
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.prompt_dir = prompt_dir
        self.preprocess_fn = preprocess_fn
        self.tokenizer_fn = tokenizer_fn
        self.is_train = is_train

        # Tự động lấy danh sách các cột label (bỏ qua các cột ID)
        self.label_cols = [col for col in self.df.columns if col not in ['subject_id', 'study_id']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Format ID: p10000032, s50414267
        subject_id = f"p{int(row['subject_id'])}"
        study_id = f"s{int(row['study_id'])}"

        # ==========================================
        # 1. XỬ LÝ ẢNH (Quét thư mục tự động)
        # ==========================================
        # Đường dẫn tới thư mục chứa các ảnh của lần chụp đó
        # Dựa vào ảnh chụp của bạn: images/p10000032/s50414267/
        study_dir = os.path.join(self.image_dir, subject_id, study_id)
        
        # (Dự phòng) Nếu thư mục prompt của bạn vẫn giữ cấu trúc cũ có p10
        folder_p = subject_id[:3] 

        # Tìm TẤT CẢ các file .jpg trong thư mục s50414267
        jpg_files = glob.glob(os.path.join(study_dir, "*.jpg"))

        if len(jpg_files) > 0:
            # Nếu có nhiều ảnh (thẳng/nghiêng), tạm thời lấy ảnh ĐẦU TIÊN (index 0)
            # Mẹo: Nếu is_train=True, bạn có thể dùng import random; random.choice(jpg_files) để tăng data augmentation
            img_path = jpg_files[0]
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = self.preprocess_fn(image)
            except Exception as e:
                logger.error(f"Lỗi đọc ảnh {img_path}: {e}")
                image_tensor = torch.zeros((3, 224, 224)) 
        else:
            # Fallback nếu thư mục rỗng hoặc không tồn tại
            # logger.warning(f"Không tìm thấy ảnh nào tại: {study_dir}")
            image_tensor = torch.zeros((3, 224, 224))

        # ==========================================
        # 2. XỬ LÝ TEXT PROMPT
        # ==========================================
        # Tuỳ thuộc vào việc thư mục prompt của bạn CÓ hay KHÔNG CÓ thư mục p10
        # Nếu có p10: os.path.join(self.prompt_dir, folder_p, subject_id, f"{study_id}.txt")
        # Nếu giống hệt ảnh: os.path.join(self.prompt_dir, subject_id, f"{study_id}.txt")
        txt_path = os.path.join(self.prompt_dir, subject_id, f"{study_id}.txt")
        
        # Nếu code chạy báo không tìm thấy prompt, hãy mở comment dòng dưới thay cho dòng trên:
        # txt_path = os.path.join(self.prompt_dir, folder_p, subject_id, f"{study_id}.txt")
        
        prompt_text = "unspecified condition" # Fallback
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                logger.info(f"Đã đọc prompt từ {txt_path}: {content[:50]}...") # In ra 50 ký tự đầu để kiểm tra
                if content and "Không có prompt" not in content:
                    prompt_text = content
                
                
        # Tokenize text
        text_tokens = self.tokenizer_fn(prompt_text)
        if isinstance(text_tokens, torch.Tensor):
            text_tokens = text_tokens.squeeze(0)


        # ==========================================
        # 3. XỬ LÝ LABELS
        # ==========================================
        labels = row[self.label_cols].values.astype(float)
        label_tensor = torch.tensor(labels, dtype=torch.float32)

        
        logger.info(f"Đã lấy nhãn từ DataFrame: {labels}")

        return {
            'image': image_tensor,
            'input_ids': text_tokens,
            'label': label_tensor
        }

# =========================================================================
# VÒNG LẶP HUẤN LUYỆN
# =========================================================================
def train_phase_1_alignment(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Phase 1 - Epoch {epoch}")

    for batch in progress_bar:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        # BỔ SUNG DÒNG NÀY: Lấy nhãn bệnh lý từ batch
        labels = batch['label'].to(device) 

        optimizer.zero_grad()

        # BỔ SUNG `labels=labels`: Truyền nhãn vào model để tính Soft-target Loss
        loss = model(images, input_ids, labels=labels, phase="alignment")

        loss.backward() 
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    logger.info(f"[Phase 1] Epoch {epoch} - Avg Alignment Loss: {avg_loss:.4f}")
    return avg_loss

def train_phase_2_tuning(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Phase 2 - Epoch {epoch}")

    for batch in progress_bar:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device) # Shape: [batch_size, num_classes]

        optimizer.zero_grad()

        # Forward pass với phase="tuning" (Chạy qua FiLM và Prototypes)
        logits = model(images, input_ids, phase="tuning")

        # Sử dụng BCEWithLogitsLoss cho bài toán Multi-label (MIMIC-CXR)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    logger.info(f"[Phase 2] Epoch {epoch} - Avg Loss: {avg_loss:.4f}")
    return avg_loss

# =========================================================================
# HÀM MAIN
# =========================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Đọc Metadata CSV đã merge (Ví dụ file chứa: subject_id, study_id, dicom_id, Pneumonia, Edema,...)
    # Lưu ý: Cần tiền xử lý CSV trước (thay -1 thành 1/0, fillna(0))
    csv_path = r'C:\Users\lebat\Documents\Github\xbone-net\data\19052026\mimic-cxr-2.0.0-chexpert.csv' 
    df = pd.read_csv(csv_path)
    
    # Lấy thử 100 dòng để test code (xoá dòng này khi train thật)
    df = df.head(300) 
    
    # Đếm số lượng bệnh lý (số cột label)
    num_classes = len(df.columns) - 2 # Trừ đi 2 cột ID (subject_id, study_id) 

    config = {
        'model': {
            'config_path': 'path/to/biomedclip_config.json',
            'checkpoint_path': 'path/to/pytorch_model.bin',
            'num_classes': num_classes 
        }
    }

    logger.info("Đang khởi tạo mô hình đa phương thức...")
    model = XBoneMultiModalModel(config).to(device)

    clip_preprocess = model.backbone.preprocess
    clip_tokenizer = model.backbone.tokenizer

    logger.info("Đang tải MIMIC-CXR Dataset...")
    train_dataset = MIMICCXRDataset(
        df=df,
        image_dir=r'C:\Users\lebat\Documents\Github\xbone-net\data\19052026\images',
        prompt_dir=r'C:\Users\lebat\Documents\Github\xbone-net\data\19052026\reports_xray_prompts',
        #prompt_dir=r'C:\Users\lebat\Documents\Github\xbone-net\data\19052026\reports',
        preprocess_fn=clip_preprocess,
        tokenizer_fn=clip_tokenizer,
        is_train=True
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )

    # --- CHẠY PHA 1: ALIGNMENT ---
    logger.info("BẮT ĐẦU PHA 1: ALIGNMENT LORA...")
    phase1_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer_phase1 = optim.AdamW(phase1_params, lr=3e-4, weight_decay=0.01)

    for epoch in range(1, 11):
        train_phase_1_alignment(model, train_dataloader, optimizer_phase1, device, epoch)

    # ==========================================
    # THÊM CODE: LƯU TRỌNG SỐ SAU PHA 1
    # ==========================================
    logger.info("Đang lưu trọng số mô hình sau Phase 1...")
    save_dir = r"C:\Users\lebat\Documents\Github\xbone-net\checkpoints"
    os.makedirs(save_dir, exist_ok=True) # Tự động tạo thư mục nếu chưa có
    
    # Cách 1: Lưu toàn bộ state_dict của mô hình (Khuyên dùng cho custom model)
    phase1_save_path = os.path.join(save_dir, "model_phase1_alignment.pth")
    torch.save(model.state_dict(), phase1_save_path)
    
    # Cách 2 (Tuỳ chọn): Nếu bạn muốn dùng thư viện PEFT để lưu riêng tệp cấu hình LoRA
    # lora_save_path = os.path.join(save_dir, "lora_adapter_phase1")
    # model.backbone.model.visual.save_pretrained(lora_save_path)
    
    logger.info(f"Đã lưu checkpoint Phase 1 tại: {save_dir}")
    # ==========================================

    # # --- CHUYỂN GIAO PHA ---
    # logger.info("ĐANG THỰC HIỆN CHUYỂN GIAO PHA...")
    # # model.backbone.model.visual = model.backbone.model.visual.merge_and_unload()
    # for param in model.backbone.parameters():
    #     param.requires_grad = False
    # logger.info("Đã đóng băng toàn bộ BioMedCLIP Backbone.")

    # # --- CHẠY PHA 2: TUNING ---
    # logger.info("BẮT ĐẦU PHA 2: TUNING FiLM & PROTOTYPES...")
    # phase2_params = list(model.fusion_layer.parameters()) + list(model.classifier.parameters())
    # optimizer_phase2 = optim.AdamW(phase2_params, lr=5e-5, weight_decay=0.05)

    # for epoch in range(1, 21):
    #     train_phase_2_tuning(model, train_dataloader, optimizer_phase2, device, epoch)

    # # (Tuỳ chọn) Bạn cũng có thể copy khối lưu model ở trên xuống đây để lưu thêm trọng số của Pha 2
    # # phase2_save_path = os.path.join(save_dir, "model_phase2_final.pth")
    # # torch.save(model.state_dict(), phase2_save_path)

    logger.info("🎉 HOÀN TẤT TOÀN BỘ QUÁ TRÌNH HUẤN LUYỆN!")

if __name__ == "__main__":
    main()