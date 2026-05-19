import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

# Import model của bạn (dựa trên cấu trúc file build_model.py đã cập nhật)
from src.models.build_model import XBoneMultiModalModel

from dataset import XBoneDataset
from torch.utils.data import DataLoader

# Cấu hình Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 1. Khởi tạo Model
model = XBoneMultiModalModel(config).to(device)

# 2. Trích xuất hàm tiền xử lý từ Backbone để đưa cho Dataset
# Lưu ý: Vì model.backbone chứa class BiomedCLIPBackbone, ta gọi thẳng các thuộc tính của nó
clip_preprocess = model.backbone.preprocess
clip_tokenizer = model.backbone.tokenizer

# 3. Khởi tạo Dataset và DataLoader
train_dataset = XBoneDataset(
    json_path='data/CTCH/Unified_Results.json',
    image_dir='data/images/',
    preprocess_fn=clip_preprocess,
    tokenizer_fn=clip_tokenizer,
    is_train=True
)

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=4, # Dùng 4 CPU cores để chạy preprocess_fn song song
    pin_memory=True # Tăng tốc độ chuyển dữ liệu từ CPU sang GPU
)

def train_phase_1_alignment(model, dataloader, optimizer, device, epoch):
    """
    PHA 1: Căn chỉnh Không gian (Alignment)
    Mục tiêu: Dạy LoRA kéo vector Ảnh X-quang về sát với vector UMLS Text.
    """
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Phase 1 - Epoch {epoch}")

    for batch in progress_bar:
        # Lấy dữ liệu (giả định dataloader trả về images và input_ids)
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)

        optimizer.zero_grad()

        # Forward pass với phase="alignment" (Chỉ chạy qua Backbone)
        loss = model(images, input_ids, phase="alignment")

        # Backward & Optimize (Chỉ cập nhật trọng số LoRA)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    logger.info(f"[Phase 1] Epoch {epoch} - Avg Alignment Loss: {avg_loss:.4f}")
    return avg_loss

def train_phase_2_tuning(model, dataloader, optimizer, device, epoch):
    """
    PHA 2: Huấn luyện Dung hợp (Tuning)
    Mục tiêu: Dạy FiLM cách điều chế ảnh bằng text và cập nhật Prototypes.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(dataloader, desc=f"Phase 2 - Epoch {epoch}")

    for batch in progress_bar:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Forward pass với phase="tuning" (Chạy qua FiLM và Prototypical Head)
        logits = model(images, input_ids, phase="tuning")

        # Tính Cross Entropy Loss dựa trên Logits (Cosine Similarity đã scale)
        loss = F.cross_entropy(logits, labels)

        # Backward & Optimize (Chỉ cập nhật FiLM và Prototypes)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Tính Accuracy
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(loss=loss.item(), acc=(correct/total)*100)

    avg_loss = total_loss / len(dataloader)
    acc = (correct / total) * 100
    logger.info(f"[Phase 2] Epoch {epoch} - Avg Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")
    return avg_loss, acc

def main():
    # 1. Cấu hình cơ bản
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        'model': {
            'config_path': 'path/to/biomedclip_config.json',
            'checkpoint_path': 'path/to/pytorch_model.bin',
            'num_classes': 10 # Thay đổi theo số lượng bệnh lý xương của bạn
        }
    }

    # 2. Khởi tạo Model
    logger.info("Đang khởi tạo mô hình đa phương thức...")
    model = XBoneMultiModalModel(config).to(device)

    # 3. Khởi tạo DataLoader (Bạn tự thay thế bằng Dataset của bạn)
    # train_dataset = YourXBoneDataset(...)
    # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    train_dataloader = [] # Placeholder

    # =========================================================================
    # CHẠY PHA 1: ALIGNMENT (Ví dụ: 10 Epochs)
    # =========================================================================
    logger.info("BẮT ĐẦU PHA 1: ALIGNMENT LORA...")
    
    # Chỉ đưa các tham số có requires_grad=True (chính là LoRA) vào Optimizer
    phase1_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer_phase1 = optim.AdamW(phase1_params, lr=1e-4, weight_decay=0.01)

    num_epochs_phase1 = 10
    for epoch in range(1, num_epochs_phase1 + 1):
        train_phase_1_alignment(model, train_dataloader, optimizer_phase1, device, epoch)
        # Khuyên dùng: Lưu checkpoint sau Phase 1 ở đây

    # =========================================================================
    # CHUYỂN GIAO PHA (TRANSITION)
    # =========================================================================
    logger.info("ĐANG THỰC HIỆN CHUYỂN GIAO PHA...")
    
    # 1. Merge trọng số LoRA vào thẳng Backbone của Image Encoder để tăng tốc suy diễn
    # Lưu ý: Lệnh này phụ thuộc vào cấu trúc peft, nếu bạn áp dụng peft cho model.visual
    model.backbone.model.visual = model.backbone.model.visual.merge_and_unload()
    
    # 2. Đóng băng toàn bộ Backbone (Không cho cập nhật gradient nữa)
    for param in model.backbone.parameters():
        param.requires_grad = False
    logger.info("Đã merge LoRA và đóng băng toàn bộ BioMedCLIP Backbone.")

    # =========================================================================
    # CHẠY PHA 2: TUNING (Ví dụ: 20 Epochs)
    # =========================================================================
    logger.info("BẮT ĐẦU PHA 2: TUNING FiLM & PROTOTYPES...")
    
    # Ở Pha 2, ta chỉ cập nhật trọng số của khối FiLM và mạng Prototype
    phase2_params = list(model.fusion_layer.parameters()) + list(model.classifier.parameters())
    
    # Dùng Learning rate nhỏ hơn cho Pha 2 để tinh chỉnh an toàn
    optimizer_phase2 = optim.AdamW(phase2_params, lr=5e-5, weight_decay=0.05)

    num_epochs_phase2 = 20
    for epoch in range(1, num_epochs_phase2 + 1):
        train_phase_2_tuning(model, train_dataloader, optimizer_phase2, device, epoch)
        # Khuyên dùng: Đánh giá trên tập Validation và lưu mô hình tốt nhất ở đây

    logger.info("🎉 HOÀN TẤT TOÀN BỘ QUÁ TRÌNH HUẤN LUYỆN!")

if __name__ == "__main__":
    main()