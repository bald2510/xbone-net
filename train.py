import os
import torch
import torch.nn as nn
import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW

# Import các builders từ hệ thống Plug-and-Play của bạn
from models.builder import build_model
from datasets.builder import build_dataloader

def seed_everything(seed=42):
    """Cố định random seed để đảm bảo tính tái lập."""
    import random
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    # Thanh tiến trình trực quan cho mỗi batch
    pbar = tqdm(dataloader, desc="  [Train Batch]")
    for images, input_ids, labels in pbar:
        images = images.to(device)
        input_ids = input_ids.to(device)
        labels = labels.to(device) # Shape: [Batch_size, Num_classes]
        
        optimizer.zero_grad()
        
        # Luồng forward qua Composer (Ví dụ: FiLM + Prototypical Head trả về logits)
        logits = model(images, input_ids)
        
        # Sử dụng BCEWithLogitsLoss cho bài toán phân loại đa nhãn (Multi-label)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, input_ids, labels in tqdm(dataloader, desc="  [Val Batch]"):
            images = images.to(device)
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            logits = model(images, input_ids)
            loss = criterion(logits, labels)
            
            running_loss += loss.item()
            
    return running_loss / len(dataloader)

@hydra.main(version_base=None, config_path="configs", config_name="experiment/exp_baseline_crsttn_prototypical")
def main(cfg: DictConfig):
    # 1. In cấu hình kiểm tra
    print("=== CẤU HÌNH THÍ NGHIỆM ĐÃ ĐƯỢC GỘP ===")
    print(OmegaConf.to_yaml(cfg))
    print("========================================\n")

    # 2. Thiết lập môi trường
    seed_everything(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}\n")

    # 3. Lắp ráp mô hình từ Config
    print("Đang lắp ráp mô hình...")
    print(cfg.model.fusion) # In ra config fusion để kiểm tra
    model = build_model(cfg.model).to(device)

    model.print_parameter_summary()

    print("Lắp ráp mô hình thành công!\n")

    # 4. Khởi tạo DataLoaders thực tế (Train & Val)
    print("Đang khởi tạo DataLoaders...")
    preprocess = model.backbone.preprocess
    tokenizer = model.backbone.tokenizer
    
    train_loader = build_dataloader(
        cfg=cfg.dataset, split="train", transform=preprocess, tokenizer=tokenizer
    )
    val_loader = build_dataloader(
        cfg=cfg.dataset, split="val", transform=preprocess, tokenizer=tokenizer
    )
    print("Khởi tạo dữ liệu hoàn tất!\n")

    # 5. Cấu hình các siêu tham số huấn luyện (Hyperparameters)
    # Lấy các tham số từ config, nếu không có sẽ dùng giá trị mặc định phòng hờ
    epochs = cfg.get("epochs", 10)
    lr = cfg.get("lr", 1e-4)
    weight_decay = cfg.get("weight_decay", 1e-2)
    
    # Định nghĩa Loss function cho bài toán phân loại đa nhãn của MIMIC-CXR
    criterion = nn.BCEWithLogitsLoss()

    # Khởi tạo Optimizer (Chỉ tối ưu các tham số được bật requires_grad=True, ví dụ: LoRA, FiLM)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    
    print(f"Bắt đầu huấn luyện với {len(trainable_params)} tensor tham số được tối ưu.")
    print(f"Tổng số Epochs: {epochs} | Learning Rate: {lr}\n")

    # Thư mục lưu checkpoint mô hình (Tự động tạo theo cấu trúc đầu ra của Hydra)
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_val_loss = float('inf')

    # 6. Vòng lặp Huấn luyện chính (Training Loop)
    for epoch in range(1, epochs + 1):
        print(f"=== Epoch [{epoch}/{epochs}] ===")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f" -> Kết quả: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Cơ chế lưu checkpoint lưu vết mô hình tốt nhất (Best Model Save)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f"{cfg.experiment_name}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': OmegaConf.to_container(cfg, resolve=True)
            }, checkpoint_path)
            print(f" [*] Đã lưu mô hình tốt nhất mới tại: {checkpoint_path}")
            
        print("-" * 40)

    print("\nQuá trình huấn luyện hoàn tất thành công!")

if __name__ == "__main__":
    main()