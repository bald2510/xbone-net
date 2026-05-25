import os
import cv2
import numpy as np
import torch
import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.builder import build_model

# Wrapper để ép luồng Forward nhận 1 input (ảnh) thay vì 2 (ảnh + text) cho thư viện Grad-CAM hiểu
class ModelWrapper(torch.nn.Module):
    def __init__(self, full_model, text_tokens):
        super().__init__()
        self.model = full_model
        self.text_tokens = text_tokens  # Cố định từ khóa truy vấn

    def forward(self, x):
        img_feats = self.model.backbone.model.encode_image(x)
        txt_feats = self.model.backbone.model.encode_text(self.text_tokens)
        fused_feats = self.model.fusion(img_feats, txt_feats)
        logits = self.model.head(fused_feats)
        return logits

@hydra.main(version_base=None, config_path="configs", config_name="experiment/exp_baseline_flim_prototypical")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # 1. KHỞI TẠO VÀ TẢI TRỌNG SỐ
    model = build_model(cfg.model).to(device)
    checkpoint_path = cfg.get("checkpoint_path", "./checkpoints/best_model.pth")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        print("Tải trọng số thành công!")
    else:
        print("[CẢNH BÁO] Không tìm thấy trọng số, mô hình sẽ chạy Zero-shot/Random.")
    model.eval()

    # 2. CHUẨN BỊ TỪ KHÓA TRUY VẤN
    query_text = "Findings consistent with Cardiomegaly" # THAY ĐỔI BỆNH LÝ Ở ĐÂY
    print(f"Truy vấn: '{query_text}'")
    tokenizer = model.backbone.tokenizer
    text_tokens = tokenizer([query_text]).to(device)

    wrapped_model = ModelWrapper(model, text_tokens)

    # 3. CHỈ ĐỊNH LỚP MẠNG VÀ CẤU HÌNH GRAD-CAM
    visual_model = model.backbone.model.visual
    
    # ========================================================
    # [FIX 1]: Bật requires_grad = True để Grad-CAM tính được đạo hàm
    # Lưu ý: Chỉ bật ở đây để vẽ biểu đồ, không ảnh hưởng đến file train
    # ========================================================
    for param in visual_model.parameters():
        param.requires_grad = True

    # Tìm lớp target (Dành cho ViT của thư viện timm/open_clip)
    if hasattr(visual_model, 'trunk') and hasattr(visual_model.trunk, 'blocks'):
        target_layers = [visual_model.trunk.blocks[-1].norm1]
    else:
        target_layers = [list(visual_model.children())[-1]]
        
    print(f"Đã khóa mục tiêu Grad-CAM vào lớp: {target_layers[0].__class__.__name__}")

    # ========================================================
    # [FIX 2]: Hàm biến đổi (Reshape) cấu trúc Transformer thành 2D
    # ViT-Base có 196 patches (14x14) + 1 token CLS = 197 tokens
    # ========================================================
    def vit_reshape_transform(tensor, height=14, width=14):
        # Nếu model trả về đúng ảnh 2D (ví dụ dùng ResNet) thì không cần biến đổi
        if len(tensor.shape) == 4:
            return tensor
            
        # Nếu là ViT: Vứt bỏ token [CLS] (thường ở index 0)
        # Lấy từ index 1 đến hết: [Batch, 196, 768]
        result = tensor[:, 1:, :] 
        
        # Sắp xếp lại thành lưới ảnh 14x14
        result = result.reshape(tensor.size(0), height, width, tensor.size(2))
        
        # Xoay trục Tensor về chuẩn PyTorch: [Batch, Channels, Height, Width]
        result = result.permute(0, 3, 1, 2)
        return result

    # Khởi tạo GradCAM và truyền hàm reshape vào
    cam = GradCAM(
        model=wrapped_model, 
        target_layers=target_layers,
        reshape_transform=vit_reshape_transform
    )

    # 4. CHUẨN BỊ ẢNH ĐẦU VÀO
    # LƯU Ý: Bạn cần sửa đường dẫn này trỏ tới một file ảnh CÓ THẬT trên máy bạn!
    img_path = r"C:\Users\lebat\Documents\Github\xbone-net\data\MIMIC_CXR\imgs\p10\p10001401\s51065211\8061113f-c019f3ae-fd1b7c54-33e8690d-be838099.jpg" 
    
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh tại: {img_path}. Vui lòng sửa lại đường dẫn!")

    rgb_img = cv2.imread(img_path, 1)[:, :, ::-1] # Đọc ảnh và chuyển BGR sang RGB
    rgb_img = cv2.resize(rgb_img, (224, 224))
    
    # Chuẩn hóa về [0, 1] cho mô hình
    input_tensor = torch.from_numpy(np.float32(rgb_img) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    # 5. TÍNH TOÁN GRAD-CAM
    # Giả sử Cardiomegaly là index số 1 trong danh sách 14 bệnh của bạn
    target_category = 1 
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

    # 6. VẼ BẢN ĐỒ NHIỆT VÀ LƯU ẢNH
    input_float_img = np.float32(rgb_img) / 255.0
    cam_image = show_cam_on_image(input_float_img, grayscale_cam, use_rgb=True)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title("Ảnh X-quang gốc")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cam_image)
    plt.title(f"Attention: '{query_text}'")
    plt.axis('off')
    
    plt.tight_layout()
    save_path = "runs/attention_output.png"
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Đã lưu kết quả tại: {save_path}")

if __name__ == "__main__":
    main()