import torch.nn as nn

class XBoneMultiModalModel(nn.Module):
    def __init__(self, backbone, fusion_module, head_module):
        super().__init__()
        # Mô hình chính chỉ làm nhiệm vụ giữ các thành phần
        self.backbone = backbone
        self.fusion = fusion_module
        self.head = head_module

    def forward(self, images, input_ids):
        # 1. Trích xuất đặc trưng
        img_feats, txt_feats = self.backbone(images, input_ids)
        
        # 2. Kết hợp đặc trưng (Nếu config là 'none', nó sẽ chạy IdentityFusion)
        fused_feats = self.fusion(img_feats, txt_feats)
        
        # 3. Phân loại (Nếu là ProtoHead thì trả về logits và cosine_sim)
        logits = self.head(fused_feats)
        
        return logits
    
    def print_parameter_summary(self):
        """
        In ra báo cáo chi tiết về số lượng tham số của mô hình.
        Hữu ích để kiểm tra xem backbone đã được đóng băng đúng cách chưa.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print("\n" + "="*50)
        print("📊 BÁO CÁO THÔNG SỐ MÔ HÌNH (MODEL SUMMARY)")
        print("="*50)
        
        # Sử dụng f-string với format :, để thêm dấu phẩy ngăn cách hàng nghìn cho dễ nhìn
        print(f"Tổng số tham số (Total)       : {total_params:,}")
        print(f"Tham số bị đóng băng (Frozen) : {frozen_params:,}")
        print(f"Tham số huấn luyện (Trainable): {trainable_params:,}")
        print(f"Tỉ lệ huấn luyện              : {(trainable_params / total_params) * 100:.4f}%\n")

        print("--- Chi tiết theo từng Module ---")
        # Quét qua 3 component chính: backbone, fusion, head
        for name, module in self.named_children():
            mod_total = sum(p.numel() for p in module.parameters())
            mod_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            # Chỉ in ra nếu module đó có tham số (tránh in Identity module nếu nó rỗng)
            if mod_total > 0:
                print(f"🔹 {name.upper()}:")
                print(f"   - Tổng      : {mod_total:,}")
                print(f"   - Trainable : {mod_train:,}")
        print("="*50 + "\n")