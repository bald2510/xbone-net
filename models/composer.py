import torch.nn as nn

class XBoneMultiModalModel(nn.Module):
    def __init__(self, backbone, fusion_module, head_module):
        super().__init__()
        # Mô hình chính chỉ làm nhiệm vụ giữ các thành phần
        self.backbone = backbone
        self.fusion = fusion_module
        self.head = head_module

    def forward(self, images, input_ids, return_features=False):
        # 1. Extract features from backbone
        img_feats, txt_feats = self.backbone(images, input_ids)
        
        # 2. Fusion (IdentityFusion when config is 'none')
        if txt_feats is None:
            fused_feats = img_feats
        else:
            fused_feats = self.fusion(img_feats, txt_feats)
        
        # 3. Classification
        if return_features and hasattr(self.head, 'prototypes'):
            logits, features = self.head(fused_feats, return_features=True)
            return logits, features, self.head.prototypes
        
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

    def print_encoder_layers(self):
        """
        In ra cấu trúc chi tiết các lớp (layers) của image encoder và text encoder.
        """
        print("\n" + "="*50)
        print("🖼️ CẤU TRÚC LỚP BỘ MÃ HÓA HÌNH ẢNH (IMAGE ENCODER LAYERS)")
        print("="*50)
        if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "visual"):
            print(self.backbone.model.visual)
        else:
            print("Không tìm thấy bộ mã hóa hình ảnh trong backbone.model.visual")
            
        print("\n" + "="*50)
        print("📝 CẤU TRÚC LỚP BỘ MÃ HÓA VĂN BẢN (TEXT ENCODER LAYERS)")
        print("="*50)
        if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "text"):
            print(self.backbone.model.text)
        else:
            print("Không tìm thấy bộ mã hóa văn bản trong backbone.model.text")
        print("="*50 + "\n")

    def gradient_checkpointing_enable(self, **kwargs):
        """
        Enables gradient checkpointing (activation checkpointing)
        on both the vision (ViT) and text encoders.
        Safely handles models without .text attribute (e.g., OpenCLIP CLIP).
        """
        print("[XBone Model] Enabling gradient checkpointing for memory optimization.")
        # Enable for OpenCLIP visual encoder (uses set_grad_checkpointing)
        if hasattr(self.backbone.model, "set_grad_checkpointing"):
            self.backbone.model.set_grad_checkpointing(True)
        elif hasattr(self.backbone.model, "visual") and hasattr(self.backbone.model.visual, "set_grad_checkpointing"):
            self.backbone.model.visual.set_grad_checkpointing(True)

        # Enable for Hugging Face text encoder (e.g., PubMedBERT in BiomedCLIP)
        text_module = getattr(self.backbone.model, "text", None)
        if text_module is not None:
            text_transformer = getattr(text_module, "transformer", None)
            if text_transformer is not None and hasattr(text_transformer, "gradient_checkpointing_enable"):
                text_transformer.gradient_checkpointing_enable(**kwargs)

        # Required for gradient checkpointing with PEFT LoRA adapters
        if hasattr(self.backbone.model, "visual"):
            visual = self.backbone.model.visual
            if hasattr(visual, "enable_input_require_grads"):
                visual.enable_input_require_grads()
        if text_module is not None:
            text_transformer = getattr(text_module, "transformer", None)
            if text_transformer is not None and hasattr(text_transformer, "enable_input_require_grads"):
                text_transformer.enable_input_require_grads()
