import torch
import torch.nn as nn

class FiLMFusion(nn.Module):
    def __init__(self, text_dim: int = 512, img_dim: int = 512):
        super().__init__()
        # Mạng nơ-ron tuyến tính để dự đoán gamma và beta từ text features
        self.gamma_net = nn.Linear(text_dim, img_dim)
        self.beta_net = nn.Linear(text_dim, img_dim)
        
        # Lớp chuẩn hóa và kích hoạt sau khi điều chế
        self.layer_norm = nn.LayerNorm(img_dim)
        self.relu = nn.ReLU()

        # Khởi tạo trọng số tối ưu ban đầu để không làm hỏng đặc trưng ảnh gốc
        nn.init.zeros_(self.gamma_net.weight)
        nn.init.ones_(self.gamma_net.bias)  # gamma ban đầu gần bằng 1
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)  # beta ban đầu gần bằng 0

    def forward(self, img_feats, txt_feats):
        """
        img_feats: [Batch_size, img_dim]
        txt_feats: [Batch_size, text_dim]
        """
        # Dự đoán hệ số điều chế từ văn bản
        gamma = self.gamma_net(txt_feats) # [Batch_size, img_dim]
        beta = self.beta_net(txt_feats)   # [Batch_size, img_dim]
        
        # Áp dụng công thức FiLM: (FiLM(x) = gamma * x + beta)
        fused_feats = gamma * img_feats + beta
        
        # Chuẩn hóa và kích hoạt phi tuyến
        fused_feats = self.layer_norm(fused_feats)
        fused_feats = self.relu(fused_feats)
        
        return fused_feats