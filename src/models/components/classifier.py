import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypicalHead(nn.Module):
    """
    Phân loại dựa trên khoảng cách Cosine từ vector truy vấn đến các Prototypes
    """
    def __init__(self, feature_dim=512, num_classes=10, temperature=10.0):
        super(PrototypicalHead, self).__init__()
        self.temperature = temperature
        
        # Khởi tạo Prototypes như một bộ trọng số có thể học được trong lúc Tuning
        # Ở Giai đoạn Online, bộ trọng số này sẽ được freeze
        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))
        
    def forward(self, x):
        # Chuẩn hóa L2 cho cả vector truy vấn và Prototypes
        x_norm = F.normalize(x, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        
        # Tính độ tương đồng Cosine (Cosine Similarity)
        # Kết quả có range [-1, 1]
        cosine_sim = torch.matmul(x_norm, proto_norm.T)
        
        # Scale với nhiệt độ để làm sắc nét phân phối xác suất
        logits = cosine_sim * self.temperature
        
        # Trả về cả logits (để tính CrossEntropy lúc train) và cosine_sim (để tính OOD lúc test)
        return logits, cosine_sim