import torch
import torch.nn as nn

class PrototypicalHead(nn.Module):
    def __init__(self, feature_dim: int = 512, num_classes: int = 14):
        super().__init__()
        self.num_classes = num_classes
        
        # Khởi tạo ma trận Prototypes như một nn.Parameter có thể huấn luyện được.
        # Mỗi class bệnh lý sẽ có một vector đại diện kích thước [feature_dim]
        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))
        
        # Khởi tạo trọng số chuẩn Xavier
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, features):
        """
        features: Đặc trưng sau fusion [Batch_size, feature_dim]
        """
        # 1. Chuẩn hóa L2 cho cả features và prototypes để tính Cosine Similarity
        normed_features = features / features.norm(dim=-1, keepdim=True)
        normed_prototypes = self.prototypes / self.prototypes.norm(dim=-1, keepdim=True)
        
        # 2. Tính Cosine Similarity giữa mỗi mẫu trong batch với 14 mẫu đại diện bệnh lý
        # Ma trận nhân: [Batch_size, feature_dim] x [feature_dim, num_classes]
        cosine_sim = normed_features @ normed_prototypes.T # Đầu ra: [Batch_size, num_classes]
        
        # 3. Nhân với một hệ số scale (nhiệt độ nghịch đảo) tương tự CLIP để kéo giãn khoảng cách logits
        logits = cosine_sim * 10.0 
        
        return logits