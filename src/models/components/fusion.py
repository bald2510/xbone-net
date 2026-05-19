import torch
import torch.nn as nn

class FiLMFusion(nn.Module):
    """
    Sử dụng đặc trưng Văn bản (UMLS Text) để điều chế đặc trưng Hình ảnh
    """
    def __init__(self, text_dim=512, img_dim=512):
        super(FiLMFusion, self).__init__()
        
        # FiLM Generator: Mạng MLP nhỏ sinh ra 2 hệ số gamma (scale) và beta (shift)
        self.film_generator = nn.Sequential(
            nn.Linear(text_dim, img_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(img_dim, img_dim * 2) # Nhân 2 để chẻ ra gamma và beta
        )
        
        # Zero-initialization ở lớp cuối giúp mô hình an toàn tuyệt đối khi thiếu data
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)

    def forward(self, img_feat, text_feat):
        # film_params: [Batch, img_dim * 2]
        film_params = self.film_generator(text_feat)
        
        # Chẻ đôi để lấy gamma và beta
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        
        # Phép biến đổi Affine (Điều chế)
        # Cộng thêm 1 vào gamma để hoạt động như một Residual connection
        fused_vector = (1 + gamma) * img_feat + beta
        
        # Chuẩn hóa lại vector sau khi dung hợp
        fused_vector = fused_vector / fused_vector.norm(dim=-1, keepdim=True)
        return fused_vector