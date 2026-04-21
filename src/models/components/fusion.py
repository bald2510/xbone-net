import torch
import torch.nn as nn

class ConcatFusion(nn.Module):
    def __init__(self):
        super(ConcatFusion, self).__init__()

    def forward(self, img_feat, text_feat):
        """
        img_feat: [B, 512]
        text_feat: [B, 512]
        return: [B, 1024]
        """
        # Đảm bảo cả 2 đều đã được normalize trước khi concat
        # (Thường CLIP đã làm bước này, nhưng làm lại cho chắc)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        
        # Nối lại
        fused_vector = torch.cat((img_feat, text_feat), dim=-1)
        return fused_vector