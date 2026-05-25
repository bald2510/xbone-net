import torch
import torch.nn as nn

class IdentityFusion(nn.Module):
    """
    Dummy module cho Fusion: Nhận gì trả nấy.
    Thêm **kwargs để hứng toàn bộ các tham số như text_dim, img_dim từ config bự ném vào.
    """
    def __init__(self, **kwargs):
        super().__init__()
        
    def forward(self, img_feats, txt_feats):
        return img_feats, txt_feats