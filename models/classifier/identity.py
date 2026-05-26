import torch
import torch.nn as nn

class IdentityHead(nn.Module):
    """
    Dummy module cho Classifier Head.
    Thêm **kwargs để hứng toàn bộ tham số như feature_dim, num_classes.
    """
    def __init__(self, **kwargs):
        super().__init__()
        
    def forward(self, features):
        return features