import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearHead(nn.Module):
    """
    Phân loại đơn giản bằng một lớp Linear trên đặc trưng đã được kết hợp.
    """
    def __init__(self, feature_dim=512, num_classes=10):
        super(LinearHead, self).__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        logits = self.classifier(x)
        return logits