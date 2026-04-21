import torch
import torch.nn as nn

class MedicalHead(nn.Module):
    """
    Module Classifier tự làm. 
    Nhận đầu vào là vector sau khi Fusion (ví dụ: 1024 dim).
    """
    def __init__(self, input_dim, num_classes, hidden_dim=512, dropout_rate=0.3):
        super(MedicalHead, self).__init__()
        
        self.classifier = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            # Layer 2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2), # Giảm dần dropout ở các lớp sâu
            
            # Output layer
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)