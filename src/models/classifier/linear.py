"""Đầu phân loại tuyến tính dùng trong cấu hình XBone-Net hiện tại."""

import torch.nn as nn


class LinearHead(nn.Module):
    """Ánh xạ embedding dung hợp sang logits của các lớp đích."""

    def __init__(self, feature_dim: int = 512, num_classes: int = 10):
        """Khởi tạo phép chiếu ``feature_dim -> num_classes``.

        Parameters
        ----------
        feature_dim : int, optional
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        num_classes : int, optional
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        """
        super().__init__()
        # Thiết lập giá trị trung gian cho bước xử lý tiếp theo.
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        """Tính logits chưa chuẩn hóa cho một batch embedding.

        Parameters
        ----------
        x : object
            Giá trị ``x`` được sử dụng trong phép xử lý.

        Returns
        -------
        torch.Tensor
            Logits có dạng ``[batch_size, num_classes]``.
        """
        return self.classifier(x)
