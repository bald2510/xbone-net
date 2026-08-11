"""Cung cấp đầu phân lớp linear cho XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

import torch.nn as nn


# ============================================================
# Thiết lập mô-đun dung hợp và đầu phân lớp theo cấu hình.
# ============================================================

class LinearHead(nn.Module):
    """Cung cấp đầu phân lớp bằng lớp ``LinearHead``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(self, feature_dim: int = 512, num_classes: int = 10):
        """Thực hiện bước init trong quy trình hiện tại.

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
        """Thực hiện lượt lan truyền xuôi của mô hình.

        Parameters
        ----------
        x : object
            Giá trị ``x`` được sử dụng trong phép xử lý.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        return self.classifier(x)
