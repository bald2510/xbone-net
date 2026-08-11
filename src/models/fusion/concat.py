"""Cung cấp cơ chế dung hợp đa phương thức concat cho XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

import torch
import torch.nn as nn


# ============================================================
# Thiết lập mô-đun dung hợp và đầu phân lớp theo cấu hình.
# ============================================================

class ConcatFusion(nn.Module):
    """Dung hợp biểu diễn đa phương thức bằng lớp ``ConcatFusion``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(self, text_dim: int = 512, img_dim: int = 512, **kwargs):
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        text_dim : int, optional
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        img_dim : int, optional
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        **kwargs : dict
            Các đối số từ khóa bổ sung.
        """
        super().__init__()
        # Thiết lập giá trị trung gian cho bước xử lý tiếp theo.
        self.projection = nn.Sequential(
            nn.Linear(img_dim + text_dim, img_dim),
            nn.LayerNorm(img_dim),
            nn.GELU(),
        )

    def forward(self, img_feats: torch.Tensor, txt_feats: torch.Tensor) -> torch.Tensor:
        """Thực hiện lượt lan truyền xuôi của mô hình.

        Parameters
        ----------
        img_feats : torch.Tensor
            Ảnh hoặc biểu diễn ảnh đầu vào.
        txt_feats : torch.Tensor
            Giá trị ``txt_feats`` được sử dụng trong phép xử lý.

        Returns
        -------
        torch.Tensor
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        # Thu thập và xử lý biểu diễn đặc trưng của mô hình.
        combined = torch.cat([img_feats, txt_feats], dim=-1)

        # Bước hỗ trợ để thực hiện lượt lan truyền xuôi của mô hình.
        return self.projection(combined)

