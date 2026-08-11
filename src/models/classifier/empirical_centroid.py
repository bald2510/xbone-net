"""Cung cấp đầu phân lớp empirical centroid cho XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmpiricalCentroidHead(nn.Module):
    """Cung cấp đầu phân lớp bằng lớp ``EmpiricalCentroidHead``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_classes: int = 14,
        scale: float = 15.0,
        use_class_bias: bool = False,
        eps: float = 1e-8,
    ) -> None:
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        feature_dim : int, optional
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        num_classes : int, optional
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        scale : float, optional
            Giá trị ``scale`` được sử dụng trong phép xử lý.
        use_class_bias : bool, optional
            Nhãn hoặc chỉ số lớp liên quan.
        eps : float, optional
            Giá trị ``eps`` được sử dụng trong phép xử lý.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        super().__init__()
        if feature_dim < 1 or num_classes < 2:
            raise ValueError("feature_dim must be positive and num_classes must be >= 2.")
        if scale <= 0:
            raise ValueError("scale must be positive.")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.scale = float(scale)
        self.use_class_bias = bool(use_class_bias)
        self.eps = float(eps)

        if self.use_class_bias:
            self.class_bias = nn.Parameter(torch.zeros(self.num_classes))
        else:
            self.register_parameter("class_bias", None)

        self.register_buffer(
            "centroids", torch.zeros(self.num_classes, self.feature_dim)
        )
        self.register_buffer(
            "centroid_counts", torch.zeros(self.num_classes, dtype=torch.long)
        )
        self.register_buffer(
            "centroids_initialized", torch.tensor(False, dtype=torch.bool)
        )

    @property
    def prototypes(self) -> torch.Tensor:
        """Thực hiện bước prototypes trong quy trình hiện tại.

        Returns
        -------
        torch.Tensor
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        return self.centroids

    @torch.no_grad()
    def set_centroids(
        self,
        centroids: torch.Tensor,
        counts: torch.Tensor,
    ) -> None:
        """Thiết lập các tâm lớp cho bước xử lý hiện tại.

        Parameters
        ----------
        centroids : torch.Tensor
            Giá trị ``centroids`` được sử dụng trong phép xử lý.
        counts : torch.Tensor
            Giá trị ``counts`` được sử dụng trong phép xử lý.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        expected_shape = (self.num_classes, self.feature_dim)
        if tuple(centroids.shape) != expected_shape:
            raise ValueError(
                f"Expected centroids with shape {expected_shape}, "
                f"got {tuple(centroids.shape)}."
            )
        if tuple(counts.shape) != (self.num_classes,):
            raise ValueError(
                f"Expected counts with shape {(self.num_classes,)}, "
                f"got {tuple(counts.shape)}."
            )
        missing = torch.nonzero(counts <= 0, as_tuple=False).flatten().tolist()
        if missing:
            raise ValueError(
                "Empirical centroid estimation requires every configured class "
                f"to occur in the training subset. Missing classes: {missing}"
            )
        if not torch.isfinite(centroids).all():
            raise ValueError("Centroids contain NaN or infinite values.")
        zero_norm = torch.nonzero(
            centroids.norm(dim=-1) <= self.eps, as_tuple=False
        ).flatten().tolist()
        if zero_norm:
            raise ValueError(f"Empirical centroids have zero norm for classes: {zero_norm}")

        self.centroids.copy_(centroids.to(self.centroids))
        self.centroid_counts.copy_(counts.to(self.centroid_counts))
        self.centroids_initialized.fill_(True)

    def forward(self, features: torch.Tensor, return_features: bool = False):
        """Thực hiện lượt lan truyền xuôi của mô hình.

        Parameters
        ----------
        features : torch.Tensor
            Giá trị ``features`` được sử dụng trong phép xử lý.
        return_features : bool, optional
            Giá trị ``return_features`` được sử dụng trong phép xử lý.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        RuntimeError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        if features.ndim != 2:
            raise ValueError(
                "EmpiricalCentroidHead expects [B,D] features, "
                f"got {tuple(features.shape)}."
            )
        if features.size(-1) != self.feature_dim:
            raise ValueError(
                f"Expected feature dimension {self.feature_dim}, "
                f"got {features.size(-1)}."
            )
        if not bool(self.centroids_initialized.item()):
            raise RuntimeError(
                "Empirical centroids are not initialized. Compute them from the "
                "labelled training set before classification."
            )

        normed_features = F.normalize(features, dim=-1, eps=self.eps)
        normed_centroids = F.normalize(self.centroids, dim=-1, eps=self.eps)
        logits = self.scale * (normed_features @ normed_centroids.T)
        if self.class_bias is not None:
            logits = logits + self.class_bias
        if return_features:
            return logits, features
        return logits
