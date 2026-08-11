"""Cung cấp thành phần mô hình drl trong kiến trúc XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DRLAuxiliaryBranch(nn.Module):
    """Đóng gói hành vi của thành phần ``DRLAuxiliaryBranch``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_classes: int = 22,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        epsilon: float = 1e-3,
        minimum_component_weight: float = 1e-4,
        covariance_shrinkage: float = 1e-4,
    ) -> None:
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        feature_dim : int, optional
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        num_classes : int, optional
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        hidden_dim : int, optional
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        dropout : float, optional
            Giá trị ``dropout`` được sử dụng trong phép xử lý.
        epsilon : float, optional
            Giá trị ``epsilon`` được sử dụng trong phép xử lý.
        minimum_component_weight : float, optional
            Giá trị ``minimum_component_weight`` được sử dụng trong phép xử lý.
        covariance_shrinkage : float, optional
            Giá trị ``covariance_shrinkage`` được sử dụng trong phép xử lý.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1:
            raise ValueError("feature_dim and hidden_dim must be positive.")
        if num_classes < 2:
            raise ValueError("num_classes must be at least two.")
        if not 0.0 < epsilon:
            raise ValueError("epsilon must be positive.")
        if not 0.0 < minimum_component_weight:
            raise ValueError("minimum_component_weight must be positive.")
        if not 0.0 <= covariance_shrinkage:
            raise ValueError("covariance_shrinkage must be non-negative.")

        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.epsilon = float(epsilon)
        self.minimum_component_weight = float(minimum_component_weight)
        self.covariance_shrinkage = float(covariance_shrinkage)

        def projector(input_dim: int) -> nn.Sequential:
            """Thực hiện bước projector trong quy trình hiện tại.

            Parameters
            ----------
            input_dim : int
                Số lượng, kích thước hoặc tỷ lệ được sử dụng.

            Returns
            -------
            nn.Sequential
                Kết quả được tạo bởi bước xử lý của hàm.
            """
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, feature_dim),
                nn.LayerNorm(feature_dim),
            )

        self.global_visual_projection = projector(feature_dim)
        self.local_visual_projection = projector(feature_dim)
        self.text_projection = projector(feature_dim)
        self.multimodal_projection = projector(feature_dim * 3)
        self.output_norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

        self.register_buffer(
            "label_covariance",
            torch.eye(feature_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "covariance_initialized",
            torch.tensor(False, dtype=torch.bool),
        )

    @staticmethod
    def _masked_local_mean(
        image_features: torch.Tensor,
        image_local_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Thực hiện bước masked local mean trong quy trình hiện tại.

        Parameters
        ----------
        image_features : torch.Tensor
            Ảnh hoặc biểu diễn ảnh đầu vào.
        image_local_padding_mask : Optional[torch.Tensor]
            Ảnh hoặc biểu diễn ảnh đầu vào.

        Returns
        -------
        torch.Tensor
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        if image_features.ndim == 2:
            return image_features
        if image_features.ndim != 3:
            raise ValueError(
                "DRL image features must have shape [B,D] or [B,T,D], got "
                f"{tuple(image_features.shape)}."
            )
        if image_features.size(1) <= 1:
            return image_features[:, 0]
        local = image_features[:, 1:]
        if image_local_padding_mask is None:
            return local.mean(dim=1)
        if tuple(image_local_padding_mask.shape) != tuple(local.shape[:2]):
            raise ValueError(
                "DRL local image padding mask has incompatible shape: "
                f"{tuple(image_local_padding_mask.shape)} vs {tuple(local.shape[:2])}."
            )
        valid = (~image_local_padding_mask.to(torch.bool)).unsqueeze(-1).to(local.dtype)
        return (local * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    @torch.no_grad()
    def set_label_covariance(self, covariance: torch.Tensor) -> None:
        """Thiết lập nhãn covariance cho bước xử lý hiện tại.

        Parameters
        ----------
        covariance : torch.Tensor
            Giá trị ``covariance`` được sử dụng trong phép xử lý.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        expected = (self.feature_dim, self.feature_dim)
        if tuple(covariance.shape) != expected:
            raise ValueError(
                f"Expected covariance shape {expected}, got {tuple(covariance.shape)}."
            )
        covariance = covariance.to(
            device=self.label_covariance.device,
            dtype=self.label_covariance.dtype,
        )
        if not torch.isfinite(covariance).all():
            raise ValueError("Label covariance contains NaN or infinity.")
        covariance = 0.5 * (covariance + covariance.T)
        if self.covariance_shrinkage:
            covariance = covariance + self.covariance_shrinkage * torch.eye(
                self.feature_dim,
                device=covariance.device,
                dtype=covariance.dtype,
            )
        self.label_covariance.copy_(covariance)
        self.covariance_initialized.fill_(True)

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        label_representation: torch.Tensor,
        image_local_padding_mask: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ):
        """Thực hiện lượt lan truyền xuôi của mô hình.

        Parameters
        ----------
        image_features : torch.Tensor
            Ảnh hoặc biểu diễn ảnh đầu vào.
        text_features : torch.Tensor
            Văn bản hoặc biểu diễn văn bản đầu vào.
        label_representation : torch.Tensor
            Nhãn hoặc chỉ số lớp liên quan.
        image_local_padding_mask : Optional[torch.Tensor]
            Ảnh hoặc biểu diễn ảnh đầu vào.
        return_details : bool, optional
            Giá trị ``return_details`` được sử dụng trong phép xử lý.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        if label_representation.ndim != 2:
            raise ValueError("label_representation must have shape [B,D].")
        if label_representation.size(-1) != self.feature_dim:
            raise ValueError(
                f"Expected label representation dimension {self.feature_dim}, "
                f"got {label_representation.size(-1)}."
            )
        if text_features is None:
            raise ValueError("The DRL auxiliary branch requires text features.")

        image_global = (
            image_features[:, 0]
            if image_features.ndim == 3
            else image_features
        )
        image_local = self._masked_local_mean(
            image_features,
            image_local_padding_mask,
        )
        text_global = (
            text_features[:, 0]
            if text_features.ndim == 3
            else text_features
        )

        components = torch.stack(
            [
                self.global_visual_projection(image_global),
                self.local_visual_projection(image_local),
                self.text_projection(text_global),
                self.multimodal_projection(
                    torch.cat([image_global, image_local, text_global], dim=-1)
                ),
            ],
            dim=1,
        )

        # Thiết lập và thực thi pha 3 học biểu diễn bổ sung.
        # Thiết lập giá trị trung gian cho bước xử lý tiếp theo.
        label_detached = label_representation.detach()
        similarity = torch.einsum("bmd,bd->bm", components, label_detached)
        component_weights = (
            1.0 - self.epsilon * similarity
        ).clamp_min(self.minimum_component_weight)
        component_weights = component_weights / component_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(self.minimum_component_weight)
        component_mean = torch.einsum(
            "bm,bmd->bd", component_weights, components
        )

        covariance_term = F.linear(
            label_detached,
            self.label_covariance,
        )
        alignment = (component_mean * label_detached).sum(dim=-1, keepdim=True)
        complementary = component_mean - self.epsilon * (
            covariance_term + alignment * component_mean
        )
        complementary = self.output_norm(complementary)
        logits = self.classifier(complementary)

        if return_details:
            return logits, complementary, {
                "components": components,
                "component_weights": component_weights,
                "component_similarity": similarity,
                "component_mean": component_mean,
            }
        return logits, complementary


def drl_ood_score(
    primary_logits: torch.Tensor,
    auxiliary_logits: torch.Tensor,
) -> torch.Tensor:
    """Thực hiện bước drl ood điểm trong quy trình hiện tại.

    Parameters
    ----------
    primary_logits : torch.Tensor
        Giá trị ``primary_logits`` được sử dụng trong phép xử lý.
    auxiliary_logits : torch.Tensor
        Giá trị ``auxiliary_logits`` được sử dụng trong phép xử lý.

    Returns
    -------
    torch.Tensor
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if primary_logits.shape != auxiliary_logits.shape:
        raise ValueError(
            "Primary and auxiliary logits must have identical shapes, got "
            f"{tuple(primary_logits.shape)} and {tuple(auxiliary_logits.shape)}."
        )
    primary_probability = torch.softmax(primary_logits, dim=-1)
    auxiliary_probability = torch.softmax(auxiliary_logits, dim=-1)
    combined_probability = 0.5 * (
        primary_probability + auxiliary_probability
    )
    return 1.0 - combined_probability.max(dim=-1).values

