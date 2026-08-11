"""Cung cấp tiện ích losses cho huấn luyện, đánh giá và phân tích XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Thiết lập và thực thi pha 1 căn chỉnh ảnh-văn bản.
# ============================================================

def get_clip_logit_params(clip_model, require_bias: bool = False):
    """Lấy clip logit params cho bước xử lý hiện tại.

    Parameters
    ----------
    clip_model : object
        Mô hình hoặc thành phần mô hình cần xử lý.
    require_bias : bool, optional
        Giá trị ``require_bias`` được sử dụng trong phép xử lý.

    Returns
    -------
    object
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    if not hasattr(clip_model, "logit_scale"):
        # Xử lý bộ mã hóa nền tảng theo giao diện tương ứng.
        # Bước hỗ trợ để lấy clip logit params cho bước xử lý hiện tại.
        import math
        logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
        setattr(clip_model, "logit_scale", logit_scale)
        print("  [Loss] Created learnable logit_scale for non-OpenCLIP backbone.")

    logit_scale = clip_model.logit_scale
    logit_bias = getattr(clip_model, "logit_bias", None)

    if logit_bias is None and require_bias:
        if hasattr(clip_model, "logit_bias"):
            logit_bias = clip_model.logit_bias
        else:
            logit_bias = nn.Parameter(torch.full([], -10.0, device=logit_scale.device))
            clip_model.register_parameter("logit_bias", logit_bias)

    return logit_scale, logit_bias


def _pairwise_logits(image_features, text_features, logit_scale, logit_bias=None):
    """Thực hiện bước pairwise logits trong quy trình hiện tại.

    Parameters
    ----------
    image_features : object
        Ảnh hoặc biểu diễn ảnh đầu vào.
    text_features : object
        Văn bản hoặc biểu diễn văn bản đầu vào.
    logit_scale : object
        Giá trị ``logit_scale`` được sử dụng trong phép xử lý.
    logit_bias : object, optional
        Giá trị ``logit_bias`` được sử dụng trong phép xử lý.

    Returns
    -------
    object
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    image_features = F.normalize(image_features, dim=-1, p=2)
    text_features = F.normalize(text_features, dim=-1, p=2)
    logits = logit_scale.exp() * image_features @ text_features.T
    if logit_bias is not None:
        logits = logits + logit_bias
    return logits


# ============================================================
# Thiết lập và thực thi pha 1 căn chỉnh ảnh-văn bản.
# ============================================================

class SoftTargetSemanticMatchingLoss(nn.Module):
    """Đóng gói hành vi của thành phần ``SoftTargetSemanticMatchingLoss``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(self, clip_model, target_similarity: float = 0.95):
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        clip_model : object
            Mô hình hoặc thành phần mô hình cần xử lý.
        target_similarity : float, optional
            Nhãn hoặc chỉ số lớp liên quan.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        super().__init__()
        if not 0.0 <= float(target_similarity) <= 1.0:
            raise ValueError("target_similarity must be in [0, 1].")
        self.logit_scale, _ = get_clip_logit_params(clip_model)
        self.target_similarity = float(target_similarity)

    def forward(self, image_features, text_features, disease_labels=None):
        """Thực hiện lượt lan truyền xuôi của mô hình.

        Parameters
        ----------
        image_features : object
            Ảnh hoặc biểu diễn ảnh đầu vào.
        text_features : object
            Văn bản hoặc biểu diễn văn bản đầu vào.
        disease_labels : object, optional
            Giá trị ``disease_labels`` được sử dụng trong phép xử lý.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        if image_features.ndim != 2 or text_features.ndim != 2:
            raise ValueError(
                "Semantic matching expects [B,D] image and text features, got "
                f"{tuple(image_features.shape)} and {tuple(text_features.shape)}."
            )
        if image_features.shape != text_features.shape:
            raise ValueError("Image and text features must have identical [B,D] shapes.")
        batch_size = image_features.size(0)
        device = image_features.device
        logits_v2t = _pairwise_logits(image_features, text_features, self.logit_scale)
        logits_t2v = logits_v2t.T

        # Thiết lập giá trị trung gian cho bước xử lý tiếp theo.
        targets_v2t = torch.eye(batch_size, device=device)
        if disease_labels is not None:
            labels_col = disease_labels.view(-1, 1)
            labels_row = disease_labels.view(1, -1)
            same_class_mask = (labels_col == labels_row) & (~torch.eye(batch_size, dtype=torch.bool, device=device))
            targets_v2t[same_class_mask] = self.target_similarity

        targets_v2t = targets_v2t / targets_v2t.sum(dim=1, keepdim=True)
        targets_t2v = targets_v2t.T / targets_v2t.T.sum(dim=1, keepdim=True)

        loss_v2t = -torch.sum(targets_v2t * F.log_softmax(logits_v2t, dim=1), dim=1).mean()
        loss_t2v = -torch.sum(targets_t2v * F.log_softmax(logits_t2v, dim=1), dim=1).mean()
        return 0.5 * (loss_v2t + loss_t2v)





LOSS_REGISTRY = {
    "semantic_matching": SoftTargetSemanticMatchingLoss,
}


def build_loss(loss_type: str, clip_model=None, **kwargs):
    """Xây dựng loss cho bước xử lý hiện tại.

    Parameters
    ----------
    loss_type : str
        Phương pháp hoặc chế độ xử lý được chọn.
    clip_model : object, optional
        Mô hình hoặc thành phần mô hình cần xử lý.
    **kwargs : dict
        Các đối số từ khóa bổ sung.

    Returns
    -------
    object
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"Loss type '{loss_type}' not supported. Choose from {list(LOSS_REGISTRY.keys())}")
    if clip_model is None:
        raise ValueError("build_loss requires `clip_model`.")
    kwargs.pop("temperature", None)
    return LOSS_REGISTRY[loss_type](clip_model=clip_model, **kwargs)


def resolve_phase2_loss_type(loss_type: str | None, classifier_type: str) -> str:
    """Xác định phase2 loss type cho bước xử lý hiện tại.

    Parameters
    ----------
    loss_type : str | None
        Phương pháp hoặc chế độ xử lý được chọn.
    classifier_type : str
        Phương pháp hoặc chế độ xử lý được chọn.

    Returns
    -------
    str
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    classifier_type = str(classifier_type).strip().lower()
    supported_classifiers = {"empirical_centroid", "linear"}
    if classifier_type not in supported_classifiers:
        raise ValueError(
            f"Unsupported Phase-2 classifier_type='{classifier_type}'. "
            f"Choose from {sorted(supported_classifiers)}."
        )
    expected = (
        "empirical_centroid_ce"
        if classifier_type == "empirical_centroid"
        else "ce"
    )
    resolved = expected if loss_type is None else str(loss_type).strip().lower()
    supported = {"ce", "empirical_centroid_ce"}
    if resolved not in supported:
        raise ValueError(
            f"Unsupported Phase-2 loss_type='{resolved}'. "
            f"Choose from {sorted(supported)}."
        )
    if resolved != expected:
        raise ValueError(
            f"Phase-2 loss_type='{resolved}' is incompatible with "
            f"classifier_type='{classifier_type}'. Expected '{expected}'."
        )
    return resolved


def build_phase2_loss(
    loss_type: str | None,
    classifier_type: str,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """Xây dựng phase2 loss cho bước xử lý hiện tại.

    Parameters
    ----------
    loss_type : str | None
        Phương pháp hoặc chế độ xử lý được chọn.
    classifier_type : str
        Phương pháp hoặc chế độ xử lý được chọn.
    class_weights : torch.Tensor | None, optional
        Nhãn hoặc chỉ số lớp liên quan.
    label_smoothing : float, optional
        Nhãn hoặc chỉ số lớp liên quan.

    Returns
    -------
    nn.Module
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    resolve_phase2_loss_type(loss_type, classifier_type)
    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(label_smoothing),
    )
