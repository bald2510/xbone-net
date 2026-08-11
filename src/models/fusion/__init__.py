"""Cung cấp cơ chế dung hợp đa phương thức   init   cho XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

from .identity import IdentityFusion
from .cross_attention import CrossAttentionFusion
from .gated_cross_attention import GatedCrossAttentionFusion
from .concat import ConcatFusion

# ============================================================
# Thiết lập mô-đun dung hợp và đầu phân lớp theo cấu hình.
# ============================================================

FUSION_REGISTRY = {
    'none': IdentityFusion,
    'cross_attention': CrossAttentionFusion,
    'gated_cross_attention': GatedCrossAttentionFusion,
    'concat': ConcatFusion,
}


# ============================================================
# Thiết lập mô-đun dung hợp và đầu phân lớp theo cấu hình.
# ============================================================

def build_fusion_module(cfg: dict):
    """Xây dựng fusion module cho bước xử lý hiện tại.

    Parameters
    ----------
    cfg : dict
        Cấu hình điều khiển bước xử lý.

    Returns
    -------
    object
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    fusion_type = cfg.get('type', 'none')
    if fusion_type not in FUSION_REGISTRY:
        raise ValueError(f"Fusion '{fusion_type}' not supported. Choose from {list(FUSION_REGISTRY.keys())}")

    return FUSION_REGISTRY[fusion_type](**cfg.get('params', {}))

