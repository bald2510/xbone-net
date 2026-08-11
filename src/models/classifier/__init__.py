"""Cung cấp đầu phân lớp   init   cho XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

from .identity import IdentityHead
from .linear import LinearHead
from .empirical_centroid import EmpiricalCentroidHead

# ============================================================
# Thiết lập mô-đun dung hợp và đầu phân lớp theo cấu hình.
# ============================================================

HEAD_REGISTRY = {
    'none': IdentityHead,
    'linear': LinearHead,
    'empirical_centroid': EmpiricalCentroidHead,
}


# ============================================================
# Thiết lập mô-đun dung hợp và đầu phân lớp theo cấu hình.
# ============================================================

def build_head_module(cfg: dict):
    """Xây dựng head module cho bước xử lý hiện tại.

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
    head_type = cfg.get('type', 'none')
    if head_type not in HEAD_REGISTRY:
        raise ValueError(f"Classifier head '{head_type}' not supported.")

    return HEAD_REGISTRY[head_type](**cfg.get('params', {}))

