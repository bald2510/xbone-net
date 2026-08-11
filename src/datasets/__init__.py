"""Cung cấp thành phần dữ liệu   init   cho XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

from .btxrd import BTXRDDataset
from .ctch import CTCHDataset
from .fracatlas import FracAtlasDataset
from .analysis import CTCHOODDataset

# ============================================================
# Chuẩn bị dữ liệu và chiến lược lấy mẫu tương ứng.
# ============================================================

# Chuẩn bị dữ liệu và chiến lược lấy mẫu tương ứng.
# Chuẩn bị dữ liệu và chiến lược lấy mẫu tương ứng.
DATASET_REGISTRY = {
    'btxrd': BTXRDDataset,       # Chuẩn bị dữ liệu và chiến lược lấy mẫu tương ứng.
    'ctch': CTCHDataset,         # Chuẩn bị dữ liệu và chiến lược lấy mẫu tương ứng.
    'ctch_ood': CTCHOODDataset,  # Tính điểm và độ đo phát hiện dữ liệu ngoài phân phối.
    'fracatlas': FracAtlasDataset,  # Tính điểm và độ đo phát hiện dữ liệu ngoài phân phối.
}

