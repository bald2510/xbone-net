"""Cung cấp thành phần mô hình   init   trong kiến trúc XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

from .composer import XBoneMultiModalModel
from .builder import build_model, setup_phase2_modules, setup_phase3_modules
from .drl import DRLAuxiliaryBranch, drl_ood_score

__all__ = [
    "XBoneMultiModalModel",
    "build_model",
    "setup_phase2_modules",
    "setup_phase3_modules",
    "DRLAuxiliaryBranch",
    "drl_ood_score",
]
