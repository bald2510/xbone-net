"""
Dataset registry and dataset exports for XBone-Net architecture.
============================================================
Provides a centralized dataset registry mapping config names to PyTorch
Dataset classes for Stage 1 (contrastive VLM alignment) and Stage 2
(prototypical classification & OOD detection) training pipelines.

Supported Datasets:
  - btxrd: BTXRD (Bone Tumor X-Ray Dataset) multi-class/multi-label loader
  - ctch: CTCH pediatric bone fracture dataset loader
  - fracatlas: FracAtlas fracture dataset loader (ID or OOD evaluation)
"""

from .btxrd import BTXRDDataset
from .ctch import CTCHDataset
from .fracatlas import FracAtlasDataset

# ============================================================
# DATASET REGISTRY
# ============================================================

# Maps dataset string key to PyTorch Dataset class constructor.
# Used by dataset builder (src/datasets/builder.py) during runtime setup.
DATASET_REGISTRY = {
    'btxrd': BTXRDDataset,       # Bone Tumor X-Ray Dataset (primary ID dataset)
    'ctch': CTCHDataset,         # CTCH pediatric fracture dataset
    'fracatlas': FracAtlasDataset,  # FracAtlas fracture atlas (OOD evaluation)
}

