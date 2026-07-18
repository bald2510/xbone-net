"""
Dataset registry and dataset exports for XBone-Net architecture.
============================================================
Provides a centralized dataset registry mapping config names to PyTorch
Dataset classes for Stage 1 (contrastive VLM alignment), Stage 2
classification, and post-hoc OOD analysis pipelines.

Supported Datasets:
  - btxrd: BTXRD (Bone Tumor X-Ray Dataset) multi-class/multi-label loader
  - ctch: CTCH pediatric bone fracture dataset loader
  - fracatlas: FracAtlas fracture dataset loader (ID or OOD evaluation)
"""

from .btxrd import BTXRDDataset
from .ctch import CTCHDataset
from .fracatlas import FracAtlasDataset
from .analysis import CTCHOODDataset

# ============================================================
# DATASET REGISTRY
# ============================================================

# Maps dataset string key to PyTorch Dataset class constructor.
# Used by dataset builder (src/datasets/builder.py) during runtime setup.
DATASET_REGISTRY = {
    'btxrd': BTXRDDataset,       # Bone Tumor X-Ray Dataset (primary ID dataset)
    'ctch': CTCHDataset,         # CTCH pediatric fracture dataset
    'ctch_ood': CTCHOODDataset,  # CTCH semantic OOD evaluation manifest
    'fracatlas': FracAtlasDataset,  # FracAtlas fracture atlas (OOD evaluation)
}

