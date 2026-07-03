"""
XBone-Net Model Architectures & Module Factory.
===============================================================================
Exposes the core model composer, builder factory, and sub-package registries:
  - Composer: Composite multi-modal architecture (XBoneMultiModalModel)
  - Builder: Factory functions (build_model, setup_phase2_modules)
  - Backbones: BiomedCLIP, OpenCLIP, ResNet-50, DenseNet-121, MedCLIP
  - Fusion: CrossAttentionFusion, ConcatFusion, IdentityFusion
  - Classifier Heads: PrototypicalHead, LinearHead, IdentityHead
  - PEFT: LoRA, QLoRA, Full FT
"""

from .composer import XBoneMultiModalModel
from .builder import build_model, setup_phase2_modules

__all__ = [
    "XBoneMultiModalModel",
    "build_model",
    "setup_phase2_modules",
]
