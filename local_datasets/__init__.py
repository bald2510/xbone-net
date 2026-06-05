from .mimic_dataset import MimicCxrDataset
from .btxrd_dataset import BTXRDDataset
from .fracatlas_dataset import FracAtlasDataset
# from .chexpert_dataset import ChexpertDataset (Bỏ comment khi có file này)

DATASET_REGISTRY = {
    'mimic_cxr': MimicCxrDataset,
    'btxrd': BTXRDDataset,
    'fracatlas': FracAtlasDataset,
}

