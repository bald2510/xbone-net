from .btxrd_dataset import BTXRDDataset
from .ctch_dataset import CTCHDataset
from .fracatlas_dataset import FracAtlasDataset

DATASET_REGISTRY = {
    'btxrd': BTXRDDataset,
    'ctch': CTCHDataset,
    'fracatlas': FracAtlasDataset,
}

