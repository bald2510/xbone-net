from .btxrd_dataset import BTXRDDataset
from .ctch_dataset import CTCHDataset

DATASET_REGISTRY = {
    'btxrd': BTXRDDataset,
    'ctch': CTCHDataset,
}
