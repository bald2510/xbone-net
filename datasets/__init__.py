from .mimic_dataset import MimicCxrDataset
# from .chexpert_dataset import ChexpertDataset (Bỏ comment khi có file này)

DATASET_REGISTRY = {
    'mimic_cxr': MimicCxrDataset,
    # 'chexpert': ChexpertDataset,
}