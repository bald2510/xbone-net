"""
Dataset builder factory for XBone-Net architecture.
============================================================
Implements a factory pattern to instantiate PyTorch DataLoaders from configuration
dictionaries, supporting multi-modal data loading (images + text reports) for both
Stage 1 (contrastive learning) and Stage 2 (prototypical classification) of XBone-Net.

Overview:
  - Dynamically resolves PyTorch Dataset classes via DATASET_REGISTRY
  - Forwards image transforms and text tokenizers to dataset instances
  - Configures DataLoader parameters (batch size, shuffling, worker count, pin_memory)
"""

from torch.utils.data import DataLoader
from . import DATASET_REGISTRY


# ============================================================
# DATALOADER BUILDER FACTORY
# ============================================================

def build_dataloader(cfg: dict, split: str = "train", transform=None, tokenizer=None) -> DataLoader:
    """Build a DataLoader from a dataset configuration dictionary.

    Resolves the dataset class via DATASET_REGISTRY, passes through
    all parameters from cfg['params'], and returns a ready-to-iterate
    DataLoader. Training splits are automatically shuffled.

    Args:
        cfg: Dataset configuration dictionary with keys:
            - 'name' (str): Dataset registry key (e.g. 'btxrd', 'ctch', 'fracatlas').
            - 'params' (dict, optional): Keyword arguments forwarded to the dataset constructor.
            - 'batch_size' (int, optional): Batch size per step. Defaults to 32.
            - 'num_workers' (int, optional): Number of DataLoader worker threads. Defaults to 4.
        split: Data split identifier ('train', 'val', or 'test').
            Determines whether the loader shuffles samples.
        transform: torchvision.transforms pipeline applied to each image sample.
        tokenizer: Text tokenizer callable used to encode report strings into token IDs.

    Returns:
        DataLoader: PyTorch DataLoader configured with the requested dataset,
            batch size, shuffle policy, and memory pinning options.

    Raises:
        ValueError: If cfg['name'] is not registered in DATASET_REGISTRY.

    Example:
        cfg = {
            'name': 'btxrd',
            'batch_size': 16,
            'num_workers': 2,
            'params': {
                'img_dir': 'data/btxrd/images',
                'report_dir': 'data/btxrd/reports',
                'csv_split_path': 'data/btxrd/splits.csv',
                'csv_labels_path': 'data/btxrd/labels.csv',
            },
        }
        loader = build_dataloader(cfg, split='train', transform=T, tokenizer=tok)
    """
    # --- Resolve dataset class from registry ---
    dataset_name = cfg.get('name')
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' not found in DATASET_REGISTRY.")

    dataset_class = DATASET_REGISTRY[dataset_name]

    # --- Instantiate Dataset with split and transformers ---
    dataset = dataset_class(
        split=split,
        transform=transform,
        tokenizer=tokenizer,
        **cfg.get('params', {})
    )

    # --- Determine shuffle policy and assemble DataLoader ---
    is_train = (split == "train")  # Only shuffle the training split

    return DataLoader(
        dataset,
        batch_size=cfg.get('batch_size', 32),
        shuffle=is_train,
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True  # Speeds up host-to-device transfer on CUDA
    )
