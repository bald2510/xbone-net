from torch.utils.data import DataLoader
from . import DATASET_REGISTRY

def build_dataloader(cfg, split="train", transform=None, tokenizer=None):
    """
    Factory tự động khởi tạo DataLoader dựa trên config.
    """
    dataset_name = cfg.get('name')
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' chưa được đăng ký trong DATASET_REGISTRY.")
        
    dataset_class = DATASET_REGISTRY[dataset_name]
    
    # 1. Khởi tạo Dataset
    dataset = dataset_class(
        split=split, # Truyền split ('train', 'val', 'test') để dataset tự lọc dữ liệu
        transform=transform,
        tokenizer=tokenizer,
        **cfg.get('params', {}) # unpack toàn bộ các thông số riêng của dataset này
    )
    
    # 2. Đóng gói vào DataLoader
    # Khi test/val thì không cần shuffle, khi train thì bật shuffle
    is_train = (split == "train")
    
    loader = DataLoader(
        dataset,
        batch_size=cfg.get('batch_size', 32),
        shuffle=is_train,
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True # Giúp chuyển dữ liệu lên GPU nhanh hơn
    )
    
    return loader