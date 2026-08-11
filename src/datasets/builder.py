"""Cung cấp thành phần dữ liệu builder cho XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

from torch.utils.data import DataLoader
from . import DATASET_REGISTRY


# ============================================================
# Chuẩn bị dữ liệu và chiến lược lấy mẫu tương ứng.
# ============================================================

def build_dataloader(cfg: dict, split: str = "train", transform=None, tokenizer=None) -> DataLoader:
    """Xây dựng dataloader cho bước xử lý hiện tại.

    Parameters
    ----------
    cfg : dict
        Cấu hình điều khiển bước xử lý.
    split : str, optional
        Giá trị ``split`` được sử dụng trong phép xử lý.
    transform : object, optional
        Giá trị ``transform`` được sử dụng trong phép xử lý.
    tokenizer : object, optional
        Giá trị ``tokenizer`` được sử dụng trong phép xử lý.

    Returns
    -------
    DataLoader
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    # Chuẩn bị dữ liệu và chiến lược lấy mẫu tương ứng.
    dataset_name = cfg.get('name')
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' not found in DATASET_REGISTRY.")

    dataset_class = DATASET_REGISTRY[dataset_name]

    # Chuẩn bị dữ liệu và chiến lược lấy mẫu tương ứng.
    dataset = dataset_class(
        split=split,
        transform=transform,
        tokenizer=tokenizer,
        **cfg.get('params', {})
    )

    # Chuẩn bị dữ liệu và chiến lược lấy mẫu tương ứng.
    is_train = (split == "train")  # Bước hỗ trợ để xây dựng dataloader cho bước xử lý hiện tại.

    return DataLoader(
        dataset,
        batch_size=cfg.get('batch_size', 32),
        shuffle=is_train,
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True  # Chọn thiết bị và độ chính xác tính toán phù hợp.
    )

