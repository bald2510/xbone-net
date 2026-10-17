"""Tiền xử lý ảnh đầu vào cho các bộ dữ liệu XBone-Net.

``direct_resize`` là tên tương thích của quy trình hiện tại: ảnh RGB gốc được
chuyển thẳng cho transform chuẩn của backbone. Ngoài ra giao thức nghiên cứu
có ``letterbox`` (đệm giữ tỉ lệ) và ``full_resize`` (co giãn toàn ảnh) để kiểm
tra độ nhạy với hình học tiền xử lý. Hai chiến lược này vẫn dùng cùng transform
chuẩn hóa của backbone sau bước hình học.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from PIL import Image, ImageOps


def prepare_image(
    image: Image.Image,
    transform: Callable[[Image.Image], Any] | None,
    config: Mapping[str, Any] | None = None,
) -> Any:
    """Áp dụng chiến lược tiền xử lý ảnh trước transform của backbone.

    Parameters
    ----------
    image : PIL.Image.Image
        Ảnh đầu vào.
    transform : callable or None
        Transform chuẩn hóa của backbone. Nếu ``None``, trả về ảnh PIL.
    config : mapping, optional
        Cấu hình chiến lược. ``direct_resize`` chuyển ảnh RGB gốc thẳng cho
        transform chuẩn của backbone; ``letterbox`` và ``full_resize`` là các
        đối chứng hình học được khai báo tường minh.

    Returns
    -------
    Any
        Ảnh PIL đã xử lý hoặc tensor do ``transform`` tạo ra.

    Raises
    ------
    ValueError
        Nếu chiến lược hoặc tham số hình học không hợp lệ.
    """
    config = config or {}
    strategy = str(config.get("strategy", "direct_resize")).lower()
    if strategy not in {"direct_resize", "native", "letterbox", "full_resize"}:
        raise ValueError(
            f"Unsupported preprocessing strategy: {strategy!r}. "
            "Use native/direct_resize, letterbox, or full_resize."
        )
    source = image.convert("RGB")
    if strategy in {"letterbox", "full_resize"}:
        size = int(config.get("image_size", 224))
        if size < 1:
            raise ValueError("image_size must be positive.")
        if strategy == "letterbox":
            fill = tuple(config.get("fill", [123, 117, 104]))
            if len(fill) != 3 or any(not 0 <= v <= 255 for v in fill):
                raise ValueError("letterbox fill must contain three RGB values in [0,255].")
            source = ImageOps.pad(source, (size, size), method=Image.Resampling.BICUBIC,
                                  color=fill, centering=(0.5, 0.5))
        else:
            source = source.resize((size, size), resample=Image.Resampling.BICUBIC)
    return transform(source) if transform else source
