"""Tiền xử lý ảnh đầu vào cho các bộ dữ liệu XBone-Net.

``direct_resize`` là tên tương thích của quy trình hiện tại: ảnh RGB gốc được
chuyển thẳng cho transform chuẩn của backbone. Với XBone-Net và BiomedCLIP,
transform này chính là resize, center-crop và chuẩn hóa gốc của BiomedCLIP.
Mô-đun không tự đổi kích thước, cắt ảnh hay đệm viền.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from PIL import Image


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
        transform chuẩn của backbone và không tự resize trước.

    Returns
    -------
    Any
        Ảnh PIL đã xử lý hoặc tensor do ``transform`` tạo ra.

    Raises
    ------
    ValueError
        Nếu chiến lược không phải ``direct_resize``.
    """
    config = config or {}
    strategy = str(config.get("strategy", "direct_resize")).lower()
    if strategy != "direct_resize":
        raise ValueError(
            f"Unsupported preprocessing strategy: {strategy!r}. "
            "Only 'direct_resize' (the backbone's native transform) is supported."
        )
    source = image.convert("RGB")
    return transform(source) if transform else source
