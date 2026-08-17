"""Tiền xử lý ảnh đầu vào cho các bộ dữ liệu XBone-Net.

Mô-đun chỉ cung cấp hai chiến lược dùng trong các thí nghiệm hiện tại:
``letterbox`` giữ nguyên tỷ lệ ảnh và đệm thành hình vuông, còn
``direct_resize`` chuyển ảnh gốc trực tiếp cho transform của backbone.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import numpy as np
from PIL import Image


def _border_pixels(image: Image.Image, border_fraction: float) -> np.ndarray:
    """Lấy các pixel nằm trong dải biên của ảnh RGB.

    Parameters
    ----------
    image : PIL.Image.Image
        Ảnh nguồn.
    border_fraction : float
        Tỷ lệ chiều ngắn dùng làm độ rộng dải biên.

    Returns
    -------
    numpy.ndarray
        Mảng pixel biên có dạng ``[N, 3]``.
    """
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = array.shape[:2]
    band = max(1, round(min(height, width) * border_fraction))
    return np.concatenate(
        [
            array[:band].reshape(-1, 3),
            array[-band:].reshape(-1, 3),
            array[:, :band].reshape(-1, 3),
            array[:, -band:].reshape(-1, 3),
        ],
        axis=0,
    )


def estimate_border_color(
    image: Image.Image,
    border_fraction: float = 0.04,
) -> tuple[int, int, int]:
    """Ước lượng màu nền bằng trung vị các pixel ở biên ảnh.

    Parameters
    ----------
    image : PIL.Image.Image
        Ảnh nguồn.
    border_fraction : float, default=0.04
        Tỷ lệ chiều ngắn dùng làm độ rộng dải biên.

    Returns
    -------
    tuple of int
        Màu RGB dùng để đệm ảnh.

    Raises
    ------
    ValueError
        Nếu ``border_fraction`` không thuộc ``(0, 0.25]``.
    """
    if not 0 < border_fraction <= 0.25:
        raise ValueError("border_fraction must be in (0, 0.25].")
    median = np.median(_border_pixels(image, border_fraction), axis=0)
    return tuple(int(round(value)) for value in median)


def _resolve_pad_color(
    image: Image.Image,
    pad_value: str | int | tuple[int, int, int],
    border_fraction: float,
) -> tuple[int, int, int]:
    """Chuẩn hóa cấu hình màu đệm thành bộ ba RGB."""
    if isinstance(pad_value, str):
        value = pad_value.lower()
        if value == "border_median":
            return estimate_border_color(image, border_fraction)
        if value == "black":
            return (0, 0, 0)
        raise ValueError(
            "pad_value must be 'border_median', 'black', an int, or RGB."
        )
    if isinstance(pad_value, int):
        value = min(255, max(0, pad_value))
        return (value, value, value)
    if len(pad_value) != 3:
        raise ValueError("RGB pad_value must contain exactly three values.")
    return tuple(min(255, max(0, int(value))) for value in pad_value)


def letterbox_square(
    image: Image.Image,
    size: int = 224,
    pad_value: str | int | tuple[int, int, int] = "black",
    border_fraction: float = 0.04,
) -> Image.Image:
    """Đổi kích thước ảnh theo tỷ lệ và đệm thành ảnh vuông.

    Parameters
    ----------
    image : PIL.Image.Image
        Ảnh nguồn với tỷ lệ bất kỳ.
    size : int, default=224
        Kích thước cạnh ảnh đầu ra.
    pad_value : str or int or tuple of int, default="black"
        Màu phần đệm. Hỗ trợ ``"black"``, ``"border_median"``, một mức xám
        hoặc bộ ba RGB.
    border_fraction : float, default=0.04
        Tỷ lệ dải biên khi ước lượng ``border_median``.

    Returns
    -------
    PIL.Image.Image
        Ảnh RGB vuông có kích thước ``size x size``.

    Raises
    ------
    ValueError
        Nếu ``size`` không dương hoặc màu đệm không hợp lệ.
    """
    source = image.convert("RGB")
    side = int(size)
    if side < 1:
        raise ValueError("size must be positive.")
    scale = min(side / source.width, side / source.height)
    resized_size = (
        max(1, min(side, round(source.width * scale))),
        max(1, min(side, round(source.height * scale))),
    )
    resized = source.resize(resized_size, resample=Image.Resampling.BICUBIC)
    canvas = Image.new(
        "RGB",
        (side, side),
        color=_resolve_pad_color(source, pad_value, border_fraction),
    )
    canvas.paste(
        resized,
        ((side - resized.width) // 2, (side - resized.height) // 2),
    )
    return canvas


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
        Cấu hình gồm ``strategy``, ``target_size``, ``pad_value`` và
        ``border_fraction``.

    Returns
    -------
    Any
        Ảnh PIL đã xử lý hoặc tensor do ``transform`` tạo ra.

    Raises
    ------
    ValueError
        Nếu chiến lược không phải ``letterbox`` hoặc ``direct_resize``.
    """
    config = config or {}
    strategy = str(config.get("strategy", "letterbox")).lower()
    if strategy == "letterbox":
        source = letterbox_square(
            image,
            size=int(config.get("target_size", 224)),
            pad_value=config.get("pad_value", "black"),
            border_fraction=float(config.get("border_fraction", 0.04)),
        )
    elif strategy == "direct_resize":
        source = image.convert("RGB")
    else:
        raise ValueError(
            f"Unsupported preprocessing strategy: {strategy!r}. "
            "Choose 'letterbox' or 'direct_resize'."
        )
    return transform(source) if transform else source
