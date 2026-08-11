"""Cung cấp thành phần dữ liệu sampling cho XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

from __future__ import annotations

import numpy as np


def deranged_donor_indices(size: int, seed: int) -> np.ndarray:
    """Thực hiện bước deranged donor indices trong quy trình hiện tại.

    Parameters
    ----------
    size : int
        Số lượng, kích thước hoặc tỷ lệ được sử dụng.
    seed : int
        Hạt giống phục vụ khả năng tái lập.

    Returns
    -------
    np.ndarray
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    RuntimeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if size < 2:
        raise ValueError("A report derangement requires at least two samples.")
    rng = np.random.default_rng(seed)
    donors = np.arange(size, dtype=np.int64)
    for index in range(size - 1, 0, -1):
        swap_index = int(rng.integers(0, index))
        donors[index], donors[swap_index] = donors[swap_index], donors[index]
    if np.any(donors == np.arange(size)) or len(np.unique(donors)) != size:
        raise RuntimeError("Failed to construct a one-to-one derangement.")
    return donors


def cross_class_donor_indices(labels: np.ndarray, seed: int) -> np.ndarray:
    """Thực hiện bước cross class donor indices trong quy trình hiện tại.

    Parameters
    ----------
    labels : np.ndarray
        Giá trị ``labels`` được sử dụng trong phép xử lý.
    seed : int
        Hạt giống phục vụ khả năng tái lập.

    Returns
    -------
    np.ndarray
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    RuntimeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    labels = np.asarray(labels).reshape(-1)
    if labels.size < 2:
        raise ValueError("Cross-class mismatch requires at least two samples.")
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    largest_class = int(counts.max())
    if largest_class * 2 > labels.size:
        majority = classes[int(np.argmax(counts))]
        raise ValueError(
            "A complete cross-class derangement is impossible because class "
            f"{majority!r} contains {largest_class}/{labels.size} samples."
        )

    ordered = np.concatenate(
        [rng.permutation(np.flatnonzero(labels == class_id)) for class_id in classes]
    )
    shifted = np.roll(ordered, -largest_class)
    donors = np.empty(labels.size, dtype=np.int64)
    donors[ordered] = shifted
    recipients = np.arange(labels.size, dtype=np.int64)
    if np.any(recipients == donors):
        raise RuntimeError("Cross-class derangement contains a fixed point.")
    if np.any(labels[recipients] == labels[donors]):
        raise RuntimeError("Cross-class derangement retained a recipient class.")
    if len(np.unique(donors)) != labels.size:
        raise RuntimeError("Cross-class donor assignments are not one-to-one.")
    return donors
