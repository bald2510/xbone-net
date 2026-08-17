"""Dataset ngoại phân phối dùng cho benchmark CTCH và BTXRD."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.analysis import MetadataDataset

from .preprocessing import prepare_image


def _file_issue(path: Path) -> Optional[str]:
    """Thực hiện bước file issue trong quy trình hiện tại.

    Parameters
    ----------
    path : Path
        Đường dẫn tài nguyên được sử dụng.

    Returns
    -------
    Optional[str]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    if not path.is_file():
        return "missing"
    try:
        with path.open("rb") as handle:
            handle.read(1)
    except OSError as error:
        return f"unreadable:{type(error).__name__}"
    return None


class CTCHOODDataset(Dataset):
    """Đọc cohort semantic-OOD CTCH từ manifest và kiểm tra đủ tệp bắt buộc."""

    def __init__(
        self,
        img_dir: str,
        xray_report_dir: str,
        clinical_report_dir: str,
        csv_manifest_path: str,
        split: str = "test",
        transform=None,
        tokenizer=None,
        preprocess: Optional[dict] = None,
        required_report_types: tuple[str, ...] = ("xray", "clinical"),
        allow_missing: bool = False,
        **_: Any,
    ) -> None:
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        img_dir : str
            Đường dẫn tài nguyên được sử dụng.
        xray_report_dir : str
            Đường dẫn tài nguyên được sử dụng.
        clinical_report_dir : str
            Đường dẫn tài nguyên được sử dụng.
        csv_manifest_path : str
            Đường dẫn tài nguyên được sử dụng.
        split : str, optional
            Giá trị ``split`` được sử dụng trong phép xử lý.
        transform : object, optional
            Giá trị ``transform`` được sử dụng trong phép xử lý.
        tokenizer : object, optional
            Giá trị ``tokenizer`` được sử dụng trong phép xử lý.
        preprocess : Optional[dict]
            Chiến lược chuẩn hóa ảnh trước transform của backbone.
        required_report_types : tuple[str, ...], optional
            Các loại báo cáo bắt buộc phải tồn tại để giữ mẫu trong tập dữ liệu.
        allow_missing : bool, optional
            Giá trị ``allow_missing`` được sử dụng trong phép xử lý.
        **_ : Any
            Giá trị ``_`` được sử dụng trong phép xử lý.

        Raises
        ------
        FileNotFoundError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        RuntimeError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        if split not in {"test", "ood"}:
            raise ValueError("CTCHOODDataset supports only split='test' or 'ood'.")
        self.img_dir = str(img_dir)
        self.xray_report_dir = str(xray_report_dir)
        self.clinical_report_dir = str(clinical_report_dir)
        self.transform = transform
        self.tokenizer = tokenizer
        self.preprocess_cfg = preprocess or {}
        self.required_report_types = tuple(required_report_types)
        unknown_report_types = set(self.required_report_types) - {
            "xray",
            "clinical",
        }
        if unknown_report_types:
            raise ValueError(
                "required_report_types only supports 'xray' and 'clinical', got "
                f"{sorted(unknown_report_types)}."
            )

        manifest = Path(csv_manifest_path)
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing CTCH OOD manifest: {manifest}")
        df = pd.read_csv(manifest)
        if "image_id" not in df.columns:
            raise ValueError("CTCH OOD manifest must contain image_id.")
        df["image_id"] = df["image_id"].astype(str)

        complete = []
        missing: list[dict[str, Any]] = []
        for index, row in df.iterrows():
            image_id = str(row["image_id"])
            stem = Path(image_id).stem
            absent = []
            required_files = {
                "image": Path(self.img_dir) / image_id,
            }
            if "xray" in self.required_report_types:
                required_files["xray_report"] = (
                    Path(self.xray_report_dir) / f"{stem}.txt"
                )
            if "clinical" in self.required_report_types:
                required_files["clinical_report"] = (
                    Path(self.clinical_report_dir) / f"{stem}.txt"
                )
            for role, path in required_files.items():
                issue = _file_issue(path)
                if issue is not None:
                    absent.append(f"{role}:{issue}")
            complete.append(not absent)
            if absent:
                missing.append({"image_id": image_id, "missing": absent})

        self.coverage = {
            "manifest_rows": int(len(df)),
            "complete_rows": int(sum(complete)),
            "missing_rows": int(len(missing)),
            "coverage_fraction": float(sum(complete) / max(len(df), 1)),
            "allow_missing": bool(allow_missing),
            "required_report_types": list(self.required_report_types),
            "missing": missing,
        }
        if missing and not allow_missing:
            preview = ", ".join(item["image_id"] for item in missing[:5])
            raise FileNotFoundError(
                "CTCH semantic-OOD is incomplete: "
                f"{len(missing)}/{len(df)} manifest rows lack an image or report "
                f"({preview}). Re-run data/CTCH/preprocess_ctch.py, or pass "
                "--allow-incomplete-ood only for an exploratory run."
            )
        self.df = df.loc[np.asarray(complete, dtype=bool)].reset_index(drop=True)
        if self.df.empty:
            raise RuntimeError("No complete CTCH OOD samples are available.")

    def __len__(self) -> int:
        """Thực hiện bước len trong quy trình hiện tại.

        Returns
        -------
        int
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        return len(self.df)

    def _report(self, directory: str, image_id: str) -> torch.Tensor | str:
        """Thực hiện bước báo cáo trong quy trình hiện tại.

        Parameters
        ----------
        directory : str
            Giá trị ``directory`` được sử dụng trong phép xử lý.
        image_id : str
            Ảnh hoặc biểu diễn ảnh đầu vào.

        Returns
        -------
        torch.Tensor | str
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        path = Path(directory) / f"{Path(image_id).stem}.txt"
        text = (
            path.read_text(encoding="utf-8").strip().lower()
            if path.is_file()
            else ""
        )
        text = text or "no clinical information available."
        return self.tokenizer([text]).squeeze(0) if self.tokenizer else text

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Thực hiện bước getitem trong quy trình hiện tại.

        Parameters
        ----------
        index : int
            Chỉ mục của phần tử cần xử lý.

        Returns
        -------
        dict[str, Any]
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        row = self.df.iloc[index]
        image_id = str(row["image_id"])
        image = Image.open(Path(self.img_dir) / image_id).convert("RGB")
        image_fields = {
            "pixel_values": prepare_image(
                image, self.transform, self.preprocess_cfg
            )
        }

        columns = list(self.df.columns)
        patient_value = row[columns[1]] if len(columns) > 1 else ""
        group_value = row[columns[-2]] if len(columns) > 2 else "semantic_ood"
        return {
            **image_fields,
            "xray_input_ids": self._report(self.xray_report_dir, image_id),
            "clinical_input_ids": self._report(self.clinical_report_dir, image_id),
            "labels": torch.tensor(-1, dtype=torch.long),
            "image_id": image_id,
            "group": str(group_value),
            "patient_id": str(patient_value),
            "report_source_id": image_id,
            "scenario": "semantic_ood",
        }


def cross_class_derangement(
    labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Thực hiện bước cross class derangement trong quy trình hiện tại.

    Parameters
    ----------
    labels : np.ndarray
        Giá trị ``labels`` được sử dụng trong phép xử lý.
    seed : int
        Hạt giống phục vụ khả năng tái lập.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
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
    maximum = int(counts.max())
    if maximum * 2 > labels.size:
        majority = classes[int(np.argmax(counts))]
        raise ValueError(
            "A complete cross-class derangement is impossible because class "
            f"{majority!r} contains {maximum}/{labels.size} samples."
        )
    ordered_parts = []
    for class_id in classes:
        indices = np.flatnonzero(labels == class_id)
        ordered_parts.append(rng.permutation(indices))
    recipients = np.concatenate(ordered_parts)
    donors = np.roll(recipients, -maximum)
    if np.any(labels[recipients] == labels[donors]) or np.any(recipients == donors):
        raise RuntimeError("Failed to construct a valid cross-class derangement.")
    return recipients.astype(np.int64), donors.astype(np.int64)


def same_class_derangement(
    labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Thực hiện bước same class derangement trong quy trình hiện tại.

    Parameters
    ----------
    labels : np.ndarray
        Giá trị ``labels`` được sử dụng trong phép xử lý.
    seed : int
        Hạt giống phục vụ khả năng tái lập.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    RuntimeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    labels = np.asarray(labels).reshape(-1)
    rng = np.random.default_rng(seed)
    recipients, donors = [], []
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        if len(indices) < 2:
            continue
        indices = rng.permutation(indices)
        recipients.extend(indices.tolist())
        donors.extend(np.roll(indices, -1).tolist())
    recipient_array = np.asarray(recipients, dtype=np.int64)
    donor_array = np.asarray(donors, dtype=np.int64)
    if recipient_array.size == 0:
        raise ValueError("No class contains at least two samples.")
    if np.any(recipient_array == donor_array):
        raise RuntimeError("Same-class derangement contains a fixed point.")
    if np.any(labels[recipient_array] != labels[donor_array]):
        raise RuntimeError("Same-class derangement crossed a class boundary.")
    return recipient_array, donor_array


class ReportMismatchDataset(Dataset):
    """Biểu diễn và truy xuất dữ liệu bằng lớp ``ReportMismatchDataset``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(
        self,
        dataset: Dataset,
        mode: str,
        seed: int,
    ) -> None:
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        dataset : Dataset
            Dữ liệu đầu vào của bước xử lý.
        mode : str
            Phương pháp hoặc chế độ xử lý được chọn.
        seed : int
            Hạt giống phục vụ khả năng tái lập.

        Raises
        ------
        TypeError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        if not hasattr(dataset, "df"):
            raise TypeError("ReportMismatchDataset requires a dataset with a DataFrame.")
        self.dataset = dataset
        self.df = dataset.df
        labels = self.df["class_id"].to_numpy(dtype=np.int64)
        if mode == "cross_class":
            self.recipient_indices, self.donor_indices = cross_class_derangement(
                labels, seed
            )
        elif mode == "same_class":
            self.recipient_indices, self.donor_indices = same_class_derangement(
                labels, seed
            )
        else:
            raise ValueError("mode must be 'cross_class' or 'same_class'.")
        self.mode = mode

    def __len__(self) -> int:
        """Thực hiện bước len trong quy trình hiện tại.

        Returns
        -------
        int
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        return len(self.recipient_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Thực hiện bước getitem trong quy trình hiện tại.

        Parameters
        ----------
        index : int
            Chỉ mục của phần tử cần xử lý.

        Returns
        -------
        dict[str, Any]
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        recipient = int(self.recipient_indices[index])
        donor = int(self.donor_indices[index])
        sample = MetadataDataset._as_dict(self.dataset[recipient])
        recipient_row = self.df.iloc[recipient]
        donor_row = self.df.iloc[donor]
        recipient_id = str(recipient_row["image_id"])
        donor_id = str(donor_row["image_id"])

        xray_text = self.dataset._load_report(self.dataset.xray_report_dir, donor_id)
        clinical_text = self.dataset._load_report(
            self.dataset.clinical_report_dir, donor_id
        )
        sample["xray_input_ids"] = self.dataset._tokenize(xray_text)
        sample["clinical_input_ids"] = self.dataset._tokenize(clinical_text)
        sample.update(
            {
                "image_id": recipient_id,
                "group": str(int(recipient_row["class_id"])),
                "patient_id": "",
                "report_source_id": donor_id,
                "scenario": f"report_mismatch_{self.mode}",
            }
        )
        return sample
