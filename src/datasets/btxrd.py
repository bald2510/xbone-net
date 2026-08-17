"""Dataset BTXRD với preprocessing ảnh thống nhất cùng CTCH."""

from __future__ import annotations

import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .preprocessing import prepare_image
from .sampling import cross_class_donor_indices, deranged_donor_indices


BTXRD_CLASS_NAMES = (
    "normal",
    "osteochondroma",
    "osteosarcoma",
    "multiple osteochondromas",
    "simple bone cyst",
    "other bt",
    "giant cell tumor",
    "synovial osteochondroma",
    "other mt",
    "osteofibroma",
)


class BTXRDDataset(Dataset):
    """Đọc ảnh u xương, báo cáo tổng hợp và nhãn đa lớp BTXRD."""

    def __init__(
        self,
        img_dir: str,
        report_dir: str,
        csv_split_path: str,
        csv_labels_path: str,
        pathologies: list | None = None,
        classes: list | None = None,
        task_type: str = "multiclass",
        num_classes: int | None = None,
        split: str = "train",
        train_ratio: float = 1.0,
        transform=None,
        tokenizer=None,
        max_text_len: int = 256,
        clinical_subdir: str = "clinical_v2",
        k_shot: int | None = None,
        seed: int = 42,
        preprocess: dict | None = None,
        **kwargs,
    ):
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        img_dir : str
            Đường dẫn tài nguyên được sử dụng.
        report_dir : str
            Đường dẫn tài nguyên được sử dụng.
        csv_split_path : str
            Đường dẫn tài nguyên được sử dụng.
        csv_labels_path : str
            Đường dẫn tài nguyên được sử dụng.
        pathologies : list | None, optional
            Danh sách tên bệnh lý hoặc lớp đích.
        classes : list | None, optional
            Giá trị ``classes`` được sử dụng trong phép xử lý.
        task_type : str, optional
            Phương pháp hoặc chế độ xử lý được chọn.
        num_classes : int | None, optional
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        split : str, optional
            Giá trị ``split`` được sử dụng trong phép xử lý.
        train_ratio : float, optional
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        transform : object, optional
            Giá trị ``transform`` được sử dụng trong phép xử lý.
        tokenizer : object, optional
            Giá trị ``tokenizer`` được sử dụng trong phép xử lý.
        max_text_len : int, optional
            Văn bản hoặc biểu diễn văn bản đầu vào.
        clinical_subdir : str, optional
            Giá trị ``clinical_subdir`` được sử dụng trong phép xử lý.
        k_shot : int | None, optional
            Giá trị ``k_shot`` được sử dụng trong phép xử lý.
        seed : int, optional
            Hạt giống phục vụ khả năng tái lập.
        preprocess : dict | None, optional
            Chiến lược chuẩn hóa ảnh trước transform của backbone.
        **kwargs : dict
            Các đối số từ khóa bổ sung.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        del num_classes
        self.img_dir = img_dir
        self.report_dir = report_dir
        self.classes = classes or pathologies or []
        self.task_type = task_type
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

        split_frame = pd.read_csv(csv_split_path)
        label_frame = pd.read_csv(csv_labels_path)
        merged = pd.merge(
            split_frame,
            label_frame,
            on=["image_id"],
            how="inner",
            validate="one_to_one",
        )

        if self.task_type == "multiclass" and "class_id" not in merged.columns:
            if not self.classes:
                raise ValueError(
                    "BTXRD multiclass loading requires dataset.params.classes when "
                    "the label manifest has no class_id column."
                )
            missing_class_columns = [
                name for name in self.classes if name not in merged.columns
            ]
            if missing_class_columns:
                raise ValueError(
                    "BTXRD label manifest does not match dataset classes. "
                    f"Missing columns: {missing_class_columns}."
                )
            label_matrix = merged[self.classes].apply(
                pd.to_numeric, errors="coerce"
            )
            # Bước hỗ trợ để khởi tạo trạng thái.
            # Thiết lập giá trị trung gian cho bước xử lý tiếp theo.
            hierarchical_children = {
                "osteochondroma": (
                    "multiple osteochondromas",
                    "synovial osteochondroma",
                ),
            }
            for parent, children in hierarchical_children.items():
                present_children = [
                    child for child in children if child in label_matrix.columns
                ]
                if parent in label_matrix.columns and present_children:
                    child_is_active = label_matrix[present_children].eq(1).any(axis=1)
                    label_matrix.loc[child_is_active, parent] = 0
            active_counts = label_matrix.eq(1).sum(axis=1)
            invalid_rows = label_matrix.isna().any(axis=1) | active_counts.ne(1)
            if invalid_rows.any():
                raise ValueError(
                    "BTXRD multiclass labels must contain exactly one active class "
                    f"per sample; invalid rows={int(invalid_rows.sum())}."
                )
            merged["class_id"] = label_matrix.to_numpy().argmax(axis=1)

        if self.task_type == "multiclass":
            class_ids = pd.to_numeric(merged["class_id"], errors="coerce")
            expected_classes = len(self.classes)
            invalid_ids = class_ids.isna() | class_ids.lt(0)
            if expected_classes:
                invalid_ids |= class_ids.ge(expected_classes)
            if invalid_ids.any():
                raise ValueError(
                    "BTXRD label manifest contains "
                    f"{int(invalid_ids.sum())} invalid class_id values."
                )
            merged["class_id"] = class_ids.astype(int)

        current_split = "validate" if split == "val" else split
        filtered = merged[merged["split"] == current_split].reset_index(drop=True)
        if current_split == "train" and k_shot is not None:
            if self.task_type != "multiclass" or "class_id" not in filtered.columns:
                raise ValueError(
                    "BTXRD k-shot sampling requires multiclass labels in class_id."
                )
            if isinstance(k_shot, bool) or not isinstance(k_shot, int) or k_shot < 1:
                raise ValueError("k_shot must be a positive integer or null.")
            sampled_groups = [
                group.sample(n=min(len(group), k_shot), random_state=seed)
                for _, group in filtered.groupby("class_id", sort=True)
            ]
            if not sampled_groups:
                raise ValueError("BTXRD training split is empty; cannot apply k-shot sampling.")
            filtered = pd.concat(sampled_groups, ignore_index=True)
            print(
                f"[Dataset] BTXRD few-shot learning: requested={k_shot}-shot, "
                f"classes={filtered['class_id'].nunique()}, "
                f"samples={len(filtered)}, seed={seed}."
            )
        elif current_split == "train" and train_ratio < 1.0:
            filtered = filtered.sample(frac=train_ratio, random_state=seed).reset_index(
                drop=True
            )
            print(f"[Dataset] Training subset: {train_ratio * 100:.1f}%.")
        self.df = filtered

        self.xray_dir = os.path.join(report_dir, "xray")
        self.clinical_dir = os.path.join(report_dir, clinical_subdir)
        self.has_dual_reports = os.path.isdir(self.xray_dir) and os.path.isdir(
            self.clinical_dir
        )

        self.preprocess_cfg = preprocess or {}
        self.text_only = bool(kwargs.get("text_only", False))
        self.shuffle_reports_all_splits = bool(
            kwargs.get("shuffle_reports", False)
        )
        self.shuffle_report_mode = str(
            kwargs.get("shuffle_report_mode", "derangement")
        ).lower()
        configured_shuffle_splits = kwargs.get(
            "shuffle_report_splits",
            [],
        ) or []
        self.shuffle_report_splits = {
            "validate" if str(name).lower() == "val" else str(name).lower()
            for name in configured_shuffle_splits
        }
        self.shuffle_reports = (
            self.shuffle_reports_all_splits
            or current_split in self.shuffle_report_splits
        )
        if self.shuffle_reports:
            if self.shuffle_report_mode == "cross_class":
                if "class_id" not in self.df.columns:
                    raise ValueError(
                        "shuffle_report_mode='cross_class' requires class_id."
                    )
                self.shuffled_report_indices = cross_class_donor_indices(
                    self.df["class_id"].to_numpy(),
                    seed,
                )
            elif self.shuffle_report_mode == "derangement":
                self.shuffled_report_indices = deranged_donor_indices(
                    len(self.df),
                    seed,
                )
            else:
                raise ValueError(
                    "shuffle_report_mode must be 'cross_class' or "
                    "'derangement'."
                )
        else:
            self.shuffled_report_indices = None

        print(
            f"[Dataset] Initialized '{split.upper()}' with {len(self.df)} samples. "
            f"Dual reports: {self.has_dual_reports}. "
            f"Preprocess: {self.preprocess_cfg.get('strategy', 'letterbox')}. "
            f"Text-only: {self.text_only}. "
            f"Reports shuffled: {self.shuffle_reports}."
        )

    def __len__(self):
        """Thực hiện bước len trong quy trình hiện tại.

        Returns
        -------
        int
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        return len(self.df)

    def _load_and_tokenize(self, path: str, default_text: str):
        """Tải and tokenize cho bước xử lý hiện tại.

        Parameters
        ----------
        path : str
            Đường dẫn tài nguyên được sử dụng.
        default_text : str
            Văn bản hoặc biểu diễn văn bản đầu vào.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        raw_text = ""
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as report_file:
                raw_text = report_file.read()
        text = raw_text.strip().lower() or default_text
        return self.tokenizer([text]).squeeze(0) if self.tokenizer else text

    def _label(self, row) -> torch.Tensor:
        """Thực hiện bước nhãn trong quy trình hiện tại.

        Parameters
        ----------
        row : object
            Giá trị ``row`` được sử dụng trong phép xử lý.

        Returns
        -------
        torch.Tensor
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        if self.task_type == "multiclass":
            return torch.tensor(int(row["class_id"]), dtype=torch.long)
        return torch.tensor(
            [float(row.get(name, 0.0)) for name in self.classes],
            dtype=torch.float32,
        )

    def __getitem__(self, index: int):
        """Thực hiện bước getitem trong quy trình hiện tại.

        Parameters
        ----------
        index : int
            Chỉ mục của phần tử cần xử lý.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        row = self.df.iloc[index]
        image_id = str(row["image_id"])
        stem = os.path.splitext(image_id)[0]
        image_path = os.path.join(self.img_dir, image_id)

        if self.text_only:
            image = Image.new("RGB", (224, 224), color="black")
        else:
            try:
                image = Image.open(image_path).convert("RGB")
            except FileNotFoundError:
                image = Image.new("RGB", (224, 224), color="black")

        report_stem = stem
        if self.shuffled_report_indices is not None:
            shuffled_id = str(
                self.df.iloc[self.shuffled_report_indices[index]]["image_id"]
            )
            report_stem = os.path.splitext(shuffled_id)[0]

        labels = self._label(row)
        image = prepare_image(image, self.transform, self.preprocess_cfg)

        if self.has_dual_reports:
            xray_ids = self._load_and_tokenize(
                os.path.join(self.xray_dir, f"{report_stem}.txt"),
                "no clear bone abnormalities or fracture identified.",
            )
            clinical_ids = self._load_and_tokenize(
                os.path.join(self.clinical_dir, f"{report_stem}.txt"),
                "no clinical information available.",
            )
            return image, xray_ids, clinical_ids, labels

        input_ids = self._load_and_tokenize(
            os.path.join(self.report_dir, f"{report_stem}.txt"),
            "no clear bone abnormalities or fracture identified.",
        )
        return image, input_ids, labels
