"""Cung cấp thành phần dữ liệu ctch cho XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .high_resolution import prepare_global_image, prepare_high_resolution_inputs
from .sampling import cross_class_donor_indices, deranged_donor_indices


# ============================================================
# Bộ nạp dữ liệu CTCH
# ============================================================

class CTCHDataset(Dataset):
    """Biểu diễn và truy xuất dữ liệu bằng lớp ``CTCHDataset``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(
        self,
        img_dir: str,
        csv_split_path: str,
        csv_labels_path: str,
        pathologies: list = None,
        classes: list = None,
        task_type: str = "multiclass",
        num_classes: int = None,
        split: str = "train",
        train_ratio: float = 1.0,
        k_shot: int = None,
        seed: int = 42,
        transform=None,
        tokenizer=None,
        max_text_len=256,
        # Hỗ trợ đồng thời hai loại báo cáo
        xray_report_dir: str = None,
        clinical_report_dir: str = None,
        # Chuẩn bị và xử lý đầu vào hoặc đặc trưng văn bản.
        report_dir: str = None,
        high_res: dict = None,
        preprocess: dict = None,
        **kwargs,
    ):
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        img_dir : str
            Đường dẫn tài nguyên được sử dụng.
        csv_split_path : str
            Đường dẫn tài nguyên được sử dụng.
        csv_labels_path : str
            Đường dẫn tài nguyên được sử dụng.
        pathologies : list, optional
            Danh sách tên bệnh lý hoặc lớp đích.
        classes : list, optional
            Giá trị ``classes`` được sử dụng trong phép xử lý.
        task_type : str, optional
            Phương pháp hoặc chế độ xử lý được chọn.
        num_classes : int, optional
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        split : str, optional
            Giá trị ``split`` được sử dụng trong phép xử lý.
        train_ratio : float, optional
            Số lượng, kích thước hoặc tỷ lệ được sử dụng.
        k_shot : int, optional
            Giá trị ``k_shot`` được sử dụng trong phép xử lý.
        seed : int, optional
            Hạt giống phục vụ khả năng tái lập.
        transform : object, optional
            Giá trị ``transform`` được sử dụng trong phép xử lý.
        tokenizer : object, optional
            Giá trị ``tokenizer`` được sử dụng trong phép xử lý.
        max_text_len : object, optional
            Văn bản hoặc biểu diễn văn bản đầu vào.
        xray_report_dir : str, optional
            Đường dẫn tài nguyên được sử dụng.
        clinical_report_dir : str, optional
            Đường dẫn tài nguyên được sử dụng.
        report_dir : str, optional
            Đường dẫn tài nguyên được sử dụng.
        high_res : dict, optional
            Giá trị ``high_res`` được sử dụng trong phép xử lý.
        preprocess : dict, optional
            Giá trị ``preprocess`` được sử dụng trong phép xử lý.
        **kwargs : dict
            Các đối số từ khóa bổ sung.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        self.img_dir = img_dir
        self.classes = classes or pathologies or []
        self.task_type = task_type
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.high_res_cfg = high_res or {}
        self.use_high_res = bool(self.high_res_cfg.get("enabled", False))
        self.cache_high_res_selection = bool(
            self.high_res_cfg.get("cache_selection", True)
        )
        self._high_res_selection_cache = {}
        self.preprocess_cfg = preprocess or {}
        self.text_only = bool(kwargs.get("text_only", False))
        self.shuffle_reports_all_splits = bool(kwargs.get("shuffle_reports", False))
        self.shuffle_report_mode = str(
            kwargs.get("shuffle_report_mode", "derangement")
        ).lower()
        configured_shuffle_splits = kwargs.get("shuffle_report_splits", []) or []
        self.shuffle_report_splits = {
            "validate" if str(name).lower() == "val" else str(name).lower()
            for name in configured_shuffle_splits
        }

        # Chuẩn bị và xử lý đầu vào hoặc đặc trưng văn bản.
        if report_dir and not xray_report_dir and not clinical_report_dir:
            # Chuẩn bị và xử lý đầu vào hoặc đặc trưng văn bản.
            self.xray_report_dir = report_dir
            self.clinical_report_dir = report_dir
        else:
            self.xray_report_dir = xray_report_dir
            self.clinical_report_dir = clinical_report_dir

        # --- Tải và kết hợp siêu dữ liệu ---
        df_split = pd.read_csv(csv_split_path)
        df_labels = pd.read_csv(csv_labels_path)
        df_merged = pd.merge(df_split, df_labels, on=["image_id"], how="inner")

        # Kiểm tra điều kiện trước khi thực hiện nhánh xử lý tương ứng.
        if task_type == "multiclass" and "class_id" not in df_merged.columns and self.classes:
            missing_class_columns = [
                class_name
                for class_name in self.classes
                if class_name not in df_merged.columns
            ]
            if missing_class_columns:
                raise ValueError(
                    "CTCH label manifest does not match dataset classes. "
                    f"Missing {len(missing_class_columns)} columns, including "
                    f"{missing_class_columns[:3]}. Regenerate ctch-labels.csv with "
                    "data/CTCH/preprocess_ctch.py."
                )

            def _get_class_id(row):
                """Lấy class id cho bước xử lý hiện tại.

                Parameters
                ----------
                row : object
                    Giá trị ``row`` được sử dụng trong phép xử lý.

                Returns
                -------
                object
                    Kết quả được tạo bởi bước xử lý của hàm.

                Raises
                ------
                ValueError
                    Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
                """
                for idx_cls, cls_name in enumerate(self.classes):
                    if row.get(cls_name, 0) == 1:
                        return idx_cls
                raise ValueError("A CTCH sample has no active multiclass label")
            df_merged["class_id"] = df_merged.apply(_get_class_id, axis=1)

        if task_type == "multiclass" and "class_id" in df_merged.columns:
            class_ids = pd.to_numeric(df_merged["class_id"], errors="coerce")
            expected_classes = num_classes or len(self.classes)
            invalid = class_ids.isna() | class_ids.lt(0)
            if expected_classes:
                invalid |= class_ids.ge(expected_classes)
            if invalid.any():
                raise ValueError(
                    f"CTCH label manifest contains {int(invalid.sum())} invalid class_id values"
                )
            df_merged["class_id"] = class_ids.astype(int)

        # Chuẩn bị dữ liệu và chiến lược lấy mẫu tương ứng.
        current_split = "validate" if split == "val" else split
        filtered_df = df_merged[df_merged["split"] == current_split].reset_index(drop=True)

        if current_split == "train" and k_shot is not None:
            if task_type != "multiclass" or "class_id" not in filtered_df.columns:
                raise ValueError(
                    "CTCH k-shot sampling requires multiclass labels in class_id."
                )
            if isinstance(k_shot, bool) or not isinstance(k_shot, int) or k_shot < 1:
                raise ValueError("k_shot must be a positive integer or null.")

            sampled_groups = [
                group.sample(n=min(len(group), k_shot), random_state=seed)
                for _, group in filtered_df.groupby("class_id", sort=True)
            ]
            filtered_df = pd.concat(sampled_groups, ignore_index=True)
            print(
                f"[Dataset] CTCH few-shot learning: requested={k_shot}-shot, "
                f"classes={filtered_df['class_id'].nunique()}, "
                f"samples={len(filtered_df)}, seed={seed}."
            )
        elif current_split == "train" and train_ratio < 1.0:
            filtered_df = filtered_df.sample(
                frac=train_ratio, random_state=seed
            ).reset_index(drop=True)
            print(
                "[Dataset] Subsampling enabled: using "
                f"{train_ratio * 100:.1f}% of training data (seed={seed})."
            )

        self.df = filtered_df
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
                    self.df["class_id"].to_numpy(), seed
                )
            elif self.shuffle_report_mode == "derangement":
                self.shuffled_report_indices = deranged_donor_indices(
                    len(self.df), seed
                )
            else:
                raise ValueError(
                    "shuffle_report_mode must be 'cross_class' or 'derangement'."
                )
        else:
            self.shuffled_report_indices = None

        # --- Kiểm tra khả năng sử dụng hai loại báo cáo ---
        has_dual = bool(self.xray_report_dir and self.clinical_report_dir)
        print(
            f"[Dataset] CTCH '{split.upper()}' initialized with {len(self.df)} samples. "
            f"Task: {task_type}, Dual reports: {has_dual}, Sparse high-res views: "
            f"{self.use_high_res}, Text-only: {self.text_only}, Reports shuffled: "
            f"{self.shuffle_reports}."
        )

    def __len__(self):
        """Thực hiện bước len trong quy trình hiện tại.

        Returns
        -------
        int
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        return len(self.df)

    def _load_report(self, report_dir: str, image_id: str) -> str:
        """Tải báo cáo cho bước xử lý hiện tại.

        Parameters
        ----------
        report_dir : str
            Đường dẫn tài nguyên được sử dụng.
        image_id : str
            Ảnh hoặc biểu diễn ảnh đầu vào.

        Returns
        -------
        str
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        if report_dir is None:
            return ""
        file_name = os.path.splitext(image_id)[0]
        report_path = os.path.join(report_dir, f"{file_name}.txt")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                return f.read().strip().lower()
        return ""

    def _tokenize(self, text: str):
        """Thực hiện bước tokenize trong quy trình hiện tại.

        Parameters
        ----------
        text : str
            Văn bản hoặc biểu diễn văn bản đầu vào.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        if not text:
            text = "no clinical information available."
        if self.tokenizer:
            return self.tokenizer([text]).squeeze(0)
        return text

    def __getitem__(self, idx: int):
        """Thực hiện bước getitem trong quy trình hiện tại.

        Parameters
        ----------
        idx : int
            Chỉ mục của phần tử cần xử lý.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])

        # --- Tải ảnh đầu vào ---
        img_path = os.path.join(self.img_dir, image_id)
        if self.text_only:
            image = Image.new("RGB", (224, 224), color="black")
        else:
            try:
                image = Image.open(img_path).convert("RGB")
            except FileNotFoundError:
                image = Image.new("RGB", (224, 224), color="black")

        if self.use_high_res:
            cache_key = "__text_only__" if self.text_only else image_id
            cached_selection = self._high_res_selection_cache.get(cache_key)
            high_res_fields, selection = prepare_high_resolution_inputs(
                image,
                self.transform,
                self.high_res_cfg,
                selection=cached_selection,
                return_selection=True,
            )
            if self.cache_high_res_selection and cached_selection is None:
                self._high_res_selection_cache[cache_key] = selection
        else:
            high_res_fields = {}
            image = prepare_global_image(
                image,
                self.transform,
                self.preprocess_cfg,
            )

        # Chuẩn hóa chuỗi token và mặt nạ đệm cho batch.
        report_image_id = image_id
        if self.shuffled_report_indices is not None:
            report_image_id = str(
                self.df.iloc[self.shuffled_report_indices[idx]]["image_id"]
            )

        xray_text = self._load_report(self.xray_report_dir, report_image_id)
        clinical_text = self._load_report(
            self.clinical_report_dir, report_image_id
        )

        xray_ids = self._tokenize(xray_text)
        clinical_ids = self._tokenize(clinical_text)

        # --- Mã hóa nhãn ---
        if self.task_type == "multiclass":
            if "class_id" in row:
                labels = torch.tensor(int(row["class_id"]), dtype=torch.long)
            else:
                class_vals = [int(row.get(c, 0)) for c in self.classes]
                class_id = class_vals.index(1) if 1 in class_vals else 0
                labels = torch.tensor(class_id, dtype=torch.long)
        else:
            # Đa nhãn được biểu diễn bằng vectơ nhị phân
            label_vals = []
            for path in self.classes:
                val = row.get(path, 0)
                label_vals.append(0.0 if pd.isna(val) else float(val))
            labels = torch.tensor(label_vals, dtype=torch.float32)

        if self.use_high_res:
            return {
                **high_res_fields,
                "xray_input_ids": xray_ids,
                "clinical_input_ids": clinical_ids,
                "labels": labels,
            }
        return image, xray_ids, clinical_ids, labels
