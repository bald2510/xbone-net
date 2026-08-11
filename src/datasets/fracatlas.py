"""Cung cấp thành phần dữ liệu fracatlas cho XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

import os
from pathlib import Path

import torch
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from .high_resolution import prepare_global_image, prepare_high_resolution_inputs


FRACATLAS_IMAGE_RESOLVER_VERSION = 3
_FRACATLAS_CLASS_DIRECTORIES = {
    0: "Non_fractured",
    1: "Fractured",
}


# ============================================================
# Chuẩn bị dữ liệu và chiến lược lấy mẫu tương ứng.
# ============================================================

class FracAtlasDataset(Dataset):
    """Biểu diễn và truy xuất dữ liệu bằng lớp ``FracAtlasDataset``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(
        self, 
        img_dir: str, 
        report_dir: str,
        csv_split_path: str,
        csv_labels_path: str,
        classes: list = None,
        task_type: str = "multiclass",
        split: str = "train",
        transform=None,
        tokenizer=None,
        max_text_len=256,
        strict_files: bool = False,
        allow_truncated_images: bool = False,
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
        classes : list, optional
            Giá trị ``classes`` được sử dụng trong phép xử lý.
        task_type : str, optional
            Phương pháp hoặc chế độ xử lý được chọn.
        split : str, optional
            Giá trị ``split`` được sử dụng trong phép xử lý.
        transform : object, optional
            Giá trị ``transform`` được sử dụng trong phép xử lý.
        tokenizer : object, optional
            Giá trị ``tokenizer`` được sử dụng trong phép xử lý.
        max_text_len : object, optional
            Văn bản hoặc biểu diễn văn bản đầu vào.
        strict_files : bool, optional
            Giá trị ``strict_files`` được sử dụng trong phép xử lý.
        allow_truncated_images : bool, optional
            Giá trị ``allow_truncated_images`` được sử dụng trong phép xử lý.
        **kwargs : dict
            Các đối số từ khóa bổ sung.

        Raises
        ------
        FileNotFoundError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        OSError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        self.img_dir = img_dir
        self.report_dir = report_dir
        self.classes = classes or ['fractured']
        self.task_type = task_type
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.strict_files = bool(strict_files)
        self.allow_truncated_images = bool(allow_truncated_images)

        # --- Tải và kết hợp siêu dữ liệu ---
        df_split = pd.read_csv(csv_split_path)
        df_labels = pd.read_csv(csv_labels_path)
        df_merged = pd.merge(df_split, df_labels, on=["image_id"], how="inner")
        
        # Thiết lập giá trị trung gian cho bước xử lý tiếp theo.
        current_split = 'validate' if split == 'val' else split
        self.df = df_merged[df_merged['split'] == current_split].reset_index(drop=True)

        # Chuẩn bị và xử lý đầu vào hoặc đặc trưng hình ảnh.
        # Bước hỗ trợ để khởi tạo trạng thái.
        # Tính điểm và độ đo phát hiện dữ liệu ngoài phân phối.
        # bằng một ảnh đen có giá trị không đổi.
        self._image_paths, image_layout, missing_images = self._resolve_image_paths()
        missing_reports = [
            str(image_id)
            for image_id in self.df["image_id"].astype(str)
            if not (
                Path(self.report_dir) / f"{Path(image_id).stem}.txt"
            ).is_file()
        ]
        truncated_images: list[str] = []
        decode_failures: list[dict[str, str]] = []
        if self.strict_files and not missing_images:
            truncated_images, decode_failures = self._validate_image_decoding()
        self.coverage = {
            "rows": int(len(self.df)),
            "resolved_images": int(len(self._image_paths)),
            "missing_images": int(len(missing_images)),
            "present_reports": int(len(self.df) - len(missing_reports)),
            "missing_reports": int(len(missing_reports)),
            "decode_validation": self.strict_files,
            "truncated_image_count": int(len(truncated_images)),
            "truncated_images": truncated_images,
            "decode_failure_count": int(len(decode_failures)),
            "decode_failures": decode_failures,
            "image_layout": image_layout,
            "resolver_version": FRACATLAS_IMAGE_RESOLVER_VERSION,
            "strict_files": self.strict_files,
            "allow_truncated_images": self.allow_truncated_images,
        }
        if missing_images:
            preview = ", ".join(missing_images[:5])
            raise FileNotFoundError(
                "FracAtlas image coverage is incomplete: "
                f"{len(missing_images)}/{len(self.df)} files are missing "
                f"({preview}). Expected either <img_dir>/<image_id> or the "
                "released Fractured/Non_fractured class-directory layout."
            )
        if self.strict_files and missing_reports:
            preview = ", ".join(missing_reports[:5])
            raise FileNotFoundError(
                "FracAtlas report coverage is incomplete: "
                f"{len(missing_reports)}/{len(self.df)} files are missing "
                f"({preview})."
            )
        if decode_failures:
            preview = ", ".join(
                item["image_id"] for item in decode_failures[:5]
            )
            raise OSError(
                "FracAtlas image decoding validation failed for "
                f"{len(decode_failures)}/{len(self.df)} files ({preview})."
            )

        # Chuẩn bị và xử lý đầu vào hoặc đặc trưng hình ảnh.
        self.high_res_cfg = kwargs.get('high_res', {})
        self.use_high_res = self.high_res_cfg.get('enabled', False)
        self.cache_high_res_selection = bool(
            self.high_res_cfg.get('cache_selection', True)
        )
        self._high_res_selection_cache = {}
        self.preprocess_cfg = kwargs.get('preprocess', {})
        
        print(
            f"[FracAtlasDataset] Loaded '{split.upper()}' split with "
            f"{len(self.df)} samples. Sparse high-res views: {self.use_high_res}. "
            f"Image layout: {image_layout}"
        )

    def _resolve_image_paths(self):
        """Xác định ảnh các đường dẫn cho bước xử lý hiện tại.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.

        Raises
        ------
        ValueError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        resolved: list[Path] = []
        missing: list[str] = []
        layout_counts = {"flat": 0, "class_directory": 0}
        root = Path(self.img_dir)

        for _, row in self.df.iterrows():
            image_id = str(row["image_id"])
            try:
                fractured = int(row["fractured"])
                class_directory = _FRACATLAS_CLASS_DIRECTORIES[fractured]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid FracAtlas fractured label for {image_id!r}: "
                    f"{row.get('fractured')!r}."
                ) from error

            flat_path = root / image_id
            class_path = root / class_directory / image_id
            if flat_path.is_file():
                resolved.append(flat_path)
                layout_counts["flat"] += 1
            elif class_path.is_file():
                resolved.append(class_path)
                layout_counts["class_directory"] += 1
            else:
                missing.append(image_id)

        active_layouts = [
            name for name, count in layout_counts.items() if count > 0
        ]
        layout = "+".join(active_layouts) if active_layouts else "unresolved"
        return resolved, layout, missing

    def _decode_rgb_image(self, path: Path) -> tuple[Image.Image, bool]:
        """Giải mã rgb ảnh cho bước xử lý hiện tại.

        Parameters
        ----------
        path : Path
            Đường dẫn tài nguyên được sử dụng.

        Returns
        -------
        tuple[Image.Image, bool]
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        try:
            with Image.open(path) as source:
                return source.convert("RGB"), False
        except OSError as error:
            if (
                not self.allow_truncated_images
                or "truncated" not in str(error).lower()
            ):
                raise

        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.load()
            return image, True
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous

    def _validate_image_decoding(self):
        """Kiểm tra tính hợp lệ của ảnh decoding cho bước xử lý hiện tại.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        recovered: list[str] = []
        failures: list[dict[str, str]] = []
        for image_id, path in zip(
            self.df["image_id"].astype(str), self._image_paths
        ):
            try:
                _, was_recovered = self._decode_rgb_image(path)
                if was_recovered:
                    recovered.append(str(image_id))
            except OSError as error:
                failures.append(
                    {"image_id": str(image_id), "error": str(error)}
                )
        return recovered, failures

    def __len__(self):
        """Thực hiện bước len trong quy trình hiện tại.

        Returns
        -------
        int
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        return len(self.df)
        
    def _clean_report(self, text: str) -> str:
        """Thực hiện bước clean báo cáo trong quy trình hiện tại.

        Parameters
        ----------
        text : str
            Văn bản hoặc biểu diễn văn bản đầu vào.

        Returns
        -------
        str
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        return text.strip().lower()

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
            with open(path, 'r', encoding='utf-8') as f:
                raw_text = f.read()

        cleaned_text = self._clean_report(raw_text) or default_text
        if self.tokenizer:
            return self.tokenizer([cleaned_text]).squeeze(0)
        return cleaned_text

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

        Raises
        ------
        RuntimeError
            Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
        """
        row = self.df.iloc[idx]
        image_id = str(row['image_id'])
        img_path = self._image_paths[idx]
        
        file_name_without_ext = os.path.splitext(image_id)[0]
        report_path = os.path.join(self.report_dir, f"{file_name_without_ext}.txt")
        
        # --- Tải ảnh đầu vào ---
        try:
            image, _ = self._decode_rgb_image(img_path)
        except OSError as error:
            raise RuntimeError(
                f"Unable to decode FracAtlas image {img_path}."
            ) from error
            
        # Kiểm tra điều kiện trước khi thực hiện nhánh xử lý tương ứng.
        if self.use_high_res:
            cached_selection = self._high_res_selection_cache.get(image_id)
            high_res_fields, selection = prepare_high_resolution_inputs(
                image,
                self.transform,
                self.high_res_cfg,
                selection=cached_selection,
                return_selection=True,
            )
            if self.cache_high_res_selection and cached_selection is None:
                self._high_res_selection_cache[image_id] = selection
        else:
            high_res_fields = {}
            image = prepare_global_image(
                image,
                self.transform,
                self.preprocess_cfg,
            )
            
        # Kiểm tra điều kiện trước khi thực hiện nhánh xử lý tương ứng.
        if self.task_type == "multiclass":
            labels = torch.tensor(int(row['fractured']), dtype=torch.long)
        else:
            labels = torch.tensor([float(row.get('fractured', 0.0))], dtype=torch.float32)

        # Chuẩn bị và xử lý đầu vào hoặc đặc trưng văn bản.
        input_ids = self._load_and_tokenize(report_path, "no fracture identified.")
        
        if self.use_high_res:
            return {
                **high_res_fields,
                "input_ids": input_ids,
                "labels": labels
            }
        return image, input_ids, labels
