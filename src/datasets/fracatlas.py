"""
FracAtlas dataset loader for XBone-Net architecture.
============================================================
Provides a PyTorch Dataset for the FracAtlas musculoskeletal X-ray dataset.
Used in the XBone-Net evaluation framework:
  - Out-of-Distribution (OOD) evaluation: tests generalization of BTXRD-trained VLM models
  - OOD detection via Mahalanobis distance feature statistics
  - Single-report contrastive text pairing
  - Multi-class (binary integer label) and multi-label target encoding
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
# FracAtlas Dataset Loader
# ============================================================

class FracAtlasDataset(Dataset):
    """Dataset loader for the FracAtlas fracture dataset.

    Returns (image, text_report, label) per sample. Operates in single-report mode
    for contrastive evaluation and OOD detection in XBone-Net.

    Attributes:
        img_dir: Root directory containing X-ray image files.
        report_dir: Directory containing text report files (.txt).
        classes: List of target class names (defaults to ['fractured']).
        task_type: Classification mode ('multiclass' or 'multilabel').
        df: Filtered pandas.DataFrame for the active data split.

    Example:
        ds = FracAtlasDataset(
            img_dir="data/fracatlas/images",
            report_dir="data/fracatlas/reports",
            csv_split_path="data/fracatlas/splits.csv",
            csv_labels_path="data/fracatlas/labels.csv",
            split="test",
        )
        image, input_ids, labels = ds[0]
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
        """Initialize the FracAtlas dataset.

        Args:
            img_dir: Path to directory containing X-ray image files.
            report_dir: Path to directory containing report .txt files.
            csv_split_path: Path to CSV with image_id and split columns.
            csv_labels_path: Path to CSV with image_id and fractured label columns.
            classes: List of target class names. Defaults to ['fractured'].
            task_type: 'multiclass' (scalar integer label) or 'multilabel' (float vector).
            split: Data split ('train', 'val', or 'test').
            transform: torchvision.transforms pipeline for image preprocessing.
            tokenizer: Text tokenizer callable returning token-ID tensors.
            max_text_len: Maximum token sequence length.
            strict_files: Require complete report coverage in addition to the
                always-strict image coverage check.
            allow_truncated_images: Permit Pillow to recover JPEG files with a
                truncated tail, while recording every recovered file.
            **kwargs: Extra unused arguments for backward compatibility.
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

        # --- Load and merge metadata ---
        df_split = pd.read_csv(csv_split_path)
        df_labels = pd.read_csv(csv_labels_path)
        df_merged = pd.merge(df_split, df_labels, on=["image_id"], how="inner")
        
        # --- Filter split ---
        current_split = 'validate' if split == 'val' else split
        self.df = df_merged[df_merged['split'] == current_split].reset_index(drop=True)

        # FracAtlas stores images below class directories in the released
        # layout (``Fractured`` and ``Non_fractured``).  Resolve every path up
        # front so domain-OOD evaluation cannot silently replace missing files
        # with a constant black image.
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

        # --- Image preprocessing configuration ---
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
        """Resolve flat or released class-directory image paths fail-closed."""
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
        """Decode one image and explicitly recover only truncated JPEG tails."""
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
        """Load every split image once and record controlled recoveries."""
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
        """Return the total number of samples in the current split."""
        return len(self.df)
        
    def _clean_report(self, text: str) -> str:
        """Clean and normalize raw report text by stripping whitespace and lowercasing."""
        return text.strip().lower()

    def _load_and_tokenize(self, path: str, default_text: str):
        """Load a text report from disk, clean it, and tokenize if tokenizer is set.

        Args:
            path: Absolute path to the .txt report file.
            default_text: Fallback string used when report is missing or empty.

        Returns:
            torch.Tensor of token IDs if tokenizer is present, else text string.
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
        """Retrieve a sample by index.

        Args:
            idx: Integer index into dataset.

        Returns:
            tuple: (image, input_ids, labels) where:
                - image: Transformed image tensor
                - input_ids: Tokenized report text tensor or raw string
                - labels: torch.long scalar (multiclass) or float vector (multilabel)
        """
        row = self.df.iloc[idx]
        image_id = str(row['image_id'])
        img_path = self._image_paths[idx]
        
        file_name_without_ext = os.path.splitext(image_id)[0]
        report_path = os.path.join(self.report_dir, f"{file_name_without_ext}.txt")
        
        # --- Load image ---
        try:
            image, _ = self._decode_rgb_image(img_path)
        except OSError as error:
            raise RuntimeError(
                f"Unable to decode FracAtlas image {img_path}."
            ) from error
            
        # --- Fixed-budget high-resolution mode ---
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
            
        # --- Encode label ---
        if self.task_type == "multiclass":
            labels = torch.tensor(int(row['fractured']), dtype=torch.long)
        else:
            labels = torch.tensor([float(row.get('fractured', 0.0))], dtype=torch.float32)

        # --- Load text report ---
        input_ids = self._load_and_tokenize(report_path, "no fracture identified.")
        
        if self.use_high_res:
            return {
                **high_res_fields,
                "input_ids": input_ids,
                "labels": labels
            }
        return image, input_ids, labels
