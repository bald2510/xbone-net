"""Công cụ trực quan hóa và xuất từng ảnh thành phần của quy trình tiền xử lý ảnh X-quang:
1. Quy trình padding (proposed):
   - 01_anh_dau_vao.png (Ảnh gốc W x H)
   - 02_buoc1_giam_kich_thuoc.png (Co cạnh dài về 224px: W' x H')
   - 03_buoc2_noi_suy_bac_ba.png (Nội suy Bicubic: W' x H')
   - 04_buoc3_them_vung_dem.png (Đệm viền đen Canvas: 224x224)

2. Quy trình center crop (biomedclip):
   - 01_anh_dau_vao.png (Ảnh gốc W x H)
   - 02_buoc1_giam_kich_thuoc.png (Co cạnh ngắn về 224px: W_res x H_res)
   - 03_buoc2_noi_suy_bac_ba.png (Nội suy Bicubic: W_res x H_res)
   - 04_buoc3_cat_vung_trung_tam.png (Cắt lấy ô vuông chính giữa: 224x224)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Phân tích các tham số dòng lệnh phục vụ xuất từng ảnh thành phần."""
    parser = argparse.ArgumentParser(
        description="Export individual preprocessing step components for Padding (proposed) and CenterCrop (biomedclip)."
    )
    parser.add_argument("--image", required=True, help="Đường dẫn tới tệp ảnh X-quang đầu vào.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "visualization" / "preprocessing_components"),
        help="Thư mục gốc lưu các ảnh thành phần.",
    )
    parser.add_argument("--target-size", type=int, default=224, help="Kích thước cạnh chuẩn (mặc định 224).")
    parser.add_argument("--dpi", type=int, default=200, help="Độ phân giải DPI khi lưu ảnh.")
    return parser.parse_args()


def export_padding_components(source: Image.Image, output_dir: Path, target_size: int = 224, dpi: int = 200) -> list[Path]:
    """Xuất từng ảnh thành phần của quy trình Letterbox Padding (proposed).

    Parameters
    ----------
    source : Image.Image
        Ảnh X-quang đầu vào gốc.
    output_dir : Path
        Thư mục lưu các tệp ảnh thành phần.
    target_size : int, optional
        Kích thước mục tiêu (mặc định 224).
    dpi : int, optional
        Độ phân giải DPI khi lưu.

    Returns
    -------
    list[Path]
        Danh sách đường dẫn các tệp ảnh thành phần đã lưu.
    """
    pad_dir = output_dir / "padding_proposed"
    pad_dir.mkdir(parents=True, exist_ok=True)

    source_rgb = source.convert("RGB")
    orig_w, orig_h = source_rgb.size

    # Bước 1: Tính tỷ lệ cạnh dài
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = max(1, min(target_size, round(orig_w * scale)))
    new_h = max(1, min(target_size, round(orig_h * scale)))

    # Bước 2: Nội suy Bicubic
    bicubic_resized = source_rgb.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

    # Bước 3: Đệm Canvas vuông màu đen
    canvas = Image.new("RGB", (target_size, target_size), color=(0, 0, 0))
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    padded_img = canvas.copy()
    padded_img.paste(bicubic_resized, (offset_x, offset_y))

    components = [
        (source_rgb, f"01_anh_dau_vao_{orig_w}x{orig_h}.png"),
        (bicubic_resized, f"02_buoc1_giam_kich_thuoc_{new_w}x{new_h}.png"),
        (bicubic_resized, f"03_buoc2_noi_suy_bac_ba_{new_w}x{new_h}.png"),
        (padded_img, f"04_buoc3_them_vung_dem_{target_size}x{target_size}.png"),
    ]

    saved_paths = []
    for img, filename in components:
        p = pad_dir / filename
        img.save(p, dpi=(dpi, dpi))
        saved_paths.append(p)

    return saved_paths


def export_centercrop_components(source: Image.Image, output_dir: Path, target_size: int = 224, dpi: int = 200) -> list[Path]:
    """Xuất từng ảnh thành phần của quy trình CenterCrop (biomedclip).

    Parameters
    ----------
    source : Image.Image
        Ảnh X-quang đầu vào gốc.
    output_dir : Path
        Thư mục lưu các tệp ảnh thành phần.
    target_size : int, optional
        Kích thước mục tiêu (mặc định 224).
    dpi : int, optional
        Độ phân giải DPI khi lưu.

    Returns
    -------
    list[Path]
        Danh sách đường dẫn các tệp ảnh thành phần đã lưu.
    """
    crop_dir = output_dir / "centercrop_biomedclip"
    crop_dir.mkdir(parents=True, exist_ok=True)

    source_rgb = source.convert("RGB")
    orig_w, orig_h = source_rgb.size

    # Bước 1: Thu nhỏ cạnh ngắn nhất về target_size
    scale_biomed = target_size / min(orig_w, orig_h)
    resized_w = max(1, round(orig_w * scale_biomed))
    resized_h = max(1, round(orig_h * scale_biomed))

    # Bước 2: Nội suy Bicubic
    resized_img = source_rgb.resize((resized_w, resized_h), resample=Image.Resampling.BICUBIC)

    # Bước 3: CenterCrop lấy vùng chính giữa target_size x target_size
    crop_x1 = (resized_w - target_size) // 2
    crop_y1 = (resized_h - target_size) // 2
    crop_x2 = crop_x1 + target_size
    crop_y2 = crop_y1 + target_size
    centercrop_img = resized_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    components = [
        (source_rgb, f"01_anh_dau_vao_{orig_w}x{orig_h}.png"),
        (resized_img, f"02_buoc1_giam_kich_thuoc_{resized_w}x{resized_h}.png"),
        (resized_img, f"03_buoc2_noi_suy_bac_ba_{resized_w}x{resized_h}.png"),
        (centercrop_img, f"04_buoc3_cat_vung_trung_tam_{target_size}x{target_size}.png"),
    ]

    saved_paths = []
    for img, filename in components:
        p = crop_dir / filename
        img.save(p, dpi=(dpi, dpi))
        saved_paths.append(p)

    return saved_paths


def main() -> None:
    """Điểm vào chính."""
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp ảnh: {image_path}")

    with Image.open(image_path) as img:
        source_image = img.convert("RGB")

    output_dir = Path(args.output_dir).expanduser().resolve()

    print("=" * 80)
    print("  XBONE-NET: XUẤT TỪNG ẢNH THÀNH PHẦN TIỀN XỬ LÝ (KHÔNG LƯU ẢNH GHÉP TỔNG HỢP)")
    print(f"  • Tệp ảnh nguồn : {image_path.name} ({source_image.width} × {source_image.height})")
    print(f"  • Thư mục xuất  : {output_dir}")
    print("=" * 80)

    # 1. Xuất các thành phần quy trình padding (proposed)
    pad_paths = export_padding_components(source_image, output_dir, args.target_size, args.dpi)
    print("  [Thành công] Đã lưu các thành phần của quy trình padding (proposed):")
    for p in pad_paths:
        print(f"    -> {p.relative_to(PROJECT_ROOT)}")

    # 2. Xuất các thành phần quy trình center crop (biomedclip)
    crop_paths = export_centercrop_components(source_image, output_dir, args.target_size, args.dpi)
    print("\n  [Thành công] Đã lưu các thành phần của quy trình center crop (biomedclip):")
    for p in crop_paths:
        print(f"    -> {p.relative_to(PROJECT_ROOT)}")

    print("\nHoàn tất xuất từng ảnh thành phần độc lập thành công!")


if __name__ == "__main__":
    main()
