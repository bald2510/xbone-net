"""Kiểm tra tính toàn vẹn và khả năng tương thích của gói tài nguyên demo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "demo" / "model"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Tính SHA-256 của một tệp mà không nạp toàn bộ tệp vào bộ nhớ."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_provenance(path: Path) -> dict:
    """Đọc provenance JSON được đóng gói trong một feature archive NPZ."""

    with np.load(path, allow_pickle=False) as archive:
        if "provenance_json" not in archive.files:
            raise ValueError(f"Feature archive không có provenance_json: {path}")
        return json.loads(str(archive["provenance_json"].item()))


def verify_artifacts(root: Path) -> None:
    """Xác minh kích thước, checksum, kiến trúc và nguồn checkpoint của gói demo."""

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Thiếu manifest demo: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for relative_name, expected in manifest["files"].items():
        path = root / relative_name
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu tài nguyên demo: {path}")
        if path.stat().st_size != int(expected["bytes"]):
            raise ValueError(f"Sai kích thước: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != str(expected["sha256"]).lower():
            raise ValueError(f"Sai SHA-256: {path}")

    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    config = metrics.get("config", {})
    if config.get("experiment_name") != manifest["experiment"]:
        raise ValueError("metrics.json không thuộc thí nghiệm đã khai báo trong manifest.")
    if int(config.get("seed", -1)) != int(manifest["seed"]):
        raise ValueError("Seed trong metrics.json không khớp manifest.")
    model = config.get("model", {})
    dataset = config.get("dataset", {}).get("params", {})
    if model.get("classifier", {}).get("type") != "linear":
        raise ValueError("Gói demo không sử dụng linear classifier.")
    if model.get("fusion", {}).get("type") != "cross_attention":
        raise ValueError("Gói demo không sử dụng cross-attention.")
    if dataset.get("preprocess", {}).get("strategy") != "direct_resize":
        raise ValueError("Gói demo không sử dụng direct_resize.")

    checkpoint_hash = manifest["files"]["best_phase2.pth"]["sha256"]
    for split in ("ctch_train", "ctch_val"):
        provenance = load_provenance(root / "features" / f"{split}.npz")
        if provenance.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError(f"{split}.npz không thuộc checkpoint demo hiện tại.")
        if provenance.get("classifier_type") != "linear":
            raise ValueError(f"{split}.npz không thuộc linear classifier.")
        if provenance.get("fusion_type") != "cross_attention":
            raise ValueError(f"{split}.npz không thuộc cross-attention.")
        if int(provenance.get("num_classes", -1)) != 22:
            raise ValueError(f"{split}.npz không có đúng 22 lớp CTCH.")

    print("Demo artifacts: OK")
    print(f"  Root       : {root}")
    print(f"  Experiment : {manifest['experiment']}")
    print(f"  Seed       : {manifest['seed']}")
    print(f"  Checkpoint : {checkpoint_hash}")


def parse_args() -> argparse.Namespace:
    """Phân tích đối số dòng lệnh của công cụ kiểm tra."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Thư mục chứa manifest.json và tài nguyên demo.",
    )
    return parser.parse_args()


def main() -> None:
    """Chạy kiểm tra gói tài nguyên demo."""

    args = parse_args()
    verify_artifacts(args.artifact_root.expanduser().resolve())


if __name__ == "__main__":
    main()
