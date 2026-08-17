"""Đóng gói checkpoint và feature archive chính thức cho Streamlit demo."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.verify_demo_artifacts import sha256_file, verify_artifacts


EXPERIMENT = "ctch/proposed/ours_xbone_net"
SEED = 42


def copy_artifacts(destination: Path) -> None:
    """Sao chép tài nguyên seed 42 và sinh manifest kiểm tra toàn vẹn."""

    checkpoint = (
        PROJECT_ROOT / "checkpoints" / EXPERIMENT / f"seed_{SEED}" / "best_phase2.pth"
    )
    result_root = PROJECT_ROOT / "results" / EXPERIMENT / f"seed_{SEED}"
    metrics = result_root / "metrics.json"
    feature_root = result_root / "analysis" / "features"
    sources = {
        "best_phase2.pth": checkpoint,
        "metrics.json": metrics,
        "features/ctch_train.npz": feature_root / "ctch_train.npz",
        "features/ctch_val.npz": feature_root / "ctch_val.npz",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Thiếu tài nguyên nguồn:\n" + "\n".join(missing))

    metrics_payload = json.loads(metrics.read_text(encoding="utf-8"))
    config = metrics_payload.get("config", {})
    model = config.get("model", {})
    preprocess = config.get("dataset", {}).get("params", {}).get("preprocess", {})
    expected = (
        config.get("experiment_name") == EXPERIMENT
        and int(config.get("seed", -1)) == SEED
        and model.get("backbone_type") == "biomedclip"
        and model.get("fusion", {}).get("type") == "cross_attention"
        and model.get("classifier", {}).get("type") == "linear"
        and preprocess.get("strategy") == "direct_resize"
    )
    if not expected:
        raise ValueError("Config nguồn không phải XBone-Net demo đã khóa.")

    destination.mkdir(parents=True, exist_ok=True)
    (destination / "features").mkdir(parents=True, exist_ok=True)
    for relative_name, source in sources.items():
        target = destination / relative_name
        source_hash = sha256_file(source)
        if (
            target.is_file()
            and target.stat().st_size == source.stat().st_size
            and sha256_file(target) == source_hash
        ):
            continue
        temporary = target.with_name(target.name + ".tmp")
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()

    files = {}
    for relative_name in sources:
        path = destination / relative_name
        files[relative_name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "seed": SEED,
        "architecture": {
            "backbone": "biomedclip",
            "preprocessing": "direct_resize",
            "peft": "lora_r16_alpha32",
            "fusion": "bidirectional_cross_attention",
            "classifier": "linear_22_classes",
        },
        "files": files,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    verify_artifacts(destination)


def parse_args() -> argparse.Namespace:
    """Phân tích đối số dòng lệnh của công cụ đóng gói."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=PROJECT_ROOT / "demo" / "model",
        help="Thư mục đích của gói demo.",
    )
    return parser.parse_args()


def main() -> None:
    """Đóng gói và kiểm tra tài nguyên demo."""

    args = parse_args()
    copy_artifacts(args.destination.expanduser().resolve())


if __name__ == "__main__":
    main()
