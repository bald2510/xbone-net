"""Xuất dự đoán theo mẫu cho toàn bộ mô hình so sánh trên BTXRD.

Notes
-----
Chương trình tự bỏ qua các tệp dự đoán đã tồn tại và dừng ngay khi một lượt
suy luận thất bại hoặc thiếu checkpoint.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 456)
EXPERIMENTS = (
    "btxrd/proposed/ours_xbone_net",
    "btxrd/baselines/peft_finetuned/lora_biomedclip",
    "btxrd/baselines/peft_finetuned/lora_pubmedclip",
    "btxrd/baselines/full_finetuned/fft_biomedclip",
    "btxrd/baselines/full_finetuned/fft_pubmedclip",
    "btxrd/baselines/full_finetuned/fft_clip",
    "btxrd/baselines/full_finetuned/fft_medclip",
    "btxrd/baselines/full_finetuned/fft_resnet50",
    "btxrd/baselines/full_finetuned/fft_densenet",
)


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh.

    Returns
    -------
    argparse.Namespace
        Các tham số điều khiển quá trình xuất dự đoán.
    """
    parser = argparse.ArgumentParser(
        description="Xuất đủ dự đoán theo mẫu cho kiểm định BTXRD."
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    return parser.parse_args()


def prediction_path(experiment: str, seed: int) -> Path:
    """Xác định đường dẫn tệp dự đoán của một lượt chạy.

    Parameters
    ----------
    experiment : str
        Tên thí nghiệm bao gồm tên bộ dữ liệu.
    seed : int
        Hạt giống huấn luyện.

    Returns
    -------
    pathlib.Path
        Đường dẫn tệp NPZ chứa dự đoán theo mẫu.
    """
    return (
        ROOT
        / "results"
        / experiment
        / f"seed_{seed}"
        / "analysis"
        / "features"
        / "btxrd_test.npz"
    )


def checkpoint_path(experiment: str, seed: int) -> Path:
    """Xác định checkpoint pha 2 cần dùng.

    Parameters
    ----------
    experiment : str
        Tên thí nghiệm bao gồm tên bộ dữ liệu.
    seed : int
        Hạt giống huấn luyện.

    Returns
    -------
    pathlib.Path
        Đường dẫn checkpoint tốt nhất của pha 2.
    """
    return ROOT / "checkpoints" / experiment / f"seed_{seed}" / "best_phase2.pth"


def main() -> None:
    """Xuất lần lượt các dự đoán còn thiếu và xác nhận đủ 27 tệp."""
    args = parse_args()
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUTF8": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )

    for experiment in EXPERIMENTS:
        for seed in SEEDS:
            output = prediction_path(experiment, seed)
            checkpoint = checkpoint_path(experiment, seed)
            if output.exists():
                print(f"[BỎ QUA] {experiment} seed={seed} đã có dự đoán.")
                continue
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Thiếu checkpoint: {checkpoint}")

            print(f"[ĐANG CHẠY] {experiment} seed={seed}", flush=True)
            command = [
                sys.executable,
                "-X",
                "utf8",
                str(ROOT / "tools" / "export_experiment_analysis_features.py"),
                "--experiment",
                experiment,
                "--seed",
                str(seed),
                "--scenarios",
                "btxrd_test",
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--device",
                args.device,
            ]
            subprocess.run(command, cwd=ROOT, env=environment, check=True)

    completed = sum(
        prediction_path(experiment, seed).is_file()
        for experiment in EXPERIMENTS
        for seed in SEEDS
    )
    print(f"[HOÀN TẤT] Đã có {completed}/27 tệp dự đoán BTXRD.")
    if completed != 27:
        raise RuntimeError("Chưa xuất đủ 27 tệp dự đoán BTXRD.")


if __name__ == "__main__":
    main()
