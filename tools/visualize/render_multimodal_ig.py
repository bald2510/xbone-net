"""Tạo hình trực quan Integrated Gradients (IG) đa phương thức chuẩn cho mọi mô hình.

Hỗ trợ mọi biến thể mô hình (proposed, xbone_letterbox, fft_biomedclip,...)
và kết xuất theo đúng cấu trúc:
- Header: Nhãn đúng (tiếng Việt) | Dự đoán (tiếng Việt) (%)
- (a) Ảnh X-quang đầu vào
- (b) Tích phân gradient trên ảnh toàn cục (Heatmap overlay turbo)
- (d) Tích phân gradient trên các đơn vị từ ngữ của bệnh sử (toàn bộ từ đã giải mã và ghép nối)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

from src.utils.online_inference import (
    OnlineInferenceEngine,
    _build_online_inputs,
    _decoded_text_attribution,
    _global_patch_grid,
    project_global_ig_to_source,
    render_global_ig_overlay,
)
from benchmark.explainability_analysis import (
    _cache_modalities,
    _integrated_gradients_image,
    _integrated_gradients_text,
    _logits_from_features,
)


TEXT_ATTRIBUTION_CMAP = LinearSegmentedColormap.from_list(
    "xbonenet_text_ig",
    ["#1f77b4", "#ffffff", "#ff7f0e"],
)


def _predict_architecture_neutral_ig(
    engine: OnlineInferenceEngine,
    image: Image.Image,
    clinical_text: str,
    ig_steps: int,
):
    """Compute input-level image/text IG through the common model interface."""
    loaded = engine.loaded
    model = loaded.model
    inputs = _build_online_inputs(loaded, image, clinical_text)
    batch = {
        "pixel_values": inputs["pixel_values"],
        "clinical_input_ids": inputs["input_ids"],
        "clinical_attention_mask": inputs["attention_mask"],
    }
    cached = _cache_modalities(model, batch)
    with torch.no_grad():
        logits = _logits_from_features(model, cached)
        probabilities = torch.softmax(logits, dim=-1)[0]
        predicted_index = int(probabilities.argmax())

    global_relevance, _ = _integrated_gradients_image(
        model,
        cached,
        inputs["pixel_values"],
        predicted_index,
        steps=int(ig_steps),
    )
    _, text_attribution = _integrated_gradients_text(
        model,
        cached,
        predicted_index,
        steps=int(ig_steps),
    )
    text_signed_scores = text_attribution.sum(dim=-1).detach().cpu().numpy()
    decoded_tokens, decoded_scores = _decoded_text_attribution(
        model.backbone.tokenizer_obj,
        inputs["input_ids"],
        inputs["attention_mask"],
        text_signed_scores,
    )
    global_ig_map = project_global_ig_to_source(
        global_relevance.detach().cpu().numpy(),
        inputs["foreground_box"],
        image.size,
        patch_grid=_global_patch_grid(model),
    )
    return SimpleNamespace(
        predicted_index=predicted_index,
        predicted_label=engine.class_labels[predicted_index],
        probabilities=probabilities.detach().cpu().numpy(),
        is_ood=False,
        ood_score=float("nan"),
        ood_threshold=float("nan"),
        global_ig_map=global_ig_map,
        text_tokens=decoded_tokens,
        text_ig_scores=decoded_scores,
        global_faithfulness=None,
        text_faithfulness=None,
        contribution_drops=None,
    )


def parse_args() -> argparse.Namespace:
    """Phân tích các tham số dòng lệnh phục vụ trực quan hóa Integrated Gradients.

    Returns
    -------
    argparse.Namespace
        Không gian tên chứa các đối số dòng lệnh.
    """
    parser = argparse.ArgumentParser(description="Render Multimodal Integrated Gradients")
    parser.add_argument("--image-id", required=True, help="Tên file ảnh (ví dụ: 57_img-06186-00001.jpg)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--experiment",
        default="ctch/proposed/ours_xbone_net",
        help="Định danh thí nghiệm (ví dụ: ctch/proposed/ours_xbone_net hoặc ctch/baselines/full_finetuned/fft_biomedclip)",
    )
    parser.add_argument(
        "--enable-ood",
        action="store_true",
        help="Áp dụng bộ phát hiện OOD trước khi giải thích lớp.",
    )
    parser.add_argument("--ood-method", default="mahalanobis_centroid")
    parser.add_argument("--ig-steps", type=int, default=24)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "visualization" / "multimodal_ig.png",
    )
    return parser.parse_args()


def _draw_token_attribution(axis: plt.Axes, tokens: list[str], scores: np.ndarray) -> None:
    """Vẽ khối token attribution dạng từ ngữ đóng khung theo màu tương ứng.

    Parameters
    ----------
    axis : plt.Axes
        Trục tọa độ Matplotlib để vẽ văn bản.
    tokens : list[str]
        Danh sách các từ ngữ bệnh sử đã giải mã.
    scores : np.ndarray
        Mảng điểm đóng góp Integrated Gradients chuẩn hóa tương ứng từng từ.
    """
    axis.set_axis_off()
    axis.set_title(
        "(d) Tích phân gradient trên các đơn vị từ ngữ của bệnh sử\n"
        "Màu xanh biểu thị tác động phản đối; màu cam biểu thị tác động ủng hộ",
        loc="center",
        fontsize=12,
    )
    figure = axis.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    x, y = 0.02, 0.88
    line_height = 0.16

    for token, score in zip(tokens, scores):
        color = TEXT_ATTRIBUTION_CMAP(
            0.5 + 0.5 * float(np.clip(score, -1.0, 1.0))
        )
        label = axis.text(
            x,
            y,
            token,
            transform=axis.transAxes,
            fontsize=12,
            va="top",
            ha="left",
            color="#111111",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": color,
                "edgecolor": "none",
            },
        )
        figure.canvas.draw()
        extent = label.get_window_extent(renderer=renderer).transformed(
            axis.transAxes.inverted()
        )
        if extent.x1 > 0.98 and x > 0.02:
            label.remove()
            x = 0.02
            y -= line_height
            label = axis.text(
                x,
                y,
                token,
                transform=axis.transAxes,
                fontsize=12,
                va="top",
                ha="left",
                color="#111111",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": color,
                    "edgecolor": "none",
                },
            )
            figure.canvas.draw()
            extent = label.get_window_extent(renderer=renderer).transformed(
                axis.transAxes.inverted()
            )
        x = extent.x1 + 0.012


def main() -> None:
    """Thực thi điểm vào chính của mô-đun."""
    args = parse_args()
    split = pd.read_csv(PROJECT_ROOT / "data" / "CTCH" / "ctch-split.csv")
    if args.image_id not in set(split["image_id"].astype(str)):
        raise ValueError(f"Unknown CTCH image_id: {args.image_id}")
    labels = pd.read_csv(PROJECT_ROOT / "data" / "CTCH" / "ctch-labels.csv")
    label_row = labels.loc[labels["image_id"].astype(str) == args.image_id]
    if label_row.empty:
        raise ValueError(f"CTCH label is unavailable for: {args.image_id}")
    ground_truth = str(label_row.iloc[0]["mapped_class"])

    image_path = PROJECT_ROOT / "data" / "CTCH" / "images" / args.image_id
    report_path = (
        PROJECT_ROOT
        / "data"
        / "CTCH"
        / "reports"
        / "clinical"
        / Path(args.image_id).with_suffix(".txt").name
    )
    image = Image.open(image_path).convert("RGB")
    clinical_text = report_path.read_text(encoding="utf-8").strip()

    engine = OnlineInferenceEngine(
        seed=args.seed,
        device=args.device,
        experiment_name=args.experiment,
        enable_ood=args.enable_ood,
    )
    if args.enable_ood:
        result = engine.predict(
            image,
            clinical_text,
            ood_method=args.ood_method,
            ig_steps=args.ig_steps,
            compute_faithfulness=True,
            compute_global_ig=True,
        )
    else:
        result = _predict_architecture_neutral_ig(
            engine,
            image,
            clinical_text,
            args.ig_steps,
        )

    if result.is_ood:
        raise RuntimeError(
            f"Sample is OOD: score={result.ood_score:.4f}, "
            f"threshold={result.ood_threshold:.4f}; class IG was not computed."
        )
    if result.text_tokens is None or result.text_ig_scores is None:
        raise RuntimeError("Text Integrated Gradients was not produced.")
    if result.global_ig_map is None:
        raise RuntimeError("Global Integrated Gradients was not produced.")

    global_overlay = render_global_ig_overlay(image, result)

    # Khởi tạo bố cục hình vẽ: Hàng trên 2 ảnh X-quang, Hàng dưới toàn bộ khối văn bản
    figure = plt.figure(figsize=(12.5, 9.5), facecolor="white")
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[1.25, 0.75],
        wspace=0.10,
        hspace=0.25,
        top=0.90,
    )
    original_axis = figure.add_subplot(grid[0, 0])
    global_axis = figure.add_subplot(grid[0, 1])
    text_axis = figure.add_subplot(grid[1, :])

    # (a) Ảnh X-quang đầu vào
    original_axis.imshow(image, cmap="gray")
    original_axis.set_title("(a) Ảnh X-quang đầu vào", fontsize=12)
    original_axis.axis("off")

    # (b) Tích phân gradient trên ảnh toàn cục
    global_axis.imshow(global_overlay)
    global_axis.set_title(
        "(b) Tích phân gradient trên ảnh toàn cục\n"
        "Màu nóng biểu thị đóng góp lớn hơn",
        fontsize=12,
    )
    global_axis.axis("off")

    # (d) Tích phân gradient trên các từ ngữ bệnh sử
    _draw_token_attribution(
        text_axis,
        result.text_tokens,
        result.text_ig_scores,
    )

    # Tra cứu tên nhãn tiếng Việt chuẩn
    predicted_label = str(result.predicted_label)
    if result.predicted_index is not None:
        class_names = (
            labels[["class_id", "mapped_class"]]
            .drop_duplicates("class_id")
            .set_index("class_id")["mapped_class"]
            .to_dict()
        )
        predicted_label = str(
            class_names.get(result.predicted_index, predicted_label)
        )

    ood_summary = ""
    if args.enable_ood:
        ood_summary = (
            f" | Điểm OOD: {result.ood_score:.2f} < "
            f"{result.ood_threshold:.2f}"
        )

    figure.suptitle(
        f"Nhãn đúng: {ground_truth} | Dự đoán: {predicted_label} "
        f"({100.0 * float(result.probabilities.max()):.1f}%){ood_summary}",
        fontsize=14,
        y=0.97,
    )

    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)

    # Lưu bảng điểm CSV
    score_table = pd.DataFrame(
        {
            "token": result.text_tokens,
            "signed_ig_normalized": result.text_ig_scores,
            "absolute_ig_normalized": np.abs(result.text_ig_scores),
        }
    ).sort_values("absolute_ig_normalized", ascending=False)
    score_path = args.output.with_name(f"{args.output.stem}_text_ig.csv")
    score_table.to_csv(score_path, index=False, encoding="utf-8-sig")

    if result.global_faithfulness is not None:
        faithfulness_path = args.output.with_name(
            f"{args.output.stem}_faithfulness.csv"
        )
        f_data = {
            "fraction": result.global_faithfulness["fractions"],
            "global_deletion": result.global_faithfulness["deletion"],
            "global_insertion": result.global_faithfulness["insertion"],
        }
        if result.text_faithfulness is not None:
            f_data["text_deletion"] = result.text_faithfulness["deletion"]
            f_data["text_insertion"] = result.text_faithfulness["insertion"]
        pd.DataFrame(f_data).to_csv(faithfulness_path, index=False, encoding="utf-8-sig")

    if result.contribution_drops is not None:
        contribution_path = args.output.with_name(
            f"{args.output.stem}_contribution.csv"
        )
        pd.DataFrame(
            {
                "source": list(result.contribution_drops),
                "probability_drop": list(result.contribution_drops.values()),
            }
        ).to_csv(contribution_path, index=False, encoding="utf-8-sig")

    top_text = np.argsort(np.abs(result.text_ig_scores))[::-1][:10]
    print(
        "Top |text IG|:",
        ", ".join(
            f"{result.text_tokens[index]}={result.text_ig_scores[index]:.3f}"
            for index in top_text
        ),
    )
    print(f"Ảnh xuất ra: {args.output}")
    print(f"Bảng điểm: {score_path}")


if __name__ == "__main__":
    main()
