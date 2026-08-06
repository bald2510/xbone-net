"""Render image- and text-token Integrated Gradients for one CTCH sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.online_inference import (
    OnlineInferenceEngine,
    render_global_ig_overlay,
    render_local_ig_overlay,
)


TEXT_ATTRIBUTION_CMAP = LinearSegmentedColormap.from_list(
    "xbonenet_text_ig",
    ["#1f77b4", "#ffffff", "#ff7f0e"],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--experiment",
        default="ctch/proposed/ours_xbone_net",
        help="Evaluated proposed experiment to load.",
    )
    parser.add_argument(
        "--enable-ood",
        action="store_true",
        help="Apply the experiment's locked OOD detector before class attribution.",
    )
    parser.add_argument("--ood-method", default="mahalanobis_centroid")
    parser.add_argument("--ig-steps", type=int, default=24)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "visualization" / "multimodal_ig.png",
    )
    return parser.parse_args()


def _draw_token_attribution(axis, tokens: list[str], scores: np.ndarray) -> None:
    axis.set_axis_off()
    axis.set_title(
        "(d) Tích phân gradient trên các đơn vị từ ngữ của bệnh sử\n"
        "Màu xanh biểu thị tác động phản đối; màu cam biểu thị tác động ủng hộ",
        loc="center",
        fontsize=11,
    )
    figure = axis.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    x, y = 0.01, 0.91
    line_height = 0.095

    for token, score in zip(tokens, scores):
        color = TEXT_ATTRIBUTION_CMAP(
            0.5 + 0.5 * float(np.clip(score, -1.0, 1.0))
        )
        label = axis.text(
            x,
            y,
            token,
            transform=axis.transAxes,
            fontsize=11,
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
        if extent.x1 > 0.99 and x > 0.01:
            label.remove()
            x = 0.01
            y -= line_height
            label = axis.text(
                x,
                y,
                token,
                transform=axis.transAxes,
                fontsize=11,
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


def _draw_faithfulness_curve(
    axis,
    title: str,
    values: dict,
    *,
    unit: str,
) -> None:
    fractions = 100.0 * np.asarray(values["fractions"])
    axis.plot(
        fractions,
        values["deletion"],
        color="#1f77b4",
        marker="o",
        linewidth=2,
        label=f"Deletion (AUC={values['deletion_auc']:.3f})",
    )
    axis.plot(
        fractions,
        values["insertion"],
        color="#ff7f0e",
        marker="s",
        linewidth=2,
        label=f"Insertion (AUC={values['insertion_auc']:.3f})",
    )
    axis.set_title(title, fontsize=11)
    axis.set_xlabel(f"Tỷ lệ {unit} bị thay thế/khôi phục (%)")
    axis.set_ylabel("Xác suất lớp dự đoán")
    axis.set_xlim(0.0, 100.0)
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=9)


def _draw_contribution(axis, values: dict[str, float]) -> None:
    labels = ["Ảnh toàn cục", "Ảnh cục bộ", "Bệnh sử lâm sàng"]
    keys = ["global_visual", "local_visual", "clinical_text"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    drops = np.asarray([float(values[key]) for key in keys])
    bars = axis.barh(labels, drops, color=colors, alpha=0.9)
    axis.axvline(0.0, color="#333333", linewidth=1)
    axis.set_title(
        "(e) Mức giảm xác suất khi thay từng nguồn bằng mốc tham chiếu",
        fontsize=11,
    )
    axis.set_xlabel(r"$\Delta p$ của lớp dự đoán")
    axis.grid(axis="x", alpha=0.25)
    axis.margins(x=0.12)
    for bar, value in zip(bars, drops):
        axis.annotate(
            f"{value:+.3f}",
            xy=(value, bar.get_y() + bar.get_height() / 2),
            xytext=(7 if value >= 0 else -7, 0),
            textcoords="offset points",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
        )


def main() -> None:
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
    result = engine.predict(
        image,
        clinical_text,
        ood_method=args.ood_method,
        ig_steps=args.ig_steps,
        compute_faithfulness=True,
        compute_global_ig=True,
    )
    if result.is_ood:
        raise RuntimeError(
            f"Sample is OOD: score={result.ood_score:.4f}, "
            f"threshold={result.ood_threshold:.4f}; class IG was not computed."
        )
    if result.text_tokens is None or result.text_ig_scores is None:
        raise RuntimeError("Text Integrated Gradients was not produced.")
    if (
        result.global_faithfulness is None
        or result.contribution_drops is None
        or result.global_ig_map is None
        or result.visual_faithfulness is None
        or result.text_faithfulness is None
    ):
        raise RuntimeError("The three-branch explanation was not produced.")

    global_overlay = render_global_ig_overlay(image, result)
    local_overlay = render_local_ig_overlay(image, result)
    # A two-by-two layout keeps every panel legible when the figure is placed on
    # an A4 page.  In particular, the clinical-text panel receives half of the
    # page width instead of one quarter as in the former horizontal layout.
    figure = plt.figure(figsize=(12.5, 10.0), facecolor="white")
    grid = figure.add_gridspec(
        2,
        2,
        wspace=0.08,
        hspace=0.22,
        top=0.88,
    )
    original_axis = figure.add_subplot(grid[0, 0])
    global_axis = figure.add_subplot(grid[0, 1])
    local_axis = figure.add_subplot(grid[1, 0])
    text_axis = figure.add_subplot(grid[1, 1])
    text_position = text_axis.get_position()
    text_axis.set_position(
        [
            text_position.x0,
            text_position.y0 - 0.075,
            text_position.width,
            text_position.height,
        ]
    )

    original_axis.imshow(image, cmap="gray")
    original_axis.set_title("(a) Ảnh X-quang đầu vào", fontsize=12)
    original_axis.axis("off")
    global_axis.imshow(global_overlay)
    global_axis.set_title(
        "(b) Tích phân gradient trên ảnh toàn cục\n"
        "Màu nóng biểu thị đóng góp lớn hơn",
        fontsize=12,
    )
    global_axis.axis("off")
    local_axis.imshow(local_overlay)
    local_axis.set_title(
        "(c) Tích phân gradient trên các đơn vị biểu diễn ảnh cục bộ\n"
        "Màu nóng biểu thị đóng góp lớn hơn",
        fontsize=12,
    )
    local_axis.axis("off")
    _draw_token_attribution(
        text_axis,
        result.text_tokens,
        result.text_ig_scores,
    )

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
    score_table = pd.DataFrame(
        {
            "token": result.text_tokens,
            "signed_ig_normalized": result.text_ig_scores,
            "absolute_ig_normalized": np.abs(result.text_ig_scores),
        }
    ).sort_values("absolute_ig_normalized", ascending=False)
    score_path = args.output.with_name(f"{args.output.stem}_text_ig.csv")
    score_table.to_csv(score_path, index=False, encoding="utf-8-sig")
    faithfulness_path = args.output.with_name(
        f"{args.output.stem}_faithfulness.csv"
    )
    pd.DataFrame(
        {
            "fraction": result.global_faithfulness["fractions"],
            "global_deletion": result.global_faithfulness["deletion"],
            "global_insertion": result.global_faithfulness["insertion"],
            "local_deletion": result.visual_faithfulness["deletion"],
            "local_insertion": result.visual_faithfulness["insertion"],
            "text_deletion": result.text_faithfulness["deletion"],
            "text_insertion": result.text_faithfulness["insertion"],
        }
    ).to_csv(faithfulness_path, index=False, encoding="utf-8-sig")
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
    print(args.output)
    print(score_path)
    print(
        "Global visual AUC:",
        f"deletion={result.global_faithfulness['deletion_auc']:.4f},",
        f"insertion={result.global_faithfulness['insertion_auc']:.4f}",
    )
    print(
        "Local visual AUC:",
        f"deletion={result.visual_faithfulness['deletion_auc']:.4f},",
        f"insertion={result.visual_faithfulness['insertion_auc']:.4f}",
    )
    print(
        "Text AUC:",
        f"deletion={result.text_faithfulness['deletion_auc']:.4f},",
        f"insertion={result.text_faithfulness['insertion_auc']:.4f}",
    )
    print(faithfulness_path)
    print(contribution_path)


if __name__ == "__main__":
    main()
