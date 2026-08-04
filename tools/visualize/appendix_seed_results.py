"""Generate report appendix tables and figures from per-seed result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 123, 456)
CLASSIFICATION_METRICS = (
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "BAcc"),
    ("f1_macro", "Macro-F1"),
    ("auroc_macro", "Macro-AUROC"),
    ("auprc_macro", "Macro-AUPRC"),
)
ABLATION_METRICS = (*CLASSIFICATION_METRICS, ("ece_15", "ECE"))
OOD_METRICS = (
    ("auroc_ood", "AUROC-OOD"),
    ("aupr_out", "AUPR-Out"),
    ("fpr_at_95tpr", "\\makecell{FPR@\\\\95\\%TPR}"),
)
FULL_SHOT_MODELS = (
    ("BioMedCLIP", "baselines/full_finetuned/fft_biomedclip"),
    ("CLIP", "baselines/full_finetuned/fft_clip"),
    ("DenseNet-121", "baselines/full_finetuned/fft_densenet"),
    ("MedCLIP", "baselines/full_finetuned/fft_medclip"),
    ("PubMedCLIP", "baselines/full_finetuned/fft_pubmedclip"),
    ("ResNet-50", "baselines/full_finetuned/fft_resnet50"),
    ("LoRA-BiomedCLIP", "baselines/peft_finetuned/lora_biomedclip"),
    ("LoRA-PubMedCLIP", "baselines/peft_finetuned/lora_pubmedclip"),
    ("XBone-Net", "proposed/ours_xbone_net"),
)
FEW_SHOT_MODELS = (
    ("LoRA-BiomedCLIP", "lora_biomedclip"),
    ("LoRA-PubMedCLIP", "lora_pubmedclip"),
    ("XBone-Net", "ours_xbone_net"),
)
ABLATION_VARIANTS = (
    ("XBone-Net", "proposed/ours_xbone_net"),
    (
        "Không dùng ảnh độ phân giải cao",
        "ablation_study/architecture/preprocess/xbone_nohighres",
    ),
    (
        "Gộp trung bình",
        "ablation_study/architecture/preprocess/xbone_mean_pooling",
    ),
    ("Chỉ pha 2", "ablation_study/architecture/phase/phase2_only"),
    ("Nối đặc trưng", "ablation_study/architecture/fusion/concat"),
    ("Đầu tuyến tính", "ablation_study/architecture/classifier/linear"),
)
OOD_SCENARIOS = (
    ("OOD ngữ nghĩa", "semantic_ood"),
    ("BTXRD", "domain_ood_btxrd"),
)
OOD_METHODS = (
    ("Cosine theo tâm lớp", "cosine_centroids"),
    ("Mahalanobis theo tâm lớp", "mahalanobis_centroid"),
    ("kNN", "knn"),
    ("Entropy", "entropy"),
)
REPORT_PALETTE = {
    "blue": "#1F77B4",
    "orange": "#FF7F0E",
    "green": "#2CA02C",
    "gray": "#7F7F7F",
}


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return (value if value.is_absolute() else ROOT / value).resolve()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing result file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def load_classification_metrics(path: Path, expected_seed: int) -> dict[str, float]:
    payload = load_json(path)
    recorded_seed = int(payload.get("seed", expected_seed))
    if recorded_seed != expected_seed:
        raise ValueError(
            f"Seed mismatch in {path}: expected {expected_seed}, found {recorded_seed}"
        )
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Missing classification metrics in: {path}")
    required = {key for key, _ in ABLATION_METRICS}
    missing = sorted(required.difference(metrics))
    if missing:
        raise KeyError(f"Missing metrics in {path}: {', '.join(missing)}")
    return {key: float(metrics[key]) for key in required}


def load_ood_metrics(
    path: Path,
    expected_seed: int,
) -> tuple[dict[str, dict[str, float]], int, int, str]:
    payload = load_json(path)
    recorded_seed = int(payload.get("seed", expected_seed))
    if recorded_seed != expected_seed:
        raise ValueError(
            f"Seed mismatch in {path}: expected {expected_seed}, found {recorded_seed}"
        )
    results = payload.get("results")
    counts = payload.get("counts")
    if not isinstance(results, dict) or not isinstance(counts, dict):
        raise ValueError(f"Missing OOD results or counts in: {path}")
    methods: dict[str, dict[str, float]] = {}
    for _, method_key in OOD_METHODS:
        method_payload = results.get(method_key)
        if not isinstance(method_payload, dict):
            raise KeyError(f"Missing OOD method '{method_key}' in: {path}")
        methods[method_key] = {
            metric_key: float(method_payload[metric_key])
            for metric_key, _ in OOD_METRICS
        }
    return (
        methods,
        int(counts["id_test"]),
        int(counts["ood_test"]),
        str(payload.get("analysis_status", "")),
    )


def number(value: float) -> str:
    return f"{value:.4f}".replace(".", "{,}")


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def longtable(
    rows: Iterable[list[str]],
    *,
    column_spec: str,
    headers: list[str],
    caption: str,
    label: str,
    shaded_rows: set[int] | None = None,
) -> str:
    shaded_rows = shaded_rows or set()
    column_count = len(headers)
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        rf"\begin{{longtable}}{{{column_spec}}}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule",
        " & ".join(rf"\textbf{{{header}}}" for header in headers) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        rf"\multicolumn{{{column_count}}}{{c}}{{\tablename\ \thetable\ (tiếp theo)}}\\",
        r"\toprule",
        " & ".join(rf"\textbf{{{header}}}" for header in headers) + r" \\",
        r"\midrule",
        r"\endhead",
        rf"\midrule \multicolumn{{{column_count}}}{{r}}{{Còn tiếp ở trang sau}}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for index, row in enumerate(rows):
        prefix = r"\rowcolor{gray!12}" if index in shaded_rows else ""
        lines.append(prefix + " & ".join(row) + r" \\")
    lines.extend([r"\end{longtable}", r"\endgroup", ""])
    return "\n".join(lines)


def collect_full_shot(results_root: Path) -> dict[str, list[dict[str, Any]]]:
    collected: dict[str, list[dict[str, Any]]] = {}
    for dataset in ("btxrd", "ctch"):
        rows: list[dict[str, Any]] = []
        for model_label, experiment_path in FULL_SHOT_MODELS:
            for seed in SEEDS:
                path = (
                    results_root
                    / dataset
                    / experiment_path
                    / f"seed_{seed}"
                    / "metrics.json"
                )
                rows.append(
                    {
                        "model": model_label,
                        "seed": seed,
                        **load_classification_metrics(path, seed),
                    }
                )
        collected[dataset] = rows
    return collected


def collect_few_shot(results_root: Path) -> dict[str, list[dict[str, Any]]]:
    collected: dict[str, list[dict[str, Any]]] = {}
    for dataset in ("btxrd", "ctch"):
        rows: list[dict[str, Any]] = []
        for shot in (1, 10, 20):
            for model_label, model_path in FEW_SHOT_MODELS:
                for seed in SEEDS:
                    path = (
                        results_root
                        / dataset
                        / "few_shot"
                        / f"{shot}_shot"
                        / model_path
                        / f"seed_{seed}"
                        / "metrics.json"
                    )
                    rows.append(
                        {
                            "shot": shot,
                            "model": model_label,
                            "seed": seed,
                            **load_classification_metrics(path, seed),
                        }
                    )
        collected[dataset] = rows
    return collected


def collect_ablation(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant_label, experiment_path in ABLATION_VARIANTS:
        for seed in SEEDS:
            path = (
                results_root
                / "ctch"
                / experiment_path
                / f"seed_{seed}"
                / "metrics.json"
            )
            rows.append(
                {
                    "variant": variant_label,
                    "seed": seed,
                    **load_classification_metrics(path, seed),
                }
            )
    return rows


def collect_ood(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_label, scenario_key in OOD_SCENARIOS:
        for seed in SEEDS:
            path = (
                results_root
                / "ctch"
                / "proposed"
                / "ours_xbone_net"
                / f"seed_{seed}"
                / "analysis"
                / "ood"
                / scenario_key
                / "ood_metrics.json"
            )
            methods, id_count, ood_count, status = load_ood_metrics(path, seed)
            for method_label, method_key in OOD_METHODS:
                rows.append(
                    {
                        "scenario": scenario_label,
                        "scenario_key": scenario_key,
                        "method": method_label,
                        "method_key": method_key,
                        "seed": seed,
                        "id_count": id_count,
                        "ood_count": ood_count,
                        "status": status,
                        **methods[method_key],
                    }
                )
    return rows


def write_full_shot_tables(
    output: Path,
    collected: dict[str, list[dict[str, Any]]],
) -> Path:
    tables: list[str] = []
    headers = [
        "Mô hình",
        "Hạt giống",
        "Accuracy",
        "BAcc",
        "Macro-F1",
        "Macro-AUROC",
        "Macro-AUPRC",
    ]
    for dataset, display in (("btxrd", "BTXRD"), ("ctch", "CTCH")):
        rows = collected[dataset]
        table_rows = [
            [
                latex_escape(str(row["model"])),
                str(row["seed"]),
                *(number(float(row[key])) for key, _ in CLASSIFICATION_METRICS),
            ]
            for row in rows
        ]
        shaded = {
            index
            for index, row in enumerate(rows)
            if row["model"] == "XBone-Net"
        }
        tables.append(
            longtable(
                table_rows,
                column_spec=r"L{3.2cm}C{1.15cm}*{5}{C{1.55cm}}",
                headers=headers,
                caption=(
                    "Kết quả khi sử dụng toàn bộ dữ liệu huấn luyện của từng hạt giống trên "
                    f"{display}; hàng XBone-Net được tô xám để dễ đối chiếu."
                ),
                label=f"tab:appendix-full-shot-seeds-{dataset}",
                shaded_rows=shaded,
            )
        )
    output.write_text("\n".join(tables), encoding="utf-8")
    return output


def write_few_shot_tables(
    output: Path,
    collected: dict[str, list[dict[str, Any]]],
) -> Path:
    tables: list[str] = []
    headers = [
        "Thiết lập",
        "Mô hình",
        "Hạt giống",
        "Accuracy",
        "BAcc",
        "Macro-F1",
        "Macro-AUROC",
        "Macro-AUPRC",
    ]
    for dataset, display in (("btxrd", "BTXRD"), ("ctch", "CTCH")):
        rows = collected[dataset]
        table_rows = [
            [
                f"{row['shot']} mẫu",
                latex_escape(str(row["model"])),
                str(row["seed"]),
                *(number(float(row[key])) for key, _ in CLASSIFICATION_METRICS),
            ]
            for row in rows
        ]
        shaded = {
            index
            for index, row in enumerate(rows)
            if row["model"] == "XBone-Net"
        }
        tables.append(
            longtable(
                table_rows,
                column_spec=r"C{1.15cm}L{2.65cm}C{1.05cm}*{5}{C{1.35cm}}",
                headers=headers,
                caption=(
                    "Kết quả mẫu học hạn chế của từng hạt giống trên "
                    f"{display}; hàng XBone-Net được tô xám để dễ đối chiếu."
                ),
                label=f"tab:appendix-few-shot-seeds-{dataset}",
                shaded_rows=shaded,
            )
        )
    output.write_text("\n".join(tables), encoding="utf-8")
    return output


def write_ablation_table(output: Path, rows: list[dict[str, Any]]) -> Path:
    table_rows = [
        [
            latex_escape(str(row["variant"])),
            str(row["seed"]),
            *(number(float(row[key])) for key, _ in ABLATION_METRICS),
        ]
        for row in rows
    ]
    shaded = {
        index for index, row in enumerate(rows) if row["variant"] == "XBone-Net"
    }
    output.write_text(
        longtable(
            table_rows,
            column_spec=r"L{2.8cm}C{1.0cm}*{6}{C{1.35cm}}",
            headers=[
                "Cấu hình",
                "Hạt giống",
                "Accuracy",
                "BAcc",
                "Macro-F1",
                "Macro-AUROC",
                "Macro-AUPRC",
                "ECE",
            ],
            caption=(
                "Kết quả của từng hạt giống trong nghiên cứu loại bỏ từng thành phần "
                "trên CTCH; cấu hình đầy đủ được tô xám."
            ),
            label="tab:appendix-ablation-seeds",
            shaded_rows=shaded,
        ),
        encoding="utf-8",
    )
    return output


def write_ood_table(output: Path, rows: list[dict[str, Any]]) -> Path:
    table_rows = [
        [
            latex_escape(str(row["scenario"])),
            latex_escape(str(row["method"])),
            str(row["seed"]),
            str(row["id_count"]),
            str(row["ood_count"]),
            *(number(float(row[key])) for key, _ in OOD_METRICS),
        ]
        for row in rows
    ]
    shaded = {
        index
        for index, row in enumerate(rows)
        if row["method_key"] == "mahalanobis_centroid"
    }
    output.write_text(
        longtable(
            table_rows,
            column_spec=r"L{1.8cm}L{2.65cm}C{0.9cm}C{0.9cm}C{0.9cm}*{3}{C{1.45cm}}",
            headers=[
                "Kịch bản",
                "Phương pháp",
                "Hạt giống",
                "$N_{ID}$",
                "$N_{OOD}$",
                "AUROC-OOD",
                "AUPR-Out",
                "\\makecell{FPR@\\\\95\\%TPR}",
            ],
            caption=(
                "Kết quả OOD hậu xử lý của từng hạt giống; các hàng "
                "Mahalanobis theo tâm lớp được tô xám."
            ),
            label="tab:appendix-ood-seeds",
            shaded_rows=shaded,
        ),
        encoding="utf-8",
    )
    return output


def pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )
    return plt


def plot_full_shot_seed_f1(
    output: Path,
    collected: dict[str, list[dict[str, Any]]],
    dpi: int,
) -> Path:
    plt = pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
    models = (
        ("LoRA-BiomedCLIP", REPORT_PALETTE["orange"], "o"),
        ("LoRA-PubMedCLIP", REPORT_PALETTE["green"], "s"),
        ("XBone-Net", REPORT_PALETTE["blue"], "^"),
    )
    for axis, (dataset, display) in zip(
        axes,
        (("btxrd", "BTXRD"), ("ctch", "CTCH")),
    ):
        rows = collected[dataset]
        seed_positions = list(range(len(SEEDS)))
        for model, color, marker in models:
            selected = [row for row in rows if row["model"] == model]
            selected.sort(key=lambda row: int(row["seed"]))
            axis.plot(
                seed_positions,
                [row["f1_macro"] for row in selected],
                marker=marker,
                markersize=6,
                linewidth=2,
                color=color,
                label=model,
            )
        axis.set_title(f"Macro-F1 khi sử dụng toàn bộ dữ liệu trên {display}")
        axis.set_xlabel("Hạt giống ngẫu nhiên")
        axis.set_xticks(seed_positions, [str(seed) for seed in SEEDS])
        axis.set_ylabel("Macro-F1")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.tight_layout(rect=(0, 0.10, 1, 1))
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_ood_seed_metrics(
    output: Path,
    rows: list[dict[str, Any]],
    dpi: int,
) -> Path:
    plt = pyplot()
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), sharex=True)
    metric_specs = (
        ("auroc_ood", "AUROC-OOD"),
        ("aupr_out", "AUPR-Out"),
        ("fpr_at_95tpr", "FPR@95%TPR"),
    )
    scenario_specs = (
        ("semantic_ood", "OOD ngữ nghĩa", REPORT_PALETTE["blue"], "o"),
        ("domain_ood_btxrd", "BTXRD", REPORT_PALETTE["orange"], "s"),
    )
    for axis, (metric_key, metric_label) in zip(axes, metric_specs):
        seed_positions = list(range(len(SEEDS)))
        for scenario_key, scenario_label, color, marker in scenario_specs:
            selected = [
                row
                for row in rows
                if row["scenario_key"] == scenario_key
                and row["method_key"] == "mahalanobis_centroid"
            ]
            selected.sort(key=lambda row: int(row["seed"]))
            axis.plot(
                seed_positions,
                [row[metric_key] for row in selected],
                marker=marker,
                markersize=6,
                linewidth=2,
                color=color,
                label=scenario_label,
            )
        axis.set_title(metric_label)
        axis.set_xlabel("Hạt giống ngẫu nhiên")
        axis.set_xticks(seed_positions, [str(seed) for seed in SEEDS])
        axis.set_ylim(0.0, 1.02)
        axis.set_ylabel("Giá trị")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.tight_layout(rect=(0, 0.11, 1, 1))
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output


def build_outputs(results_root: Path, output_dir: Path, dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_shot = collect_full_shot(results_root)
    few_shot = collect_few_shot(results_root)
    ablation = collect_ablation(results_root)
    ood = collect_ood(results_root)
    outputs = [
        write_full_shot_tables(
            output_dir / "table_full_shot_per_seed.tex",
            full_shot,
        ),
        write_few_shot_tables(
            output_dir / "table_few_shot_per_seed.tex",
            few_shot,
        ),
        write_ablation_table(
            output_dir / "table_ablation_per_seed.tex",
            ablation,
        ),
        write_ood_table(output_dir / "table_ood_per_seed.tex", ood),
        plot_full_shot_seed_f1(
            output_dir / "full_shot_macro_f1_by_seed.png",
            full_shot,
            dpi,
        ),
        plot_ood_seed_metrics(
            output_dir / "ood_mahalanobis_by_seed.png",
            ood,
            dpi,
        ),
    ]
    manifest = {
        "seeds": list(SEEDS),
        "full_shot_rows": sum(len(rows) for rows in full_shot.values()),
        "few_shot_rows": sum(len(rows) for rows in few_shot.values()),
        "ablation_rows": len(ablation),
        "ood_rows": len(ood),
        "ood_counts": {
            scenario_key: {
                str(row["seed"]): {
                    "id_test": row["id_count"],
                    "ood_test": row["ood_count"],
                    "analysis_status": row["status"],
                }
                for row in ood
                if row["scenario_key"] == scenario_key
                and row["method_key"] == "mahalanobis_centroid"
            }
            for _, scenario_key in OOD_SCENARIOS
        },
        "outputs": [path.name for path in outputs],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    outputs.append(manifest_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/summary/appendix_seed_results"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    outputs = build_outputs(
        resolve_path(args.results_root),
        resolve_path(args.output_dir),
        args.dpi,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
