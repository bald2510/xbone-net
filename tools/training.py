"""Cung cấp công cụ nghiên cứu training cho XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np


# ============================================================
# Sổ đăng ký thí nghiệm và bộ xác định đường dẫn
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs" / "experiment"
RESULTS_ROOT = PROJECT_ROOT / "results"


def discover_experiment_configs(config_root: Path = CONFIG_ROOT) -> list[str]:
    """Thực hiện bước discover experiment configs trong quy trình hiện tại.

    Parameters
    ----------
    config_root : Path, optional
        Cấu hình điều khiển bước xử lý.

    Returns
    -------
    list[str]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    return sorted(
        path.relative_to(config_root).with_suffix("").as_posix()
        for path in config_root.rglob("*.yaml")
    )


def _experiment_group(experiment: str) -> str:
    """Thực hiện bước experiment group trong quy trình hiện tại.

    Parameters
    ----------
    experiment : str
        Giá trị ``experiment`` được sử dụng trong phép xử lý.

    Returns
    -------
    str
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    parts = experiment.split("/")
    dataset = parts[0]
    if len(parts) >= 4 and parts[1:3] == ["baselines", "zeroshot"]:
        return f"{dataset}_zeroshot"
    if len(parts) >= 4 and parts[1:3] == ["baselines", "full_finetuned"]:
        return f"{dataset}_finetuned"
    if len(parts) >= 4 and parts[1:3] == ["baselines", "peft_finetuned"]:
        return f"{dataset}_peft_finetuned"
    if len(parts) >= 4 and parts[1] == "few_shot":
        return f"{dataset}_few_shot_{parts[2].removesuffix('_shot')}"
    if len(parts) >= 3 and parts[1] == "proposed":
        return f"{dataset}_proposed"
    if len(parts) >= 3 and parts[1] == "ablation_study":
        return f"{dataset}_ablation"
    return f"{dataset}_other"


def _build_experiment_groups(experiments: list[str]) -> OrderedDict:
    """Xây dựng experiment groups cho bước xử lý hiện tại.

    Parameters
    ----------
    experiments : list[str]
        Giá trị ``experiments`` được sử dụng trong phép xử lý.

    Returns
    -------
    OrderedDict
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for experiment in experiments:
        groups[_experiment_group(experiment)].append(experiment)
    return OrderedDict(
        (name, sorted(values))
        for name, values in sorted(groups.items())
    )


_ALL_REGISTERED_EXPERIMENTS = discover_experiment_configs()
EXPERIMENTS = _build_experiment_groups(_ALL_REGISTERED_EXPERIMENTS)


def _registered_group(*experiments: str) -> list[str]:
    """Thực hiện bước registered group trong quy trình hiện tại.

    Parameters
    ----------
    *experiments : str
        Giá trị ``experiments`` được sử dụng trong phép xử lý.

    Returns
    -------
    list[str]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    RuntimeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    known = set(_ALL_REGISTERED_EXPERIMENTS)
    missing = sorted(set(experiments) - known)
    if missing:
        raise RuntimeError(
            "Research-question group references missing configs: "
            + ", ".join(missing)
        )
    return list(dict.fromkeys(experiments))


# Thiết lập thành phần dùng chung cho quy trình xử lý của mô-đun.
# Thiết lập thành phần dùng chung cho quy trình xử lý của mô-đun.
# Thiết lập giá trị trung gian cho bước xử lý tiếp theo.
RESEARCH_GROUPS = OrderedDict({
    "rq1_btxrd_modality": _registered_group(
        "btxrd/proposed/ours_xbone_net",
        "btxrd/ablation_study/modality/image_only",
        "btxrd/ablation_study/modality/text_only",
        "btxrd/ablation_study/modality/shuffled_report",
    ),
    "rq2_visual_encoding": _registered_group(
        "ctch/proposed/ours_xbone_net",
        "ctch/ablation_study/architecture/preprocess/xbone_nohighres",
        "ctch/ablation_study/architecture/preprocess/xbone_letterbox_mlp_classifier",
    ),
    "rq2_training_strategy": _registered_group(
        "ctch/proposed/ours_xbone_net",
        "ctch/ablation_study/architecture/phase/phase2_only",
        "ctch/ablation_study/architecture/phase/phase1_merged",
    ),
    "rq2_fusion": _registered_group(
        "ctch/proposed/ours_xbone_net",
        "ctch/ablation_study/architecture/fusion/concat",
        "ctch/ablation_study/architecture/fusion/image_to_text",
        "ctch/ablation_study/architecture/fusion/text_to_image",
    ),
    "rq2_classifier": _registered_group(
        "ctch/proposed/ours_xbone_net",
        "ctch/ablation_study/architecture/classifier/linear",
        "ctch/ablation_study/architecture/classifier/no_class_weight",
        "ctch/ablation_study/architecture/classifier/no_class_bias",
    ),
    "rq3_lora_vs_full_ft": _registered_group(
        "ctch/ablation_study/finetune/xbone_highres_no_ft",
        "ctch/proposed/ours_xbone_net",
        "ctch/ablation_study/finetune/xbone_highres_full_ft",
    ),
    "rq4_ood_representation_baselines": _registered_group(
        "ctch/proposed/ours_xbone_net",
        "ctch/baselines/zeroshot/biomedclip_zeroshot",
        "ctch/ablation_study/architecture/phase/phase2_only",
        "ctch/ablation_study/architecture/fusion/concat",
        "ctch/ablation_study/architecture/classifier/linear",
    ),
})
RESEARCH_GROUPS["rq2_all_components"] = list(dict.fromkeys(
    experiment
    for group_name in (
        "rq2_visual_encoding",
        "rq2_training_strategy",
        "rq2_fusion",
        "rq2_classifier",
    )
    for experiment in RESEARCH_GROUPS[group_name]
))

# Thiết lập giá trị trung gian cho bước xử lý tiếp theo.
PRIORITY_GROUPS = OrderedDict({
    "important": [
        experiment
        for experiment in _ALL_REGISTERED_EXPERIMENTS
        if "/proposed/" in experiment or experiment.endswith("/ours_xbone_net")
    ],
    "less_important": [
        experiment
        for experiment in _ALL_REGISTERED_EXPERIMENTS
        if "/proposed/" not in experiment
        and not experiment.endswith("/ours_xbone_net")
    ],
})

DEFAULT_SEEDS = [42, 123, 456]
DEFAULT_EXPERIMENT_FILE = Path(__file__).with_name("experiments.txt")

# Kiểm tra và xử lý checkpoint tương ứng của mô hình.
# Thiết lập thành phần dùng chung cho quy trình xử lý của mô-đun.
# Thiết lập giá trị trung gian cho bước xử lý tiếp theo.
CHECKPOINT_SOURCE_EXPERIMENTS = {
    "btxrd/ablation_study/modality/shuffled_report":
        "btxrd/proposed/ours_xbone_net",
    "ctch/ablation_study/modality/shuffled_report":
        "ctch/proposed/ours_xbone_net",
}

# Thiết lập thành phần dùng chung cho quy trình xử lý của mô-đun.
# Chuẩn bị và ghi tài nguyên đầu ra theo định dạng yêu cầu.
# Thiết lập thành phần dùng chung cho quy trình xử lý của mô-đun.
# Chuẩn bị và xử lý đầu vào hoặc đặc trưng văn bản.
RQ3_EXISTING_RESULT_EXPERIMENTS = {
    "ctch/ablation_study/finetune/xbone_highres_no_ft",
    "ctch/proposed/ours_xbone_net",
    "ctch/ablation_study/finetune/xbone_highres_full_ft",
}

RQ4_EXISTING_RESULT_EXPERIMENTS = {
    "ctch/proposed/ours_xbone_net",
    "ctch/baselines/zeroshot/biomedclip_zeroshot",
    "ctch/ablation_study/architecture/phase/phase2_only",
    "ctch/ablation_study/architecture/fusion/concat",
    "ctch/ablation_study/architecture/classifier/linear",
}

METRIC_KEYS = [
    "f1_macro", "accuracy", "sensitivity", "specificity",
    "precision", "auroc_macro", "auprc_macro", "balanced_accuracy",
    "ece_15", "adaptive_ece_15", "nll", "brier_score",
    "param_total", "param_trainable", "param_trainable_pct",
    "training_phase_runtime_seconds", "training_wall_clock_seconds",
    "training_gpu_hours", "training_peak_allocated_mb",
    "training_peak_reserved_mb",
]


def get_experiments(groups: list[str] | None) -> list[str]:
    """Lấy experiments cho bước xử lý hiện tại.

    Parameters
    ----------
    groups : list[str] | None
        Giá trị ``groups`` được sử dụng trong phép xử lý.

    Returns
    -------
    list[str]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    if not groups:
        return list(_ALL_REGISTERED_EXPERIMENTS)

    result = []
    for raw_group in groups:
        g = raw_group.strip().replace("\\", "/")
        g = g.removeprefix("configs/experiment/").removesuffix(".yaml")
        g = g.removeprefix("group:")
        if g in RESEARCH_GROUPS:
            result.extend(RESEARCH_GROUPS[g])
        elif g in EXPERIMENTS:
            result.extend(EXPERIMENTS[g])
        elif g in ("less_important", "less-important"):
            result.extend(PRIORITY_GROUPS["less_important"])
        elif g == "important":
            result.extend(PRIORITY_GROUPS["important"])
        elif g in ("zero_shot_baselines", "zeroshot"):
            for key, exps in EXPERIMENTS.items():
                if "zeroshot" in key:
                    result.extend(exps)
        elif g in ("finetuned_baselines", "finetuned"):
            for key, exps in EXPERIMENTS.items():
                if "finetuned" in key:
                    result.extend(exps)
        elif g == "proposed":
            for key, exps in EXPERIMENTS.items():
                if "proposed" in key:
                    result.extend(exps)
        elif g == "ablation":
            for key, exps in EXPERIMENTS.items():
                if "ablation" in key:
                    result.extend(exps)
        elif g in ("btxrd", "ctch"):
            result.extend(
                experiment
                for experiment in _ALL_REGISTERED_EXPERIMENTS
                if experiment.startswith(f"{g}/")
            )
        elif g == "all":
            result.extend(_ALL_REGISTERED_EXPERIMENTS)
        else:
            result.append(g)

    seen = set()
    return [exp for exp in result if not (exp in seen or seen.add(exp))]


def load_experiment_switches(path: Path) -> tuple[list[str], list[str]]:
    """Tải danh sách lựa chọn thí nghiệm từ tệp switch file hoặc cấu hình YAML.

    Parameters
    ----------
    path : Path
        Đường dẫn tới tệp chuyển mạch hoặc tệp cấu hình YAML.

    Returns
    -------
    tuple[list[str], list[str]]
        Cặp danh sách (enabled, disabled) chứa các tên thí nghiệm được bật và bị tắt.

    Raises
    ------
    ValueError
        Khi định dạng tệp chuyển mạch không hợp lệ.
    """
    if path.suffix in (".yaml", ".yml"):
        try:
            rel = path.resolve().relative_to(CONFIG_ROOT.resolve()).with_suffix("").as_posix()
        except ValueError:
            rel = path.stem
        return [rel], []

    enabled: list[str] = []
    disabled: list[str] = []
    section: list[str] | None = None
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        lowered = line.casefold()
        if lowered == "[enabled]":
            section = enabled
            continue
        if lowered == "[disabled]":
            section = disabled
            continue
        if line.startswith("+"):
            section = enabled
            line = line[1:].strip()
        elif line.startswith("-"):
            section = disabled
            line = line[1:].strip()
        elif lowered.startswith("on "):
            section = enabled
            line = line[3:].strip()
        elif lowered.startswith("off "):
            section = disabled
            line = line[4:].strip()
        if line.startswith("[") and line.endswith("]"):
            raise ValueError(
                f"{path}:{line_number}: unknown section {line!r}; "
                "use [enabled] or [disabled]."
            )
        if section is None:
            raise ValueError(
                f"{path}:{line_number}: selector appears before [enabled]/[disabled]."
            )
        selector = line.removeprefix("group:").strip()
        if not selector:
            raise ValueError(f"{path}:{line_number}: empty selector.")
        section.append(selector)
    return enabled, disabled


def select_experiments(
    groups: list[str] | None,
    experiment_file: Path | None,
) -> list[str]:
    """Chọn experiments cho bước xử lý hiện tại.

    Parameters
    ----------
    groups : list[str] | None
        Giá trị ``groups`` được sử dụng trong phép xử lý.
    experiment_file : Path | None
        Đường dẫn tài nguyên được sử dụng.

    Returns
    -------
    list[str]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    if experiment_file is None:
        return get_experiments(groups)

    enabled, disabled = load_experiment_switches(experiment_file)
    selected = (
        get_experiments(groups)
        if groups
        else (get_experiments(enabled) if enabled else [])
    )
    disabled_set = set(get_experiments(disabled)) if disabled else set()
    return [experiment for experiment in selected if experiment not in disabled_set]


def validate_experiment_configs(experiments: list[str]) -> None:
    """Kiểm tra tính hợp lệ của experiment configs cho bước xử lý hiện tại.

    Parameters
    ----------
    experiments : list[str]
        Giá trị ``experiments`` được sử dụng trong phép xử lý.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    known = set(_ALL_REGISTERED_EXPERIMENTS)
    unknown = sorted(set(experiments) - known)
    if unknown:
        raise ValueError(
            "Unknown experiment config(s): "
            + ", ".join(unknown)
            + ". Use --list-configs to inspect discovered configs."
        )


def seed_dir(experiment: str, seed: int) -> str:
    """Thực hiện bước seed dir trong quy trình hiện tại.

    Parameters
    ----------
    experiment : str
        Giá trị ``experiment`` được sử dụng trong phép xử lý.
    seed : int
        Hạt giống phục vụ khả năng tái lập.

    Returns
    -------
    str
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    checkpoint_experiment = CHECKPOINT_SOURCE_EXPERIMENTS.get(
        experiment,
        experiment,
    )
    return os.path.join(
        PROJECT_ROOT,
        "checkpoints",
        checkpoint_experiment,
        f"seed_{seed}",
    )


# ============================================================
# Thực thi và thu thập độ đo
# ============================================================

def run_one(script: str, experiment: str, seed: int, extra_args: list | None = None) -> bool:
    """Thực hiện one cho bước xử lý hiện tại.

    Parameters
    ----------
    script : str
        Giá trị ``script`` được sử dụng trong phép xử lý.
    experiment : str
        Giá trị ``experiment`` được sử dụng trong phép xử lý.
    seed : int
        Hạt giống phục vụ khả năng tái lập.
    extra_args : list | None, optional
        Các đối số vị trí bổ sung.

    Returns
    -------
    bool
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    sd = seed_dir(experiment, seed)
    output_dir = os.path.join(PROJECT_ROOT, "results", experiment, f"seed_{seed}")
    cmd = [
        sys.executable, script,
        f"+experiment={experiment}",
        f"++seed={seed}",
        f"++params.model_dir={sd}/",
        f"++params.phase2.checkpoint_path={os.path.join(sd, 'best_phase2.pth')}",
    ]

    if extra_args:
        cmd.extend(extra_args)

    if "evaluate.py" in script:
        cmd.extend(["--output-dir", output_dir])

    tag = "TRAIN" if "train" in script else "EVAL"
    exp_short = experiment.split("/")[-1]
    print(f"\n  [{tag}] {exp_short} seed={seed}")
    print(f"    -> {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"    [FAIL] FAILED (exit {result.returncode})")
        return False
    print("    [OK] Done")
    return True


def load_metrics(experiment: str, seed: int) -> dict | None:
    """Tải các độ đo cho bước xử lý hiện tại.

    Parameters
    ----------
    experiment : str
        Giá trị ``experiment`` được sử dụng trong phép xử lý.
    seed : int
        Hạt giống phục vụ khả năng tái lập.

    Returns
    -------
    dict | None
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    sd = seed_dir(experiment, seed)
    candidates = [  
        os.path.join(PROJECT_ROOT, "results", experiment, f"seed_{seed}", "metrics.json"),
        os.path.join(sd, "metrics.json"),
        os.path.join(sd, "eval_results.json"),
    ]
    for full in candidates:
        if os.path.exists(full):
            try:
                with open(full, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, dict):
                    metrics_dict = data.get("metrics", data)
                else:
                    metrics_dict = {}

                # Duy trì khả năng tương thích với cấu hình hoặc dữ liệu phiên bản cũ.
                for key, macro_key in [("sensitivity", "sensitivity_macro"), 
                                        ("specificity", "specificity_macro"), 
                                        ("precision", "precision_macro")]:
                    if key in metrics_dict and macro_key not in metrics_dict:
                        metrics_dict[macro_key] = metrics_dict[key]
                    if macro_key in metrics_dict and key not in metrics_dict:
                        metrics_dict[key] = metrics_dict[macro_key]

                training_path = Path(sd) / "training_summary.json"
                if training_path.is_file():
                    training = json.loads(training_path.read_text(encoding="utf-8"))
                    phases = [
                        phase
                        for phase in training.get("phases", {}).values()
                        if isinstance(phase, dict)
                    ]
                    peak_allocated = [
                        float(phase["peak_allocated_mb"])
                        for phase in phases
                        if phase.get("peak_allocated_mb") is not None
                    ]
                    peak_reserved = [
                        float(phase["peak_reserved_mb"])
                        for phase in phases
                        if phase.get("peak_reserved_mb") is not None
                    ]
                    timing_metrics = {
                        "training_phase_runtime_seconds": training.get(
                            "phase_runtime_seconds"
                        ),
                        "training_wall_clock_seconds": training.get(
                            "orchestration_wall_clock_seconds"
                        ),
                        "training_gpu_hours": training.get("gpu_hours"),
                        "training_peak_allocated_mb": (
                            max(peak_allocated) if peak_allocated else None
                        ),
                        "training_peak_reserved_mb": (
                            max(peak_reserved) if peak_reserved else None
                        ),
                    }
                    metrics_dict.update({
                        key: float(value)
                        for key, value in timing_metrics.items()
                        if value is not None
                    })

                return metrics_dict
            except Exception as e:
                print(f"    [WARN] Failed to load {full}: {e}")
    return None


def aggregate(all_metrics: dict[int, dict]) -> dict:
    """Tổng hợp kết quả cho bước xử lý hiện tại.

    Parameters
    ----------
    all_metrics : dict[int, dict]
        Giá trị ``all_metrics`` được sử dụng trong phép xử lý.

    Returns
    -------
    dict
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    agg = {}
    for key in METRIC_KEYS:
        vals = [m[key] for m in all_metrics.values()
                if m and key in m and m[key] is not None]
        if vals:
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            agg[key] = {
                "mean": float(np.mean(vals)),
                "std": std,
                "std_ddof": 1,
                "n": len(vals),
            }
    return agg


def save_results(experiment: str, seeds_metrics: dict, agg: dict) -> str:
    """Lưu các kết quả cho bước xử lý hiện tại.

    Parameters
    ----------
    experiment : str
        Giá trị ``experiment`` được sử dụng trong phép xử lý.
    seeds_metrics : dict
        Giá trị ``seeds_metrics`` được sử dụng trong phép xử lý.
    agg : dict
        Giá trị ``agg`` được sử dụng trong phép xử lý.

    Returns
    -------
    str
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    out_dir = os.path.join(PROJECT_ROOT, "results", experiment)
    os.makedirs(out_dir, exist_ok=True)

    summary = {
        "experiment": experiment,
        "seeds": list(seeds_metrics.keys()),
        "per_seed": {str(s): m for s, m in seeds_metrics.items()},
        "aggregated": agg,
    }
    path = os.path.join(out_dir, "aggregated_results.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return path


def discover_result_experiments(results_root: Path = RESULTS_ROOT) -> list[str]:
    """Thực hiện bước discover kết quả experiments trong quy trình hiện tại.

    Parameters
    ----------
    results_root : Path, optional
        Đường dẫn tài nguyên được sử dụng.

    Returns
    -------
    list[str]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    experiments: set[str] = set()
    if not results_root.is_dir():
        return []
    for path in results_root.rglob("metrics.json"):
        relative = path.parent.relative_to(results_root)
        if relative.name.startswith("seed_"):
            experiments.add(relative.parent.as_posix())
    for path in results_root.rglob("aggregated_results.json"):
        experiments.add(path.parent.relative_to(results_root).as_posix())
    known = set(_ALL_REGISTERED_EXPERIMENTS)
    return sorted(experiment for experiment in experiments if experiment in known)


def discover_result_seeds(experiment: str) -> list[int]:
    """Thực hiện bước discover kết quả seeds trong quy trình hiện tại.

    Parameters
    ----------
    experiment : str
        Giá trị ``experiment`` được sử dụng trong phép xử lý.

    Returns
    -------
    list[int]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    directory = RESULTS_ROOT / experiment
    seeds = []
    if directory.is_dir():
        for seed_directory in directory.glob("seed_*"):
            try:
                seed = int(seed_directory.name.removeprefix("seed_"))
            except ValueError:
                continue
            if (seed_directory / "metrics.json").is_file():
                seeds.append(seed)
    return sorted(set(seeds))


def load_table_results(
    experiments: list[str],
    requested_seeds: list[int] | None = None,
) -> tuple[OrderedDict, dict[str, list[int]]]:
    """Tải table các kết quả cho bước xử lý hiện tại.

    Parameters
    ----------
    experiments : list[str]
        Giá trị ``experiments`` được sử dụng trong phép xử lý.
    requested_seeds : list[int] | None, optional
        Giá trị ``requested_seeds`` được sử dụng trong phép xử lý.

    Returns
    -------
    tuple[OrderedDict, dict[str, list[int]]]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    results: OrderedDict[str, dict] = OrderedDict()
    used_seeds: dict[str, list[int]] = {}
    for experiment in sorted(experiments):
        seeds = requested_seeds or discover_result_seeds(experiment)
        per_seed = {}
        for seed in seeds:
            metrics = load_metrics(experiment, seed)
            if metrics:
                per_seed[seed] = metrics
        if per_seed:
            results[experiment] = aggregate(per_seed)
            used_seeds[experiment] = sorted(per_seed)
            continue

        aggregate_path = RESULTS_ROOT / experiment / "aggregated_results.json"
        if requested_seeds is None and aggregate_path.is_file():
            try:
                payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
                aggregated = payload.get("aggregated", {})
                if aggregated:
                    results[experiment] = aggregated
                    used_seeds[experiment] = [
                        int(seed) for seed in payload.get("seeds", [])
                    ]
            except (OSError, ValueError, json.JSONDecodeError) as error:
                print(f"  [WARN] Failed to load {aggregate_path}: {error}")
    return results, used_seeds


def experiment_metadata(experiment: str) -> tuple[str, str, str]:
    """Thực hiện bước experiment metadata trong quy trình hiện tại.

    Parameters
    ----------
    experiment : str
        Giá trị ``experiment`` được sử dụng trong phép xử lý.

    Returns
    -------
    tuple[str, str, str]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    parts = experiment.split("/")
    dataset = parts[0].upper()
    if len(parts) >= 4 and parts[1] == "baselines":
        labels = {
            "zeroshot": "Baselines / Zero-shot",
            "full_finetuned": "Baselines / Full fine-tuning",
            "peft_finetuned": "Baselines / PEFT",
        }
        category = labels.get(parts[2], f"Baselines / {parts[2]}")
    elif len(parts) >= 4 and parts[1] == "few_shot":
        category = f"Few-shot / {parts[2].replace('_', ' ')}"
    elif len(parts) >= 3 and parts[1] == "proposed":
        category = "Proposed"
    elif len(parts) >= 3 and parts[1] == "ablation_study":
        category = "Ablation / " + " / ".join(parts[2:-1])
    else:
        category = "Other"
    return dataset, category, parts[-1]


def _format_metric(aggregated: dict, key: str, digits: int = 3) -> str:
    """Định dạng độ đo cho bước xử lý hiện tại.

    Parameters
    ----------
    aggregated : dict
        Giá trị ``aggregated`` được sử dụng trong phép xử lý.
    key : str
        Tên hoặc khóa định danh của giá trị.
    digits : int, optional
        Giá trị ``digits`` được sử dụng trong phép xử lý.

    Returns
    -------
    str
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    metric = aggregated.get(key)
    if not metric or metric.get("n", 0) < 1:
        return "—"
    return f"{metric['mean']:.{digits}f}±{metric['std']:.{digits}f}"


def _format_parameter(aggregated: dict, key: str) -> str:
    """Định dạng parameter cho bước xử lý hiện tại.

    Parameters
    ----------
    aggregated : dict
        Giá trị ``aggregated`` được sử dụng trong phép xử lý.
    key : str
        Tên hoặc khóa định danh của giá trị.

    Returns
    -------
    str
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    metric = aggregated.get(key)
    if not metric:
        return "—"
    value = float(metric["mean"])
    if key == "param_trainable_pct":
        return f"{value:.2f}%"
    return f"{value / 1_000_000.0:.2f}M"


def _format_duration(aggregated: dict, key: str) -> str:
    """Định dạng duration cho bước xử lý hiện tại.

    Parameters
    ----------
    aggregated : dict
        Giá trị ``aggregated`` được sử dụng trong phép xử lý.
    key : str
        Tên hoặc khóa định danh của giá trị.

    Returns
    -------
    str
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    metric = aggregated.get(key)
    if not metric or metric.get("n", 0) < 1:
        return "—"
    mean_hours = float(metric["mean"]) / 3600.0
    std_hours = float(metric["std"]) / 3600.0
    return f"{mean_hours:.2f}±{std_hours:.2f}h(n={metric['n']})"


def _format_memory(aggregated: dict, key: str) -> str:
    """Định dạng memory cho bước xử lý hiện tại.

    Parameters
    ----------
    aggregated : dict
        Giá trị ``aggregated`` được sử dụng trong phép xử lý.
    key : str
        Tên hoặc khóa định danh của giá trị.

    Returns
    -------
    str
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    metric = aggregated.get(key)
    if not metric or metric.get("n", 0) < 1:
        return "—"
    mean_gib = float(metric["mean"]) / 1024.0
    std_gib = float(metric["std"]) / 1024.0
    return f"{mean_gib:.2f}±{std_gib:.2f}GiB(n={metric['n']})"


def _format_metric_with_n(
    aggregated: dict,
    key: str,
    digits: int = 2,
) -> str:
    """Định dạng độ đo with n cho bước xử lý hiện tại.

    Parameters
    ----------
    aggregated : dict
        Giá trị ``aggregated`` được sử dụng trong phép xử lý.
    key : str
        Tên hoặc khóa định danh của giá trị.
    digits : int, optional
        Giá trị ``digits`` được sử dụng trong phép xử lý.

    Returns
    -------
    str
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    metric = aggregated.get(key)
    if not metric or metric.get("n", 0) < 1:
        return "—"
    return (
        f"{metric['mean']:.{digits}f}±{metric['std']:.{digits}f}"
        f"(n={metric['n']})"
    )


def _result_seed_count(aggregated: dict) -> int:
    """Thực hiện bước kết quả seed count trong quy trình hiện tại.

    Parameters
    ----------
    aggregated : dict
        Giá trị ``aggregated`` được sử dụng trong phép xử lý.

    Returns
    -------
    int
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    return max(
        (
            int(metric.get("n", 0))
            for metric in aggregated.values()
            if isinstance(metric, dict)
        ),
        default=0,
    )


def export_results_csv(
    results: dict[str, dict],
    used_seeds: dict[str, list[int]],
    output: Path,
) -> None:
    """Xuất các kết quả csv cho bước xử lý hiện tại.

    Parameters
    ----------
    results : dict[str, dict]
        Giá trị ``results`` được sử dụng trong phép xử lý.
    used_seeds : dict[str, list[int]]
        Giá trị ``used_seeds`` được sử dụng trong phép xử lý.
    output : Path
        Vị trí hoặc cấu trúc nhận kết quả.
    """
    rows = []
    for experiment, aggregated in sorted(results.items()):
        dataset, category, config = experiment_metadata(experiment)
        row = {
            "dataset": dataset,
            "category": category,
            "config": config,
            "experiment": experiment,
            "seeds": " ".join(str(seed) for seed in used_seeds.get(experiment, [])),
            "n_seeds": _result_seed_count(aggregated),
        }
        for metric in METRIC_KEYS:
            summary = aggregated.get(metric, {})
            row[f"{metric}_mean"] = summary.get("mean")
            row[f"{metric}_std"] = summary.get("std")
        rows.append(row)
    if not rows:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_results_table(
    results: dict[str, dict],
    used_seeds: dict[str, list[int]] | None = None,
) -> None:
    """In các kết quả table cho bước xử lý hiện tại.

    Parameters
    ----------
    results : dict[str, dict]
        Giá trị ``results`` được sử dụng trong phép xử lý.
    used_seeds : dict[str, list[int]] | None, optional
        Giá trị ``used_seeds`` được sử dụng trong phép xử lý.
    """
    if not results:
        print("\nNo result metrics were found for the requested scope.")
        return

    used_seeds = used_seeds or {}
    grouped: dict[str, dict[str, list[tuple[str, str, dict]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for experiment, aggregated in results.items():
        dataset, category, config = experiment_metadata(experiment)
        grouped[dataset][category].append((config, experiment, aggregated))

    print("\n" + "=" * 142)
    print("RESULT REVIEW — mean±std across seeds")
    print("=" * 142)
    for dataset in sorted(grouped):
        print(f"\nDATASET: {dataset}")
        print("=" * 142)
        for category in sorted(grouped[dataset]):
            rows = sorted(grouped[dataset][category], key=lambda item: item[0])
            print(f"\n[{category}]")
            print(
                f"{'Config':<32} {'Seeds':>11} {'F1 macro':>13} {'Bal. acc.':>13} "
                f"{'Accuracy':>13} {'Sensitivity':>13} {'Specificity':>13}"
            )
            print("-" * 115)
            for config, experiment, aggregated in rows:
                seed_label = ",".join(
                    str(seed) for seed in used_seeds.get(experiment, [])
                ) or f"n={_result_seed_count(aggregated)}"
                print(
                    f"{config:<32} {seed_label:>11} "
                    f"{_format_metric(aggregated, 'f1_macro'):>13} "
                    f"{_format_metric(aggregated, 'balanced_accuracy'):>13} "
                    f"{_format_metric(aggregated, 'accuracy'):>13} "
                    f"{_format_metric(aggregated, 'sensitivity'):>13} "
                    f"{_format_metric(aggregated, 'specificity'):>13}"
                )

            print()
            print(
                f"{'Config':<32} {'Precision':>13} {'AUROC':>13} {'AUPRC':>13} "
                f"{'ECE':>13} {'Adapt. ECE':>13} {'NLL':>13} {'Brier':>13}"
            )
            print("-" * 142)
            for config, _, aggregated in rows:
                print(
                    f"{config:<32} "
                    f"{_format_metric(aggregated, 'precision'):>13} "
                    f"{_format_metric(aggregated, 'auroc_macro'):>13} "
                    f"{_format_metric(aggregated, 'auprc_macro'):>13} "
                    f"{_format_metric(aggregated, 'ece_15'):>13} "
                    f"{_format_metric(aggregated, 'adaptive_ece_15'):>13} "
                    f"{_format_metric(aggregated, 'nll'):>13} "
                    f"{_format_metric(aggregated, 'brier_score'):>13}"
                )

            print()
            print(
                f"{'Config':<32} {'Parameters':>13} {'Trainable':>13} "
                f"{'Trainable %':>13}"
            )
            print("-" * 75)
            for config, _, aggregated in rows:
                print(
                    f"{config:<32} "
                    f"{_format_parameter(aggregated, 'param_total'):>13} "
                    f"{_format_parameter(aggregated, 'param_trainable'):>13} "
                    f"{_format_parameter(aggregated, 'param_trainable_pct'):>13}"
                )

            print()
            print(
                f"{'Config':<32} {'Phase runtime':>24} {'Wall clock':>24} "
                f"{'GPU hours':>19} {'Peak alloc.':>24} {'Peak reserved':>24}"
            )
            print("-" * 151)
            for config, _, aggregated in rows:
                print(
                    f"{config:<32} "
                    f"{_format_duration(aggregated, 'training_phase_runtime_seconds'):>24} "
                    f"{_format_duration(aggregated, 'training_wall_clock_seconds'):>24} "
                    f"{_format_metric_with_n(aggregated, 'training_gpu_hours'):>19} "
                    f"{_format_memory(aggregated, 'training_peak_allocated_mb'):>24} "
                    f"{_format_memory(aggregated, 'training_peak_reserved_mb'):>24}"
                )
    print("\n" + "=" * 142)


# ============================================================
# Điểm vào chính và phân tích tham số dòng lệnh
# ============================================================

def main():
    """Thực thi điểm vào chính của mô-đun."""
    parser = argparse.ArgumentParser(
        description="Run all experiments (baselines + ablations) with multi-seed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--group", "-g", nargs="+", default=None,
                        help="Experiment groups to run or direct config path.")
    parser.add_argument(
        "--experiment-file",
        type=Path,
        default=DEFAULT_EXPERIMENT_FILE,
        help=(
            "Text file with [enabled]/[disabled] selectors "
            f"(default: {DEFAULT_EXPERIMENT_FILE})."
        ),
    )
    parser.add_argument(
        "--ignore-experiment-file",
        action="store_true",
        help="Ignore the text switches and select experiments only from --group.",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List every discovered config grouped by dataset/category and exit.",
    )
    parser.add_argument(
        "--priority-group",
        choices=tuple(PRIORITY_GROUPS),
        default=None,
        help=(
            "Run only experiments assigned to this priority. When used with "
            "--group, the intersection is selected."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected experiment paths without launching subprocesses.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Do not retrain a seed when its best_phase2.pth already exists.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help=(
            f"Seeds to run (training default: {DEFAULT_SEEDS}); "
            "--table discovers all available seeds when omitted."
        ),
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--train-only", action="store_true",
                            help="Only train models and save checkpoints")
    mode_group.add_argument("--eval-only", action="store_true",
                            help="Only evaluate existing checkpoints")
    mode_group.add_argument("--table", action="store_true",
                            help="Review existing results without training/evaluation")
    parser.add_argument("--phase2-only", "-p2", action="store_true",
                        help="Skip Phase 1 contrastive training, only train Phase 2 classifier")
    parser.add_argument(
        "--table-selected-only",
        action="store_true",
        help="With --table, review only configs enabled by experiments.txt.",
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=Path("results/summary/run_all_table.csv"),
        help="CSV exported by --table.",
    )
    parser.add_argument("--bootstrap", action="store_true",
                        help="Compute paired sample-level bootstrap CIs during evaluation")
    parser.add_argument("--n-bootstrap", type=int, default=10_000,
                        help="Bootstrap resamples passed to evaluate.py (default: 10000)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override per-device train/eval batch size for every selected config.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Keep effective batch size fixed on memory-constrained GPUs.",
    )
    parser.add_argument("configs", nargs="*", default=[],
                        help="Optional direct experiment config names or YAML paths.")
    args = parser.parse_args()
    if args.configs:
        args.group = (args.group or []) + args.configs
        args.ignore_experiment_file = True
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if (
        args.gradient_accumulation_steps is not None
        and args.gradient_accumulation_steps < 1
    ):
        parser.error("--gradient-accumulation-steps must be positive")
    if args.phase2_only and (args.eval_only or args.table):
        parser.error("--phase2-only is only valid for training modes")
    if args.table_selected_only and not args.table:
        parser.error("--table-selected-only requires --table")

    if args.list_configs:
        print("\nRESEARCH-QUESTION GROUPS")
        print("========================")
        for group, values in RESEARCH_GROUPS.items():
            print(f"\n[{group}]")
            for experiment in values:
                print(f"  {experiment}")
        for dataset in ("BTXRD", "CTCH"):
            print(f"\n{dataset}")
            print("=" * len(dataset))
            for group, values in EXPERIMENTS.items():
                if group.startswith(dataset.lower()):
                    print(f"\n[{group}]")
                    for experiment in values:
                        print(f"  {experiment}")
        return

    experiment_file = None
    uses_switch_file = not args.ignore_experiment_file and (
        not args.table or args.table_selected_only
    )
    if uses_switch_file:
        experiment_file = args.experiment_file.expanduser().resolve()
        if not experiment_file.is_file():
            parser.error(f"Experiment switch file not found: {experiment_file}")
    try:
        if args.table and not args.table_selected_only:
            experiments = (
                get_experiments(args.group)
                if args.group
                else discover_result_experiments()
            )
        else:
            experiments = select_experiments(args.group, experiment_file)
        validate_experiment_configs(experiments)
    except ValueError as error:
        parser.error(str(error))
    if args.priority_group is not None:
        priority_set = set(PRIORITY_GROUPS[args.priority_group])
        experiments = [
            experiment for experiment in experiments if experiment in priority_set
        ]
    if not experiments:
        scope = "result files" if args.table else "requested switches/groups"
        parser.error(f"No experiments matched the {scope}.")

    if args.table:
        table_results, used_seeds = load_table_results(
            experiments,
            requested_seeds=args.seeds,
        )
        print_results_table(table_results, used_seeds)
        table_output = (
            args.table_output
            if args.table_output.is_absolute()
            else PROJECT_ROOT / args.table_output
        )
        export_results_csv(table_results, used_seeds, table_output)
        if table_results:
            print(f"Review CSV -> {table_output}")
        missing = sorted(set(experiments) - set(table_results))
        if missing:
            print(f"\nConfigs without readable metrics ({len(missing)}):")
            for experiment in missing:
                print(f"  {experiment}")
        return

    if args.dry_run:
        priority_label = args.priority_group
        if priority_label is None and args.group and len(args.group) == 1:
            if args.group[0] in PRIORITY_GROUPS:
                priority_label = args.group[0]
        priority_label = priority_label or "custom/all"
        print(f"Selected {len(experiments)} experiments for {priority_label}:")
        if experiment_file is not None:
            print(f"Switch file: {experiment_file}")
        for experiment in experiments:
            print(f"  {experiment}")
        return
    seeds = args.seeds or DEFAULT_SEEDS
    total_runs = sum(
        1 if "zeroshot" in experiment else len(seeds)
        for experiment in experiments
    )

    mode_str = "TABLE ONLY" if args.table else (
        "PHASE2 ONLY" if args.phase2_only else (
            "TRAIN ONLY" if args.train_only else (
                "EVAL ONLY" if args.eval_only else "TRAIN + EVAL"
            )
        )
    )

    print("+----------------------------------------------------------+")
    print("|  XBone-Net -- Full Experiment Suite                      |")
    print("+----------------------------------------------------------+")
    print(f"|  Experiments : {len(experiments):<40d} |")
    print(f"|  Seeds       : {str(seeds):<40s} |")
    print(f"|  Total runs  : {total_runs:<40d} |")
    print(f"|  Mode        : {mode_str:<40s} |")
    print("+----------------------------------------------------------+")

    all_results = OrderedDict()

    for i, experiment in enumerate(experiments, 1):
        exp_short = experiment.split("/")[-1]
        print(f"\n{'-'*60}")
        print(f"  [{i}/{len(experiments)}] {experiment}")
        print(f"{'-'*60}")

        seeds_metrics = {}
        is_zeroshot = "zeroshot" in experiment
        checkpoint_source = CHECKPOINT_SOURCE_EXPERIMENTS.get(experiment)
        experiment_seeds = seeds[:1] if is_zeroshot else seeds
        if is_zeroshot and len(seeds) > 1:
            print(
                "  [ZERO-SHOT] Deterministic foundation evaluation uses one "
                f"seed ({experiment_seeds[0]}); repeated seeds would be "
                "pseudo-replication."
            )

        for seed in experiment_seeds:
            if args.table:
                metrics = load_metrics(experiment, seed)
                if metrics:
                    seeds_metrics[seed] = metrics
                continue

            if (
                args.skip_completed
                and experiment
                in (
                    RQ3_EXISTING_RESULT_EXPERIMENTS
                    | RQ4_EXISTING_RESULT_EXPERIMENTS
                )
            ):
                existing_metrics = load_metrics(experiment, seed)
                if existing_metrics:
                    print(
                        "\n  [REUSE EXISTING RESULT] Existing metrics"
                        f"/runtime: {experiment} seed={seed}"
                    )
                    seeds_metrics[seed] = existing_metrics
                    continue

            sd = seed_dir(experiment, seed)
            ckpt_p2 = os.path.join(sd, "best_phase2.pth")
            ckpt_p1 = os.path.join(sd, "best_phase1.pth")

            # Kiểm tra điều kiện trước khi thực hiện nhánh xử lý tương ứng.
            if checkpoint_source is not None and not args.eval_only:
                print(
                    "\n  [CHECKPOINT REUSE] Test-time intervention uses "
                    f"{checkpoint_source} seed={seed}; training is skipped."
                )
                if not os.path.exists(ckpt_p2) and not os.path.exists(ckpt_p1):
                    print(
                        "    [ERROR] Source checkpoint is missing. Run the "
                        f"source experiment first: {checkpoint_source}"
                    )
                    continue
                if args.train_only:
                    continue
            elif not args.eval_only and not is_zeroshot:
                if args.skip_completed and os.path.exists(ckpt_p2):
                    print(
                        f"\n  [SKIP] Existing Phase-2 checkpoint: {ckpt_p2}"
                    )
                    if args.train_only:
                        continue
                else:
                    train_extra = []
                    if args.phase2_only:
                        train_extra.extend([
                            "++params.run_phase1=false",
                            "++params.phase1.enabled=false",
                            "++params.run_phase2=true",
                            "++params.phase2.enabled=true",
                        ])
                    if args.batch_size is not None:
                        train_extra.append(f"++dataset.batch_size={args.batch_size}")
                    if args.gradient_accumulation_steps is not None:
                        train_extra.append(
                            "++params.gradient_accumulation_steps="
                            f"{args.gradient_accumulation_steps}"
                        )
                    print(f"\n  [STEP 1/2: TRAINING] Launching train.py for {experiment} (seed={seed})...")
                    sys.stdout.flush()
                    train_success = run_one(
                        "train.py", experiment, seed, extra_args=train_extra or None
                    )

                    # Kiểm tra điều kiện và dữ liệu trước khi tiếp tục xử lý.
                    if not train_success:
                        print(f"    [ERROR] train.py failed for seed={seed}. ABORTING evaluation for this seed.")
                        continue

                    # Kiểm tra và xử lý checkpoint tương ứng của mô hình.
                    if not os.path.exists(ckpt_p2) and not os.path.exists(ckpt_p1):
                        print(f"    [ERROR] train.py completed but no trained checkpoint (.pth) was found in '{sd}'.")
                        print("            ABORTING evaluation to prevent evaluating untrained random weights.")
                        continue

                    print(f"    [OK] Training complete! Verified checkpoint on disk: {ckpt_p2 if os.path.exists(ckpt_p2) else ckpt_p1}")

            elif is_zeroshot and not args.eval_only:
                exp_short = experiment.split("/")[-1]
                print(f"\n  [ZERO-SHOT BASELINE] {exp_short} seed={seed}")
                if args.train_only:
                    print("    -> Zero-shot baseline requires no fine-tuning. Skipping.")
                    continue
                else:
                    print("    -> Foundation model requires no fine-tuning. Launching evaluate.py directly.")

            elif args.eval_only and not is_zeroshot:
                if not os.path.exists(ckpt_p2) and not os.path.exists(ckpt_p1):
                    print(f"    [WARN] Mode is --eval-only but no trained checkpoint exists at '{sd}'. Skipping evaluation.")
                    continue

            if args.train_only:
                print(f"    [OK] Mode is --train-only. Training complete for seed={seed}. Skipping evaluation.")
                continue

            # Bước hỗ trợ để thực thi điểm vào chính của mô-đun.
            print(f"\n  [STEP 2/2: EVALUATION] Launching evaluate.py for {experiment} (seed={seed})...")
            sys.stdout.flush()
            eval_extra = []
            if args.batch_size is not None:
                eval_extra.append(f"++dataset.batch_size={args.batch_size}")
            if args.bootstrap:
                eval_extra.extend(
                    ["--bootstrap", "--n-bootstrap", str(args.n_bootstrap)]
                )
            eval_success = run_one(
                "evaluate.py", experiment, seed, extra_args=eval_extra or None
            )
            if not eval_success:
                print(f"    [ERROR] evaluate.py failed for seed={seed}.")
                continue

            metrics = load_metrics(experiment, seed)
            if metrics:
                seeds_metrics[seed] = metrics
            else:
                print(f"    [WARN] No metrics found for seed={seed}")

        if seeds_metrics:
            agg = aggregate(seeds_metrics)
            save_path = save_results(experiment, seeds_metrics, agg)
            all_results[experiment] = agg

            f1 = agg.get("f1_macro", {})
            if f1:
                print(f"\n  > {exp_short}: F1={f1['mean']:.4f}+/-{f1['std']:.4f} (n={f1['n']})")
                print(f"    Saved -> {save_path}")
        else:
            print(f"\n  > {exp_short}: No results collected")
            all_results[experiment] = {}

    print_results_table(all_results)

    master_path = os.path.join(PROJECT_ROOT, "results", "all_results.json")
    os.makedirs(os.path.dirname(master_path), exist_ok=True)
    with open(master_path, "w") as f:
        json.dump(
            {exp: agg for exp, agg in all_results.items()},
            f, indent=2,
        )
    print(f"  Master results -> {master_path}")


if __name__ == "__main__":
    main()
