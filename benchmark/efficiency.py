"""Thực hiện benchmark efficiency cho quy trình nghiên cứu XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _parse_args() -> argparse.Namespace:
    """Phân tích các tham số dòng lệnh.

    Returns
    -------
    argparse.Namespace
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    parser = argparse.ArgumentParser(
        description="Measure inference efficiency for one Hydra experiment."
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--skip-flops",
        action="store_true",
        help="Skip FlopCounterMode when only latency/params are needed.",
    )
    parser.add_argument(
        "--export-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a report-ready efficiency.csv beside efficiency.json.",
    )
    parser.add_argument(
        "--export-latex",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a report-ready efficiency_table.tex beside efficiency.json.",
    )
    args, hydra_args = parser.parse_known_args()
    if args.batch_size < 1 or args.num_batches < 1:
        parser.error("--batch-size and --num-batches must be positive.")
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be non-negative and --repeats must be positive.")
    sys.argv = [sys.argv[0], *hydra_args]
    return args


ARGS = _parse_args()

# Thiết lập thành phần dùng chung cho quy trình xử lý của mô-đun.
# Kiểm tra và xử lý checkpoint tương ứng của mô hình.
import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from evaluate import load_model_checkpoint, seed_everything  # noqa: E402
from src.datasets.builder import build_dataloader  # noqa: E402
from src.models.builder import (  # noqa: E402
    build_model,
    checkpoint_model_config,
    setup_phase2_modules,
)
from src.utils.efficiency import (  # noqa: E402
    batch_metadata,
    count_supported_flops,
    measure_cuda_memory_mb,
    measure_latency_ms,
    parameter_summary,
    summarize_measurements,
)
from src.utils.prompts import generate_clip_class_prompts  # noqa: E402
from src.utils.trainer import BioMedCLIPDataCollator, resolve_pad_token_id  # noqa: E402


def _resolve_device(name: str) -> torch.device:
    """Xác định thiết bị cho bước xử lý hiện tại.

    Parameters
    ----------
    name : str
        Tên hoặc khóa định danh của giá trị.

    Returns
    -------
    torch.device
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    RuntimeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable.")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Thực hiện bước move batch trong quy trình hiện tại.

    Parameters
    ----------
    batch : dict[str, Any]
        Batch dữ liệu đầu vào.
    device : torch.device
        Thiết bị thực thi phép tính.

    Returns
    -------
    dict[str, Any]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _tokenize_prompts(tokenizer, texts: list[str], device: torch.device):
    """Thực hiện bước tokenize prompts trong quy trình hiện tại.

    Parameters
    ----------
    tokenizer : object
        Giá trị ``tokenizer`` được sử dụng trong phép xử lý.
    texts : list[str]
        Giá trị ``texts`` được sử dụng trong phép xử lý.
    device : torch.device
        Thiết bị thực thi phép tính.

    Returns
    -------
    object
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    encoded = tokenizer(texts)
    attention_mask = None
    if isinstance(encoded, dict):
        attention_mask = encoded.get("attention_mask")
        encoded = encoded["input_ids"]
    input_ids = encoded.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


def _encode_zero_shot_prompts(
    model,
    class_names: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Mã hóa zero shot prompts cho bước xử lý hiện tại.

    Parameters
    ----------
    model : object
        Mô hình hoặc thành phần mô hình cần xử lý.
    class_names : list[str]
        Nhãn hoặc chỉ số lớp liên quan.
    device : torch.device
        Thiết bị thực thi phép tính.

    Returns
    -------
    torch.Tensor
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    AttributeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    tokenizer = getattr(model.backbone, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Zero-shot efficiency benchmarking requires a tokenizer.")

    prompts = generate_clip_class_prompts(class_names)
    flat_prompts = [prompts[class_name] for class_name in class_names]

    input_ids, attention_mask = _tokenize_prompts(tokenizer, flat_prompts, device)
    encoder = getattr(model.backbone, "encode_text", None)
    if encoder is None:
        encoder = getattr(getattr(model.backbone, "model", None), "encode_text", None)
    if encoder is None:
        raise AttributeError("The zero-shot backbone does not expose encode_text().")

    with torch.inference_mode():
        try:
            features = encoder(input_ids, attention_mask=attention_mask)
        except TypeError:
            features = encoder(input_ids)
        features = F.normalize(features, dim=-1)
    return features


def _select_report_inputs(
    batch: dict[str, Any],
    is_classifier: bool,
    use_text: bool,
    report_type: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Chọn báo cáo inputs cho bước xử lý hiện tại.

    Parameters
    ----------
    batch : dict[str, Any]
        Batch dữ liệu đầu vào.
    is_classifier : bool
        Giá trị ``is_classifier`` được sử dụng trong phép xử lý.
    use_text : bool
        Văn bản hoặc biểu diễn văn bản đầu vào.
    report_type : str
        Văn bản hoặc biểu diễn văn bản đầu vào.

    Returns
    -------
    tuple[torch.Tensor | None, torch.Tensor | None]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    NotImplementedError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if not is_classifier or not use_text:
        return None, None
    if report_type in ("both", "xray_clinical"):
        raise NotImplementedError(
            "The benchmark follows evaluation.py, which does not support "
            "simultaneous token-level dual-report inference."
        )
    prefix = "xray" if report_type == "xray" else "clinical"
    return batch[f"{prefix}_input_ids"], batch.get(f"{prefix}_attention_mask")


def _build_forward(
    model,
    batch: dict[str, Any],
    is_classifier: bool,
    use_text: bool,
    report_type: str,
    zero_shot_features: torch.Tensor | None,
    temperature: float,
):
    """Xây dựng forward cho bước xử lý hiện tại.

    Parameters
    ----------
    model : object
        Mô hình hoặc thành phần mô hình cần xử lý.
    batch : dict[str, Any]
        Batch dữ liệu đầu vào.
    is_classifier : bool
        Giá trị ``is_classifier`` được sử dụng trong phép xử lý.
    use_text : bool
        Văn bản hoặc biểu diễn văn bản đầu vào.
    report_type : str
        Văn bản hoặc biểu diễn văn bản đầu vào.
    zero_shot_features : torch.Tensor | None
        Giá trị ``zero_shot_features`` được sử dụng trong phép xử lý.
    temperature : float
        Giá trị ``temperature`` được sử dụng trong phép xử lý.

    Returns
    -------
    object
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    AttributeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    input_ids, attention_mask = _select_report_inputs(
        batch, is_classifier, use_text, report_type
    )
    if is_classifier:
        def forward():
            """Thực hiện lượt lan truyền xuôi của mô hình.

            Returns
            -------
            object
                Kết quả được tạo bởi bước xử lý của hàm.
            """
            return model(
                images=batch["pixel_values"],
                input_ids=input_ids,
                attention_mask=attention_mask,
                tile_values=batch.get("tile_values"),
                tile_mask=batch.get("tile_mask"),
                tile_boxes=batch.get("tile_boxes"),
            )

        return forward, attention_mask

    image_encoder = getattr(getattr(model.backbone, "model", None), "encode_image", None)
    if image_encoder is None or zero_shot_features is None:
        raise AttributeError("Zero-shot benchmarking requires image and text encoders.")

    def forward():
        """Thực hiện lượt lan truyền xuôi của mô hình.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        image_features = F.normalize(image_encoder(batch["pixel_values"]), dim=-1)
        class_logits = image_features @ zero_shot_features.T
        return torch.softmax(class_logits / temperature, dim=-1)

    return forward, None


def _with_precision(forward_fn, device: torch.device, precision: str):
    """Thực hiện bước with precision trong quy trình hiện tại.

    Parameters
    ----------
    forward_fn : object
        Giá trị ``forward_fn`` được sử dụng trong phép xử lý.
    device : torch.device
        Thiết bị thực thi phép tính.
    precision : str
        Giá trị ``precision`` được sử dụng trong phép xử lý.

    Returns
    -------
    object
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if precision == "fp32":
        return forward_fn
    if device.type == "cpu" and precision == "fp16":
        raise ValueError("FP16 autocast is not supported by this CPU benchmark.")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16

    def autocast_forward():
        """Thực hiện bước autocast forward trong quy trình hiện tại.

        Returns
        -------
        object
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        with torch.autocast(device_type=device.type, dtype=dtype):
            return forward_fn()

    return autocast_forward


def _aggregate_input_profile(batch_profiles: list[dict[str, Any]]) -> dict[str, Any]:
    """Tổng hợp đầu vào profile cho bước xử lý hiện tại.

    Parameters
    ----------
    batch_profiles : list[dict[str, Any]]
        Batch dữ liệu đầu vào.

    Returns
    -------
    dict[str, Any]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    valid_tiles = [
        value
        for profile in batch_profiles
        for value in profile.get("valid_tiles_per_sample", [])
    ]
    valid_text = [
        value
        for profile in batch_profiles
        for value in profile.get("valid_text_tokens_per_sample", [])
    ]
    result: dict[str, Any] = {
        "batches": len(batch_profiles),
        "samples": sum(profile["batch_size"] for profile in batch_profiles),
        "observed_batches": batch_profiles,
    }
    if valid_tiles:
        result["valid_tiles_per_sample"] = summarize_measurements(valid_tiles)
    if valid_text:
        result["valid_text_tokens_per_sample"] = summarize_measurements(valid_text)
    return result


def _output_path(cfg: DictConfig) -> Path:
    """Thực hiện bước đầu ra đường dẫn trong quy trình hiện tại.

    Parameters
    ----------
    cfg : DictConfig
        Cấu hình điều khiển bước xử lý.

    Returns
    -------
    Path
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    experiment_name = str(cfg.get("experiment_name", "default"))
    params_cfg = cfg.get("params", {}) or {}
    seed = int(params_cfg.get("seed", cfg.get("seed", 42)))
    if ARGS.output_dir:
        directory = Path(ARGS.output_dir)
        if not directory.is_absolute():
            directory = ROOT / directory
    else:
        directory = ROOT / "results" / experiment_name / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "efficiency.json"


def _report_row(result: dict[str, Any]) -> dict[str, Any]:
    """Thực hiện bước báo cáo row trong quy trình hiện tại.

    Parameters
    ----------
    result : dict[str, Any]
        Giá trị ``result`` được sử dụng trong phép xử lý.

    Returns
    -------
    dict[str, Any]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    flops = result["supported_gflops_per_sample"]
    memory = result["cuda_memory"]
    return {
        "experiment": result["experiment_name"],
        "seed": result["seed"],
        "device": result["protocol"]["device_name"],
        "precision": result["protocol"]["precision"],
        "batch_size": result["protocol"]["batch_size_requested"],
        "parameters_total": result["parameters"]["total"],
        "parameters_trainable": result["parameters"]["trainable"],
        "gflops_per_sample": None if flops is None else flops["mean"],
        "latency_ms_mean": result["latency_ms_per_sample"]["mean"],
        "latency_ms_p95": result["latency_ms_per_sample"]["p95"],
        "throughput_samples_s": result["throughput_samples_per_second"]["mean"],
        "peak_gpu_memory_mib": (
            None if memory is None else memory["peak_allocated_mb"]
        ),
    }


def _write_csv_report(path: Path, row: dict[str, Any]) -> None:
    """Ghi csv báo cáo cho bước xử lý hiện tại.

    Parameters
    ----------
    path : Path
        Đường dẫn tài nguyên được sử dụng.
    row : dict[str, Any]
        Giá trị ``row`` được sử dụng trong phép xử lý.
    """
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _latex_escape(value: Any) -> str:
    """Thực hiện bước latex escape trong quy trình hiện tại.

    Parameters
    ----------
    value : Any
        Giá trị ``value`` được sử dụng trong phép xử lý.

    Returns
    -------
    str
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    text = str(value)
    for source, replacement in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ):
        text = text.replace(source, replacement)
    return text


def _write_latex_report(path: Path, row: dict[str, Any]) -> None:
    """Ghi latex báo cáo cho bước xử lý hiện tại.

    Parameters
    ----------
    path : Path
        Đường dẫn tài nguyên được sử dụng.
    row : dict[str, Any]
        Giá trị ``row`` được sử dụng trong phép xử lý.
    """
    def number(key: str, digits: int = 2) -> str:
        """Thực hiện bước number trong quy trình hiện tại.

        Parameters
        ----------
        key : str
            Tên hoặc khóa định danh của giá trị.
        digits : int, optional
            Giá trị ``digits`` được sử dụng trong phép xử lý.

        Returns
        -------
        str
            Kết quả được tạo bởi bước xử lý của hàm.
        """
        value = row[key]
        return "--" if value is None else f"{float(value):.{digits}f}"

    columns = (
        "Experiment & Params & GFLOPs/sample & Latency (ms) & "
        "Throughput (sample/s) & Peak GPU (MiB)"
    )
    values = " & ".join(
        [
            _latex_escape(row["experiment"]),
            f"{int(row['parameters_total']):,}",
            number("gflops_per_sample", 3),
            number("latency_ms_mean", 3),
            number("throughput_samples_s", 2),
            number("peak_gpu_memory_mib", 1),
        ]
    )
    table = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Inference efficiency benchmark.}\n"
        "\\label{tab:efficiency}\n"
        "\\begin{tabular}{lrrrrr}\n"
        "\\toprule\n"
        f"{columns} \\\\\n"
        "\\midrule\n"
        f"{values} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    path.write_text(table, encoding="utf-8")


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Thực thi điểm vào chính của mô-đun.

    Parameters
    ----------
    cfg : DictConfig
        Cấu hình điều khiển bước xử lý.

    Raises
    ------
    FileNotFoundError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    RuntimeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    TypeError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    OmegaConf.set_struct(cfg, False)
    cfg.dataset.batch_size = ARGS.batch_size

    params_cfg = cfg.get("params", {}) or {}
    seed = int(params_cfg.get("seed", cfg.get("seed", 42)))
    seed_everything(seed)
    device = _resolve_device(ARGS.device)
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    print(f"Building model on {device} for efficiency benchmark...")
    model = build_model(checkpoint_model_config(cfg)).to(device)
    model, classifier_type, fusion_type, _ = setup_phase2_modules(model, cfg, device)
    model.eval()

    is_classifier = classifier_type != "none"
    experiment_name = str(cfg.get("experiment_name", "")).lower()
    is_zero_shot = (not is_classifier) or ("zeroshot" in experiment_name)
    checkpoint_loaded = load_model_checkpoint(
        model, cfg, device, is_zero_shot=is_zero_shot
    )
    if not checkpoint_loaded and not is_zero_shot:
        raise FileNotFoundError(
            "A trained checkpoint is required to benchmark this classifier."
        )

    tokenizer = getattr(
        model.backbone,
        "tokenizer_obj",
        getattr(model.backbone, "tokenizer", None),
    )
    test_loader = build_dataloader(
        cfg=cfg.dataset,
        split="test",
        transform=model.backbone.preprocess,
        tokenizer=tokenizer,
    )
    pad_id = resolve_pad_token_id(tokenizer) if tokenizer is not None else 0
    test_loader.collate_fn = BioMedCLIPDataCollator(pad_token_id=pad_id)

    dataset_params = cfg.dataset.get("params", {}) or {}
    class_names = list(
        dataset_params.get("prompt_classes", dataset_params.get("classes", []))
        if is_zero_shot
        else dataset_params.get("classes", dataset_params.get("pathologies", []))
    )
    if not class_names:
        raise ValueError("The dataset config does not define classes/pathologies.")

    zero_shot_features = None
    if is_zero_shot:
        zero_shot_features = _encode_zero_shot_prompts(
            model,
            class_names,
            device,
        )

    phase2_cfg = params_cfg.get("phase2", {}) or {}
    use_text = bool(phase2_cfg.get("use_text", True))
    report_type = str(phase2_cfg.get("p2_report_type", "clinical"))
    temperature = float(params_cfg.get("temperature", 0.07))

    latency_per_batch = []
    latency_per_sample = []
    throughput = []
    gflops_per_sample = []
    input_profiles = []
    cuda_memory_profiles = []
    warnings = []

    batches = itertools.islice(test_loader, ARGS.num_batches)
    for batch_index, cpu_batch in enumerate(batches, start=1):
        if not isinstance(cpu_batch, dict):
            raise TypeError("Efficiency benchmark expects dictionary batches.")
        batch = _move_batch(cpu_batch, device)
        forward_fn, attention_mask = _build_forward(
            model=model,
            batch=batch,
            is_classifier=is_classifier,
            use_text=use_text,
            report_type=report_type,
            zero_shot_features=zero_shot_features,
            temperature=temperature,
        )
        forward_fn = _with_precision(forward_fn, device, ARGS.precision)
        profile = batch_metadata(batch, attention_mask)
        input_profiles.append(profile)
        batch_size = int(profile["batch_size"])

        if not ARGS.skip_flops:
            try:
                supported_flops = count_supported_flops(forward_fn, model)
                gflops_per_sample.append(supported_flops / batch_size / 1.0e9)
            except Exception as exc:
                warning = (
                    f"FLOP counting failed for batch {batch_index}: "
                    f"{type(exc).__name__}: {exc}"
                )
                print(f"[Warning] {warning}")
                warnings.append(warning)

        timings = measure_latency_ms(
            forward_fn,
            device=device,
            warmup=ARGS.warmup,
            repeats=ARGS.repeats,
        )
        latency_per_batch.extend(timings)
        latency_per_sample.extend(value / batch_size for value in timings)
        throughput.extend(batch_size * 1000.0 / value for value in timings)

        memory = measure_cuda_memory_mb(forward_fn, device)
        if memory is not None:
            cuda_memory_profiles.append(memory)
        print(
            f"  batch {batch_index}/{ARGS.num_batches}: "
            f"latency={sum(timings) / len(timings):.3f} ms, "
            f"tiles={profile.get('valid_tiles_per_sample', 'n/a')}"
        )

    if not input_profiles:
        raise RuntimeError("The test dataloader did not yield any batches.")

    result: dict[str, Any] = {
        "experiment_name": str(cfg.get("experiment_name", "default")),
        "seed": seed,
        "model": {
            "backbone_type": str(cfg.model.get("backbone_type", "unknown")),
            "fusion_type": str(fusion_type),
            "classifier_type": str(classifier_type),
            "zero_shot": bool(is_zero_shot),
        },
        "parameters": parameter_summary(model),
        "input_profile": _aggregate_input_profile(input_profiles),
        "latency_ms_per_batch": summarize_measurements(latency_per_batch),
        "latency_ms_per_sample": summarize_measurements(latency_per_sample),
        "throughput_samples_per_second": summarize_measurements(throughput),
        "supported_gflops_per_sample": (
            summarize_measurements(gflops_per_sample)
            if gflops_per_sample
            else None
        ),
        "cuda_memory": (
            {
                key: max(profile[key] for profile in cuda_memory_profiles)
                for key in cuda_memory_profiles[0]
            }
            if cuda_memory_profiles
            else None
        ),
        "protocol": {
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "CPU"
            ),
            "torch_version": torch.__version__,
            "precision": ARGS.precision,
            "batch_size_requested": ARGS.batch_size,
            "num_batches_requested": ARGS.num_batches,
            "warmup_per_batch": ARGS.warmup,
            "repeats_per_batch": ARGS.repeats,
            "latency_scope": "model forward only; excludes dataloader and H2D transfer",
            "zero_shot_prompt_precomputation_included": False,
            "flop_definition": (
                "PyTorch FlopCounterMode supported operations; multiply-add=2 FLOPs; "
                "unsupported pointwise/custom operations are excluded"
            ),
        },
        "warnings": warnings,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    output_path = _output_path(cfg)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, default=str)
    report_row = _report_row(result)
    csv_path = output_path.with_name("efficiency.csv")
    latex_path = output_path.with_name("efficiency_table.tex")
    if ARGS.export_csv:
        _write_csv_report(csv_path, report_row)
    if ARGS.export_latex:
        _write_latex_report(latex_path, report_row)

    params = result["parameters"]
    latency = result["latency_ms_per_sample"]
    print("\nEfficiency benchmark complete")
    print(f"  Parameters: {params['total']:,} total, {params['trainable']:,} trainable")
    if result["supported_gflops_per_sample"] is not None:
        print(
            "  Supported GFLOPs/sample: "
            f"{result['supported_gflops_per_sample']['mean']:.3f}"
        )
    print(f"  Latency/sample: {latency['mean']:.3f} ms (p95={latency['p95']:.3f} ms)")
    throughput_summary = result["throughput_samples_per_second"]
    print(
        "  Throughput: "
        f"{throughput_summary['mean']:.3f} samples/s "
        f"(p50={throughput_summary['p50']:.3f})"
    )
    cuda_memory = result["cuda_memory"]
    if cuda_memory is not None:
        print(
            "  Peak GPU memory: "
            f"{cuda_memory['peak_allocated_mb']:.1f} MiB allocated "
            f"(forward delta={cuda_memory['forward_peak_delta_mb']:.1f} MiB)"
        )
    print(f"  Saved to: {output_path}")
    if ARGS.export_csv:
        print(f"  CSV: {csv_path}")
    if ARGS.export_latex:
        print(f"  LaTeX: {latex_path}")


if __name__ == "__main__":
    main()
