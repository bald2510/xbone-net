"""Utilities for reproducible model-efficiency measurements.

The FLOP counter reports operations supported by PyTorch's
``FlopCounterMode``.  This is deliberately described as an estimate because
pointwise operations, normalisation and some third-party/custom kernels may not
be registered by PyTorch's counter.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn


def parameter_summary(model: nn.Module) -> dict[str, Any]:
    """Return total/trainable parameter counts and top-level breakdown."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    modules = {}
    for name, module in model.named_children():
        module_total = sum(parameter.numel() for parameter in module.parameters())
        module_trainable = sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        )
        modules[name] = {
            "total": int(module_total),
            "trainable": int(module_trainable),
        }

    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(total - trainable),
        "trainable_percent": 100.0 * trainable / max(total, 1),
        "by_module": modules,
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot compute a percentile of an empty sequence.")
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_measurements(values: Sequence[float]) -> dict[str, float | int]:
    """Summarise repeated scalar measurements without requiring NumPy."""
    numeric = [float(value) for value in values]
    if not numeric:
        raise ValueError("At least one measurement is required.")
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "std": statistics.pstdev(numeric) if len(numeric) > 1 else 0.0,
        "min": min(numeric),
        "max": max(numeric),
        "p50": _percentile(numeric, 50.0),
        "p95": _percentile(numeric, 95.0),
    }


def count_supported_flops(
    forward_fn: Callable[[], Any],
    model: nn.Module | None = None,
) -> int:
    """Count forward FLOPs supported by the installed PyTorch registry."""
    # Kept in the public signature for callers that want to associate a model
    # with the measurement. Recent PyTorch versions no longer need ``mods`` to
    # compute the global total.
    del model
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError as exc:  # pragma: no cover - pinned Torch provides it.
        raise RuntimeError(
            "This benchmark requires torch.utils.flop_counter.FlopCounterMode."
        ) from exc

    with torch.inference_mode():
        with FlopCounterMode(display=False) as counter:
            forward_fn()
    return int(counter.get_total_flops())


def measure_latency_ms(
    forward_fn: Callable[[], Any],
    device: torch.device,
    warmup: int = 10,
    repeats: int = 50,
) -> list[float]:
    """Measure model-only forward latency, excluding host-to-device transfer."""
    if warmup < 0:
        raise ValueError("warmup must be non-negative.")
    if repeats < 1:
        raise ValueError("repeats must be positive.")

    with torch.inference_mode():
        for _ in range(warmup):
            forward_fn()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
            for start, end in zip(starts, ends):
                start.record()
                forward_fn()
                end.record()
            torch.cuda.synchronize(device)
            return [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]

        elapsed = []
        for _ in range(repeats):
            start = time.perf_counter()
            forward_fn()
            elapsed.append((time.perf_counter() - start) * 1000.0)
        return elapsed


def measure_cuda_memory_mb(
    forward_fn: Callable[[], Any],
    device: torch.device,
) -> dict[str, float] | None:
    """Measure peak allocated/reserved CUDA memory for one forward pass."""
    if device.type != "cuda":
        return None

    torch.cuda.synchronize(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        forward_fn()
    torch.cuda.synchronize(device)

    divisor = 1024.0**2
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    return {
        "baseline_allocated_mb": baseline_allocated / divisor,
        "peak_allocated_mb": peak_allocated / divisor,
        "forward_peak_delta_mb": max(0, peak_allocated - baseline_allocated) / divisor,
        "baseline_reserved_mb": baseline_reserved / divisor,
        "peak_reserved_mb": peak_reserved / divisor,
    }


def batch_metadata(
    batch: dict[str, Any],
    attention_mask: torch.Tensor | None,
) -> dict[str, Any]:
    """Record the dynamic image/tile/text dimensions that determine compute."""
    images = batch["pixel_values"]
    metadata: dict[str, Any] = {
        "batch_size": int(images.shape[0]),
        "global_image_shape": [int(value) for value in images.shape[1:]],
    }

    tiles = batch.get("tile_values")
    tile_mask = batch.get("tile_mask")
    if tiles is not None:
        metadata["tile_shape"] = [int(value) for value in tiles.shape[2:]]
        metadata["padded_tiles_per_sample"] = int(tiles.shape[1])
        if tile_mask is None:
            valid_tiles = [int(tiles.shape[1])] * int(tiles.shape[0])
        else:
            valid_tiles = [int(value) for value in tile_mask.sum(dim=1).tolist()]
        metadata["valid_tiles_per_sample"] = valid_tiles

    if attention_mask is not None:
        metadata["padded_text_length"] = int(attention_mask.shape[1])
        metadata["valid_text_tokens_per_sample"] = [
            int(value) for value in attention_mask.sum(dim=1).tolist()
        ]
    return metadata
