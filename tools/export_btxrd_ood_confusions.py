"""Cung cấp công cụ nghiên cứu export btxrd ood confusions cho XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.btxrd import BTXRD_CLASS_NAMES
from src.utils.analysis import (
    SOURCE_EXPERIMENT,
    SOURCE_SEEDS,
    analysis_root,
    load_feature_archive,
    sha256_file,
)

SCENARIO = "domain_ood_btxrd"
ARCHIVE = "btxrd_test"


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Ghi csv cho bước xử lý hiện tại.

    Parameters
    ----------
    path : Path
        Đường dẫn tài nguyên được sử dụng.
    rows : list[dict[str, Any]]
        Giá trị ``rows`` được sử dụng trong phép xử lý.
    fields : list[str]
        Giá trị ``fields`` được sử dụng trong phép xử lý.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _ctch_class_names(provenance: dict[str, Any]) -> list[str]:
    """Thực hiện bước ctch class names trong quy trình hiện tại.

    Parameters
    ----------
    provenance : dict[str, Any]
        Giá trị ``provenance`` được sử dụng trong phép xử lý.

    Returns
    -------
    list[str]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    names = (
        provenance.get("resolved_config", {})
        .get("dataset", {})
        .get("params", {})
        .get("classes", [])
    )
    names = [str(name) for name in names]
    if not names:
        raise ValueError("BTXRD feature provenance has no CTCH class names.")
    return names


def _load_seed(seed: int):
    """Tải seed cho bước xử lý hiện tại.

    Parameters
    ----------
    seed : int
        Hạt giống phục vụ khả năng tái lập.

    Returns
    -------
    object
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    FileNotFoundError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    feature_path = analysis_root(seed) / "features" / f"{ARCHIVE}.npz"
    result_root = analysis_root(seed) / "ood" / SCENARIO
    metrics_path = result_root / "ood_metrics.json"
    scores_path = result_root / "ood_scores.npz"
    arrays, provenance = load_feature_archive(feature_path)
    if provenance.get("scenario") != ARCHIVE:
        raise ValueError(f"Unexpected feature scenario for seed {seed}.")
    if int(provenance.get("seed", -1)) != seed:
        raise ValueError(f"Unexpected feature seed for {feature_path}.")
    if not metrics_path.is_file() or not scores_path.is_file():
        raise FileNotFoundError(
            f"Missing BTXRD OOD metrics/scores for seed {seed}: {result_root}"
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        metrics.get("source_experiment") != SOURCE_EXPERIMENT
        or metrics.get("scenario") != SCENARIO
        or int(metrics.get("seed", -1)) != seed
    ):
        raise ValueError(f"Unexpected OOD result provenance in {metrics_path}.")
    if metrics.get("feature_archive_sha256", {}).get("ood_test") != sha256_file(
        feature_path
    ):
        raise ValueError(f"BTXRD feature archive changed after {metrics_path} was made.")
    with np.load(scores_path, allow_pickle=False) as stored:
        scores = {key: stored[key] for key in stored.files}
    return arrays, provenance, metrics, scores, result_root


def _case_and_pair_rows(seed: int):
    """Thực hiện bước case and pair rows trong quy trình hiện tại.

    Parameters
    ----------
    seed : int
        Hạt giống phục vụ khả năng tái lập.

    Returns
    -------
    object
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    arrays, provenance, metrics, scores, result_root = _load_seed(seed)
    image_ids = arrays["image_id"].astype(str)
    btxrd_ids = np.asarray(arrays["labels"]).reshape(-1).astype(np.int64)
    logits = np.asarray(arrays["logits"], dtype=np.float64)
    if "probabilities" in arrays:
        probabilities = np.asarray(arrays["probabilities"], dtype=np.float64)
    else:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    predictions = logits.argmax(axis=1).astype(np.int64)
    confidence = probabilities.max(axis=1)
    ctch_names = _ctch_class_names(provenance)
    primary_methods = [str(value) for value in metrics["protocol"]["primary_methods"]]
    feature_key = str(metrics.get("feature_key", ""))
    if not feature_key:
        raise ValueError("BTXRD OOD metrics do not identify their feature_key.")
    if not primary_methods:
        raise ValueError("BTXRD OOD protocol has no primary methods.")
    if len(image_ids) != len(btxrd_ids) or len(image_ids) != len(predictions):
        raise ValueError("BTXRD feature arrays have inconsistent row counts.")
    if np.any(btxrd_ids < 0) or np.any(btxrd_ids >= len(BTXRD_CLASS_NAMES)):
        raise ValueError("BTXRD archive contains an invalid class ID.")
    if np.any(predictions < 0) or np.any(predictions >= len(ctch_names)):
        raise ValueError("CTCH predictions contain an invalid class ID.")

    accepted: dict[str, np.ndarray] = {}
    method_scores: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    for method in primary_methods:
        score_key = f"{method}_ood"
        threshold_key = f"{method}_threshold"
        if score_key not in scores or threshold_key not in scores:
            raise ValueError(f"OOD score archive is missing {method!r} outputs.")
        method_scores[method] = np.asarray(scores[score_key], dtype=np.float64)
        thresholds[method] = float(np.asarray(scores[threshold_key]).item())
        accepted[method] = method_scores[method] <= thresholds[method]

    any_accepted = np.logical_or.reduce(list(accepted.values()))
    all_accepted = np.logical_and.reduce(list(accepted.values()))
    case_rows: list[dict[str, Any]] = []
    for index in np.flatnonzero(any_accepted):
        row: dict[str, Any] = {
            "seed": seed,
            "ood_feature_key": feature_key,
            "image_id": image_ids[index],
            "btxrd_class_id": int(btxrd_ids[index]),
            "btxrd_label": BTXRD_CLASS_NAMES[int(btxrd_ids[index])],
            "ctch_predicted_class_id": int(predictions[index]),
            "ctch_predicted_label": ctch_names[int(predictions[index])],
            "ctch_prediction_confidence": float(confidence[index]),
            "accepted_by_any_primary": True,
            "accepted_by_all_primary": bool(all_accepted[index]),
        }
        for method in primary_methods:
            row[f"{method}_ood_score"] = float(method_scores[method][index])
            row[f"{method}_threshold"] = thresholds[method]
            row[f"{method}_accepted_as_ctch_id"] = bool(accepted[method][index])
        case_rows.append(row)

    pair_stats: dict[tuple[int, int], dict[str, Any]] = {}
    for btxrd_id, prediction in zip(btxrd_ids, predictions):
        key = (int(btxrd_id), int(prediction))
        pair_stats.setdefault(
            key,
            {
                "total_cases": 0,
                "accepted_any": 0,
                "accepted_all": 0,
                **{f"accepted_{method}": 0 for method in primary_methods},
            },
        )["total_cases"] += 1
    for index, (btxrd_id, prediction) in enumerate(zip(btxrd_ids, predictions)):
        stats = pair_stats[(int(btxrd_id), int(prediction))]
        stats["accepted_any"] += int(any_accepted[index])
        stats["accepted_all"] += int(all_accepted[index])
        for method in primary_methods:
            stats[f"accepted_{method}"] += int(accepted[method][index])

    pair_rows: list[dict[str, Any]] = []
    for (btxrd_id, prediction), stats in sorted(pair_stats.items()):
        total = int(stats["total_cases"])
        row = {
            "seed": seed,
            "ood_feature_key": feature_key,
            "btxrd_class_id": btxrd_id,
            "btxrd_label": BTXRD_CLASS_NAMES[btxrd_id],
            "ctch_predicted_class_id": prediction,
            "ctch_predicted_label": ctch_names[prediction],
            "total_cases": total,
            "accepted_by_any_primary": int(stats["accepted_any"]),
            "accepted_by_any_primary_rate": stats["accepted_any"] / total,
            "accepted_by_all_primary": int(stats["accepted_all"]),
            "accepted_by_all_primary_rate": stats["accepted_all"] / total,
        }
        for method in primary_methods:
            count = int(stats[f"accepted_{method}"])
            row[f"{method}_accepted_as_ctch_id"] = count
            row[f"{method}_accepted_as_ctch_id_rate"] = count / total
        pair_rows.append(row)
    return case_rows, pair_rows, primary_methods, feature_key, result_root


def export(seeds: list[int]) -> None:
    """Xuất kết quả cho bước xử lý hiện tại.

    Parameters
    ----------
    seeds : list[int]
        Giá trị ``seeds`` được sử dụng trong phép xử lý.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    all_cases: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []
    primary_methods: list[str] | None = None
    feature_key: str | None = None
    for seed in seeds:
        cases, pairs, methods, seed_feature_key, result_root = _case_and_pair_rows(seed)
        if primary_methods is None:
            primary_methods = methods
        elif methods != primary_methods:
            raise ValueError("Primary BTXRD OOD methods differ across seeds.")
        if feature_key is None:
            feature_key = seed_feature_key
        elif seed_feature_key != feature_key:
            raise ValueError("BTXRD OOD feature keys differ across seeds.")
        feature_tag = seed_feature_key.removesuffix("_embeddings")
        case_fields = [
            "seed", "ood_feature_key", "image_id", "btxrd_class_id", "btxrd_label",
            "ctch_predicted_class_id", "ctch_predicted_label",
            "ctch_prediction_confidence", "accepted_by_any_primary",
            "accepted_by_all_primary",
        ]
        for method in methods:
            case_fields.extend(
                [
                    f"{method}_ood_score",
                    f"{method}_threshold",
                    f"{method}_accepted_as_ctch_id",
                ]
            )
        _write_csv(
            result_root / f"btxrd_confused_as_ctch_cases_{feature_tag}.csv",
            cases,
            case_fields,
        )
        _write_csv(
            result_root / f"btxrd_confusion_by_label_{feature_tag}.csv",
            pairs,
            list(pairs[0]),
        )
        all_cases.extend(cases)
        all_pairs.extend(pairs)

    if primary_methods is None or feature_key is None:
        raise ValueError("No seeds were provided.")
    if set(seeds) != set(SOURCE_SEEDS):
        print(
            "Per-seed BTXRD audit exported; aggregate CSV requires all locked "
            f"seeds {list(SOURCE_SEEDS)}."
        )
        return
    feature_tag = feature_key.removesuffix("_embeddings")
    root = analysis_root()
    _write_csv(
        root / f"btxrd_confused_as_ctch_cases_{feature_tag}.csv",
        all_cases,
        case_fields,
    )
    _write_csv(
        root / f"btxrd_confusion_by_label_{feature_tag}.csv",
        all_pairs,
        list(all_pairs[0]),
    )

    label_stats: dict[tuple[int, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for row in all_pairs:
        key = (int(row["btxrd_class_id"]), str(row["btxrd_label"]))
        stats = label_stats[key]
        stats["total_cases"] += int(row["total_cases"])
        stats["accepted_any"] += int(row["accepted_by_any_primary"])
        stats["accepted_all"] += int(row["accepted_by_all_primary"])
        for method in primary_methods:
            stats[f"accepted_{method}"] += int(
                row[f"{method}_accepted_as_ctch_id"]
            )
    label_rows = []
    for (class_id, label), stats in sorted(label_stats.items()):
        total = int(stats["total_cases"])
        row = {
            "btxrd_class_id": class_id,
            "btxrd_label": label,
            "evaluations_across_seeds": total,
            "accepted_by_any_primary": int(stats["accepted_any"]),
            "accepted_by_any_primary_rate": stats["accepted_any"] / total,
            "accepted_by_all_primary": int(stats["accepted_all"]),
            "accepted_by_all_primary_rate": stats["accepted_all"] / total,
        }
        for method in primary_methods:
            count = int(stats[f"accepted_{method}"])
            row[f"{method}_accepted_as_ctch_id"] = count
            row[f"{method}_accepted_as_ctch_id_rate"] = count / total
        label_rows.append(row)
    _write_csv(
        root / f"btxrd_ood_label_summary_{feature_tag}.csv",
        label_rows,
        list(label_rows[0]),
    )
    print(
        "Detailed confused cases: "
        f"{root / f'btxrd_confused_as_ctch_cases_{feature_tag}.csv'}"
    )
    print(
        "Label-pair summary: "
        f"{root / f'btxrd_confusion_by_label_{feature_tag}.csv'}"
    )
    print(
        "BTXRD label summary: "
        f"{root / f'btxrd_ood_label_summary_{feature_tag}.csv'}"
    )


def main() -> None:
    """Thực thi điểm vào chính của mô-đun.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(SOURCE_SEEDS)
    )
    args = parser.parse_args()
    invalid = sorted(set(args.seeds) - set(SOURCE_SEEDS))
    if invalid:
        raise ValueError(f"Unsupported CTCH proposed seeds: {invalid}")
    export([int(seed) for seed in args.seeds])


if __name__ == "__main__":
    main()
