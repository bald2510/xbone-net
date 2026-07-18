"""
XBone-Net Full Experiment Suite Orchestrator.
===============================================================================
Orchestrates multi-seed training and evaluation across the full XBone-Net experiment suite:
  - Experiment Groups: Configures zero-shot baselines, fine-tuned models, and proposed XBone-Net.
  - Subprocess Management: Spawns sequential train.py and evaluate.py jobs per seed.
  - Metric Aggregation: Parses per-seed metric JSONs and computes mean ± std across runs.
  - Master Export: Saves per-experiment and master summary JSON files and displays console tables.
"""

import argparse
import json
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np


# ============================================================
# Experiment Registry & Path Resolvers
# ============================================================

EXPERIMENTS = OrderedDict({
    "btxrd_zeroshot": [
        "btxrd/baselines/zeroshot/biomedclip_zeroshot",
        "btxrd/baselines/zeroshot/clip_zeroshot",
        "btxrd/baselines/zeroshot/pubmedclip_zeroshot",
        "btxrd/baselines/zeroshot/medclip_zeroshot",
    ],
    "btxrd_finetuned": [
        "btxrd/baselines/full_finetuned/fft_resnet50",
        "btxrd/baselines/full_finetuned/fft_densenet",
        "btxrd/baselines/full_finetuned/fft_clip",
        "btxrd/baselines/full_finetuned/fft_pubmedclip",
        "btxrd/baselines/full_finetuned/fft_medclip",
        "btxrd/baselines/full_finetuned/fft_biomedclip",
    ],
    "btxrd_peft_finetuned": [
        "btxrd/baselines/peft_finetuned/lora_pubmedclip",
        "btxrd/baselines/peft_finetuned/lora_biomedclip",
    ],
    "btxrd_few_shot_1": [
        "btxrd/few_shot/1_shot/lora_pubmedclip",
        "btxrd/few_shot/1_shot/lora_biomedclip",
        "btxrd/few_shot/1_shot/ours_xbone_net",
    ],
    "btxrd_few_shot_10": [
        "btxrd/few_shot/10_shot/lora_pubmedclip",
        "btxrd/few_shot/10_shot/lora_biomedclip",
        "btxrd/few_shot/10_shot/ours_xbone_net",
    ],
    "btxrd_few_shot_20": [
        "btxrd/few_shot/20_shot/lora_pubmedclip",
        "btxrd/few_shot/20_shot/lora_biomedclip",
        "btxrd/few_shot/20_shot/ours_xbone_net",
    ],
    "btxrd_proposed": [
        "btxrd/proposed/ours_xbone_net",
    ],
    "ctch_zeroshot": [
        "ctch/baselines/zeroshot/biomedclip_zeroshot",
        "ctch/baselines/zeroshot/clip_zeroshot",
        "ctch/baselines/zeroshot/pubmedclip_zeroshot",
        "ctch/baselines/zeroshot/medclip_zeroshot",
    ],
    "ctch_finetuned": [
        "ctch/baselines/full_finetuned/fft_biomedclip",
        "ctch/baselines/full_finetuned/fft_clip",
        "ctch/baselines/full_finetuned/fft_pubmedclip",
        "ctch/baselines/full_finetuned/fft_medclip",
        "ctch/baselines/full_finetuned/fft_resnet50",
        "ctch/baselines/full_finetuned/fft_densenet",
    ],
    "ctch_peft_finetuned": [
        "ctch/baselines/peft_finetuned/lora_pubmedclip",
        "ctch/baselines/peft_finetuned/lora_biomedclip",
    ],
    "ctch_few_shot_1": [
        "ctch/few_shot/1_shot/lora_pubmedclip",
        "ctch/few_shot/1_shot/lora_biomedclip",
        "ctch/few_shot/1_shot/ours_xbone_net",
    ],
    "ctch_few_shot_10": [
        "ctch/few_shot/10_shot/lora_pubmedclip",
        "ctch/few_shot/10_shot/lora_biomedclip",
        "ctch/few_shot/10_shot/ours_xbone_net",
    ],
    "ctch_few_shot_20": [
        "ctch/few_shot/20_shot/lora_pubmedclip",
        "ctch/few_shot/20_shot/lora_biomedclip",
        "ctch/few_shot/20_shot/ours_xbone_net",
    ],
    "ctch_proposed": [
        "ctch/proposed/ours_xbone_net",
    ],
    "ctch_ablation": [
        "ctch/ablation_study/modality/image_only",
        "ctch/ablation_study/modality/text_only",
        "ctch/ablation_study/modality/shuffled_report",
        "ctch/ablation_study/modality/phase1_xray_phase2_clinical",
        "ctch/ablation_study/finetune/xbone_highres_full_ft",
        "ctch/ablation_study/finetune/xbone_highres_no_ft",
        "ctch/ablation_study/architecture/preprocess/xbone_nohighres",
        "ctch/ablation_study/architecture/preprocess/xbone_letterbox",
        "ctch/ablation_study/architecture/phase/phase2_only",
        "ctch/ablation_study/architecture/phase/phase1_merged",
        "ctch/ablation_study/architecture/fusion/concat",
        "ctch/ablation_study/architecture/fusion/image_to_text",
        "ctch/ablation_study/architecture/fusion/text_to_image",
        "ctch/ablation_study/architecture/classifier/no_class_weight",
        "ctch/ablation_study/architecture/classifier/linear",
    ],
})

# Run this subset on one fixed, locally controlled GPU when wall-clock training
# time is a reported outcome. Performance-only experiments may run elsewhere.
CTCH_TRAINING_TIME_EXPERIMENTS = [
    "ctch/proposed/ours_xbone_net",
    "ctch/baselines/peft_finetuned/lora_biomedclip",
    "ctch/baselines/full_finetuned/fft_biomedclip",
    "ctch/ablation_study/architecture/preprocess/xbone_nohighres",
    "ctch/ablation_study/architecture/preprocess/xbone_letterbox",
    "ctch/ablation_study/finetune/xbone_highres_no_ft",
    "ctch/ablation_study/finetune/xbone_highres_full_ft",
    "ctch/ablation_study/architecture/phase/phase2_only",
]

BTXRD_TRAINING_TIME_EXPERIMENTS = [
    "btxrd/proposed/ours_xbone_net",
    "btxrd/baselines/peft_finetuned/lora_biomedclip",
    "btxrd/baselines/full_finetuned/fft_biomedclip",
]

TRAINING_TIME_EXPERIMENTS = (
    BTXRD_TRAINING_TIME_EXPERIMENTS + CTCH_TRAINING_TIME_EXPERIMENTS
)

_ALL_REGISTERED_EXPERIMENTS = list(dict.fromkeys(
    experiment
    for experiments in EXPERIMENTS.values()
    for experiment in experiments
))
_TRAINING_TIME_SET = set(TRAINING_TIME_EXPERIMENTS)
NON_TIMING_EXPERIMENTS = [
    experiment
    for experiment in _ALL_REGISTERED_EXPERIMENTS
    if experiment not in _TRAINING_TIME_SET
]

DEFAULT_SEEDS = [42, 123, 456]

METRIC_KEYS = [
    "f1_macro", "accuracy", "sensitivity", "specificity",
    "precision", "auroc_macro", "auprc_macro", "balanced_accuracy",
    "ece_15", "adaptive_ece_15", "nll", "brier_score",
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_experiments(groups: list[str] | None) -> list[str]:
    """Return flat list of experiment config paths for requested groups.

    Args:
        groups: List of group names, category keywords, or experiment paths.

    Returns:
        list[str]: Deduplicated list of experiment config path strings.
    """
    if not groups:
        seen = set()
        return [exp for group in EXPERIMENTS.values() for exp in group if not (exp in seen or seen.add(exp))]

    result = []
    for g in groups:
        if g in EXPERIMENTS:
            result.extend(EXPERIMENTS[g])
        elif g == "training_time":
            result.extend(TRAINING_TIME_EXPERIMENTS)
        elif g in ("ctch_training_time", "training_time_ctch"):
            result.extend(CTCH_TRAINING_TIME_EXPERIMENTS)
        elif g in ("btxrd_training_time", "training_time_btxrd"):
            result.extend(BTXRD_TRAINING_TIME_EXPERIMENTS)
        elif g in ("non_timing", "performance_only"):
            result.extend(NON_TIMING_EXPERIMENTS)
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
            for key, exps in EXPERIMENTS.items():
                if key.startswith(g):
                    result.extend(exps)
        else:
            result.append(g)

    seen = set()
    return [exp for exp in result if not (exp in seen or seen.add(exp))]


def seed_dir(experiment: str, seed: int) -> str:
    """Generate absolute checkpoint directory path for experiment and seed.

    Args:
        experiment: Experiment configuration name.
        seed: Integer seed value.

    Returns:
        str: Absolute checkpoint path rooted at PROJECT_ROOT.
    """
    return os.path.join(PROJECT_ROOT, "checkpoints", experiment, f"seed_{seed}")


# ============================================================
# Execution & Metric Collection
# ============================================================

def run_one(script: str, experiment: str, seed: int, extra_args: list | None = None) -> bool:
    """Execute train.py or evaluate.py subprocess for given experiment and seed.

    Args:
        script: Python script filename ('train.py' or 'evaluate.py').
        experiment: Experiment configuration identifier.
        seed: Integer seed value.
        extra_args: Optional list of extra CLI argument strings.

    Returns:
        bool: True if process executed with exit code 0; False otherwise.
    """
    sd = seed_dir(experiment, seed)
    output_dir = os.path.join(PROJECT_ROOT, "results", experiment, f"seed_{seed}")
    cmd = [
        sys.executable, script,
        f"+experiment={experiment}",
        f"++seed={seed}",
        f"++params.model_dir={sd}/",
        f"++params.phase1.checkpoint_path={os.path.join(sd, 'best_phase1.pth')}",
        f"++params.phase1.merged_checkpoint_path={os.path.join(sd, 'merged_phase1.pth')}",
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
    """Load and normalize metric keys from result JSON file.

    Args:
        experiment: Experiment identifier.
        seed: Random seed integer.

    Returns:
        dict | None: Loaded metric dictionary if found; None otherwise.
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

                # Legacy alias mapping (kept for backward compatibility)
                for key, macro_key in [("sensitivity", "sensitivity_macro"), 
                                        ("specificity", "specificity_macro"), 
                                        ("precision", "precision_macro")]:
                    if key in metrics_dict and macro_key not in metrics_dict:
                        metrics_dict[macro_key] = metrics_dict[key]
                    if macro_key in metrics_dict and key not in metrics_dict:
                        metrics_dict[key] = metrics_dict[macro_key]

                return metrics_dict
            except Exception as e:
                print(f"    [WARN] Failed to load {full}: {e}")
    return None


def aggregate(all_metrics: dict[int, dict]) -> dict:
    """Compute metric mean and standard deviation across seeds.

    Args:
        all_metrics: Dict mapping seed integer to metrics dict.

    Returns:
        dict: Aggregated dictionary containing mean, std, and sample count per metric.
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
    """Save per-seed and aggregated metrics to JSON.

    Args:
        experiment: Experiment identifier.
        seeds_metrics: Dict mapping seed to metrics dict.
        agg: Aggregated metrics dictionary.

    Returns:
        str: File path to saved aggregated_results.json.
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


def print_results_table(results: dict[str, dict]):
    """Print console summary table of experiment results.

    Args:
        results: Dict mapping experiment names to aggregated metrics dicts.

    Returns:
        None
    """
    print(f"\n{'='*90}")
    print("  RESULTS SUMMARY (mean +/- std)")
    print(f"{'='*90}")

    header = f"  {'Experiment':<35s} {'F1 Macro':>14s} {'Accuracy':>14s} {'Sensitivity':>14s} {'AUROC':>14s}"
    print(header)
    print(f"  {'-'*35} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")

    for exp, agg in results.items():
        name = exp.split("/")[-1]
        cols = []
        for key in ["f1_macro", "accuracy", "sensitivity", "auroc_macro"]:
            if key in agg and agg[key].get("n", 0) > 0:
                cols.append(f"{agg[key]['mean']:.4f}+/-{agg[key]['std']:.4f}")
            else:
                cols.append("N/A")
        print(f"  {name:<35s} {cols[0]:>14s} {cols[1]:>14s} {cols[2]:>14s} {cols[3]:>14s}")

    print(f"{'='*90}\n")


# ============================================================
# Main Entry Point & CLI Parsing
# ============================================================

def main():
    """CLI entry-point: parse CLI flags and run full experiment suite."""
    parser = argparse.ArgumentParser(
        description="Run all experiments (baselines + ablations) with multi-seed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--group", "-g", nargs="+", default=None,
                        help="Experiment groups to run or direct config path.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help=f"Seeds to run (default: {DEFAULT_SEEDS})")
    parser.add_argument("--train-only", action="store_true",
                        help="Only train models and save checkpoints, skip evaluation")
    parser.add_argument("--phase2-only", "-p2", action="store_true",
                        help="Skip Phase 1 contrastive training, only train Phase 2 classifier")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only evaluate + aggregate")
    parser.add_argument("--table", action="store_true",
                        help="Only print results from existing aggregated JSON files")
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
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if (
        args.gradient_accumulation_steps is not None
        and args.gradient_accumulation_steps < 1
    ):
        parser.error("--gradient-accumulation-steps must be positive")

    experiments = get_experiments(args.group)
    seeds = args.seeds
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

            sd = seed_dir(experiment, seed)
            ckpt_p2 = os.path.join(sd, "best_phase2.pth")
            ckpt_p1 = os.path.join(sd, "best_phase1.pth")

            # --- STEP 1: Train model ---
            if not args.eval_only and not is_zeroshot:
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

                # Strictly verify train.py process exited cleanly with code 0
                if not train_success:
                    print(f"    [ERROR] train.py failed for seed={seed}. ABORTING evaluation for this seed.")
                    continue

                # Strictly verify trained checkpoint file actually exists on disk before evaluate.py
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

            # --- STEP 2: Evaluate model (Only after train.py is 100% finished & verified) ---
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

