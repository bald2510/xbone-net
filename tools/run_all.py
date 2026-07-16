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
    # --- BTXRD Zero-Shot Baselines --- Done
    # "btxrd_zeroshot": [
    #     "btxrd/baselines/zeroshot/biomedclip_zeroshot",# done
    #     "btxrd/baselines/zeroshot/clip_zeroshot", # done
    #     "btxrd/baselines/zeroshot/pubmedclip_zeroshot", # done
    #     "btxrd/baselines/zeroshot/medclip_zeroshot", # done
    # ],
    # #--- BTXRD Fine-Tuned Baselines --- Done
    # "btxrd_finetuned": [
    #     "btxrd/baselines/full_finetuned/fft_resnet50", # done
    #     "btxrd/baselines/full_finetuned/fft_densenet", # done
    #     "btxrd/baselines/full_finetuned/fft_clip", # done
    #     "btxrd/baselines/full_finetuned/fft_pubmedclip", # done
    #     "btxrd/baselines/full_finetuned/fft_medclip", # done
    #     "btxrd/baselines/full_finetuned/fft_biomedclip", # done
    # ],
    # # --- BTXRD PEFT Fine-Tuned Baselines ---
    # "btxrd_peft_finetuned": [
    #     "btxrd/baselines/peft_finetuned/lora_pubmedclip", # done
    #     "btxrd/baselines/peft_finetuned/lora_biomedclip", # done
    # ],
    # # --- BTXRD Proposed Model ---
    # "btxrd_proposed": [
    #     "btxrd/proposed/ours_xbone_net",
    # ],
    # # --- BTXRD Few-Shot Learning ---
    # "btxrd_few_shot_1": [
    #     "btxrd/few_shot/1_shot/lora_pubmedclip", # done
    #     "btxrd/few_shot/1_shot/lora_biomedclip", # done
    #     "btxrd/few_shot/1_shot/ours_xbone_net", # done
    # ],
    # "btxrd_few_shot_10": [
    #     "btxrd/few_shot/10_shot/lora_pubmedclip", # done
    #     "btxrd/few_shot/10_shot/lora_biomedclip", # done
    #     "btxrd/few_shot/10_shot/ours_xbone_net", # done
    # ],
    # "btxrd_few_shot_20": [
    #     "btxrd/few_shot/20_shot/lora_pubmedclip", # done
    #     "btxrd/few_shot/20_shot/lora_biomedclip", # done
    #     "btxrd/few_shot/20_shot/ours_xbone_net", # done
    # ],
    # --- CTCH Zero-Shot Baselines ---
    # "ctch_zeroshot": [
    #     "ctch/baselines/zeroshot/biomedclip_zeroshot", # done
    #     "ctch/baselines/zeroshot/clip_zeroshot", # done
    #     "ctch/baselines/zeroshot/pubmedclip_zeroshot", # done
    #     "ctch/baselines/zeroshot/medclip_zeroshot", # done
    # ],
    # --- CTCH Fine-Tuned Baselines ---
    "ctch_finetuned": [
        # "ctch/baselines/full_finetuned/fft_biomedclip", # done
        # "ctch/baselines/full_finetuned/fft_clip", # done
        # "ctch/baselines/full_finetuned/fft_pubmedclip", # done
        # "ctch/baselines/full_finetuned/fft_medclip", # done
        # "ctch/baselines/full_finetuned/fft_resnet50", # done
        # "ctch/baselines/full_finetuned/fft_densenet", # done
    ],
    "ctch_peft_finetuned": [
        # "ctch/baselines/peft_finetuned/lora_pubmedclip", # done
        # "ctch/baselines/peft_finetuned/lora_biomedclip", # done
    ],
    "ctch_few_shot_1": [
        # "ctch/few_shot/1_shot/lora_pubmedclip", # done
        # "ctch/few_shot/1_shot/lora_biomedclip", # done
        # "ctch/few_shot/1_shot/ours_xbone_net", # done
    ],
    "ctch_few_shot_10": [
        # "ctch/few_shot/10_shot/lora_pubmedclip", # done
        # "ctch/few_shot/10_shot/lora_biomedclip", # done
        # "ctch/few_shot/10_shot/ours_xbone_net", # done
    ],
    "ctch_few_shot_20": [
        # "ctch/few_shot/20_shot/lora_pubmedclip", # done
        # "ctch/few_shot/20_shot/lora_biomedclip", # done
        # "ctch/few_shot/20_shot/ours_xbone_net",
    ],
    # --- CTCH Proposed Model ---
    "ctch_proposed": [
        "ctch/proposed/ours_xbone_net",
    ],
    # --- CTCH Ablation Studies (uncomment experiments to schedule them) ---
    "ctch_ablation": [
        # "ctch/ablation_study/modality/image_only",
        # "ctch/ablation_study/modality/text_only",
        # "ctch/ablation_study/modality/shuffled_report",
        # "ctch/ablation_study/modality/phase1_xray_phase2_clinical",
        # "ctch/ablation_study/finetune/xbone_highres_full_ft",
        # "ctch/ablation_study/finetune/xbone_highres_no_ft",
        # "ctch/ablation_study/architecture/preprocess/xbone_nohighres",
        # "ctch/ablation_study/architecture/preprocess/xbone_letterbox",
        # "ctch/ablation_study/architecture/phase/phase2_only",
        # "ctch/ablation_study/architecture/phase/phase1_merged",
        # "ctch/ablation_study/architecture/fusion/concat",
        # "ctch/ablation_study/architecture/fusion/image_to_text",
        # "ctch/ablation_study/architecture/fusion/text_to_image",
        # "ctch/ablation_study/architecture/classifier/no_class_weight",
        # "ctch/ablation_study/architecture/classifier/linear",
    ],
})

DEFAULT_SEEDS = [456]

METRIC_KEYS = [
    "f1_macro", "accuracy", "sensitivity", "specificity",
    "precision", "auroc_macro",
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
            agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
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
    args = parser.parse_args()

    experiments = get_experiments(args.group)
    seeds = args.seeds

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
    print(f"|  Total runs  : {len(experiments) * len(seeds):<40d} |")
    print(f"|  Mode        : {mode_str:<40s} |")
    print("+----------------------------------------------------------+")

    all_results = OrderedDict()

    for i, experiment in enumerate(experiments, 1):
        exp_short = experiment.split("/")[-1]
        print(f"\n{'-'*60}")
        print(f"  [{i}/{len(experiments)}] {experiment}")
        print(f"{'-'*60}")

        seeds_metrics = {}

        for seed in seeds:
            if args.table:
                metrics = load_metrics(experiment, seed)
                if metrics:
                    seeds_metrics[seed] = metrics
                continue

            is_zeroshot = "zeroshot" in experiment
            sd = seed_dir(experiment, seed)
            ckpt_p2 = os.path.join(sd, "best_phase2.pth")
            ckpt_p1 = os.path.join(sd, "best_phase1.pth")

            # --- STEP 1: Train model ---
            if not args.eval_only and not is_zeroshot:
                p2_extra = ["++params.run_phase1=false", "++params.run_phase2=true"] if args.phase2_only else None
                print(f"\n  [STEP 1/2: TRAINING] Launching train.py for {experiment} (seed={seed})...")
                sys.stdout.flush()
                train_success = run_one("train.py", experiment, seed, extra_args=p2_extra)

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
            eval_success = run_one("evaluate.py", experiment, seed)
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

