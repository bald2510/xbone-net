"""
Multi-seed experiment runner for statistical evaluation.
=========================================================
Runs each experiment config across multiple seeds, then aggregates
results (mean ± std) for thesis-quality reporting.

Usage:
    python run_multi_seed.py +experiment=1_baseline/B1_resnet50
    python run_multi_seed.py +experiment=1_baseline/B1_resnet50 --seeds 42 123 456
    python run_multi_seed.py +experiment=1_baseline/B1_resnet50 --eval-only
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SEEDS = [42, 123, 456]

METRICS_KEYS = [
    "f1_macro", "accuracy", "sensitivity_macro", "specificity_macro",
    "precision_macro", "auroc_macro", "hamming_loss",
]


def run_training(experiment: str, seed: int, extra_args: list[str] = None):
    """Run train_2stage.py with a specific seed."""
    cmd = [
        sys.executable, "train_2stage.py",
        f"+experiment={experiment}",
        f"++seed={seed}",
        # Override model_dir to include seed
        f"++params.model_dir=checkpoints/{experiment}/seed_{seed}/",
    ]
    
    # Update checkpoint paths to include seed
    cmd.extend([
        f"++params.phase1.checkpoint_path=checkpoints/{experiment}/seed_{seed}/best_phase1.pth",
        f"++params.phase2.checkpoint_path=checkpoints/{experiment}/seed_{seed}/best_phase2.pth",
    ])
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"\n{'='*60}")
    print(f"  TRAINING: {experiment} | seed={seed}")
    print(f"{'='*60}")
    print(f"  Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"  [ERROR] Training failed for seed={seed}")
        return False
    return True


def load_and_normalize_metrics(metrics_path: str) -> dict | None:
    """Load metrics from JSON, extract the 'metrics' sub-dictionary if nested,
    and map 'sensitivity', 'specificity', 'precision' to their macro equivalents."""
    if not os.path.exists(metrics_path):
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [ERROR] Failed to load {metrics_path}: {e}")
        return None

    # Handle nesting (eval script puts metrics under "metrics" key)
    if isinstance(data, dict):
        metrics_dict = data.get("metrics", data)
    else:
        metrics_dict = {}

    # Map keys to macro variants expected by orchestrator
    for key, macro_key in [("sensitivity", "sensitivity_macro"), 
                            ("specificity", "specificity_macro"), 
                            ("precision", "precision_macro")]:
        if key in metrics_dict and macro_key not in metrics_dict:
            metrics_dict[macro_key] = metrics_dict[key]

    return metrics_dict


def run_evaluation(experiment: str, seed: int, extra_args: list[str] = None):
    """Run evaluate_model.py with a specific seed and save metrics."""
    output_dir = f"results/{experiment}/seed_{seed}"
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        sys.executable, "evaluate_model.py",
        f"+experiment={experiment}",
        f"++seed={seed}",
        f"++params.model_dir=checkpoints/{experiment}/seed_{seed}/",
        f"++params.phase2.checkpoint_path=checkpoints/{experiment}/seed_{seed}/best_phase2.pth",
        "--output-dir", output_dir,
    ]
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"\n{'='*60}")
    print(f"  EVALUATING: {experiment} | seed={seed}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"  [ERROR] Evaluation failed for seed={seed}")
        return None
    
    # Load metrics from output_dir
    metrics_path = os.path.join(output_dir, "metrics.json")
    metrics = load_and_normalize_metrics(metrics_path)
    if metrics:
        return metrics
    
    print(f"  [WARNING] No metrics.json found for seed={seed}")
    return None


def aggregate_results(all_metrics: dict[int, dict], experiment: str):
    """Compute mean ± std across seeds and print formatted table."""
    if not all_metrics:
        print("[ERROR] No metrics to aggregate.")
        return
    
    print(f"\n{'='*60}")
    print(f"  AGGREGATED RESULTS: {experiment}")
    print(f"  Seeds: {list(all_metrics.keys())}")
    print(f"{'='*60}\n")
    
    aggregated = {}
    for key in METRICS_KEYS:
        values = []
        for seed, metrics in all_metrics.items():
            if metrics and key in metrics:
                val = metrics[key]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    values.append(val)
        
        if values:
            mean = np.mean(values)
            std = np.std(values)
            aggregated[key] = {"mean": mean, "std": std, "values": values}
            print(f"  {key:25s}: {mean:.4f} ± {std:.4f}  (n={len(values)})")
        else:
            print(f"  {key:25s}: N/A")
    
    # Save aggregated results
    output_dir = f"results/{experiment}"
    os.makedirs(output_dir, exist_ok=True)
    
    summary = {
        "experiment": experiment,
        "seeds": list(all_metrics.keys()),
        "per_seed": {str(s): m for s, m in all_metrics.items()},
        "aggregated": {
            k: {"mean": v["mean"], "std": v["std"]}
            for k, v in aggregated.items()
        },
    }
    
    summary_path = os.path.join(output_dir, "aggregated_results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved to: {summary_path}")
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed experiment runner",
        usage="%(prog)s --experiment <config_path> [--seeds N ...] [--eval-only]"
    )
    parser.add_argument("--experiment", "-e", dest="experiment", default=None,
                       help="Experiment config path (e.g., 1_baseline/B1_resnet50)")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                       help=f"List of seeds (default: {SEEDS})")
    parser.add_argument("--eval-only", action="store_true",
                       help="Skip training, only run evaluation")
    parser.add_argument("--aggregate-only", action="store_true",
                       help="Skip training and evaluation, only aggregate existing results")
    
    args, extra = parser.parse_known_args()
    
    # Also support Hydra-style +experiment=... from CLI
    experiment = args.experiment
    if experiment is None:
        for arg in sys.argv[1:]:
            if arg.startswith("+experiment="):
                experiment = arg.split("=", 1)[1]
                break
    
    if experiment is None:
        parser.error("experiment is required: --experiment 1_baseline/B1_resnet50 or +experiment=1_baseline/B1_resnet50")
    
    # Filter out args we handle explicitly to avoid duplicates in subprocesses
    extra = [a for a in extra if not a.startswith("+experiment=") and not a.startswith("seed=")]
    
    seeds = args.seeds
    
    print(f"Multi-seed runner: {experiment}")
    print(f"Seeds: {seeds}")
    print(f"Eval-only: {args.eval_only}")
    print(f"Aggregate-only: {args.aggregate_only}")
    
    all_metrics = {}
    
    for seed in seeds:
        # Training
        if not args.eval_only:
            success = run_training(experiment, seed, extra)
            if not success:
                print(f"Skipping seed {seed} due to training failure")
                continue
        
        # Evaluation
        if not args.aggregate_only:
            metrics = run_evaluation(experiment, seed, extra)
            if metrics:
                all_metrics[seed] = metrics
        else:
            # Try loading existing metrics
            metrics_path = f"results/{experiment}/seed_{seed}/metrics.json"
            metrics = load_and_normalize_metrics(metrics_path)
            if metrics:
                all_metrics[seed] = metrics
    
    # Aggregate
    aggregate_results(all_metrics, experiment)


if __name__ == "__main__":
    main()
