"""
Run All Experiments — Baselines + Ablations + Multi-seed.
=========================================================

Orchestrates the full experiment suite: trains and evaluates every config
across multiple seeds, then prints an aggregated results table.

Usage:
    python run_all.py                          # Run everything (3 seeds)
    python run_all.py --group baseline         # Only baselines
    python run_all.py --group ablation         # Only ablations
    python run_all.py --seeds 42               # Single seed (fast debug)
    python run_all.py --eval-only              # Skip training, only evaluate
    python run_all.py --table                  # Only print results table
"""

import argparse
import json
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

# ── Experiment registry ──────────────────────────────────────────────────────
# Ordered so baselines run first, then ablations (dependencies respected).

EXPERIMENTS = OrderedDict({
    # ─── 1. Đánh giá đối sánh hiệu năng (Baseline Comparison) ───
    "baseline": [
        # "1_baseline/B1_resnet50",
        "1_baseline/B2_medclip", 
        # "1_baseline/B3_pubmedclip",
        # "1_baseline/B4_biomedclip",
        # "1_baseline/B5_clip",
        # "1_baseline/ours_xbone_net",
    ],
    # ─── 2. Đánh giá nguồn văn bản lâm sàng và rò rỉ nhãn (Modality Ablation) ───
    # "ablation_modality": [
    #     "2_ablation_modality/T2_img_only",
    #     "2_ablation_modality/T2_clinical_only",
    #     "2_ablation_modality/T2_xray_only",
    #     "2_ablation_modality/T2_xray_clinical",
    #     "2_ablation_modality/T2_both"
    # ],
    # ─── 3. Đánh giá tinh chỉnh backbone và trôi lệch biểu diễn (PEFT & Drift Ablation) ───
    "ablation_finetune": [
        "3_ablation_finetune/T3_no_ft",
        "3_ablation_finetune/T3_lora_ft",
        "3_ablation_finetune/T3_full_ft",
        "3_ablation_finetune/T3_lora_ft_clinical_only",
        "3_ablation_finetune/T3_full_ft_clinical_only"
    ],
    # # ─── 4. So sánh hiệu quả giữa các đầu phân loại (Classifier Ablation) ───
    "ablation_classifier": [
        "4_ablation_classifier/T4_crossattn_linear",
        "4_ablation_classifier/T4_concat_linear",
        "4_ablation_classifier/T4_crossattn_proto",
        "4_ablation_classifier/T4_concat_proto"
    ],
    # # ─── 6. Đánh giá hàm mất mát và tối ưu (Loss Ablation) ───
    # "ablation_loss": [
    #     "6_ablation_loss/T6_bce_proto",
    #     "6_ablation_loss/T6_infonce",
    # ],
})

DEFAULT_SEEDS = [42, 123, 456]

METRIC_KEYS = [
    "f1_macro", "accuracy", "sensitivity_macro", "specificity_macro",
    "precision_macro", "auroc_macro",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_experiments(groups: list[str] | None) -> list[str]:
    """Return flat list of experiment config paths for the requested groups."""
    if not groups:
        # All experiments (preserving order, removing duplicates)
        seen = set()
        return [exp for group in EXPERIMENTS.values() for exp in group if not (exp in seen or seen.add(exp))]
    
    result = []
    for g in groups:
        if g in EXPERIMENTS:
            result.extend(EXPERIMENTS[g])
        elif g == "ablation":
            # Shorthand for all ablation groups
            for key, exps in EXPERIMENTS.items():
                if key.startswith("ablation") or key == "proposed":
                    result.extend(exps)
        else:
            # Treat as a direct experiment path
            result.append(g)
            
    seen = set()
    return [exp for exp in result if not (exp in seen or seen.add(exp))]


def seed_dir(experiment: str, seed: int) -> str:
    return f"checkpoints/{experiment}/seed_{seed}"


def run_one(script: str, experiment: str, seed: int) -> bool:
    """Run train_2stage.py or evaluate_model.py for one (experiment, seed)."""
    sd = seed_dir(experiment, seed)
    cmd = [
        sys.executable, script,
        f"+experiment={experiment}",
        f"++seed={seed}",
        f"++params.model_dir={sd}/",
        f"++params.phase1.checkpoint_path={sd}/best_phase1.pth",
        f"++params.phase2.checkpoint_path={sd}/best_phase2.pth",
    ]
    
    if "evaluate_model.py" in script:
        cmd.extend(["--output-dir", f"results/{experiment}/seed_{seed}"])
        
    tag = "TRAIN" if "train" in script else "EVAL"
    exp_short = experiment.split("/")[-1]
    print(f"\n  [{tag}] {exp_short} seed={seed}")
    print(f"    → {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"    ✗ FAILED (exit {result.returncode})")
        return False
    print(f"    ✓ Done")
    return True


def load_metrics(experiment: str, seed: int) -> dict | None:
    """Try to load metrics.json from various possible locations and normalize keys."""
    candidates = [  
        f"results/{experiment}/seed_{seed}/metrics.json",
        f"{seed_dir(experiment, seed)}/metrics.json",
        f"{seed_dir(experiment, seed)}/eval_results.json",
    ]
    for path in candidates:
        full = os.path.join(SCRIPT_DIR, path)
        if os.path.exists(full):
            try:
                with open(full, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
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
            except Exception as e:
                print(f"    ⚠ Failed to load {full}: {e}")
    return None


def aggregate(all_metrics: dict[int, dict]) -> dict:
    """Compute mean ± std across seeds."""
    agg = {}
    for key in METRIC_KEYS:
        vals = [m[key] for m in all_metrics.values()
                if m and key in m and m[key] is not None]
        if vals:
            agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    return agg


def save_results(experiment: str, seeds_metrics: dict, agg: dict):
    """Save per-seed and aggregated results to JSON."""
    out_dir = os.path.join(SCRIPT_DIR, "results", experiment)
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
    """Print a formatted table of all aggregated results."""
    print(f"\n{'='*90}")
    print(f"  RESULTS SUMMARY (mean ± std)")
    print(f"{'='*90}")
    
    header = f"  {'Experiment':<35s} {'F1 Macro':>14s} {'Accuracy':>14s} {'Sensitivity':>14s} {'AUROC':>14s}"
    print(header)
    print(f"  {'-'*35} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")
    
    for exp, agg in results.items():
        name = exp.split("/")[-1]
        cols = []
        for key in ["f1_macro", "accuracy", "sensitivity_macro", "auroc_macro"]:
            if key in agg and agg[key].get("n", 0) > 0:
                cols.append(f"{agg[key]['mean']:.4f}±{agg[key]['std']:.4f}")
            else:
                cols.append("N/A")
        print(f"  {name:<35s} {cols[0]:>14s} {cols[1]:>14s} {cols[2]:>14s} {cols[3]:>14s}")
    
    print(f"{'='*90}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run all experiments (baselines + ablations) with multi-seed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--group", "-g", nargs="+", default=None,
                       help="Experiment groups: baseline, ablation_modality, "
                            "ablation_finetune, ablation_classifier, "
                            "ablation_fusion, ablation_loss. "
                            "Or 'ablation' to run all ablation groups. Or a direct config path.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                       help=f"Seeds to run (default: {DEFAULT_SEEDS})")
    parser.add_argument("--eval-only", action="store_true",
                       help="Skip training, only evaluate + aggregate")
    parser.add_argument("--table", action="store_true",
                       help="Only print results from existing aggregated JSON files")
    args = parser.parse_args()
    
    experiments = get_experiments(args.group)
    seeds = args.seeds
    
    print(f"╔══════════════════════════════════════════════════════════╗")
    print(f"║  XBone-Net — Full Experiment Suite                      ║")
    print(f"╠══════════════════════════════════════════════════════════╣")
    print(f"║  Experiments : {len(experiments):<40d} ║")
    print(f"║  Seeds       : {str(seeds):<40s} ║")
    print(f"║  Total runs  : {len(experiments) * len(seeds):<40d} ║")
    print(f"║  Mode        : {'TABLE ONLY' if args.table else 'EVAL ONLY' if args.eval_only else 'TRAIN + EVAL':<40s} ║")
    print(f"╚══════════════════════════════════════════════════════════╝")
    
    all_results = OrderedDict()
    
    for i, experiment in enumerate(experiments, 1):
        exp_short = experiment.split("/")[-1]
        print(f"\n{'━'*60}")
        print(f"  [{i}/{len(experiments)}] {experiment}")
        print(f"{'━'*60}")
        
        seeds_metrics = {}
        
        for seed in seeds:
            if args.table:
                # Only load existing results
                metrics = load_metrics(experiment, seed)
                if metrics:
                    seeds_metrics[seed] = metrics
                continue
            
            # Train
            if not args.eval_only:
                success = run_one("train_2stage.py", experiment, seed)
                if not success:
                    print(f"    ⚠ Skipping eval for seed={seed}")
                    continue
            
            # Evaluate
            run_one("evaluate_model.py", experiment, seed)
            
            # Load metrics
            metrics = load_metrics(experiment, seed)
            if metrics:
                seeds_metrics[seed] = metrics
            else:
                print(f"    ⚠ No metrics found for seed={seed}")
        
        # Aggregate
        if seeds_metrics:
            agg = aggregate(seeds_metrics)
            save_path = save_results(experiment, seeds_metrics, agg)
            all_results[experiment] = agg
            
            # Print per-experiment summary
            f1 = agg.get("f1_macro", {})
            if f1:
                print(f"\n  ► {exp_short}: F1={f1['mean']:.4f}±{f1['std']:.4f} (n={f1['n']})")
                print(f"    Saved → {save_path}")
        else:
            print(f"\n  ► {exp_short}: No results collected")
            all_results[experiment] = {}
    
    # Final table
    print_results_table(all_results)
    
    # Save master summary
    master_path = os.path.join(SCRIPT_DIR, "results", "all_results.json")
    os.makedirs(os.path.dirname(master_path), exist_ok=True)
    with open(master_path, "w") as f:
        json.dump(
            {exp: agg for exp, agg in all_results.items()},
            f, indent=2,
        )
    print(f"  Master results → {master_path}")


if __name__ == "__main__":
    main()
