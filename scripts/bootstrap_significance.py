"""
Bootstrap Significance Testing for XBone-Net Experiments
=========================================================
Compares two experiments using paired bootstrap tests with Bonferroni correction.

Usage:
    # Compare two experiments
    python scripts/bootstrap_significance.py \
        --exp-a results/claim2_loss_infonce_fracatlas/ \
        --exp-b results/claim2_loss_semantic_fracatlas/ \
        --metric auroc_macro \
        --output results/significance/infonce_vs_semantic.json

    # Compare all pairs in a claim
    python scripts/bootstrap_significance.py \
        --results-dir results/ \
        --claim claim2_loss \
        --output results/significance/claim2_loss_pairwise.json
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import argparse
import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def load_predictions(results_dir: str) -> list[dict]:
    """Load per-seed prediction files from an experiment directory."""
    results = []
    results_path = Path(results_dir)

    for json_path in sorted(results_path.rglob("metrics.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results.append(data)

    return results


def paired_bootstrap_test(
    metric_values_a: list[float],
    metric_values_b: list[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap test on pre-computed metric values across seeds.

    Args:
        metric_values_a: metric values for experiment A (one per seed)
        metric_values_b: metric values for experiment B (one per seed)
        n_bootstrap: number of bootstrap iterations
        seed: random seed

    Returns:
        dict with test results
    """
    rng = np.random.RandomState(seed)
    a = np.array(metric_values_a)
    b = np.array(metric_values_b)
    n = len(a)

    observed_diff = float(np.mean(b) - np.mean(a))

    # Bootstrap the mean difference
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_diff = np.mean(b[idx]) - np.mean(a[idx])
        boot_diffs.append(boot_diff)

    boot_diffs = np.array(boot_diffs)

    # Two-sided p-value
    p_value = float(np.mean(np.abs(boot_diffs - np.mean(boot_diffs)) >= np.abs(observed_diff)))

    # 95% CI for the difference
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    return {
        "mean_a": float(np.mean(a)),
        "std_a": float(np.std(a)),
        "mean_b": float(np.mean(b)),
        "std_b": float(np.std(b)),
        "observed_diff": observed_diff,
        "p_value": p_value,
        "ci_95": [ci_lower, ci_upper],
        "n_seeds": n,
        "significant_at_005": p_value < 0.05,
    }


def extract_metric_values(results: list[dict], metric: str) -> list[float]:
    """Extract metric values from experiment results."""
    values = []
    for r in results:
        metrics = r.get("metrics", {})
        if metric in metrics and metrics[metric] is not None:
            values.append(float(metrics[metric]))
    return values


def run_pairwise_comparisons(
    experiments: dict[str, list[dict]],
    metric: str,
    n_bootstrap: int = 10000,
    bonferroni: bool = True,
) -> list[dict]:
    """
    Run pairwise bootstrap tests between all experiment pairs.

    Args:
        experiments: {experiment_name: [results_per_seed]}
        metric: metric to compare
        n_bootstrap: bootstrap iterations
        bonferroni: apply Bonferroni correction

    Returns:
        list of comparison results
    """
    exp_names = sorted(experiments.keys())
    pairs = list(combinations(exp_names, 2))
    n_comparisons = len(pairs)

    results = []
    for exp_a, exp_b in pairs:
        vals_a = extract_metric_values(experiments[exp_a], metric)
        vals_b = extract_metric_values(experiments[exp_b], metric)

        if not vals_a or not vals_b:
            print(f"  [Skip] {exp_a} vs {exp_b}: missing metric values")
            continue

        # Ensure same number of seeds
        n = min(len(vals_a), len(vals_b))
        vals_a = vals_a[:n]
        vals_b = vals_b[:n]

        test_result = paired_bootstrap_test(vals_a, vals_b, n_bootstrap)

        # Bonferroni correction
        corrected_alpha = 0.05 / n_comparisons if bonferroni else 0.05
        test_result["bonferroni_corrected"] = bonferroni
        test_result["corrected_alpha"] = corrected_alpha
        test_result["significant_bonferroni"] = test_result["p_value"] < corrected_alpha

        comparison = {
            "experiment_a": exp_a,
            "experiment_b": exp_b,
            "metric": metric,
            **test_result,
        }

        results.append(comparison)

        sig_marker = "✓" if test_result["significant_bonferroni"] else "✗"
        print(f"  {exp_a} vs {exp_b}: Δ={test_result['observed_diff']:.4f} "
              f"(p={test_result['p_value']:.4f}) {sig_marker}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Bootstrap significance testing")
    parser.add_argument("--exp-a", default=None, help="Experiment A results directory")
    parser.add_argument("--exp-b", default=None, help="Experiment B results directory")
    parser.add_argument("--results-dir", default="results/", help="Root results directory")
    parser.add_argument("--claim", default=None,
                        help="Filter experiment names by claim prefix (e.g., claim2_loss)")
    parser.add_argument("--metric", default="auroc_macro", help="Metric to compare")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--no-bonferroni", action="store_true")
    parser.add_argument("--output", default="results/significance/significance_test.json")
    args = parser.parse_args()

    print("=" * 60)
    print("BOOTSTRAP SIGNIFICANCE TESTING")
    print("=" * 60)
    print(f"Metric: {args.metric}")
    print(f"Bootstrap samples: {args.n_bootstrap:,}")
    print(f"Bonferroni correction: {not args.no_bonferroni}")

    if args.exp_a and args.exp_b:
        # Direct comparison of two experiments
        results_a = load_predictions(args.exp_a)
        results_b = load_predictions(args.exp_b)

        vals_a = extract_metric_values(results_a, args.metric)
        vals_b = extract_metric_values(results_b, args.metric)

        print(f"\nExp A ({args.exp_a}): {len(vals_a)} seeds, {args.metric} = {vals_a}")
        print(f"Exp B ({args.exp_b}): {len(vals_b)} seeds, {args.metric} = {vals_b}")

        n = min(len(vals_a), len(vals_b))
        result = paired_bootstrap_test(vals_a[:n], vals_b[:n], args.n_bootstrap)

        print(f"\nResult: Δ = {result['observed_diff']:.4f}")
        print(f"  95% CI: [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")
        print(f"  p-value: {result['p_value']:.4f}")
        print(f"  Significant (α=0.05): {'Yes' if result['significant_at_005'] else 'No'}")

        output = {
            "experiment_a": args.exp_a,
            "experiment_b": args.exp_b,
            "metric": args.metric,
            **result,
        }
        comparisons = [output]

    elif args.claim:
        # Pairwise comparisons within a claim
        experiments = {}
        results_path = Path(args.results_dir)

        for exp_dir in sorted(results_path.iterdir()):
            if exp_dir.is_dir() and args.claim in exp_dir.name:
                exp_results = load_predictions(str(exp_dir))
                if exp_results:
                    experiments[exp_dir.name] = exp_results

        print(f"\nFound {len(experiments)} experiments matching '{args.claim}':")
        for name in experiments:
            n_seeds = len(experiments[name])
            print(f"  {name}: {n_seeds} seeds")

        print(f"\nRunning pairwise comparisons...")
        comparisons = run_pairwise_comparisons(
            experiments, args.metric, args.n_bootstrap,
            bonferroni=not args.no_bonferroni,
        )
    else:
        print("\nProvide either --exp-a/--exp-b or --claim.")
        return

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        "type": "significance_test",
        "metric": args.metric,
        "n_bootstrap": args.n_bootstrap,
        "bonferroni": not args.no_bonferroni,
        "timestamp": datetime.datetime.now().isoformat(),
        "comparisons": comparisons,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
