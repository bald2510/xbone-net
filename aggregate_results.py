"""
Results Aggregation for XBone-Net Experiments
=============================================
Reads JSON results from results/ directory and produces:
  1. Summary CSV with mean ± std across seeds
  2. Paired bootstrap significance tests
  3. LaTeX tables for paper insertion

Usage:
    python aggregate_results.py --results-dir results/
    python aggregate_results.py --results-dir results/ --compare claim2_loss_infonce claim2_loss_semantic
    python aggregate_results.py --results-dir results/ --latex --output tables/
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


def load_all_results(results_dir: str) -> list[dict]:
    """Load all metrics.json files from results directory tree."""
    results = []
    results_path = Path(results_dir)

    for json_path in results_path.rglob("metrics.json"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_source_file"] = str(json_path)
            results.append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  [Warning] Failed to read {json_path}: {e}")

    print(f"Loaded {len(results)} result files from {results_dir}")
    return results


def aggregate_by_experiment(results: list[dict]) -> pd.DataFrame:
    """
    Group results by experiment name and compute mean ± std across seeds.

    Returns DataFrame with columns:
        experiment_name, dataset, n_seeds, metric_mean, metric_std
    """
    groups = defaultdict(list)

    for r in results:
        key = r.get("experiment_name", "unknown")
        groups[key].append(r)

    rows = []
    for exp_name, runs in sorted(groups.items()):
        metrics_list = [r.get("metrics", {}) for r in runs]
        dataset = runs[0].get("dataset", "unknown")
        seeds = [r.get("seed", -1) for r in runs]

        row = {
            "experiment": exp_name,
            "dataset": dataset,
            "n_seeds": len(runs),
            "seeds": str(seeds),
        }

        # Aggregate each metric
        for metric_key in ["auroc_macro", "f1_macro", "accuracy", "sensitivity", "specificity", "precision", "hamming_loss"]:
            values = [m.get(metric_key) for m in metrics_list if m.get(metric_key) is not None]
            if values:
                row[f"{metric_key}_mean"] = np.mean(values)
                row[f"{metric_key}_std"] = np.std(values)
                row[f"{metric_key}_formatted"] = f"{np.mean(values):.4f} ± {np.std(values):.4f}"
            else:
                row[f"{metric_key}_mean"] = None
                row[f"{metric_key}_std"] = None
                row[f"{metric_key}_formatted"] = "—"

        rows.append(row)

    return pd.DataFrame(rows)


def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
    metric_fn,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap significance test.

    Args:
        scores_a: (N,) prediction scores from model A
        scores_b: (N,) prediction scores from model B
        labels: (N,) ground truth binary labels
        metric_fn: callable(labels, scores) -> float (e.g., roc_auc_score)
        n_bootstrap: number of bootstrap iterations
        seed: random seed for reproducibility

    Returns:
        dict with observed_diff, p_value, ci_95
    """
    rng = np.random.RandomState(seed)
    n = len(labels)

    observed_a = metric_fn(labels, scores_a)
    observed_b = metric_fn(labels, scores_b)
    observed_diff = observed_b - observed_a

    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_labels = labels[idx]

        # Skip degenerate bootstrap samples
        if len(np.unique(boot_labels)) < 2:
            continue

        boot_a = metric_fn(boot_labels, scores_a[idx])
        boot_b = metric_fn(boot_labels, scores_b[idx])
        diffs.append(boot_b - boot_a)

    diffs = np.array(diffs)

    # Two-sided p-value
    p_value = np.mean(np.abs(diffs) >= np.abs(observed_diff))

    # 95% CI for the difference
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)

    return {
        "metric_a": float(observed_a),
        "metric_b": float(observed_b),
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "ci_95": [float(ci_lower), float(ci_upper)],
        "n_bootstrap": n_bootstrap,
        "significant_at_005": p_value < 0.05,
    }


def generate_latex_table(df: pd.DataFrame, metric_cols: list[str], caption: str = "") -> str:
    """
    Generate a LaTeX table from aggregated results DataFrame.

    Args:
        df: DataFrame with experiment results
        metric_cols: list of column suffixes to include (e.g., ["auroc_macro", "f1_macro"])
        caption: table caption

    Returns:
        LaTeX table string
    """
    # Column headers
    headers = ["Experiment", "Dataset"] + [col.replace("_", " ").title() for col in metric_cols]
    col_spec = "l l " + " ".join(["c"] * len(metric_cols))

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]

    # Find best values per metric for bolding
    best_vals = {}
    for col in metric_cols:
        mean_col = f"{col}_mean"
        if mean_col in df.columns:
            valid = df[mean_col].dropna()
            if len(valid) > 0:
                best_vals[col] = valid.max()

    for _, row in df.iterrows():
        cells = [row.get("experiment", ""), row.get("dataset", "")]
        for col in metric_cols:
            formatted = row.get(f"{col}_formatted", "—")
            mean_val = row.get(f"{col}_mean")
            # Bold the best result
            if mean_val is not None and col in best_vals and abs(mean_val - best_vals[col]) < 1e-6:
                formatted = f"\\textbf{{{formatted}}}"
            cells.append(formatted)
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument("--results-dir", default="results/", help="Root results directory")
    parser.add_argument("--output", default="results/summary/", help="Output directory for summary files")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX tables")
    parser.add_argument("--compare", nargs=2, metavar=("EXP_A", "EXP_B"),
                        help="Run significance test between two experiments")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load all results
    results = load_all_results(args.results_dir)
    if not results:
        print("No results found. Run experiments first.")
        return

    # Aggregate
    df = aggregate_by_experiment(results)

    # Save summary CSV
    csv_path = os.path.join(args.output, "summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n📊 Summary saved to: {csv_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    display_cols = ["experiment", "dataset", "n_seeds",
                    "auroc_macro_formatted", "f1_macro_formatted", "accuracy_formatted"]
    available_cols = [c for c in display_cols if c in df.columns]
    print(df[available_cols].to_string(index=False))

    # Generate LaTeX tables
    if args.latex:
        metrics = ["auroc_macro", "f1_macro", "accuracy", "sensitivity", "specificity"]
        latex = generate_latex_table(df, metrics, caption="Experiment Results")
        latex_path = os.path.join(args.output, "results_table.tex")
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(latex)
        print(f"\n📄 LaTeX table saved to: {latex_path}")

    print("\n✅ Aggregation complete!")


if __name__ == "__main__":
    main()
