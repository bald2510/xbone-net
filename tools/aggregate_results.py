"""
XBone-Net Multi-Seed Results Aggregator.
===============================================================================
Aggregates multi-seed experiment metrics and exports tables:
  - Metric Gathering: Recursively scans results directory for metrics.json files across seeds.
  - Statistical Aggregation: Groups metrics by experiment and computes mean ± std values.
  - Table Export: Exports aggregated summary CSV and publication-ready LaTeX tables.
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


# ============================================================
# Metric Loading & Data Aggregation
# ============================================================

def load_all_results(results_dir: str) -> list[dict]:
    """Recursively search for and load all metrics.json files.

    Args:
        results_dir: Root directory path containing experiment output subdirectories.

    Returns:
        list[dict]: List of loaded metric dictionaries with tracked source paths.
    """
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
    """Group result dictionaries by experiment name and compute mean ± std across seeds.

    Args:
        results: List of metric dictionaries produced by load_all_results.

    Returns:
        pd.DataFrame: DataFrame containing experiment configurations, seed lists, and statistics.
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

        for metric_key in [
            "auroc_macro", "auprc_macro", "f1_macro", "accuracy",
            "balanced_accuracy", "sensitivity", "specificity", "precision",
            "ece_15", "adaptive_ece_15", "nll", "brier_score", "hamming_loss",
        ]:
            values = [m.get(metric_key) for m in metrics_list if m.get(metric_key) is not None]
            if values:
                row[f"{metric_key}_mean"] = np.mean(values)
                row[f"{metric_key}_std"] = (
                    np.std(values, ddof=1) if len(values) > 1 else 0.0
                )

                row[f"{metric_key}_formatted"] = (
                    f"{np.mean(values):.4f} ± "
                    f"{row[f'{metric_key}_std']:.4f}"
                )
            else:
                row[f"{metric_key}_mean"] = None
                row[f"{metric_key}_std"] = None
                row[f"{metric_key}_formatted"] = "—"

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Table Export
# ============================================================
def generate_latex_table(df: pd.DataFrame, metric_cols: list[str], caption: str = "") -> str:
    """Generate LaTeX table string from aggregated results DataFrame.

    Args:
        df: Aggregated metrics DataFrame.
        metric_cols: List of metric column names to include.
        caption: Table caption text.

    Returns:
        str: Formatted LaTeX table code string.
    """
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


# ============================================================
# Main Entry Point & CLI Parsing
# ============================================================

def main():
    """CLI entry-point: parse arguments, aggregate metrics, and export summary files."""
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument("--results-dir", default="results/", help="Root results directory")
    parser.add_argument("--output", default="results/summary/", help="Output directory for summary files")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX tables")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    results = load_all_results(args.results_dir)
    if not results:
        print("No results found. Run experiments first.")
        return

    df = aggregate_by_experiment(results)

    csv_path = os.path.join(args.output, "summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to: {csv_path}")

    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    display_cols = ["experiment", "dataset", "n_seeds",
                    "auroc_macro_formatted", "f1_macro_formatted", "accuracy_formatted"]
    available_cols = [c for c in display_cols if c in df.columns]
    print(df[available_cols].to_string(index=False))

    if args.latex:
        metrics = ["auroc_macro", "f1_macro", "accuracy", "sensitivity", "specificity"]
        latex = generate_latex_table(df, metrics, caption="Experiment Results")
        latex_path = os.path.join(args.output, "results_table.tex")
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(latex)
        print(f"\nLaTeX table saved to: {latex_path}")

    print("\nAggregation complete!")


if __name__ == "__main__":
    main()
