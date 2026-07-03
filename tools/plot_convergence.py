"""
XBone-Net Training Convergence Plotter.
===============================================================================
Generates convergence plots from TensorBoard event logs or fallback CSV metrics:
  - Event Parsing: Reads TensorBoard scalar events (loss, gradient norm, learning rate).
  - Smoothing & Formatting: Applies exponential/moving-average smoothing for long training runs.
  - Figure Export: Saves multi-panel PDF/PNG publication-ready figures.
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import os
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ============================================================
# TensorBoard & CSV Readers
# ============================================================

def read_tensorboard_events(log_dir: str) -> dict[str, list[tuple[int, float]]]:
    """Read TensorBoard event files and extract scalar metrics.

    Args:
        log_dir: Directory path containing TensorBoard event files.

    Returns:
        dict[str, list[tuple[int, float]]]: Mapping scalar tag to list of (step, value) tuples.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("[Error] tensorboard not installed. Run: pip install tensorboard")
        return {}

    ea = EventAccumulator(log_dir)
    ea.Reload()

    data = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        data[tag] = [(e.step, e.value) for e in events]

    return data


def read_csv_metrics(csv_path: str) -> dict[str, list[tuple[int, float]]]:
    """Fallback reader for CSV metrics files.

    Args:
        csv_path: File path to CSV file containing metrics.

    Returns:
        dict[str, list[tuple[int, float]]]: Mapping metric name to list of (step, value) tuples.
    """
    import csv

    data = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row.get("step", row.get("epoch", 0)))
            for key, value in row.items():
                if key in ("step", "epoch"):
                    continue
                try:
                    data[key].append((step, float(value)))
                except (ValueError, TypeError):
                    pass
    return dict(data)


def find_run_dirs(runs_dir: str, name_filter: str = None) -> list[tuple[str, str]]:
    """Find TensorBoard run directories matching optional name filter.

    Args:
        runs_dir: Path to root directory containing runs.
        name_filter: Optional substring filter for directory names.

    Returns:
        list[tuple[str, str]]: List of (display_name, run_dir) tuples.
    """
    runs = []
    runs_path = Path(runs_dir)

    for entry in sorted(runs_path.rglob("events.out.tfevents.*")):
        run_dir = str(entry.parent)
        display_name = str(entry.parent.relative_to(runs_path))

        if name_filter and name_filter not in display_name:
            continue

        runs.append((display_name, run_dir))

    return runs


# ============================================================
# Convergence Plotting Engine
# ============================================================

def plot_convergence(
    all_data: dict[str, dict[str, list[tuple[int, float]]]],
    output_path: str,
    title: str = "Training Convergence",
):
    """Generate convergence plots for loss, gradient norm, and learning rate.

    Args:
        all_data: Dict mapping run names to metric dicts.
        output_path: Output file path for saved figure.
        title: Figure title string (default: 'Training Convergence').

    Returns:
        None
    """
    all_tags = set()
    for run_data in all_data.values():
        all_tags.update(run_data.keys())

    plot_configs = [
        ("train/loss", "Training Loss", "Loss"),
        ("train/grad_norm", "Gradient Norm", "‖∇‖"),
        ("train/learning_rate", "Learning Rate", "LR"),
    ]

    available_plots = [(tag, title, ylabel) for tag, title, ylabel in plot_configs if tag in all_tags]

    alt_names = {"loss": "train/loss", "grad_norm": "train/grad_norm", "learning_rate": "train/learning_rate"}
    for alt, canonical in alt_names.items():
        if alt in all_tags and canonical not in all_tags:
            available_plots.append((alt, alt.replace("_", " ").title(), alt))

    if not available_plots:
        for tag in sorted(all_tags)[:3]:
            available_plots.append((tag, tag, tag))

    n_plots = len(available_plots)
    if n_plots == 0:
        print("[Warning] No metrics found to plot.")
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4.5), squeeze=False)
    axes = axes[0]

    colors = plt.cm.Set2(np.linspace(0, 1, len(all_data)))

    for ax_idx, (tag, plot_title, ylabel) in enumerate(available_plots):
        ax = axes[ax_idx]

        for (run_name, run_data), color in zip(all_data.items(), colors):
            if tag not in run_data:
                continue

            steps = [s for s, v in run_data[tag]]
            values = [v for s, v in run_data[tag]]

            if len(values) > 20 and tag in ("train/loss", "loss", "train/grad_norm", "grad_norm"):
                window = max(5, len(values) // 20)
                smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
                smooth_steps = steps[window - 1:]
                ax.plot(smooth_steps, smoothed, label=run_name, color=color, linewidth=2)
                ax.plot(steps, values, color=color, alpha=0.15, linewidth=0.5)
            else:
                ax.plot(steps, values, label=run_name, color=color, linewidth=2)

        ax.set_title(plot_title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

        if tag in ("train/learning_rate", "learning_rate"):
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(all_data), 4),
                   fontsize=9, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")
    plt.close(fig)


# ============================================================
# Main Entry Point & CLI Parsing
# ============================================================

def main():
    """CLI entry-point: parse arguments and plot convergence curves."""
    parser = argparse.ArgumentParser(description="Plot training convergence curves")
    parser.add_argument("--runs-dir", default="runs/", help="TensorBoard runs directory")
    parser.add_argument("--filter", default=None, help="Filter run names by substring")
    parser.add_argument("--output", default="figures/convergence.pdf", help="Output figure path")
    parser.add_argument("--title", default="Training Convergence", help="Figure title")
    args = parser.parse_args()

    print("Scanning for TensorBoard runs...")
    runs = find_run_dirs(args.runs_dir, args.filter)
    print(f"Found {len(runs)} runs")

    if not runs:
        print("No runs found. Check --runs-dir and --filter.")
        return

    all_data = {}
    for name, path in runs:
        print(f"  Reading: {name}")
        data = read_tensorboard_events(path)
        if not data:
            csv_path = os.path.join(path, "metrics.csv")
            if os.path.exists(csv_path):
                data = read_csv_metrics(csv_path)
        if data:
            all_data[name] = data

    if not all_data:
        print("No data found in any run.")
        return

    plot_convergence(all_data, args.output, args.title)


if __name__ == "__main__":
    main()

