"""
Dual-Backend Training Logger for XBone-Net Experiments.
===============================================================================
Provides two complementary logging sinks during XBone-Net model training:
  - TensorBoard: Real-time scalar, text, and GPU-memory tracking.
  - CSV: Archival per-epoch metrics written alongside event files.

The XBoneTrainerCallback bridges HuggingFace Trainer callbacks to TrainingLogger
so training, evaluation, and hardware metrics are automatically recorded.
"""

import csv
import os
import time
from datetime import datetime

import torch
from transformers import TrainerCallback


# ============================================================
# Centralized Training Logger
# ============================================================

class TrainingLogger:
    """Centralized training logger writing to TensorBoard and CSV.

    Creates a timestamped run directory under log_dir with TensorBoard
    event files, a per-epoch CSV metrics archive, and a hyperparameters snapshot.

    Attributes:
        tb_dir (str): Directory path where TensorBoard logs and outputs are stored.
        writer (SummaryWriter): PyTorch TensorBoard SummaryWriter instance.
        csv_path (str): File path for saving CSV metric records.

    Example:
        logger = TrainingLogger(log_dir="runs", experiment_name="biomedclip_phase1")
        logger.log_hyperparams({"lr": 1e-4, "batch_size": 32})
    """

    def __init__(self, log_dir: str, experiment_name: str, phase: str = "", use_timestamp: bool = False):
        """Initialize the logger and create the run directory structure.

        Args:
            log_dir: Root directory for all experiment logs.
            experiment_name: Name of the current experiment / model configuration.
            phase: Training phase identifier (e.g., 'phase1', 'phase2').
            use_timestamp: If True, appends timestamp. If False, logs directly into log_dir/logs/phase.
        """
        # --- Create directory ---
        self.log_dir = log_dir
        self._trainable_params = None

        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{experiment_name}/{phase}_{timestamp}" if phase else f"{experiment_name}/{timestamp}"
            self.tb_dir = os.path.join(log_dir, run_name)
        else:
            self.tb_dir = os.path.join(log_dir, "logs", phase) if phase else os.path.join(log_dir, "logs")
            
        os.makedirs(self.tb_dir, exist_ok=True)

        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=self.tb_dir)

        # --- Setup CSV output file ---
        self.csv_path = os.path.join(self.tb_dir, "metrics.csv")
        self._csv_file = None
        self._csv_writer = None
        self._csv_fields_written = False

        self._start_time = time.time()
        self._epoch_start_time = None

        print(f"[Logger] TensorBoard logs: {self.tb_dir}")
        print(f"[Logger] CSV metrics:      {self.csv_path}")

    def log_hyperparams(self, hparams: dict):
        """Log hyperparameters to TensorBoard and a local text file.

        Writes a Markdown table to the TensorBoard Text tab for interactive
        inspection, and persists a plain-text hparams.txt file for archival.

        Args:
            hparams: Dictionary mapping hyperparameter names to values.
        """
        # --- Write Markdown table to TensorBoard ---
        lines = [f"| {k} | {v} |" for k, v in hparams.items()]
        table = "| Parameter | Value |\n|---|---|\n" + "\n".join(lines)
        self.writer.add_text("hyperparameters", table, global_step=0)

        # --- Write plain-text hparams.txt ---
        hparams_path = os.path.join(self.tb_dir, "hparams.txt")
        with open(hparams_path, "w", encoding="utf-8") as f:
            f.write(f"Experiment started: {datetime.now().isoformat()}\n")
            f.write("=" * 50 + "\n")
            for k, v in hparams.items():
                f.write(f"{k}: {v}\n")

        print(f"[Logger] Hyperparameters saved to {hparams_path}")

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a single scalar value to TensorBoard.

        Args:
            tag: Metric tag / name.
            value: Scalar metric value.
            step: Training step or epoch index.
        """
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalar values under a main tag to TensorBoard.

        Args:
            main_tag: Main group name for scalars.
            tag_scalar_dict: Dictionary mapping sub-tags to scalar values.
            step: Training step or epoch index.
        """
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_gpu_memory(self, step: int):
        """Log CUDA GPU memory allocation statistics to TensorBoard.

        Args:
            step: Current training step index.
        """
        if torch.cuda.is_available():
            allocated_mb = torch.cuda.memory_allocated() / 1024 ** 2
            reserved_mb = torch.cuda.memory_reserved() / 1024 ** 2
            max_allocated_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
            self.writer.add_scalar("gpu/memory_allocated_mb", allocated_mb, step)
            self.writer.add_scalar("gpu/memory_reserved_mb", reserved_mb, step)
            self.writer.add_scalar("gpu/memory_peak_mb", max_allocated_mb, step)

    def _get_latest_checkpoint_size(self) -> float:
        """Find the latest checkpoint directory or file and return its size in MB."""
        if not hasattr(self, "log_dir") or not self.log_dir or not os.path.exists(self.log_dir):
            return 0.0
        
        checkpoint_dirs = []
        try:
            for d in os.listdir(self.log_dir):
                path = os.path.join(self.log_dir, d)
                if os.path.isdir(path) and d.startswith("checkpoint-"):
                    checkpoint_dirs.append(path)
        except Exception:
            return 0.0
            
        if not checkpoint_dirs:
            # Look for direct .pth or .safetensors files in log_dir
            try:
                pth_files = [os.path.join(self.log_dir, f) for f in os.listdir(self.log_dir) if f.endswith(".pth") or f.endswith(".safetensors")]
                if not pth_files:
                    return 0.0
                latest_file = max(pth_files, key=os.path.getmtime)
                return os.path.getsize(latest_file) / (1024 ** 2)
            except Exception:
                return 0.0
             
        try:   
            latest_dir = max(checkpoint_dirs, key=os.path.getmtime)
            total_size = 0
            for root, dirs, files in os.walk(latest_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    if os.path.exists(fp):
                        total_size += os.path.getsize(fp)
            return total_size / (1024 ** 2)
        except Exception:
            return 0.0

    def start_epoch(self):
        """Record the start timestamp for the current epoch."""
        self._epoch_start_time = time.time()

    def log_epoch_metrics(self, metrics: dict, epoch: int):
        """Log epoch summary metrics to TensorBoard and CSV.

        Enriches raw metrics with timing info, GPU stats, checkpoint size, and trainable parameters before recording.

        Args:
            metrics: Dictionary of metric names to values.
            epoch: Current epoch number (0-indexed).
        """
        # --- Compute elapsed timing ---
        elapsed = time.time() - self._start_time
        epoch_time = time.time() - self._epoch_start_time if self._epoch_start_time else 0

        enriched = {
            "epoch": epoch,
            "epoch_time_sec": round(epoch_time, 1),
            "total_time_sec": round(elapsed, 1),
            **metrics,
        }

        # Add trainable parameters if available
        if hasattr(self, "_trainable_params") and self._trainable_params is not None:
            enriched["trainable_params"] = self._trainable_params

        # Compute and add checkpoint size
        checkpoint_size_mb = self._get_latest_checkpoint_size()
        if checkpoint_size_mb > 0:
            enriched["checkpoint_size_mb"] = round(checkpoint_size_mb, 1)

        if torch.cuda.is_available():
            enriched["gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024 ** 2, 1)
            enriched["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)

        # --- Write scalars to TensorBoard ---
        for key, value in enriched.items():
            if key == "epoch":
                continue
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"epoch/{key}", value, epoch)

        # --- Append row to CSV file ---
        self._write_csv_row(enriched)

        # --- Console output ---
        parts = [f"Epoch {epoch}"]
        for key in ("train_loss", "eval_loss", "eval_f1_macro", "f1_macro", "learning_rate"):
            if key in metrics and metrics[key] is not None:
                is_metric = "f1" in key or "accuracy" in key
                parts.append(f"{key}={metrics[key]:.6f}" if not is_metric else f"{key}={metrics[key]:.4f}")
        parts.append(f"time={epoch_time:.1f}s")
        if torch.cuda.is_available():
            parts.append(f"gpu={enriched.get('gpu_allocated_mb', 0):.0f}MB (peak={enriched.get('gpu_peak_mb', 0):.0f}MB)")
        if "checkpoint_size_mb" in enriched:
            parts.append(f"ckpt={enriched['checkpoint_size_mb']:.1f}MB")
        if "trainable_params" in enriched:
            parts.append(f"trainable={enriched['trainable_params']:,}")
        print(f"[Logger] {' | '.join(parts)}")

        self.writer.flush()

    def _write_csv_row(self, row: dict):
        """Append a single row to the CSV metrics file.

        Args:
            row: Column-name to value mapping for one epoch.
        """
        if self._csv_file is None:
            self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")

        if not self._csv_fields_written:
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            self._csv_writer.writeheader()
            self._csv_fields_written = True

        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def log_model_summary(self, model):
        """Log model parameter statistics to TensorBoard.

        Args:
            model: PyTorch model instance to summarize.
        """
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        
        self._trainable_params = trainable

        self.writer.add_scalar("model/total_params", total, 0)
        self.writer.add_scalar("model/trainable_params", trainable, 0)
        self.writer.add_scalar("model/frozen_params", frozen, 0)
        self.writer.add_scalar("model/trainable_ratio_pct", (trainable / total) * 100 if total > 0 else 0, 0)
        
        # --- Save full model architecture to text file ---
        arch_path = os.path.join(self.tb_dir, "model_architecture.txt")
        try:
            with open(arch_path, "w", encoding="utf-8") as f:
                f.write(str(model))
        except Exception as e:
            print(f"[Logger] Warning: Could not save model architecture: {e}")

    def close(self):
        """Close TensorBoard writer and CSV file handles."""
        self.writer.flush()
        self.writer.close()
        if self._csv_file:
            self._csv_file.close()
        elapsed = time.time() - self._start_time
        print(f"[Logger] Training completed in {elapsed:.1f}s. Logs saved to: {self.tb_dir}")


# ============================================================
# HuggingFace Trainer Callback Integration
# ============================================================

class XBoneTrainerCallback(TrainerCallback):
    """HuggingFace Trainer callback that routes events to TrainingLogger.

    Maps Trainer lifecycle callbacks (on_train_begin, on_epoch_begin,
    on_log, on_evaluate, on_train_end) to logger calls.

    Attributes:
        logger (TrainingLogger): Active TrainingLogger instance.
    """

    def __init__(self, logger: TrainingLogger):
        """Initialize the callback.

        Args:
            logger: The TrainingLogger instance to delegate event handling to.
        """
        super().__init__()
        self.logger = logger
        self._current_epoch = 0

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Callback triggered at start of training."""
        if model is not None:
            self.logger.log_model_summary(model)

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Callback triggered at start of each epoch."""
        self._current_epoch = int(state.epoch) if state.epoch else 0
        self.logger.start_epoch()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Callback triggered when Trainer logs metrics."""
        if logs is None:
            return

        step = state.global_step
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.logger.log_scalar(f"train/{key}", value, step)

        self.logger.log_gpu_memory(step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Callback triggered when evaluation completes."""
        if metrics is None:
            return

        epoch = int(state.epoch) if state.epoch else self._current_epoch

        epoch_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                clean_key = key.replace("eval_", "eval/") if key.startswith("eval_") else key
                epoch_metrics[key] = value
                self.logger.log_scalar(clean_key, value, epoch)

        epoch_metrics["train_loss"] = None
        epoch_metrics["learning_rate"] = None
        if state.log_history:
            for entry in reversed(state.log_history):
                if "loss" in entry and "eval_loss" not in entry:
                    epoch_metrics["train_loss"] = entry["loss"]
                    if "learning_rate" in entry:
                        epoch_metrics["learning_rate"] = entry["learning_rate"]
                    break

        self.logger.log_epoch_metrics(epoch_metrics, epoch)

    def on_train_end(self, args, state, control, **kwargs):
        """Callback triggered at end of training."""
        self.logger.writer.flush()


