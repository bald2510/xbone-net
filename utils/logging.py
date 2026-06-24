"""
Training logger for XBone-Net two-stage finetuning.

Provides:
  - TensorBoard logging (scalars, hyperparameters, GPU memory)
  - CSV logging (per-epoch metrics saved to disk)
  - GPU memory tracking
  - Hyperparameter recording
  - TrainerCallback integration with HuggingFace Trainer
"""

import csv
import os
import time
from datetime import datetime

import torch
from transformers import TrainerCallback


class TrainingLogger:
    """
    Centralized training logger that writes to TensorBoard and CSV simultaneously.

    Usage:
        logger = TrainingLogger(log_dir="runs/my_experiment", experiment_name="qlora_r16")
        logger.log_hyperparams({...})
        logger.log_scalar("train/loss", 0.5, step=100)
        logger.log_epoch_metrics({"train_loss": 0.5, "eval_loss": 0.6}, epoch=1)
        logger.close()
    """

    def __init__(self, log_dir: str, experiment_name: str, phase: str = ""):
        """
        Args:
            log_dir: Base directory for logs (e.g. "runs/").
            experiment_name: Name of the experiment (e.g. "finetune_biomedclip_qlora_r16").
            phase: Training phase identifier (e.g. "phase1", "phase2").
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{experiment_name}/{phase}_{timestamp}" if phase else f"{experiment_name}/{timestamp}"
        self.tb_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.tb_dir, exist_ok=True)

        # TensorBoard writer
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=self.tb_dir)

        # CSV log file
        self.csv_path = os.path.join(self.tb_dir, "metrics.csv")
        self._csv_file = None
        self._csv_writer = None
        self._csv_fields_written = False

        # Timing
        self._start_time = time.time()
        self._epoch_start_time = None

        print(f"[Logger] TensorBoard logs: {self.tb_dir}")
        print(f"[Logger] CSV metrics:      {self.csv_path}")

    def log_hyperparams(self, hparams: dict):
        """Log hyperparameters to TensorBoard and as a text summary."""
        # Write as text for easy viewing
        lines = [f"| {k} | {v} |" for k, v in hparams.items()]
        table = "| Parameter | Value |\n|---|---|\n" + "\n".join(lines)
        self.writer.add_text("hyperparameters", table, global_step=0)

        # Also write to a separate hparams.txt file for quick reference
        hparams_path = os.path.join(self.tb_dir, "hparams.txt")
        with open(hparams_path, "w", encoding="utf-8") as f:
            f.write(f"Experiment started: {datetime.now().isoformat()}\n")
            f.write("=" * 50 + "\n")
            for k, v in hparams.items():
                f.write(f"{k}: {v}\n")

        print(f"[Logger] Hyperparameters saved to {hparams_path}")

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a single scalar value to TensorBoard."""
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalars under the same main tag (overlaid in TensorBoard)."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_gpu_memory(self, step: int):
        """Log current GPU memory usage to TensorBoard."""
        if torch.cuda.is_available():
            allocated_mb = torch.cuda.memory_allocated() / 1024 ** 2
            reserved_mb = torch.cuda.memory_reserved() / 1024 ** 2
            max_allocated_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
            self.writer.add_scalar("gpu/memory_allocated_mb", allocated_mb, step)
            self.writer.add_scalar("gpu/memory_reserved_mb", reserved_mb, step)
            self.writer.add_scalar("gpu/memory_peak_mb", max_allocated_mb, step)

    def start_epoch(self):
        """Call at the start of each epoch to begin timing."""
        self._epoch_start_time = time.time()

    def log_epoch_metrics(self, metrics: dict, epoch: int):
        """
        Log a complete set of epoch-level metrics to both TensorBoard and CSV.

        Args:
            metrics: Dict of metric_name -> value. Keys like 'train_loss', 'eval_loss', etc.
            epoch: Epoch number (1-indexed).
        """
        # Compute elapsed time
        elapsed = time.time() - self._start_time
        epoch_time = time.time() - self._epoch_start_time if self._epoch_start_time else 0

        # Enrich metrics with timing
        enriched = {
            "epoch": epoch,
            "epoch_time_sec": round(epoch_time, 1),
            "total_time_sec": round(elapsed, 1),
            **metrics,
        }

        # GPU memory
        if torch.cuda.is_available():
            enriched["gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024 ** 2, 1)
            enriched["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)

        # TensorBoard scalars
        for key, value in enriched.items():
            if key == "epoch":
                continue
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"epoch/{key}", value, epoch)

        # CSV
        self._write_csv_row(enriched)

        # Console summary
        parts = [f"Epoch {epoch}"]
        for key in ("train_loss", "eval_loss", "learning_rate"):
            if key in metrics:
                parts.append(f"{key}={metrics[key]:.6f}")
        parts.append(f"time={epoch_time:.1f}s")
        if torch.cuda.is_available():
            parts.append(f"gpu={enriched.get('gpu_allocated_mb', 0):.0f}MB")
        print(f"[Logger] {' | '.join(parts)}")

        self.writer.flush()

    def _write_csv_row(self, row: dict):
        """Append a row to the CSV log file."""
        if self._csv_file is None:
            self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")

        if not self._csv_fields_written:
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            self._csv_writer.writeheader()
            self._csv_fields_written = True

        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def log_model_summary(self, model):
        """Log model parameter counts as scalars."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable

        self.writer.add_scalar("model/total_params", total, 0)
        self.writer.add_scalar("model/trainable_params", trainable, 0)
        self.writer.add_scalar("model/frozen_params", frozen, 0)
        self.writer.add_scalar("model/trainable_ratio_pct", (trainable / total) * 100 if total > 0 else 0, 0)

    def close(self):
        """Flush and close all writers."""
        self.writer.flush()
        self.writer.close()
        if self._csv_file:
            self._csv_file.close()
        elapsed = time.time() - self._start_time
        print(f"[Logger] Training completed in {elapsed:.1f}s. Logs saved to: {self.tb_dir}")


class XBoneTrainerCallback(TrainerCallback):
    """
    HuggingFace TrainerCallback that bridges the Trainer events to TrainingLogger.

    Logs:
      - Per-step: training loss, learning rate, grad norm
      - Per-epoch: eval loss, GPU memory
      - On train begin: hyperparameters, model summary
      - On train end: total training time
    """

    def __init__(self, logger: TrainingLogger):
        super().__init__()
        self.logger = logger
        self._current_epoch = 0

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Log model summary at the start of training."""
        if model is not None:
            self.logger.log_model_summary(model)

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Start epoch timer."""
        self._current_epoch = int(state.epoch) if state.epoch else 0
        self.logger.start_epoch()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Called by Trainer whenever it logs metrics.
        This captures per-step training metrics (loss, lr, grad_norm).
        """
        if logs is None:
            return

        step = state.global_step

        # Log individual scalars
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.logger.log_scalar(f"train/{key}", value, step)

        # Log GPU memory periodically
        self.logger.log_gpu_memory(step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Called after each evaluation. Logs eval metrics as epoch-level summary.
        """
        if metrics is None:
            return

        epoch = int(state.epoch) if state.epoch else self._current_epoch

        # Flatten HF Trainer metric names (remove 'eval_' prefix for some)
        epoch_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                clean_key = key.replace("eval_", "eval/") if key.startswith("eval_") else key
                epoch_metrics[key] = value
                self.logger.log_scalar(clean_key, value, epoch)

        # Also collect train loss from state if available
        if state.log_history:
            for entry in reversed(state.log_history):
                if "loss" in entry and "eval_loss" not in entry:
                    epoch_metrics["train_loss"] = entry["loss"]
                    if "learning_rate" in entry:
                        epoch_metrics["learning_rate"] = entry["learning_rate"]
                    break

        self.logger.log_epoch_metrics(epoch_metrics, epoch)

    def on_train_end(self, args, state, control, **kwargs):
        """Flush all logs at the end of training."""
        self.logger.writer.flush()
