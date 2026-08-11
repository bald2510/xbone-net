"""Cung cấp tiện ích logging cho huấn luyện, đánh giá và phân tích XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""

import csv
import os
import time
from datetime import datetime

import torch
from transformers import TrainerCallback


# ============================================================
# Bộ ghi nhật ký huấn luyện tập trung
# ============================================================

class TrainingLogger:
    """Ghi nhận thông tin huấn luyện bằng lớp ``TrainingLogger``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(self, log_dir: str, experiment_name: str, phase: str = "", use_timestamp: bool = False):
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        log_dir : str
            Đường dẫn tài nguyên được sử dụng.
        experiment_name : str
            Tên hoặc khóa định danh của giá trị.
        phase : str, optional
            Giá trị ``phase`` được sử dụng trong phép xử lý.
        use_timestamp : bool, optional
            Giá trị ``use_timestamp`` được sử dụng trong phép xử lý.
        """
        # --- Tạo thư mục đầu ra ---
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

        # --- Thiết lập tệp CSV đầu ra ---
        self.csv_path = os.path.join(self.tb_dir, "metrics.csv")
        self._csv_file = None
        self._csv_writer = None
        self._csv_fields_written = False
        self._csv_fieldnames = []

        self._start_time = time.time()
        self._epoch_start_time = None

        print(f"[Logger] TensorBoard logs: {self.tb_dir}")
        print(f"[Logger] CSV metrics:      {self.csv_path}")

    def log_hyperparams(self, hparams: dict):
        """Thực hiện bước log hyperparams trong quy trình hiện tại.

        Parameters
        ----------
        hparams : dict
            Giá trị ``hparams`` được sử dụng trong phép xử lý.
        """
        # Chuẩn bị và ghi tài nguyên đầu ra theo định dạng yêu cầu.
        lines = [f"| {k} | {v} |" for k, v in hparams.items()]
        table = "| Parameter | Value |\n|---|---|\n" + "\n".join(lines)
        self.writer.add_text("hyperparameters", table, global_step=0)

        # Chuẩn bị và xử lý đầu vào hoặc đặc trưng văn bản.
        hparams_path = os.path.join(self.tb_dir, "hparams.txt")
        with open(hparams_path, "w", encoding="utf-8") as f:
            f.write(f"Experiment started: {datetime.now().isoformat()}\n")
            f.write("=" * 50 + "\n")
            for k, v in hparams.items():
                f.write(f"{k}: {v}\n")

        print(f"[Logger] Hyperparameters saved to {hparams_path}")

    def log_scalar(self, tag: str, value: float, step: int):
        """Thực hiện bước log scalar trong quy trình hiện tại.

        Parameters
        ----------
        tag : str
            Giá trị ``tag`` được sử dụng trong phép xử lý.
        value : float
            Giá trị ``value`` được sử dụng trong phép xử lý.
        step : int
            Giá trị ``step`` được sử dụng trong phép xử lý.
        """
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Thực hiện bước log scalars trong quy trình hiện tại.

        Parameters
        ----------
        main_tag : str
            Giá trị ``main_tag`` được sử dụng trong phép xử lý.
        tag_scalar_dict : dict
            Giá trị ``tag_scalar_dict`` được sử dụng trong phép xử lý.
        step : int
            Giá trị ``step`` được sử dụng trong phép xử lý.
        """
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_gpu_memory(self, step: int):
        """Thực hiện bước log gpu memory trong quy trình hiện tại.

        Parameters
        ----------
        step : int
            Giá trị ``step`` được sử dụng trong phép xử lý.
        """
        if torch.cuda.is_available():
            allocated_mb = torch.cuda.memory_allocated() / 1024 ** 2
            reserved_mb = torch.cuda.memory_reserved() / 1024 ** 2
            max_allocated_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
            self.writer.add_scalar("gpu/memory_allocated_mb", allocated_mb, step)
            self.writer.add_scalar("gpu/memory_reserved_mb", reserved_mb, step)
            self.writer.add_scalar("gpu/memory_peak_mb", max_allocated_mb, step)

    def _get_latest_checkpoint_size(self) -> float:
        """Lấy latest checkpoint size cho bước xử lý hiện tại.

        Returns
        -------
        float
            Kết quả được tạo bởi bước xử lý của hàm.
        """
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
            # Chuẩn bị và ghi tài nguyên đầu ra theo định dạng yêu cầu.
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
        """Thực hiện bước start epoch trong quy trình hiện tại."""
        self._epoch_start_time = time.time()

    def log_epoch_metrics(self, metrics: dict, epoch: int):
        """Thực hiện bước log epoch các độ đo trong quy trình hiện tại.

        Parameters
        ----------
        metrics : dict
            Giá trị ``metrics`` được sử dụng trong phép xử lý.
        epoch : int
            Giá trị ``epoch`` được sử dụng trong phép xử lý.
        """
        # --- Tính thời gian đã sử dụng ---
        elapsed = time.time() - self._start_time
        epoch_time = time.time() - self._epoch_start_time if self._epoch_start_time else 0

        enriched = {
            "epoch": epoch,
            "epoch_time_sec": round(epoch_time, 1),
            "total_time_sec": round(elapsed, 1),
            **metrics,
        }

        # Thiết lập trạng thái và thống kê các tham số mô hình.
        if hasattr(self, "_trainable_params") and self._trainable_params is not None:
            enriched["trainable_params"] = self._trainable_params

        # Kiểm tra và xử lý checkpoint tương ứng của mô hình.
        checkpoint_size_mb = self._get_latest_checkpoint_size()
        if checkpoint_size_mb > 0:
            enriched["checkpoint_size_mb"] = round(checkpoint_size_mb, 1)

        if torch.cuda.is_available():
            enriched["gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024 ** 2, 1)
            enriched["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)

        # --- Ghi các giá trị vô hướng vào TensorBoard ---
        for key, value in enriched.items():
            if key == "epoch":
                continue
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"epoch/{key}", value, epoch)

        # --- Thêm một dòng vào tệp CSV ---
        self._write_csv_row(enriched)

        # --- Xuất thông tin ra dòng lệnh ---
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
        """Ghi csv row cho bước xử lý hiện tại.

        Parameters
        ----------
        row : dict
            Giá trị ``row`` được sử dụng trong phép xử lý.
        """
        if self._csv_file is None:
            self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")

        if not self._csv_fields_written:
            self._csv_fieldnames = list(row.keys())
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=self._csv_fieldnames,
            )
            self._csv_writer.writeheader()
            self._csv_fields_written = True

        new_fields = [
            field for field in row.keys() if field not in self._csv_fieldnames
        ]
        if new_fields:
            # Bước hỗ trợ để ghi csv row cho bước xử lý hiện tại.
            # Thiết lập mô-đun dung hợp và đầu phân lớp theo cấu hình.
            # Chuẩn bị và ghi tài nguyên đầu ra theo định dạng yêu cầu.
            self._csv_file.flush()
            self._csv_file.close()
            with open(self.csv_path, "r", newline="", encoding="utf-8") as handle:
                existing_rows = list(csv.DictReader(handle))

            self._csv_fieldnames.extend(new_fields)
            self._csv_file = open(
                self.csv_path,
                "w",
                newline="",
                encoding="utf-8",
            )
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=self._csv_fieldnames,
            )
            self._csv_writer.writeheader()
            self._csv_writer.writerows(existing_rows)

        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def log_model_summary(self, model):
        """Thực hiện bước log mô hình summary trong quy trình hiện tại.

        Parameters
        ----------
        model : object
            Mô hình hoặc thành phần mô hình cần xử lý.
        """
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        
        self._trainable_params = trainable

        self.writer.add_scalar("model/total_params", total, 0)
        self.writer.add_scalar("model/trainable_params", trainable, 0)
        self.writer.add_scalar("model/frozen_params", frozen, 0)
        self.writer.add_scalar("model/trainable_ratio_pct", (trainable / total) * 100 if total > 0 else 0, 0)
        
        # Chuẩn bị và xử lý đầu vào hoặc đặc trưng văn bản.
        arch_path = os.path.join(self.tb_dir, "model_architecture.txt")
        try:
            with open(arch_path, "w", encoding="utf-8") as f:
                f.write(str(model))
        except Exception as e:
            print(f"[Logger] Warning: Could not save model architecture: {e}")

    def close(self):
        """Thực hiện bước close trong quy trình hiện tại."""
        self.writer.flush()
        self.writer.close()
        if self._csv_file:
            self._csv_file.close()
        elapsed = time.time() - self._start_time
        print(f"[Logger] Training completed in {elapsed:.1f}s. Logs saved to: {self.tb_dir}")


# ============================================================
# Thiết lập thành phần dùng chung cho quy trình xử lý của mô-đun.
# ============================================================

class XBoneTrainerCallback(TrainerCallback):
    """Điều phối quá trình huấn luyện bằng lớp ``XBoneTrainerCallback``.

    Notes
    -----
    Lớp này đóng gói trạng thái và hành vi để các thành phần khác có thể tái sử dụng nhất quán.
    """

    def __init__(self, logger: TrainingLogger):
        """Thực hiện bước init trong quy trình hiện tại.

        Parameters
        ----------
        logger : TrainingLogger
            Giá trị ``logger`` được sử dụng trong phép xử lý.
        """
        super().__init__()
        self.logger = logger
        self._current_epoch = 0

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Thực hiện bước on train begin trong quy trình hiện tại.

        Parameters
        ----------
        args : object
            Các đối số vị trí bổ sung.
        state : object
            Giá trị ``state`` được sử dụng trong phép xử lý.
        control : object
            Giá trị ``control`` được sử dụng trong phép xử lý.
        model : object, optional
            Mô hình hoặc thành phần mô hình cần xử lý.
        **kwargs : dict
            Các đối số từ khóa bổ sung.
        """
        if model is not None:
            self.logger.log_model_summary(model)

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Thực hiện bước on epoch begin trong quy trình hiện tại.

        Parameters
        ----------
        args : object
            Các đối số vị trí bổ sung.
        state : object
            Giá trị ``state`` được sử dụng trong phép xử lý.
        control : object
            Giá trị ``control`` được sử dụng trong phép xử lý.
        **kwargs : dict
            Các đối số từ khóa bổ sung.
        """
        self._current_epoch = int(state.epoch) if state.epoch else 0
        self.logger.start_epoch()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Thực hiện bước on log trong quy trình hiện tại.

        Parameters
        ----------
        args : object
            Các đối số vị trí bổ sung.
        state : object
            Giá trị ``state`` được sử dụng trong phép xử lý.
        control : object
            Giá trị ``control`` được sử dụng trong phép xử lý.
        logs : object, optional
            Giá trị ``logs`` được sử dụng trong phép xử lý.
        **kwargs : dict
            Các đối số từ khóa bổ sung.
        """
        if logs is None:
            return

        step = state.global_step
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.logger.log_scalar(f"train/{key}", value, step)

        self.logger.log_gpu_memory(step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Thực hiện bước on evaluate trong quy trình hiện tại.

        Parameters
        ----------
        args : object
            Các đối số vị trí bổ sung.
        state : object
            Giá trị ``state`` được sử dụng trong phép xử lý.
        control : object
            Giá trị ``control`` được sử dụng trong phép xử lý.
        metrics : object, optional
            Giá trị ``metrics`` được sử dụng trong phép xử lý.
        **kwargs : dict
            Các đối số từ khóa bổ sung.
        """
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
        """Thực hiện bước on train end trong quy trình hiện tại.

        Parameters
        ----------
        args : object
            Các đối số vị trí bổ sung.
        state : object
            Giá trị ``state`` được sử dụng trong phép xử lý.
        control : object
            Giá trị ``control`` được sử dụng trong phép xử lý.
        **kwargs : dict
            Các đối số từ khóa bổ sung.
        """
        self.logger.writer.flush()
