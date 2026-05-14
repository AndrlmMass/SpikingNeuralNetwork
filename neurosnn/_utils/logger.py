from dataclasses import dataclass, field
from datetime import datetime
import json
import os


@dataclass
class HistoryTracker:
    ts_spec: str
    image_dataset: str = "unknown"
    acc_history: dict = field(default_factory=dict)
    _acc_log_dir: str = None
    _acc_log_file: str = None

    def _ensure_acc_logger(self):
        if self._acc_log_file is None:
            from datetime import datetime

            self.image_dataset = getattr(self, "image_dataset", "unknown")
            date = datetime.now().strftime("%Y.%m.%d")
            try:
                self._acc_log_dir = os.path.join(
                    "results",
                    "acc_history",
                    f"{self.image_dataset}",
                    f"{date}",
                    f"acc_{self.ts_spec}",
                )
                os.makedirs(self._acc_log_dir, exist_ok=True)
            except Exception:
                pass
            self._acc_log_file = os.path.join(
                self._acc_log_dir, f"acc_{self.ts_spec}.jsonl"
            )

    def _record_accuracy(
        self, split: str, value, epoch: int | None = None, method: str | None = None
    ):
        try:
            acc_val = float(value) if value is not None else None
        except Exception:
            acc_val = None
        try:
            if acc_val is not None:
                method_key = method if method is not None else "default"
                if split not in self.acc_history:
                    self.acc_history[split] = {}
                if method_key not in self.acc_history[split]:
                    self.acc_history[split][method_key] = []
                self.acc_history[split][method_key].append(acc_val)
        except Exception:
            pass
        try:
            self._ensure_acc_logger()
            rec = {
                "timestamp": datetime.now().isoformat(),
                "split": str(split),
                "epoch": int(epoch) if epoch is not None else None,
                "accuracy": acc_val,
            }
            if method is not None:
                rec["method"] = method
            with open(self._acc_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
        except Exception as e:
            print(f"Warning: failed to persist accuracy record ({e})")

    def _record_phi(self, split: str, phi_value, epoch: int | None = None):
        try:
            phi_val = float(phi_value) if phi_value is not None else None
        except Exception:
            phi_val = None
        try:
            self._ensure_acc_logger()
            rec = {
                "timestamp": datetime.now().isoformat(),
                "split": str(split),
                "epoch": int(epoch) if epoch is not None else None,
                "phi": phi_val,
            }
            with open(self._acc_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
        except Exception as e:
            print(f"Warning: failed to persist phi record ({e})")

    def _record_mcc(self, split: str, mcc_value, epoch: int | None = None):
        try:
            mcc_val = float(mcc_value) if mcc_value is not None else None
        except Exception:
            mcc_val = None
        try:
            self._ensure_acc_logger()
            rec = {
                "timestamp": datetime.now().isoformat(),
                "split": str(split),
                "epoch": int(epoch) if epoch is not None else None,
                "mcc": mcc_val,
            }
            with open(self._acc_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
        except Exception as e:
            print(f"Warning: failed to persist mcc record ({e})")

    def _read_jsonl(self, path):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
