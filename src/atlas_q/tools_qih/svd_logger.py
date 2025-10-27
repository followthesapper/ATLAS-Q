import gzip
import json
import os
import threading
import time
from pathlib import Path


class _NullLogger:
    def log(self, **kwargs):
        pass


class _JSONLGZLogger:
    def __init__(self, out_dir: str):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        run = time.strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        self.path = self.dir / f"svd_events_{run}_{pid}.jsonl.gz"
        self._fh = gzip.open(self.path, "at", encoding="utf-8")
        self._lock = threading.Lock()
        self.meta = {
            "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
            "pid": pid,
            "run_ts": run,
        }

    def log(self, **kwargs):
        # Keep it lean; big tensors should be truncated upstream
        rec = {**self.meta, **kwargs}
        with self._lock:
            self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            self._fh.flush()


def get_logger():
    out = os.getenv("QIH_SVD_LOG_DIR")
    if not out:
        return _NullLogger()
    return _JSONLGZLogger(out)
