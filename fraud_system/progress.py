from __future__ import annotations

import sys
import time


class ProgressReporter:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.started_at = time.time()

    def _emit(self, message: str) -> None:
        if not self.enabled:
            return
        elapsed = time.time() - self.started_at
        print(f"[{elapsed:6.1f}s] {message}", file=sys.stderr, flush=True)

    def stage(self, message: str) -> None:
        self._emit(message)

    def progress(self, label: str, current: int, total: int) -> None:
        if total <= 0:
            self._emit(f"{label}: {current}")
            return
        percent = 100.0 * current / total
        self._emit(f"{label}: {current}/{total} ({percent:.1f}%)")
