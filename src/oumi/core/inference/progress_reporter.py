# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from oumi.utils.logging import logger


class ProgressFileReporter:
    """Writes inference progress counters to a JSON file for external pollers.

    The snapshot format is::

        {"total": N, "completed": n, "failed": f, "updated_at": "<iso8601 utc>"}

    The run is complete when ``completed + failed == total``. Writes are atomic
    (temp file + ``os.replace``), so a polling process never observes partial
    JSON, and are throttled to at most one per ``min_write_interval`` seconds
    (``start()`` and ``finalize()`` always write).

    Filesystem failures are logged and swallowed: a broken progress path must
    never kill inference. Counter updates are thread-safe.
    """

    def __init__(self, path: str, total: int, min_write_interval: float = 1.0):
        """Initializes the reporter.

        Args:
            path: Destination file for the JSON snapshot.
            total: Total number of rows in the run.
            min_write_interval: Minimum seconds between snapshot writes.
        """
        self._path = Path(path)
        self._tmp_path = self._path.with_name(self._path.name + ".tmp")
        self._total = total
        self._min_write_interval = min_write_interval
        self._completed = 0
        self._failed = 0
        self._last_write_time = 0.0
        self._warned = False
        self._lock = threading.Lock()

    def start(self, completed: int = 0, failed: int = 0) -> None:
        """Initializes counters and writes the first snapshot."""
        with self._lock:
            self._completed = completed
            self._failed = failed
            self._write_snapshot()

    def record_completed(self, n: int = 1) -> None:
        """Records n successfully completed rows."""
        with self._lock:
            self._completed += n
            self._maybe_write_snapshot()

    def record_failed(self, n: int = 1) -> None:
        """Records n failed rows."""
        with self._lock:
            self._failed += n
            self._maybe_write_snapshot()

    def finalize(self) -> None:
        """Writes a final snapshot, bypassing the throttle."""
        with self._lock:
            self._write_snapshot()

    def _maybe_write_snapshot(self) -> None:
        if time.monotonic() - self._last_write_time >= self._min_write_interval:
            self._write_snapshot()

    def _write_snapshot(self) -> None:
        self._last_write_time = time.monotonic()
        snapshot = {
            "total": self._total,
            "completed": self._completed,
            "failed": self._failed,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temp file in the same directory so the replace is atomic.
            with open(self._tmp_path, "w") as f:
                json.dump(snapshot, f)
            self._tmp_path.replace(self._path)
        except Exception as e:
            if not self._warned:
                logger.warning(f"Failed to write inference progress file: {e}")
                self._warned = True
            else:
                logger.debug(f"Failed to write inference progress file: {e}")
