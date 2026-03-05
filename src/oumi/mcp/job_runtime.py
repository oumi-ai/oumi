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

"""Ephemeral per-job runtime state (in-memory only, never persisted)."""

import asyncio
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oumi.core.launcher.base_cluster import BaseCluster
from oumi.core.launcher.base_cluster import JobStatus as OumiJobStatus


@dataclass
class JobRuntime:
    """Ephemeral per-job state, lives only in memory, never persisted."""

    process: "subprocess.Popen[Any] | None" = None
    cluster_obj: BaseCluster | None = None
    oumi_status: OumiJobStatus | None = None
    stdout_f: Any = None
    stderr_f: Any = None
    log_dir: Path | None = None
    run_dir: Path | None = None
    staged_config_path: str = ""
    cancel_requested: bool = False
    error_message: str | None = None
    runner_task: asyncio.Task[Any] | None = None

    def close_log_files(self) -> None:
        """Close open stdout/stderr file handles."""
        for f in (self.stdout_f, self.stderr_f):
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
        self.stdout_f = None
        self.stderr_f = None


_runtimes: dict[str, JobRuntime] = {}
_runtimes_lock = asyncio.Lock()


async def get_runtime(job_id: str) -> JobRuntime:
    """Return the runtime for *job_id*, creating one if needed."""
    async with _runtimes_lock:
        if job_id not in _runtimes:
            _runtimes[job_id] = JobRuntime()
        return _runtimes[job_id]


async def evict_runtime(job_id: str) -> None:
    """Remove a runtime entry, closing any open handles."""
    async with _runtimes_lock:
        rt = _runtimes.pop(job_id, None)
    if rt is None:
        return
    rt.close_log_files()
