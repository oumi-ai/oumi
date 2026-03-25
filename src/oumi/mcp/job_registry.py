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

"""Job registry: persistence layer for job identity mapping."""

import dataclasses
import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from oumi.mcp.config_service import parse_yaml
from oumi.mcp.constants import DEFAULT_JOBS_FILE

logger = logging.getLogger(__name__)

_MAX_REGISTRY_AGE_DAYS = 7
_MAX_REGISTRY_SIZE = 200


@dataclass
class JobRecord:
    """Persisted job metadata (identity mapping only, no status).

    Status is always queried live. The ``cloud`` field disambiguates
    whether ``job_id`` is a SkyPilot ID or an MCP-generated ID.
    """

    job_id: str
    command: str
    config_path: str
    cloud: str
    cluster_name: str
    model_name: str
    submit_time: str  # ISO 8601
    output_dir: str = ""
    log_dir: str = ""


class JobRegistry:
    """Single-file JSON registry mapping MCP job IDs to cloud identities."""

    def __init__(self, path: Path) -> None:
        """Initialize the registry from *path*, loading existing records."""
        self._path = path
        self._jobs: dict[str, JobRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for entry in data:
                entry.pop("status", None)
                r = JobRecord(**entry)
                self._jobs[r.job_id] = r
        except Exception:
            logger.warning("Could not load %s, starting fresh", self._path)
        pruned = self._prune()
        if pruned:
            logger.info("Pruned %d stale job records from registry", pruned)
            self._save()

    def _prune(self) -> int:
        """Remove entries older than the age cutoff, then cap total size."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=_MAX_REGISTRY_AGE_DAYS)
        to_remove: list[str] = []
        for jid, rec in self._jobs.items():
            try:
                # Python 3.10 fromisoformat() doesn't accept the "Z" suffix.
                raw = rec.submit_time.replace("Z", "+00:00")
                ts = datetime.fromisoformat(raw)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < cutoff:
                    to_remove.append(jid)
            except (ValueError, TypeError):
                to_remove.append(jid)
        for jid in to_remove:
            del self._jobs[jid]
        removed = len(to_remove)

        if len(self._jobs) > _MAX_REGISTRY_SIZE:
            by_time = sorted(self._jobs.items(), key=lambda x: x[1].submit_time)
            while len(self._jobs) > _MAX_REGISTRY_SIZE and by_time:
                jid, _ = by_time.pop(0)
                del self._jobs[jid]
                removed += 1

        return removed

    def _save(self) -> None:
        records = [dataclasses.asdict(r) for r in self._jobs.values()]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self._path.parent),
            suffix=".tmp",
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(records, f, indent=2)
            Path(tmp_path).replace(self._path)
        except BaseException:
            Path(tmp_path).unlink()
            raise

    def add(self, record: JobRecord) -> None:
        """Add a job record and persist."""
        self._jobs[record.job_id] = record
        self._save()

    def update(self, job_id: str, **fields: Any) -> None:
        """Update fields on an existing record and persist."""
        record = self._jobs.get(job_id)
        if record is None:
            logger.warning("Registry.update: job_id %s not found, skipping", job_id)
            return
        for k, v in fields.items():
            setattr(record, k, v)
        self._save()

    def get(self, job_id: str) -> JobRecord | None:
        """Look up a record by job ID."""
        return self._jobs.get(job_id)

    def find_by_cloud(self, cloud: str, job_id: str) -> JobRecord | None:
        """Find a record by cloud provider and job ID."""
        for r in self._jobs.values():
            if r.cloud == cloud and r.job_id == job_id:
                return r
        return None

    def all(self) -> list[JobRecord]:
        """Return all records."""
        return list(self._jobs.values())

    def remove(self, job_id: str) -> None:
        """Remove a record by job ID and persist."""
        self._jobs.pop(job_id, None)
        self._save()


_registry: JobRegistry | None = None


def get_registry() -> JobRegistry:
    """Return the global ``JobRegistry``, creating it on first access."""
    global _registry
    if _registry is None:
        _registry = JobRegistry(DEFAULT_JOBS_FILE)
    return _registry


def reset_registry() -> None:
    """Reset the global registry singleton (for test teardown)."""
    global _registry
    _registry = None


def make_job_id(command: str, job_name: str | None = None) -> str:
    """Generate a human-friendly job ID.

    Format: ``{command}_{YYYYMMDD_HHMMSS}_{6-hex}`` or the caller-supplied
    *job_name* if provided (sanitized to prevent path traversal).
    """
    if job_name:
        sanitized = job_name.replace("..", "_").replace("/", "_").replace("\\", "_")
        sanitized = sanitized.strip("._- ")
        if not sanitized:
            raise ValueError(f"Invalid job_name after sanitization: {job_name!r}")
        return sanitized
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{command}_{ts}_{short}"


def extract_job_metadata(config_path: str) -> dict[str, Any]:
    """Extract model_name and output_dir from an Oumi YAML config.

    Returns a dict with ``model_name`` and ``output_dir`` keys.
    Missing values default to ``"unknown"`` / ``"./output"``.
    """
    config = parse_yaml(config_path)
    model_name = config.get("model", {}).get("model_name", "unknown") or "unknown"

    output_dir = (
        config.get("training", {}).get("output_dir")
        or config.get("output_dir")
        or "./output"
    )
    return {"model_name": model_name, "output_dir": output_dir}
