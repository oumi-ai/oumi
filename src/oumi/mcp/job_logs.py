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

"""Job log retrieval: local file tailing and cloud log streaming."""

import asyncio
import io
import logging
from collections.abc import AsyncIterator
from pathlib import Path

import oumi.launcher as launcher
from oumi.mcp.constants import LOG_TAIL_INTERVAL_SECONDS
from oumi.mcp.job_registry import JobRecord
from oumi.mcp.job_runtime import JobRuntime

logger = logging.getLogger(__name__)

_CLOUD_LOG_TIMEOUT = 30.0


def get_log_paths(record: JobRecord, rt: JobRuntime) -> dict[str, Path | None]:
    """Return paths to the stdout and stderr log files for a job.

    Returns a dict with ``"stdout"`` and ``"stderr"`` keys, each
    mapping to a ``Path`` or ``None`` if the file doesn't exist yet.
    Falls back to ``record.log_dir`` when the runtime has been evicted.
    """
    result: dict[str, Path | None] = {"stdout": None, "stderr": None}
    log_dir = rt.log_dir
    if log_dir is None and record.log_dir:
        log_dir = Path(record.log_dir)
    if log_dir is None or not log_dir.is_dir():
        return result

    id_candidates = [record.job_id]

    for suffix in ("stdout", "stderr"):
        for candidate_id in id_candidates:
            matches = sorted(log_dir.glob(f"*_{candidate_id}.{suffix}"))
            if matches:
                result[suffix] = matches[-1]
                break
        else:
            matches = sorted(log_dir.glob(f"*.{suffix}"))
            if matches:
                result[suffix] = matches[-1]

    return result


async def tail_log_file(
    path: Path,
    done_event: asyncio.Event,
    poll_interval: float = LOG_TAIL_INTERVAL_SECONDS,
) -> AsyncIterator[str]:
    """Async generator that yields new lines from *path* as they appear.

    Behaves like ``tail -f``: opens the file, seeks to the current end,
    then yields new complete lines as they are written.  Stops when
    *done_event* is set **and** no more data is available.

    If the file does not exist yet, waits up to ``poll_interval`` between
    checks until it appears or *done_event* fires.
    """
    while not path.exists():
        if done_event.is_set():
            return
        await asyncio.sleep(poll_interval)

    position = 0
    partial = ""

    while True:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0

        if size > position:
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    f.seek(position)
                    chunk = f.read()
                    position = f.tell()
            except OSError:
                chunk = ""

            if chunk:
                partial += chunk
                while "\n" in partial:
                    line, partial = partial.split("\n", 1)
                    yield line

        if done_event.is_set():
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    f.seek(position)
                    remaining = f.read()
            except OSError:
                remaining = ""
            if remaining:
                partial += remaining
            if partial:
                yield partial
            return

        await asyncio.sleep(poll_interval)


async def get_cloud_logs(
    record: JobRecord,
    rt: JobRuntime,
    lines: int = 200,
) -> tuple[str, int] | None:
    """Fetch the last *lines* of logs from a cloud job.

    Uses ``cluster.get_logs_stream()`` which calls ``sky.tail_logs(follow=True)``.
    Since ``follow=True`` means the stream never ends for running jobs, we read
    with a timeout and return whatever was accumulated (partial logs are better
    than nothing).
    """
    cluster = rt.cluster_obj

    if cluster is None and record.cloud and record.cluster_name:
        try:
            cloud_obj = await asyncio.to_thread(launcher.get_cloud, record.cloud)
            cluster = await asyncio.to_thread(
                cloud_obj.get_cluster, record.cluster_name
            )
        except Exception:
            logger.debug(
                "Failed to reconstruct cluster for %s/%s",
                record.cloud,
                record.cluster_name,
                exc_info=True,
            )
            return None

    if cluster is None:
        logger.debug(
            "Cluster %s/%s not found (may have been deleted)",
            record.cloud,
            record.cluster_name,
        )
        return None

    try:
        stream: io.TextIOBase = await asyncio.to_thread(
            cluster.get_logs_stream,
            record.cluster_name,
            record.job_id or None,
        )
    except NotImplementedError:
        logger.debug(
            "Cloud %s does not support get_logs_stream for job %s",
            record.cloud,
            record.job_id,
        )
        return None
    except Exception:
        logger.debug(
            "get_logs_stream failed for job %s",
            record.job_id,
            exc_info=True,
        )
        return None

    chunks: list[str] = []

    def _read_stream() -> str:
        try:
            while True:
                line = stream.readline()
                if not line:
                    break
                chunks.append(line)
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass
        return "".join(chunks)

    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(_read_stream),
            timeout=_CLOUD_LOG_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raw = "".join(chunks)
        if raw:
            logger.debug(
                "Cloud log read timed out for job %s after %.0fs, "
                "returning %d partial lines",
                record.job_id,
                _CLOUD_LOG_TIMEOUT,
                raw.count("\n"),
            )

    if not raw:
        return None

    all_lines = raw.splitlines()
    tail = all_lines[-lines:] if lines > 0 else all_lines
    return ("\n".join(tail), len(tail))


async def stream_cloud_logs(
    record: JobRecord,
    rt: JobRuntime,
    done_event: asyncio.Event,
) -> AsyncIterator[str]:
    """Yield log lines from ``cluster.get_logs_stream()`` for cloud jobs.

    Falls back silently (returns without yielding) if the cluster does not
    support log streaming (raises ``NotImplementedError``).
    """
    cluster = rt.cluster_obj
    if cluster is None:
        return

    try:
        stream: io.TextIOBase = await asyncio.to_thread(
            cluster.get_logs_stream,
            record.cluster_name,
            record.job_id or None,
        )
    except NotImplementedError:
        logger.debug(
            "Cloud %s does not support get_logs_stream for job %s",
            record.cloud,
            record.job_id,
        )
        return
    except Exception:
        logger.debug(
            "get_logs_stream failed for job %s",
            record.job_id,
            exc_info=True,
        )
        return

    def _read_lines() -> list[str]:
        """Read available lines from the stream (blocking)."""
        lines: list[str] = []
        try:
            while True:
                line = stream.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n"))
        except Exception:
            pass
        return lines

    try:
        while not done_event.is_set():
            lines = await asyncio.to_thread(_read_lines)
            for line in lines:
                yield line
            if not lines:
                await asyncio.sleep(LOG_TAIL_INTERVAL_SECONDS)
    finally:
        try:
            stream.close()
        except Exception:
            pass


def read_log_tail(stdout_path: Path, lines: int) -> tuple[str, int]:
    """Read the trailing *lines* from *stdout_path* efficiently."""
    lines = min(lines, 10000)
    if lines <= 0:
        return ("", 0)
    block_size = 8192
    data = b""
    newline_count = 0

    with stdout_path.open("rb") as f:
        pos = f.seek(0, 2)
        while pos > 0 and newline_count <= lines:
            read_size = min(block_size, pos)
            pos -= read_size
            f.seek(pos)
            chunk = f.read(read_size)
            data = chunk + data
            newline_count = data.count(b"\n")

    text = data.decode("utf-8", errors="replace")
    all_lines = text.splitlines()
    if not all_lines:
        return ("", 0)
    tail_lines = all_lines[-lines:]
    return ("\n".join(tail_lines), len(tail_lines))
