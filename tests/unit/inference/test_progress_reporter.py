import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from oumi.core.inference import ProgressFileReporter


@pytest.fixture
def progress_path():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield str(Path(temp_dir) / "progress.json")


def _read_snapshot(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def test_start_writes_initial_snapshot(progress_path):
    reporter = ProgressFileReporter(progress_path, total=10)
    reporter.start()

    snapshot = _read_snapshot(progress_path)
    assert snapshot["total"] == 10
    assert snapshot["completed"] == 0
    assert snapshot["failed"] == 0
    assert "updated_at" in snapshot


def test_start_with_resumed_counts(progress_path):
    reporter = ProgressFileReporter(progress_path, total=10)
    reporter.start(completed=4, failed=1)

    snapshot = _read_snapshot(progress_path)
    assert snapshot["completed"] == 4
    assert snapshot["failed"] == 1


def test_finalize_writes_final_counts(progress_path):
    reporter = ProgressFileReporter(progress_path, total=3)
    reporter.start()
    reporter.record_completed()
    reporter.record_completed()
    reporter.record_failed()
    reporter.finalize()

    snapshot = _read_snapshot(progress_path)
    assert snapshot["total"] == 3
    assert snapshot["completed"] == 2
    assert snapshot["failed"] == 1
    assert snapshot["completed"] + snapshot["failed"] == snapshot["total"]


def test_record_batch_counts(progress_path):
    reporter = ProgressFileReporter(progress_path, total=10)
    reporter.start()
    reporter.record_completed(n=7)
    reporter.record_failed(n=3)
    reporter.finalize()

    snapshot = _read_snapshot(progress_path)
    assert snapshot["completed"] == 7
    assert snapshot["failed"] == 3


def test_writes_throttled_between_ticks(progress_path):
    reporter = ProgressFileReporter(progress_path, total=100, min_write_interval=60.0)
    reporter.start()
    # Ticks within the interval should not rewrite the file.
    reporter.record_completed()
    reporter.record_completed()

    snapshot = _read_snapshot(progress_path)
    assert snapshot["completed"] == 0

    # finalize bypasses the throttle.
    reporter.finalize()
    snapshot = _read_snapshot(progress_path)
    assert snapshot["completed"] == 2


def test_writes_not_throttled_after_interval(progress_path):
    reporter = ProgressFileReporter(progress_path, total=100, min_write_interval=0.0)
    reporter.start()
    reporter.record_completed()

    snapshot = _read_snapshot(progress_path)
    assert snapshot["completed"] == 1


def test_atomic_write_leaves_no_temp_file(progress_path):
    reporter = ProgressFileReporter(progress_path, total=1, min_write_interval=0.0)
    reporter.start()
    reporter.record_completed()
    reporter.finalize()

    parent = Path(progress_path).parent
    assert [p.name for p in parent.iterdir()] == ["progress.json"]


def test_creates_parent_directories(progress_path):
    nested = str(Path(progress_path).parent / "a" / "b" / "progress.json")
    reporter = ProgressFileReporter(nested, total=1)
    reporter.start()

    assert _read_snapshot(nested)["total"] == 1


def test_write_failure_never_raises(progress_path):
    reporter = ProgressFileReporter(progress_path, total=2, min_write_interval=0.0)
    # Patch Path.replace (what _write_snapshot actually calls). Patching
    # os.replace would miss the failure on Python 3.10, where Path.replace
    # does not route through os.replace.
    with patch.object(Path, "replace", side_effect=OSError("disk full")):
        reporter.start()
        reporter.record_completed()
        reporter.record_failed()
        reporter.finalize()

    assert not os.path.exists(progress_path)


def test_write_failure_warns_once(progress_path):
    reporter = ProgressFileReporter(progress_path, total=2, min_write_interval=0.0)
    with patch("oumi.core.inference.progress_reporter.logger.warning") as mock_warning:
        with patch.object(Path, "replace", side_effect=OSError("disk full")):
            reporter.start()
            reporter.record_completed()
            reporter.finalize()

    assert mock_warning.call_count == 1


def test_recovers_after_transient_write_failure(progress_path):
    reporter = ProgressFileReporter(progress_path, total=2, min_write_interval=0.0)
    with patch.object(Path, "replace", side_effect=OSError("disk full")):
        reporter.start()
    reporter.record_completed()
    reporter.finalize()

    snapshot = _read_snapshot(progress_path)
    assert snapshot["completed"] == 1
