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
import pathlib
from unittest.mock import patch

import pytest

from oumi.core.callbacks.metrics_logger_callback import MetricsLoggerCallback


@pytest.fixture
def temp_output_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Creates a temporary output directory for tests."""
    return tmp_path / "metrics_output"


class TestMetricsLoggerCallback:
    """Tests for MetricsLoggerCallback."""

    def test_init(self, temp_output_dir: pathlib.Path):
        """Tests that the callback initializes correctly."""
        callback = MetricsLoggerCallback(output_dir=temp_output_dir)
        assert callback._output_dir == temp_output_dir
        assert callback._metrics_log_file is None

    @patch("oumi.core.callbacks.metrics_logger_callback.is_world_process_zero")
    def test_on_log_writes_metrics(
        self, mock_is_world_zero, temp_output_dir: pathlib.Path
    ):
        """Tests that on_log writes metrics to JSONL file."""
        mock_is_world_zero.return_value = True

        callback = MetricsLoggerCallback(output_dir=temp_output_dir)
        metrics = {"loss": 1.5, "learning_rate": 0.001, "epoch": 1.0}

        callback.on_log(args=None, state=None, control=None, logs=metrics)

        # Verify file was created
        assert callback._metrics_log_file is not None
        assert callback._metrics_log_file.exists()

        # Verify content
        with open(callback._metrics_log_file) as f:
            lines = f.readlines()
        assert len(lines) == 1
        logged_metrics = json.loads(lines[0])
        assert logged_metrics == metrics

    @patch("oumi.core.callbacks.metrics_logger_callback.is_world_process_zero")
    def test_on_log_appends_multiple_entries(
        self, mock_is_world_zero, temp_output_dir: pathlib.Path
    ):
        """Tests that multiple on_log calls append to the same file."""
        mock_is_world_zero.return_value = True

        callback = MetricsLoggerCallback(output_dir=temp_output_dir)

        metrics_list = [
            {"loss": 2.0, "step": 100},
            {"loss": 1.5, "step": 200},
            {"loss": 1.0, "step": 300},
        ]

        for metrics in metrics_list:
            callback.on_log(args=None, state=None, control=None, logs=metrics)

        # Verify all entries were written
        assert callback._metrics_log_file is not None
        with open(callback._metrics_log_file) as f:
            lines = f.readlines()
        assert len(lines) == 3

        for i, line in enumerate(lines):
            logged_metrics = json.loads(line)
            assert logged_metrics == metrics_list[i]

    @patch("oumi.core.callbacks.metrics_logger_callback.is_world_process_zero")
    def test_on_log_skipped_for_non_zero_rank(
        self, mock_is_world_zero, temp_output_dir: pathlib.Path
    ):
        """Tests that on_log does nothing for non-zero ranks."""
        mock_is_world_zero.return_value = False

        callback = MetricsLoggerCallback(output_dir=temp_output_dir)
        metrics = {"loss": 1.5}

        callback.on_log(args=None, state=None, control=None, logs=metrics)

        # File should not be created
        assert callback._metrics_log_file is None
        assert not temp_output_dir.exists()

    @patch("oumi.core.callbacks.metrics_logger_callback.is_world_process_zero")
    def test_on_log_skipped_without_logs_kwarg(
        self, mock_is_world_zero, temp_output_dir: pathlib.Path
    ):
        """Tests that on_log does nothing when logs kwarg is missing."""
        mock_is_world_zero.return_value = True

        callback = MetricsLoggerCallback(output_dir=temp_output_dir)

        # Call without logs kwarg
        callback.on_log(args=None, state=None, control=None)

        # File should not be created
        assert callback._metrics_log_file is None

    @patch("oumi.core.callbacks.metrics_logger_callback.is_world_process_zero")
    @patch("oumi.core.callbacks.metrics_logger_callback.get_device_rank_info")
    def test_on_log_creates_output_directory(
        self, mock_rank_info, mock_is_world_zero, temp_output_dir: pathlib.Path
    ):
        """Tests that on_log creates the output directory if it doesn't exist."""
        mock_is_world_zero.return_value = True
        mock_rank_info.return_value.rank = 0

        callback = MetricsLoggerCallback(output_dir=temp_output_dir)

        assert not temp_output_dir.exists()

        callback.on_log(args=None, state=None, control=None, logs={"loss": 1.0})

        assert temp_output_dir.exists()

    @patch("oumi.core.callbacks.metrics_logger_callback.is_world_process_zero")
    @patch("oumi.core.callbacks.metrics_logger_callback.get_device_rank_info")
    def test_metrics_file_naming(
        self, mock_rank_info, mock_is_world_zero, temp_output_dir: pathlib.Path
    ):
        """Tests that metrics file is named correctly with rank."""
        mock_is_world_zero.return_value = True
        mock_rank_info.return_value.rank = 3

        callback = MetricsLoggerCallback(output_dir=temp_output_dir)
        callback.on_log(args=None, state=None, control=None, logs={"loss": 1.0})

        expected_file = temp_output_dir / "metrics_rank0003.jsonl"
        assert callback._metrics_log_file == expected_file

    @patch("oumi.core.callbacks.metrics_logger_callback.is_world_process_zero")
    def test_on_log_handles_non_serializable_values(
        self, mock_is_world_zero, temp_output_dir: pathlib.Path
    ):
        """Tests that on_log handles non-JSON-serializable values using default=str."""
        mock_is_world_zero.return_value = True

        callback = MetricsLoggerCallback(output_dir=temp_output_dir)
        metrics = {
            "loss": 1.5,
            "path": pathlib.Path("/some/path"),  # Not JSON serializable by default
        }

        # Should not raise
        callback.on_log(args=None, state=None, control=None, logs=metrics)

        assert callback._metrics_log_file is not None
        with open(callback._metrics_log_file) as f:
            logged = json.loads(f.readline())

        assert logged["loss"] == 1.5
        assert logged["path"] == "/some/path"  # Converted to string
