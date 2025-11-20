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

import math
import pathlib
import tempfile
from unittest.mock import patch

from oumi.core.callbacks.sequence_length_stats_callback import (
    SequenceLengthStatsCallback,
)
from oumi.core.configs import TrainingParams


def test_callback_initialization():
    """Test that callback initializes with correct default values."""
    callback = SequenceLengthStatsCallback()
    assert callback._count == 0
    assert callback._sum == 0.0
    assert callback._sum_of_squares == 0.0
    assert callback._min == float("inf")
    assert callback._max == float("-inf")
    assert callback._output_dir is None


def test_callback_initialization_with_output_dir():
    """Test that callback initializes with output directory."""
    output_dir = pathlib.Path("/tmp/test")
    callback = SequenceLengthStatsCallback(output_dir=output_dir)
    assert callback._output_dir == output_dir


def test_update_stats():
    """Test that statistics are updated correctly."""
    callback = SequenceLengthStatsCallback()

    # Add first value
    callback._update_stats(10.0)
    assert callback._count == 1
    assert callback._sum == 10.0
    assert callback._sum_of_squares == 100.0
    assert callback._min == 10.0
    assert callback._max == 10.0

    # Add second value
    callback._update_stats(20.0)
    assert callback._count == 2
    assert callback._sum == 30.0
    assert callback._sum_of_squares == 500.0  # 100 + 400
    assert callback._min == 10.0
    assert callback._max == 20.0

    # Add third value
    callback._update_stats(5.0)
    assert callback._count == 3
    assert callback._sum == 35.0
    assert callback._sum_of_squares == 525.0  # 100 + 400 + 25
    assert callback._min == 5.0
    assert callback._max == 20.0


def test_compute_statistics_no_data():
    """Test that compute_statistics returns None when no data collected."""
    callback = SequenceLengthStatsCallback()
    stats = callback._compute_statistics()
    assert stats is None


def test_compute_statistics_single_value():
    """Test statistics computation with single value."""
    callback = SequenceLengthStatsCallback()
    callback._update_stats(15.0)

    stats = callback._compute_statistics()
    assert stats is not None
    assert stats["mean"] == 15.0
    assert stats["std"] == 0.0  # Single value has no variation
    assert stats["min"] == 15.0
    assert stats["max"] == 15.0
    assert stats["count"] == 1
    assert stats["total_tokens"] == 15.0


def test_compute_statistics_multiple_values():
    """Test statistics computation with multiple values."""
    callback = SequenceLengthStatsCallback()
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    for value in values:
        callback._update_stats(value)

    stats = callback._compute_statistics()
    assert stats is not None

    # Mean should be 30
    assert abs(stats["mean"] - 30.0) < 1e-6

    # Standard deviation of [10, 20, 30, 40, 50] is sqrt(200) ≈ 14.142
    expected_std = math.sqrt(200.0)
    assert abs(stats["std"] - expected_std) < 1e-6

    assert stats["min"] == 10.0
    assert stats["max"] == 50.0
    assert stats["count"] == 5
    assert stats["total_tokens"] == 150.0


def test_on_log_no_logs():
    """Test on_log when no logs are provided."""
    callback = SequenceLengthStatsCallback()
    args = TrainingParams()

    # Call without logs kwarg
    callback.on_log(args, None, None)
    assert callback._count == 0

    # Call with empty logs
    callback.on_log(args, None, None, logs={})
    assert callback._count == 0


def test_on_log_no_sequence_length():
    """Test on_log when logs don't contain sequence_length."""
    callback = SequenceLengthStatsCallback()
    args = TrainingParams()

    logs = {"loss": 1.5, "learning_rate": 0.001}
    callback.on_log(args, None, None, logs=logs)
    assert callback._count == 0


def test_on_log_with_sequence_length():
    """Test on_log when logs contain sequence_length."""
    callback = SequenceLengthStatsCallback()
    args = TrainingParams()

    # First log
    logs1 = {"sequence_length": 128.0, "loss": 1.5}
    callback.on_log(args, None, None, logs=logs1)
    assert callback._count == 1
    assert callback._sum == 128.0

    # Second log
    logs2 = {"sequence_length": 256.0, "loss": 1.2}
    callback.on_log(args, None, None, logs=logs2)
    assert callback._count == 2
    assert callback._sum == 384.0


@patch("oumi.core.callbacks.sequence_length_stats_callback.is_world_process_zero")
@patch("oumi.core.callbacks.sequence_length_stats_callback.logger")
def test_on_train_end_no_data(mock_logger, mock_is_zero):
    """Test on_train_end when no data was collected."""
    mock_is_zero.return_value = True
    callback = SequenceLengthStatsCallback()
    args = TrainingParams()

    callback.on_train_end(args, None, None)

    # Should log a warning
    mock_logger.warning.assert_called_once()
    assert "No sequence length statistics" in str(mock_logger.warning.call_args)


@patch("oumi.core.callbacks.sequence_length_stats_callback.is_world_process_zero")
@patch("oumi.core.callbacks.sequence_length_stats_callback.save_json")
@patch("oumi.core.callbacks.sequence_length_stats_callback.logger")
@patch(
    "oumi.core.callbacks.sequence_length_stats_callback.torch.distributed.is_initialized"
)
def test_on_train_end_with_data(
    mock_dist_init, mock_logger, mock_save_json, mock_is_zero
):
    """Test on_train_end with collected data."""
    mock_is_zero.return_value = True
    mock_dist_init.return_value = False  # No distributed training

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = pathlib.Path(tmpdir)
        callback = SequenceLengthStatsCallback(output_dir=output_dir)

        # Add some data
        callback._update_stats(100.0)
        callback._update_stats(200.0)
        callback._update_stats(150.0)

        args = TrainingParams()
        callback.on_train_end(args, None, None)

        # Check that save_json was called
        mock_save_json.assert_called_once()
        saved_stats = mock_save_json.call_args[0][0]

        assert abs(saved_stats["mean"] - 150.0) < 1e-6
        assert saved_stats["min"] == 100.0
        assert saved_stats["max"] == 200.0
        assert saved_stats["count"] == 3

        # Check that logger was called with summary
        logger_calls = [str(call) for call in mock_logger.info.call_args_list]
        assert any("Sequence Length Statistics" in str(call) for call in logger_calls)
        assert any("Average" in str(call) for call in logger_calls)


@patch("oumi.core.callbacks.sequence_length_stats_callback.is_world_process_zero")
def test_on_train_end_not_rank_zero(mock_is_zero):
    """Test that non-zero ranks don't save files."""
    mock_is_zero.return_value = False

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = pathlib.Path(tmpdir)
        callback = SequenceLengthStatsCallback(
            output_dir=output_dir,
            world_process_zero_only=False,  # Don't disable callback
        )

        # Add data
        callback._update_stats(100.0)

        args = TrainingParams()

        with patch(
            "oumi.core.callbacks.sequence_length_stats_callback.save_json"
        ) as mock_save:
            with patch(
                "oumi.core.callbacks.sequence_length_stats_callback.torch.distributed.is_initialized"
            ) as mock_init:
                mock_init.return_value = False
                callback.on_train_end(args, None, None)

                # Should not save since not rank 0
                mock_save.assert_not_called()


@patch(
    "oumi.core.callbacks.sequence_length_stats_callback.torch.distributed.is_initialized"
)
@patch(
    "oumi.core.callbacks.sequence_length_stats_callback.torch.distributed.get_world_size"
)
@patch(
    "oumi.core.callbacks.sequence_length_stats_callback.torch.distributed.all_reduce"
)
@patch("oumi.core.callbacks.sequence_length_stats_callback.torch.cuda.is_available")
def test_aggregate_stats_across_ranks(
    mock_cuda, mock_all_reduce, mock_world_size, mock_dist_init
):
    """Test aggregation of statistics across multiple ranks."""
    mock_dist_init.return_value = True
    mock_world_size.return_value = 2
    mock_cuda.return_value = False  # Use CPU to avoid MPS/CUDA issues

    # Mock the all_reduce operation to simulate aggregation
    def mock_reduce(tensor, op):
        # Simulate aggregating from 2 ranks with identical stats
        if len(tensor) == 3:  # count, sum, sum_of_squares
            tensor[0] *= 2  # Double the count
            tensor[1] *= 2  # Double the sum
            tensor[2] *= 2  # Double the sum of squares
        # For min/max, keep the same
        return tensor

    mock_all_reduce.side_effect = mock_reduce

    callback = SequenceLengthStatsCallback()
    callback._update_stats(100.0)

    local_stats = callback._compute_statistics()
    assert local_stats is not None  # Type guard for pyright
    global_stats = callback._aggregate_stats_across_ranks(local_stats)

    # After aggregation from 2 ranks, count should be doubled
    assert global_stats["count"] == 2


def test_compute_statistics_variance_edge_case():
    """Test that variance is clamped to zero for floating point errors."""
    callback = SequenceLengthStatsCallback()

    # Add values that could cause small negative variance due to floating point errors
    callback._update_stats(10.0)
    callback._update_stats(10.0)
    callback._update_stats(10.0)

    stats = callback._compute_statistics()
    assert stats is not None
    assert stats["std"] >= 0.0  # Should never be negative
    assert abs(stats["std"]) < 1e-10  # Should be very close to zero


@patch("oumi.core.callbacks.sequence_length_stats_callback.is_world_process_zero")
def test_world_process_zero_only_flag(mock_is_zero):
    """Test that world_process_zero_only flag disables callback correctly."""
    mock_is_zero.return_value = False  # Not rank 0

    callback = SequenceLengthStatsCallback(world_process_zero_only=True)
    assert callback._permanently_disabled is True

    # on_log should do nothing when disabled
    args = TrainingParams()
    logs = {"sequence_length": 128.0}
    callback.on_log(args, None, None, logs=logs)
    assert callback._count == 0  # Should not update

    # on_train_end should do nothing when disabled
    callback.on_train_end(args, None, None)  # Should not raise any errors
