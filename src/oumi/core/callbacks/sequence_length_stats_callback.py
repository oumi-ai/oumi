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

"""Callback to track and report sequence length statistics during training."""

import math
import pathlib
from typing import Optional, Union

import torch
import transformers

from oumi.core.callbacks.base_trainer_callback import BaseTrainerCallback
from oumi.core.configs import TrainingParams
from oumi.core.distributed import get_device_rank_info, is_world_process_zero
from oumi.utils.io_utils import save_json
from oumi.utils.logging import logger

_LOGS_KWARG = "logs"


class SequenceLengthStatsCallback(BaseTrainerCallback):
    """Trainer callback to track and report sequence length statistics.

    This callback tracks sequence lengths during training and reports summary
    statistics (mean, std, min, max) at the end of training. The statistics
    are saved to a JSON file and logged to the console.

    Statistics are computed incrementally to minimize memory overhead.
    """

    def __init__(
        self,
        output_dir: Optional[pathlib.Path] = None,
        world_process_zero_only: bool = True,
    ):
        """Initializes the SequenceLengthStatsCallback.

        Args:
            output_dir: If specified, stats will be written to this directory as JSON.
            world_process_zero_only: Whether to collect stats on the main process only.
        """
        self._output_dir: Optional[pathlib.Path] = output_dir
        self._permanently_disabled: bool = (
            world_process_zero_only and not is_world_process_zero()
        )

        # Running statistics for incremental computation
        self._count: int = 0
        self._sum: float = 0.0
        self._sum_of_squares: float = 0.0
        self._min: float = float("inf")
        self._max: float = float("-inf")

    def on_log(
        self,
        args: Union[transformers.TrainingArguments, TrainingParams],
        state: Optional[transformers.TrainerState] = None,
        control: Optional[transformers.TrainerControl] = None,
        **kwargs,
    ):
        """Event called when logging happens.

        Extracts sequence length from logs and updates running statistics.
        """
        if self._permanently_disabled:
            return

        if _LOGS_KWARG not in kwargs:
            return

        logs = kwargs[_LOGS_KWARG]
        if not isinstance(logs, dict):
            return

        # Extract sequence length from logs
        sequence_length = logs.get("sequence_length")
        if sequence_length is None:
            return

        # Update running statistics
        self._update_stats(sequence_length)

    def _update_stats(self, value: float):
        """Updates running statistics with a new value.

        Args:
            value: The sequence length value to add to statistics.
        """
        self._count += 1
        self._sum += value
        self._sum_of_squares += value * value
        self._min = min(self._min, value)
        self._max = max(self._max, value)

    def _compute_statistics(self) -> Optional[dict[str, float]]:
        """Computes final statistics from running values.

        Returns:
            Dictionary with mean, std, min, max, count, or None if no data.
        """
        if self._count == 0:
            return None

        mean = self._sum / self._count

        # Compute standard deviation using Welford's method
        # Var = E[X^2] - (E[X])^2
        variance = (self._sum_of_squares / self._count) - (mean * mean)
        # Clamp to 0 to handle floating point errors
        variance = max(0.0, variance)
        std = math.sqrt(variance)

        return {
            "mean": mean,
            "std": std,
            "min": self._min,
            "max": self._max,
            "count": self._count,
            "total_tokens": self._sum,
        }

    def _aggregate_stats_across_ranks(
        self, local_stats: dict[str, float]
    ) -> dict[str, float]:
        """Aggregates statistics across all ranks in distributed training.

        Args:
            local_stats: Statistics from the local rank.

        Returns:
            Aggregated statistics across all ranks.
        """
        # Check if distributed training is active
        if not torch.distributed.is_initialized():
            return local_stats

        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

        # Gather statistics from all ranks
        # Format: [count, sum, sum_of_squares, min, max]
        local_tensor = torch.tensor(
            [
                local_stats["count"],
                local_stats["total_tokens"],
                self._sum_of_squares,
                local_stats["min"],
                local_stats["max"],
            ],
            dtype=torch.float64,
            device=device,
        )

        # All-reduce operations
        count_sum_tensor = local_tensor[:3].clone()
        torch.distributed.all_reduce(
            count_sum_tensor, op=torch.distributed.ReduceOp.SUM
        )

        min_tensor = local_tensor[3:4].clone()
        torch.distributed.all_reduce(min_tensor, op=torch.distributed.ReduceOp.MIN)

        max_tensor = local_tensor[4:5].clone()
        torch.distributed.all_reduce(max_tensor, op=torch.distributed.ReduceOp.MAX)

        # Compute global statistics
        global_count = int(count_sum_tensor[0].item())
        global_sum = count_sum_tensor[1].item()
        global_sum_of_squares = count_sum_tensor[2].item()
        global_min = min_tensor.item()
        global_max = max_tensor.item()

        if global_count == 0:
            return local_stats

        global_mean = global_sum / global_count
        global_variance = (global_sum_of_squares / global_count) - (
            global_mean * global_mean
        )
        global_variance = max(0.0, global_variance)
        global_std = math.sqrt(global_variance)

        return {
            "mean": global_mean,
            "std": global_std,
            "min": global_min,
            "max": global_max,
            "count": global_count,
            "total_tokens": global_sum,
        }

    def on_train_end(
        self,
        args: Union[transformers.TrainingArguments, TrainingParams],
        state: Optional[transformers.TrainerState] = None,
        control: Optional[transformers.TrainerControl] = None,
        **kwargs,
    ):
        """Event called at the end of training.

        Computes final statistics, aggregates across ranks if needed,
        and saves to file and logs to console.
        """
        if self._permanently_disabled:
            return

        # Compute local statistics
        local_stats = self._compute_statistics()

        if local_stats is None:
            if is_world_process_zero():
                logger.warning(
                    "No sequence length statistics collected during training. "
                    "This may indicate that sequence_length was not added to metrics."
                )
            return

        # Aggregate across ranks if distributed training
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            global_stats = self._aggregate_stats_across_ranks(local_stats)
        else:
            global_stats = local_stats

        # Only rank 0 saves and logs
        if not is_world_process_zero():
            return

        # Log to console
        logger.info("=" * 80)
        logger.info("Sequence Length Statistics:")
        logger.info(f"  Average: {global_stats['mean']:.2f} tokens")
        logger.info(f"  Std Dev: {global_stats['std']:.2f} tokens")
        logger.info(f"  Min: {global_stats['min']:.2f} tokens")
        logger.info(f"  Max: {global_stats['max']:.2f} tokens")
        logger.info(f"  Total Sequences: {global_stats['count']}")
        logger.info(f"  Total Tokens: {global_stats['total_tokens']:.0f}")
        logger.info("=" * 80)

        # Save to JSON file
        if self._output_dir:
            device_rank_info = get_device_rank_info()
            output_file = (
                self._output_dir
                / f"sequence_length_stats_rank{device_rank_info.rank:04}.json"
            )
            logger.info(f"Saving sequence length statistics to {output_file}...")
            save_json(global_stats, output_file)
