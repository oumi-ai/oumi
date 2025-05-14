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

"""Collects sub-step/step/epoch timings."""

import copy
import pathlib
import sys
from pprint import pformat
from typing import Optional, Union

import transformers

import wandb  # isort: skip
from oumi.core.callbacks.base_trainer_callback import BaseTrainerCallback
from oumi.core.configs import TrainingParams
from oumi.core.distributed import get_device_rank_info, is_world_process_zero
from oumi.performance.telemetry import TelemetryTracker, TimerContext
from oumi.utils.device_utils import (
    log_nvidia_gpu_runtime_info,
)
from oumi.utils.io_utils import save_json
from oumi.utils.logging import logger

_LOGS_KWARG = "logs"


class TelemetryCallback(BaseTrainerCallback):
    """Trainer callback to collect sub-step/step/epoch timings.

    Based on `oumi.performance.telemetry.TelemetryTracker`.
    """

    def __init__(
        self,
        skip_first_steps: int = 1,
        world_process_zero_only: bool = True,
        include_timer_metrics: bool = False,
        track_gpu_temperature: bool = False,
        output_dir: Optional[pathlib.Path] = None,
    ):
        """Initializes the TelemetryCallback.

        Args:
            skip_first_steps: The number of initial steps to exclude from stats.
            world_process_zero_only: Whether to collect stats on the main process only.
            include_timer_metrics: Whether to add timer stats to reported metrics.
                The timings stats can be verbose/distracting, so `False` by default.
                The timings will be written to a file at the end of training regardless
                of the value of this flag.
            track_gpu_temperature:  Whether to record GPU temperature.
            output_dir: If specified, then telemetry stats will be written to
                the directory as JSON files.
        """
        self._telemetry = TelemetryTracker()
        self._microstep_timer: Optional[TimerContext] = None
        self._step_timer: Optional[TimerContext] = None
        self._epoch_timer: Optional[TimerContext] = None

        self._skip_first_steps: int = skip_first_steps
        self._include_timer_metrics = include_timer_metrics
        self._track_gpu_temperature = track_gpu_temperature
        self._output_dir: Optional[pathlib.Path] = output_dir
        self._permanently_disabled: bool = (
            world_process_zero_only and not is_world_process_zero()
        )
        self._world_process_zero_only = world_process_zero_only
        self._step: int = 0

        self._last_metrics_dict: Optional[dict[str, float]] = None

    def on_step_begin(
        self,
        args: Union[transformers.TrainingArguments, TrainingParams],
        state: Optional[transformers.TrainerState] = None,
        control: Optional[transformers.TrainerControl] = None,
        **kwargs,
    ):
        """Event called at the beginning of a training step.

        If using gradient accumulation, one training step might take several inputs.
        """
        self._step += 1
        if self._callback_disabled():
            return

        self._complete_previous_microstep_if_needed()
        self._start_microstep()
        self._complete_previous_step_if_needed()
        self._start_step()

    def on_substep_end(
        self,
        args: Union[transformers.TrainingArguments, TrainingParams],
        state: Optional[transformers.TrainerState] = None,
        control: Optional[transformers.TrainerControl] = None,
        **kwargs,
    ):
        """Event called at the end of a substep during gradient accumulation."""
        if self._callback_disabled():
            return

        self._complete_previous_microstep_if_needed()
        self._start_microstep()

    def on_step_end(
        self,
        args: Union[transformers.TrainingArguments, TrainingParams],
        state: Optional[transformers.TrainerState] = None,
        control: Optional[transformers.TrainerControl] = None,
        **kwargs,
    ):
        """Event called at the end of each train step.

        Note that this will be called after all gradient accumulation substeps.
        """
        if self._callback_disabled():
            return

        self._complete_previous_microstep_if_needed()
        self._complete_previous_step_if_needed()
        if self._track_gpu_temperature:
            self._telemetry.record_gpu_temperature()

    def on_epoch_begin(
        self,
        args: Union[transformers.TrainingArguments, TrainingParams],
        state: Optional[transformers.TrainerState] = None,
        control: Optional[transformers.TrainerControl] = None,
        **kwargs,
    ):
        """Event called at the beginning of an epoch."""
        if self._permanently_disabled:
            return

        self._complete_previous_epoch_if_needed()
        self._start_epoch()

    def on_epoch_end(
        self,
        args: Union[transformers.TrainingArguments, TrainingParams],
        state: Optional[transformers.TrainerState] = None,
        control: Optional[transformers.TrainerControl] = None,
        **kwargs,
    ):
        """Event called at the end of an epoch."""
        if self._permanently_disabled:
            return
        self._complete_previous_epoch_if_needed()

        log_nvidia_gpu_runtime_info(log_prefix="On epoch end:")

    def on_log(
        self,
        args: Union[transformers.TrainingArguments, TrainingParams],
        state: Optional[transformers.TrainerState] = None,
        control: Optional[transformers.TrainerControl] = None,
        **kwargs,
    ):
        """Event called after logging the last logs."""
        if self._callback_disabled():
            return

        device_rank_info = get_device_rank_info()
        basename = f"telemetry_rank{device_rank_info.rank:03}"

        summary = self._telemetry.get_summary()
        if (
            self._include_timer_metrics
            and "timers" in summary
            and _LOGS_KWARG in kwargs
        ):
            for name, stats in summary["timers"].items():
                for stats_key in ("mean", "median", "std_dev", "min", "max", "count"):
                    if stats_key in stats:
                        metric_name = f"{basename}_{name}_{stats_key}"
                        kwargs[_LOGS_KWARG][metric_name] = float(stats[stats_key])

        if (
            self._track_gpu_temperature
            and "gpu_temperature" in summary
            and summary["gpu_temperature"]
            and _LOGS_KWARG in kwargs
        ):
            stats = summary["gpu_temperature"]
            for stats_key in ("mean", "median", "std_dev", "min", "max", "count"):
                metric_name = f"{basename}_gpu_temperature_{stats_key}"
                kwargs[_LOGS_KWARG][metric_name] = float(stats[stats_key])

        if _LOGS_KWARG in kwargs and is_world_process_zero():
            self._last_metrics_dict = copy.deepcopy(kwargs[_LOGS_KWARG])

    def on_train_end(
        self,
        args: Union[transformers.TrainingArguments, TrainingParams],
        state: Optional[transformers.TrainerState] = None,
        control: Optional[transformers.TrainerControl] = None,
        **kwargs,
    ):
        """Event called at the end of training."""
        if self._callback_disabled() or not self._output_dir:
            return

        device_rank_info = get_device_rank_info()

        if is_world_process_zero():
            metrics_dict = self._last_metrics_dict or {}
            save_json(
                metrics_dict,
                self._output_dir
                / f"telemetry_callback_metrics_rank{device_rank_info.rank:04}.json",
            )
            if wandb.run:
                save_json(
                    {
                        "id": wandb.run.id,
                        "name": wandb.run.name,
                        "url": wandb.run.get_url(),
                    },
                    self._output_dir
                    / f"telemetry_callback_wandb_rank{device_rank_info.rank:04}.json",
                )

        if self._world_process_zero_only:
            if is_world_process_zero():
                summary = self._telemetry.get_summary()
                telemetry_file = (
                    self._output_dir
                    / f"telemetry_callback_rank{device_rank_info.rank:04}.json"
                )
                logger.info(f"Saving telemetry callback summary to {telemetry_file}...")
                save_json(summary, telemetry_file)
        else:
            # The function has to be called by all ranks.
            summaries = self._telemetry.get_summaries_from_all_ranks()
            if is_world_process_zero():
                summaries_dict = {
                    f"rank{rank:04}": summary for rank, summary in enumerate(summaries)
                }
                telemetry_file = self._output_dir / "telemetry_callback_all_ranks.json"
                logger.info(
                    "Saving telemetry callback summaries "
                    f"for all ranks to {telemetry_file}..."
                )
                save_json(summaries_dict, telemetry_file)

                gpu_temperature_info_dict = (
                    self._telemetry.compute_cross_rank_summaries(
                        summaries,
                        measurement_names={
                            "gpu_temperature": {"max", "mean", "median"},
                        },
                    )
                )
                logger.info(
                    f"GPU temperature summary:\n{pformat(gpu_temperature_info_dict)}"
                )
                save_json(
                    gpu_temperature_info_dict,
                    self._output_dir
                    / "telemetry_callback_gpu_temperature_summary.json",
                )

    def _callback_disabled(self) -> bool:
        """Check if the callback should be disabled."""
        if self._permanently_disabled:
            return True
        if self._skip_first_steps > 0 and self._step <= self._skip_first_steps:
            return True
        return False

    @staticmethod
    def _exit_timer_if_needed(timer: Optional[TimerContext]) -> Optional[TimerContext]:
        if timer is not None:
            timer.__exit__(*sys.exc_info())
        return None

    def _start_timer(self, timer_name: str) -> TimerContext:
        timer: TimerContext = self._telemetry.timer(timer_name)
        timer.__enter__()
        return timer

    def _complete_previous_microstep_if_needed(self):
        self._microstep_timer = TelemetryCallback._exit_timer_if_needed(
            self._microstep_timer
        )

    def _start_microstep(self):
        self._microstep_timer = self._start_timer("microsteps")

    def _complete_previous_step_if_needed(self):
        self._step_timer = TelemetryCallback._exit_timer_if_needed(self._step_timer)

    def _start_step(self):
        self._step_timer = self._start_timer("steps")

    def _complete_previous_epoch_if_needed(self):
        self._epoch_timer = TelemetryCallback._exit_timer_if_needed(self._epoch_timer)

    def _start_epoch(self):
        self._epoch_timer = self._start_timer("epochs")
