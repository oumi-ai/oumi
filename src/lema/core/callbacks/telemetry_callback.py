"""Collects sub-step/step/epoch timings."""

import pathlib
import sys
from typing import Optional, Union

import transformers

from lema.core.distributed import get_device_rank_info, is_world_process_zero
from lema.core.types import TrainingParams
from lema.performance.telemetry import TelemetryTracker, TimerContext
from lema.utils.io_utils import save_json
from lema.utils.logging import logger

_LOGS_KWARG = "logs"


class TelemetryCallback(transformers.TrainerCallback):
    """Trainer callback to collect sub-step/step/epoch timings.

    Based on `lema.performance.telemetry.TelemetryTracker`.
    """

    def __init__(
        self,
        skip_first_steps: int = 1,
        world_process_zero_only: bool = True,
        output_dir: Optional[pathlib.Path] = None,
    ):
        """Initializes the TelemetryCallback.

        Args:
            skip_first_steps: The number of initial steps to exclude from stats.
            world_process_zero_only: Whether collect stats on the main process only.
            output_dir: If specified, then telemetry stats will be written to
                the directory as JSON files.
        """
        self._telemetry = TelemetryTracker()
        self._microstep_timer: Optional[TimerContext] = None
        self._step_timer: Optional[TimerContext] = None
        self._epoch_timer: Optional[TimerContext] = None

        self._skip_first_steps: int = skip_first_steps
        self._output_dir: Optional[pathlib.Path] = output_dir
        self._permanently_disabled: bool = (
            world_process_zero_only and not is_world_process_zero()
        )
        self._step: int = 0

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

        summary = self._telemetry.get_summary()
        if not ("timers" in summary and _LOGS_KWARG in kwargs):
            return

        device_rank_info = get_device_rank_info()
        basename = f"telemetry_rank{device_rank_info.rank:03}"
        for name, stats in summary["timers"].items():
            for stats_key in ("mean", "median", "std_dev", "min", "max", "count"):
                if stats_key in stats:
                    metric_name = f"{basename}_{name}_{stats_key}"
                    kwargs[_LOGS_KWARG][metric_name] = float(stats[stats_key])

    def on_train_end(
        self,
        args: Union[transformers.TrainingArguments, TrainingParams],
        state: Optional[transformers.TrainerState] = None,
        control: Optional[transformers.TrainerControl] = None,
        **kwargs,
    ):
        """Event called at the end of training."""
        if self._callback_disabled():
            return

        summary = self._telemetry.get_summary()
        if "timers" not in summary:
            return

        device_rank_info = get_device_rank_info()
        basename = f"telemetry_rank{device_rank_info.rank:03}"
        if self._output_dir is not None:
            telemetry_file = self._output_dir / (basename + ".json")
            logger.info(f"Saving telemetry stats to {telemetry_file}...")
            save_json(summary["timers"], telemetry_file)

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
