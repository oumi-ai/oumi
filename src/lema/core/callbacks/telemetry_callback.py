"""Collects sub-step/step/epoch timings."""

import pathlib
import sys
from typing import Optional, Union

import transformers

from lema.core.distributed import get_device_rank_info, is_world_process_zero
from lema.core.types import TrainingParams
from lema.performance.telemetry import TelemetryTracker
from lema.utils.io_utils import save_json

_LOGS_KWARG = "logs"


class TelemetryCallback(transformers.TrainerCallback):
    """Trainer callback to collect sub-step/step/epoch timings.

    Uses `lema.performance.telemetry.TelemetryTracker` object.
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
        self._microstep_timer = None
        self._step_timer = None
        self._epoch_timer = None

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
        if self._callback_disabled():
            return

        self._complete_previous_microstep_if_needed()
        self._start_microstep()
        self._complete_previous_step_if_needed()
        self._start_step()
        self._step += 1

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
        for name, stats in summary["timers"].items():
            basename = f"telemetry_rank_{device_rank_info.rank:03}"
            for stats_key in ("mean", "median", "std_dev", "min", "max", "count"):
                if stats_key in stats:
                    metric_name = f"{basename}_{name}_{stats_key}"
                    kwargs[_LOGS_KWARG][metric_name] = float(stats[stats_key])

            if self._output_dir is not None:
                save_json(stats, self._output_dir / (basename + ".json"))

    def _callback_disabled(self) -> bool:
        """Check if the callback should be disabled."""
        if self._permanently_disabled:
            return True
        if self._skip_first_steps > 0 and self._step < self._skip_first_steps:
            return True
        return False

    def _complete_previous_microstep_if_needed(self):
        if self._microstep_timer is None:
            return

        self._microstep_timer.__exit__(*sys.exc_info())
        self._microstep_timer = None

    def _start_microstep(self):
        self._microstep_timer = self._telemetry.timer("microsteps")
        self._microstep_timer.__enter__()

    def _complete_previous_step_if_needed(self):
        if self._step_timer is None:
            return

        self._step_timer.__exit__(*sys.exc_info())
        self._step_timer = None

    def _start_step(self):
        self._step_timer = self._telemetry.timer("steps")
        self._step_timer.__enter__()

    def _complete_previous_epoch_if_needed(self):
        if self._epoch_timer is None:
            return

        self._epoch_timer.__exit__(*sys.exc_info())
        self._epoch_timer = None

    def _start_epoch(self):
        self._epoch_timer = self._telemetry.timer("epochs")
        self._epoch_timer.__enter__()
