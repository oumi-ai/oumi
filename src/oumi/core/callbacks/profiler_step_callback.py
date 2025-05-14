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

"""Calls `profiler.step()`  at the end of each training step."""

import sys
from typing import Optional, Union

import torch
import transformers

from oumi.core.callbacks.base_trainer_callback import BaseTrainerCallback
from oumi.core.configs import TrainingParams


class ProfilerStepCallback(BaseTrainerCallback):
    """Trainer callback to notify PyTorch profiler about training steps completion.

    Also, adds microstep function labels using `torch.profiler.record_function()`.
    """

    def __init__(self, profiler):
        """Initialize the ProfilerStepCallback.

        Args:
            profiler: PyTorch profiler object.
        """
        self._profiler = profiler
        self._microstep_function = None

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
        self._complete_previous_microstep_if_needed()
        self._start_microstep()

    def on_substep_end(
        self,
        args: Union[transformers.TrainingArguments, TrainingParams],
        state: Optional[transformers.TrainerState] = None,
        control: Optional[transformers.TrainerControl] = None,
        **kwargs,
    ):
        """Event called at the end of an substep during gradient accumulation."""
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
        self._complete_previous_microstep_if_needed()
        self._profiler.step()

    def _complete_previous_microstep_if_needed(self):
        if self._microstep_function is None:
            return

        self._microstep_function.__exit__(*sys.exc_info())
        self._microstep_function = None

    def _start_microstep(self):
        self._microstep_function = torch.profiler.record_function("microstep")
        self._microstep_function.__enter__()
