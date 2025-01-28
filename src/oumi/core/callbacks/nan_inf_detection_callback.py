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

"""A callback to detect NaN/INF metric values."""

import copy
from typing import Optional, Union

import numpy as np
import transformers

from oumi.core.callbacks.base_trainer_callback import BaseTrainerCallback
from oumi.core.configs import TrainingParams
from oumi.utils.logging import logger

_LOGS_KWARG = "logs"


class NanInfDetectionCallback(BaseTrainerCallback):
    """Trainer callback to detect abnormal values (NaN, INF) of selected metrics.

    For example, `NaN` loss value is an almost certain indication of a training process
    going badly, in which cases it's best to detect the condition early, and fail.
    """

    def __init__(
        self,
        metrics: list[str],
    ):
        """Initializes the NanInfDetectionCallback.

        Args:
            metrics: The list of metrics to monitor.
        """
        self._metrics = copy.deepcopy(metrics)

    def on_log(
        self,
        args: Union[transformers.TrainingArguments, TrainingParams],
        state: Optional[transformers.TrainerState] = None,
        control: Optional[transformers.TrainerControl] = None,
        **kwargs,
    ):
        """Event called after logging the last logs."""
        metrics_dict = kwargs.pop(_LOGS_KWARG, None)
        if not metrics_dict:
            return

        # Now check for NaN or Inf.
        for metric in self._metrics:
            metric_val = metrics_dict.get(metric, None)
            if metric_val is not None and (
                np.isnan(metric_val) or np.isinf(metric_val)
            ):
                error_message = (
                    "NaN" if np.isnan(metric_val) else "INF"
                ) + f" is detected for the '{metric}' metric."
                logger.error(error_message)
                raise RuntimeError(error_message)
