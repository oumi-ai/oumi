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

"""Logs training metrics to a JSONL file at each logging step."""

import json
import pathlib

import transformers

from oumi.core.callbacks.base_trainer_callback import BaseTrainerCallback
from oumi.core.configs import TrainingParams
from oumi.core.distributed import get_device_rank_info, is_world_process_zero

_LOGS_KWARG = "logs"


class MetricsLoggerCallback(BaseTrainerCallback):
    """Callback that logs training metrics to a JSONL file at each logging step.

    Each line in the output file is a valid JSON object containing all metrics
    logged at that step (loss, learning rate, epoch, MFU if enabled, etc.).
    """

    def __init__(self, output_dir: pathlib.Path):
        """Initializes the MetricsLoggerCallback.

        Args:
            output_dir: Directory where the metrics JSONL file will be written.
        """
        self._output_dir = output_dir
        self._metrics_log_file: pathlib.Path | None = None

    def on_log(
        self,
        args: transformers.TrainingArguments | TrainingParams | None,
        state: transformers.TrainerState | None = None,
        control: transformers.TrainerControl | None = None,
        **kwargs,
    ):
        """Event called after logging metrics."""
        if not is_world_process_zero():
            return

        if _LOGS_KWARG not in kwargs:
            return

        metrics = kwargs[_LOGS_KWARG]
        self._write_metrics_to_jsonl(metrics)

    def _write_metrics_to_jsonl(self, metrics: dict) -> None:
        """Writes metrics to a JSONL file.

        Each call appends a new line with the metrics as a JSON object.

        Args:
            metrics: Dictionary of metrics to write.
        """
        if self._metrics_log_file is None:
            device_rank_info = get_device_rank_info()
            self._output_dir.mkdir(parents=True, exist_ok=True)
            self._metrics_log_file = (
                self._output_dir / f"metrics_rank{device_rank_info.rank:04}.jsonl"
            )

        with open(self._metrics_log_file, "a") as f:
            f.write(json.dumps(metrics, default=str) + "\n")
