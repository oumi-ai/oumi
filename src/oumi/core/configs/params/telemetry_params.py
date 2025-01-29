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

from dataclasses import dataclass
from typing import Optional

from oumi.core.configs.params.base_params import BaseParams


@dataclass
class TelemetryParams(BaseParams):
    telemetry_dir: Optional[str] = "telemetry"
    """Directory where the telemetry data will be saved to.

    If not specified, then telemetry files will be written under `output_dir`.
    If a relative path is specified, then files will be written in a `telemetry_dir`
    sub-directory in `output_dir`.
    """

    collect_telemetry_for_all_ranks: bool = False
    """Whether to collect telemetry for all ranks.

    By default, only the main rank's telemetry stats are collected and saved.
    """

    track_gpu_temperature: bool = False
    """Whether to record GPU temperature.

    If `save_telemetry_for_all_ranks` is `False`, only the first GPU's temperature
    is tracked. Otherwise, temperature is recorded for all GPUs.
    """
