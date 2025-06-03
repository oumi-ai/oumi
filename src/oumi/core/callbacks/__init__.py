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

"""Trainer callbacks module for the Oumi (Open Universal Machine Intelligence) library.

This module provides trainer callbacks, which can be used to customize
the behavior of the training loop in the Oumi Trainer
that can inspect the training loop state for progress reporting, logging,
early stopping, etc.
"""

from oumi.core.callbacks.base_trainer_callback import BaseTrainerCallback
from oumi.core.callbacks.bitnet_callback import BitNetCallback
from oumi.core.callbacks.hf_mfu_callback import HfMfuTrainerCallback
from oumi.core.callbacks.mfu_callback import MfuTrainerCallback
from oumi.core.callbacks.nan_inf_detection_callback import NanInfDetectionCallback
from oumi.core.callbacks.profiler_step_callback import ProfilerStepCallback
from oumi.core.callbacks.telemetry_callback import TelemetryCallback

__all__ = [
    "BaseTrainerCallback",
    "BitNetCallback",
    "HfMfuTrainerCallback",
    "MfuTrainerCallback",
    "NanInfDetectionCallback",
    "ProfilerStepCallback",
    "TelemetryCallback",
]
