"""Trainer callbacks module for the LeMa (Learning Machines) library.

This module provides trainer callbacks, which can be used to customize
the behavior of the training loop in the LeMa Trainer
that can inspect the training loop state for progress reporting, logging,
early stopping, etc.
"""

from lema.core.callbacks.base_trainer_callback import BaseTrainerCallback
from lema.core.callbacks.hf_mfu_callback import HfMfuTrainerCallback
from lema.core.callbacks.mfu_callback import MfuTrainerCallback
from lema.core.callbacks.nan_inf_detection_callback import NanInfDetectionCallback
from lema.core.callbacks.profiler_step_callback import ProfilerStepCallback
from lema.core.callbacks.telemetry_callback import TelemetryCallback

__all__ = [
    "BaseTrainerCallback",
    "HfMfuTrainerCallback",
    "MfuTrainerCallback",
    "NanInfDetectionCallback",
    "ProfilerStepCallback",
    "TelemetryCallback",
]
