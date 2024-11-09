"""Trainer callbacks module for the Oumi (Open Universal Machine Intelligence) library.

This module provides trainer callbacks, which can be used to customize
the behavior of the training loop in the Oumi Trainer
that can inspect the training loop state for progress reporting, logging,
early stopping, etc.
"""

from oumi.core.callbacks.aggregate_metric_callback import AggregateMetricCallback
from oumi.core.callbacks.autoclip_gradnorm_callback import AutoClipGradNormCallback
from oumi.core.callbacks.base_trainer_callback import BaseTrainerCallback
from oumi.core.callbacks.hf_mfu_callback import HfMfuTrainerCallback
from oumi.core.callbacks.mfu_callback import MfuTrainerCallback
from oumi.core.callbacks.nan_inf_detection_callback import NanInfDetectionCallback
from oumi.core.callbacks.profiler_step_callback import ProfilerStepCallback
from oumi.core.callbacks.telemetry_callback import TelemetryCallback

__all__ = [
    "AggregateMetricCallback",
    "AutoClipGradNormCallback",
    "BaseTrainerCallback",
    "HfMfuTrainerCallback",
    "MfuTrainerCallback",
    "NanInfDetectionCallback",
    "ProfilerStepCallback",
    "TelemetryCallback",
]
