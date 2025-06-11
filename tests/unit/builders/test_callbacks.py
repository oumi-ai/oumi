from unittest.mock import patch

import torch

from oumi.builders.callbacks import build_training_callbacks
from oumi.core.callbacks.mfu_callback import MfuTrainerCallback
from oumi.core.callbacks.nan_inf_detection_callback import NanInfDetectionCallback
from oumi.core.callbacks.telemetry_callback import TelemetryCallback
from oumi.core.configs import (
    TrainingConfig,
)


def test_build_training_callbacks_no_performance_metrics():
    config = TrainingConfig()
    model = torch.nn.Module()
    result = build_training_callbacks(config, model, None)
    assert result == []


def test_build_training_callbacks_mfu_callback():
    config = TrainingConfig()
    config.training.include_performance_metrics = True
    config.data.train.pack = True
    config.model.model_max_length = 128
    model = torch.nn.Sequential(torch.nn.Linear(10, 10))
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.get_device_name", return_value="NVIDIA A100-PCIE-40GB"):
            result = build_training_callbacks(config, model, None)
    assert len(result) == 3
    assert isinstance(result[0], MfuTrainerCallback)
    assert isinstance(result[1], NanInfDetectionCallback)
    assert isinstance(result[2], TelemetryCallback)


@patch("oumi.utils.logging.logger.warning")
def test_build_training_callbacks_no_cuda(mock_logger_warning):
    config = TrainingConfig()
    config.training.include_performance_metrics = True
    model = torch.nn.Sequential(torch.nn.Linear(10, 10))
    with patch("torch.cuda.is_available", return_value=False):
        result = build_training_callbacks(config, model, None)
    assert len(result) == 2
    assert isinstance(result[0], NanInfDetectionCallback)
    assert isinstance(result[1], TelemetryCallback)
    mock_logger_warning.assert_called_with(
        "MFU logging is only supported on GPU. Skipping MFU callbacks."
    )


@patch("oumi.utils.logging.logger.warning")
def test_build_training_callbacks_peft(mock_logger_warning):
    config = TrainingConfig()
    config.training.include_performance_metrics = True
    config.training.use_peft = True
    model = torch.nn.Sequential(torch.nn.Linear(10, 10))
    with patch("torch.cuda.is_available", return_value=True):
        result = build_training_callbacks(config, model, None)
    assert len(result) == 2
    assert isinstance(result[0], NanInfDetectionCallback)
    assert isinstance(result[1], TelemetryCallback)
    mock_logger_warning.assert_called_with(
        "MFU logging is not supported for PEFT. Skipping MFU callbacks."
    )


@patch("oumi.utils.logging.logger.warning")
def test_build_training_callbacks_no_pack(mock_logger_warning):
    config = TrainingConfig()
    config.training.include_performance_metrics = True
    model = torch.nn.Sequential(torch.nn.Linear(10, 10))
    with patch("torch.cuda.is_available", return_value=True):
        result = build_training_callbacks(config, model, None)
    assert len(result) == 2
    assert isinstance(result[0], NanInfDetectionCallback)
    assert isinstance(result[1], TelemetryCallback)
    mock_logger_warning.assert_called_with(
        "MFU logging requires packed datasets. Skipping MFU callbacks."
    )


@patch("oumi.utils.logging.logger.warning")
def test_build_training_callbacks_unknown_device_name(mock_logger_warning):
    config = TrainingConfig()
    config.training.include_performance_metrics = True
    config.data.train.pack = True
    model = torch.nn.Sequential(torch.nn.Linear(10, 10))
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.get_device_name", return_value="Foo"):
            result = build_training_callbacks(config, model, None)
    assert len(result) == 2
    assert isinstance(result[0], NanInfDetectionCallback)
    assert isinstance(result[1], TelemetryCallback)
    mock_logger_warning.assert_called_with(
        "MFU logging is currently not supported for device Foo. Skipping MFU callbacks."
    )


@patch("oumi.utils.logging.logger.warning")
def test_build_training_callbacks_no_model_max_length(mock_logger_warning):
    config = TrainingConfig()
    config.training.include_performance_metrics = True
    config.data.train.pack = True
    model = torch.nn.Sequential(torch.nn.Linear(10, 10))
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.get_device_name", return_value="NVIDIA A100-PCIE-40GB"):
            result = build_training_callbacks(config, model, None)
    assert len(result) == 2
    assert isinstance(result[0], NanInfDetectionCallback)
    assert isinstance(result[1], TelemetryCallback)
    mock_logger_warning.assert_called_with(
        "model_max_length must be set to log MFU performance information."
    )
