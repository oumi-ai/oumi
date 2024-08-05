import logging
from unittest.mock import patch

import pytest
import torch

from lema.builders.lr_schedules import build_lr_scheduler
from lema.core.types import SchedulerType, TrainingParams

#
# Fixtures
#


@pytest.fixture
def optimizer():
    return torch.optim.Adam(params=[torch.nn.Parameter(torch.randn(2, 2))])


@pytest.fixture
def training_params():
    return TrainingParams(
        lr_scheduler_type=SchedulerType.LINEAR,
        warmup_steps=100,
        warmup_ratio=None,
        lr_scheduler_kwargs={},
    )


#
# Tests
#
@pytest.mark.parametrize(
    "scheduler_type",
    [
        SchedulerType.LINEAR,
        SchedulerType.COSINE,
        SchedulerType.CONSTANT,
        SchedulerType.COSINE_WITH_RESTARTS,
    ],
)
def test_build_schedulers(scheduler_type, optimizer, training_params):
    training_params.lr_scheduler_type = scheduler_type
    num_training_steps = 1000
    scheduler = build_lr_scheduler(optimizer, training_params, num_training_steps)
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)


def test_build_schedulers_with_unknown_type(optimizer, training_params):
    training_params.lr_scheduler_type = "unknown_type"
    with pytest.raises(ValueError, match="Unknown scheduler type"):
        build_lr_scheduler(optimizer, training_params)


@pytest.mark.parametrize(
    ("scheduler_type", "missing_ok"),
    [
        (SchedulerType.LINEAR, False),
        (SchedulerType.COSINE, False),
        (SchedulerType.CONSTANT, True),
        (SchedulerType.COSINE_WITH_RESTARTS, False),
    ],
)
def test_missing_num_training_steps_for_scheduler(
    scheduler_type, missing_ok, optimizer, training_params
):
    training_params.lr_scheduler_type = scheduler_type

    if missing_ok:
        scheduler = build_lr_scheduler(optimizer, training_params)
        assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
    else:
        with pytest.raises(ValueError, match="num_training_steps must be provided"):
            build_lr_scheduler(optimizer, training_params)


@pytest.mark.parametrize(
    "scheduler_type",
    [
        SchedulerType.LINEAR,
        SchedulerType.COSINE,
        SchedulerType.CONSTANT,
        SchedulerType.COSINE_WITH_RESTARTS,
    ],
)
def test_warmup_ratio(scheduler_type, optimizer, training_params):
    num_training_steps = 1000
    training_params.warmup_steps = None
    training_params.warmup_ratio = 0.1
    training_params.lr_scheduler_type = scheduler_type
    scheduler = build_lr_scheduler(optimizer, training_params, num_training_steps)
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)


@pytest.mark.parametrize(
    "scheduler_type",
    [
        SchedulerType.LINEAR,
        SchedulerType.COSINE,
        SchedulerType.CONSTANT,
        SchedulerType.COSINE_WITH_RESTARTS,
    ],
)
def test_missing_num_training_steps_for_warmup_ratio(
    scheduler_type, optimizer, training_params
):
    training_params.warmup_steps = None
    training_params.warmup_ratio = 0.1
    training_params.lr_scheduler_type = scheduler_type
    with pytest.raises(ValueError, match="num_training_steps must be provided"):
        build_lr_scheduler(optimizer, training_params)


def test_both_warmup_steps_and_ratio_provided(optimizer, training_params):
    training_params.warmup_steps = 100
    training_params.warmup_ratio = 0.1
    with pytest.raises(
        ValueError, match="Only one of warmup_steps and warmup_ratio should be provided"
    ):
        build_lr_scheduler(optimizer, training_params, num_training_steps=1000)


def test_invalid_warmup_ratio(optimizer, training_params):
    training_params.warmup_steps = None
    training_params.warmup_ratio = 1.5
    with pytest.raises(ValueError, match=r"warmup_ratio must be in \[0, 1\]"):
        build_lr_scheduler(optimizer, training_params, num_training_steps=1000)


def test_no_warmup_provided(optimizer, training_params):
    training_params.warmup_steps = None
    training_params.warmup_ratio = None
    scheduler = build_lr_scheduler(optimizer, training_params, num_training_steps=1000)
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)


@patch("transformers.get_linear_schedule_with_warmup")
def test_linear_scheduler_params(mock_get_linear, optimizer, training_params):
    num_training_steps = 1000
    current_epoch = 5
    build_lr_scheduler(optimizer, training_params, num_training_steps, current_epoch)
    mock_get_linear.assert_called_once_with(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps,
        last_epoch=4,
    )


@patch("transformers.get_cosine_schedule_with_warmup")
def test_cosine_scheduler_params(mock_get_cosine, optimizer, training_params):
    training_params.lr_scheduler_type = SchedulerType.COSINE
    num_training_steps = 1000
    current_epoch = 5
    build_lr_scheduler(optimizer, training_params, num_training_steps, current_epoch)
    mock_get_cosine.assert_called_once_with(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps,
        last_epoch=4,
        num_cycles=0.5,
    )


@patch("transformers.get_cosine_with_hard_restarts_schedule_with_warmup")
def test_cosine_with_restarts_scheduler_params(
    mock_get_cosine_restarts, optimizer, training_params
):
    training_params.lr_scheduler_type = SchedulerType.COSINE_WITH_RESTARTS
    num_training_steps = 1000
    current_epoch = 5
    training_params.lr_scheduler_kwargs = {"num_cycles": 3}
    build_lr_scheduler(optimizer, training_params, num_training_steps, current_epoch)
    mock_get_cosine_restarts.assert_called_once_with(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps,
        last_epoch=4,
        num_cycles=3,
    )


@pytest.mark.parametrize(
    "scheduler_type",
    [
        SchedulerType.LINEAR,
        SchedulerType.COSINE,
        SchedulerType.CONSTANT,
        SchedulerType.COSINE_WITH_RESTARTS,
    ],
)
def test_scheduler_specific_kwargs_warning(
    scheduler_type, optimizer, training_params, caplog
):
    training_params.lr_scheduler_kwargs = {"unknown_param": 42}
    training_params.lr_scheduler_type = scheduler_type

    # Enable propagation of log messages to the pytest logger for testing
    LOGGER = logging.getLogger("lema")
    LOGGER.propagate = True

    with caplog.at_level("WARNING"):
        build_lr_scheduler(optimizer, training_params, num_training_steps=1000)
        assert "Unrecognized scheduler kwargs" in caplog.text


@patch("lema.utils.logging.logger.info")
def test_warmup_ratio_logging(mock_logger_info, optimizer, training_params):
    training_params.warmup_steps = None
    training_params.warmup_ratio = 0.1
    num_training_steps = 1000
    build_lr_scheduler(optimizer, training_params, num_training_steps)
    mock_logger_info.assert_called_with(
        "Using warmup_steps=100 based on 0.1 warmup_ratio and 1000 max steps."
    )


@patch("lema.utils.logging.logger.info")
def test_no_warmup_logging(mock_logger_info, optimizer, training_params):
    training_params.warmup_steps = None
    training_params.warmup_ratio = None
    build_lr_scheduler(optimizer, training_params, num_training_steps=1000)
    mock_logger_info.assert_called_with(
        "No warmup steps provided. Setting warmup_steps=0."
    )
