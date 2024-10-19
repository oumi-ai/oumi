import logging
from unittest.mock import patch

import pytest
import torch

from oumi.builders.lr_schedules import (
    build_lr_scheduler,
)
from oumi.core.configs import SchedulerType, TrainingParams

#
# Fixtures
#


@pytest.fixture
def optimizer():
    params = [torch.nn.Parameter(torch.randn(2, 2), requires_grad=True)]
    return torch.optim.Adam(params=params, lr=0.001)


@pytest.fixture
def training_params():
    return TrainingParams(
        lr_scheduler_type=SchedulerType.LINEAR,
        warmup_steps=10,
        warmup_ratio=None,
        lr_scheduler_kwargs={},
        learning_rate=0.001,
    )


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


def test_linear_scheduler_params(optimizer, training_params):
    num_training_steps = 1000
    # Do epoch 0 first to initialize `initial_lr` in param_groups
    with patch(
        "oumi.builders.lr_schedules.get_linear_schedule_with_warmup"
    ) as mock_get_linear:
        build_lr_scheduler(optimizer, training_params, num_training_steps, 0)
        mock_get_linear.assert_called()
        mock_get_linear.assert_called_once_with(
            optimizer=optimizer,
            num_warmup_steps=10,
            num_training_steps=num_training_steps,
            last_epoch=-1,
        )

    current_epoch = 5
    with patch(
        "oumi.builders.lr_schedules.get_linear_schedule_with_warmup"
    ) as mock_get_linear:
        build_lr_scheduler(
            optimizer, training_params, num_training_steps, current_epoch
        )
        mock_get_linear.assert_called()
        mock_get_linear.assert_called_once_with(
            optimizer=optimizer,
            num_warmup_steps=10,
            num_training_steps=num_training_steps,
            last_epoch=4,
        )


def test_cosine_scheduler_params(optimizer, training_params):
    training_params.lr_scheduler_type = SchedulerType.COSINE
    num_training_steps = 1000

    # Do epoch 0 first to initialize `initial_lr` in param_groups
    with patch(
        "oumi.builders.lr_schedules.get_cosine_schedule_with_warmup"
    ) as mock_get_cosine:
        build_lr_scheduler(optimizer, training_params, num_training_steps, 0)
        mock_get_cosine.assert_called()
        mock_get_cosine.assert_called_once_with(
            optimizer=optimizer,
            num_warmup_steps=10,
            num_training_steps=num_training_steps,
            last_epoch=-1,
            num_cycles=0.5,
        )

    current_epoch = 5
    with patch(
        "oumi.builders.lr_schedules.get_cosine_schedule_with_warmup"
    ) as mock_get_cosine:
        build_lr_scheduler(
            optimizer, training_params, num_training_steps, current_epoch
        )
        mock_get_cosine.assert_called()
        mock_get_cosine.assert_called_once_with(
            optimizer=optimizer,
            num_warmup_steps=10,
            num_training_steps=num_training_steps,
            last_epoch=4,
            num_cycles=0.5,
        )


def test_cosine_with_restarts_scheduler_params(optimizer, training_params):
    training_params.lr_scheduler_type = SchedulerType.COSINE_WITH_RESTARTS
    num_training_steps = 1000
    training_params.lr_scheduler_kwargs = {"num_cycles": 3}
    # Do epoch 0 first to initialize `initial_lr` in param_groups
    with patch(
        "oumi.builders.lr_schedules.get_cosine_with_hard_restarts_schedule_with_warmup"
    ) as mock_get_cosine_restarts:
        build_lr_scheduler(optimizer, training_params, num_training_steps, 0)
        mock_get_cosine_restarts.assert_called()
        mock_get_cosine_restarts.assert_called_once_with(
            optimizer=optimizer,
            num_warmup_steps=10,
            num_training_steps=num_training_steps,
            last_epoch=-1,
            num_cycles=3,
        )

    current_epoch = 5
    with patch(
        "oumi.builders.lr_schedules.get_cosine_with_hard_restarts_schedule_with_warmup"
    ) as mock_get_cosine_restarts:
        build_lr_scheduler(
            optimizer, training_params, num_training_steps, current_epoch
        )
        mock_get_cosine_restarts.assert_called()
        mock_get_cosine_restarts.assert_called_once_with(
            optimizer=optimizer,
            num_warmup_steps=10,
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
    LOGGER = logging.getLogger("oumi")
    LOGGER.propagate = True

    with caplog.at_level("WARNING"):
        build_lr_scheduler(optimizer, training_params, num_training_steps=1000)
        assert "Unrecognized scheduler kwargs" in caplog.text


@patch("oumi.utils.logging.logger.info")
def test_warmup_ratio_logging(mock_logger_info, optimizer, training_params):
    training_params.warmup_steps = None
    training_params.warmup_ratio = 0.1
    num_training_steps = 1000
    build_lr_scheduler(optimizer, training_params, num_training_steps)
    mock_logger_info.assert_called_with(
        "Using warmup_steps=100 based on 0.1 warmup_ratio and 1000 max steps."
    )


@patch("oumi.utils.logging.logger.info")
def test_no_warmup_logging(mock_logger_info, optimizer, training_params):
    training_params.warmup_steps = None
    training_params.warmup_ratio = None
    build_lr_scheduler(optimizer, training_params, num_training_steps=1000)
    mock_logger_info.assert_called_with(
        "No warmup steps provided. Setting warmup_steps=0."
    )


def test_linear_scheduler_lr_values(optimizer, training_params):
    num_training_steps = 100
    scheduler = build_lr_scheduler(optimizer, training_params, num_training_steps)
    scheduler.step()

    # Check initial LR
    assert optimizer.param_groups[0]["lr"] == 0.0001  # 10% of 0.001 (warmup)

    # Move to end of warmup
    for _ in range(10 - 1):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.001  # Full LR

    # Move to end of training
    for _ in range(90):
        scheduler.step()
    assert pytest.approx(optimizer.param_groups[0]["lr"], 0.0001) == 0.0  # Final LR


def test_cosine_scheduler_lr_values(optimizer, training_params):
    training_params.lr_scheduler_type = SchedulerType.COSINE
    num_training_steps = 100
    scheduler = build_lr_scheduler(optimizer, training_params, num_training_steps)
    scheduler.step()

    # Check initial LR
    assert (
        pytest.approx(optimizer.param_groups[0]["lr"], 0.0001) == 0.0001
    )  # 10% of 0.001 (warmup)

    # Move to end of warmup
    for _ in range(10 - 1):
        scheduler.step()
    assert pytest.approx(optimizer.param_groups[0]["lr"], 0.0001) == 0.001  # Full LR

    # Move to end of training
    for _ in range(90):
        scheduler.step()
    assert pytest.approx(optimizer.param_groups[0]["lr"], 0.0001) == 0.0  # Final LR


def test_constant_scheduler_lr_values(optimizer, training_params):
    training_params.lr_scheduler_type = SchedulerType.CONSTANT
    scheduler = build_lr_scheduler(optimizer, training_params)
    scheduler.step()

    # Check initial LR
    assert (
        pytest.approx(optimizer.param_groups[0]["lr"], 0.0001) == 0.0001
    )  # 10% of 0.001 (warmup)

    # Move to end of warmup
    for _ in range(10 - 1):
        scheduler.step()
    assert pytest.approx(optimizer.param_groups[0]["lr"], 0.0001) == 0.001  # Full LR

    # Move further (should remain constant)
    for _ in range(90):
        scheduler.step()
    assert (
        pytest.approx(optimizer.param_groups[0]["lr"], 0.0001) == 0.001
    )  # Still full LR
