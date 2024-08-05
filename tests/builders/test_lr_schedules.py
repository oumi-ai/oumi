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
        lr_scheduler_type=SchedulerType.LINEAR.value, warmup_steps=100, warmup_ratio=0.1
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
def test_linear_scheduler(scheduler_type, optimizer, training_params):
    num_training_steps = 1000
    training_params.lr_scheduler_type = scheduler_type
    scheduler = build_lr_scheduler(optimizer, training_params, num_training_steps)
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)


def test_warmup_ratio(optimizer, training_params):
    num_training_steps = 1000
    training_params.warmup_steps = 0
    training_params.warmup_ratio = 0.1
    scheduler = build_lr_scheduler(optimizer, training_params, num_training_steps)
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)


def test_invalid_scheduler_type(optimizer, training_params):
    training_params.lr_scheduler_type = "invalid_type"
    with pytest.raises(ValueError, match="Unknown scheduler type"):
        build_lr_scheduler(optimizer, training_params, num_training_steps=100)


def test_missing_num_training_steps_for_cosine(optimizer, training_params):
    training_params.lr_scheduler_type = SchedulerType.COSINE
    with pytest.raises(ValueError, match="num_training_steps must be provided"):
        build_lr_scheduler(optimizer, training_params)


def test_missing_num_training_steps_for_warmup_ratio(optimizer, training_params):
    training_params.warmup_steps = 0
    training_params.warmup_ratio = 0.1
    with pytest.raises(ValueError, match="num_training_steps must be provided"):
        build_lr_scheduler(optimizer, training_params)


@patch("transformers.get_linear_schedule_with_warmup")
def test_linear_scheduler_params(mock_get_linear, optimizer, training_params):
    num_training_steps = 1000
    last_epoch = 10
    build_lr_scheduler(optimizer, training_params, num_training_steps, last_epoch)
    mock_get_linear.assert_called_once_with(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps,
        last_epoch=last_epoch,
    )


@patch("transformers.get_cosine_schedule_with_warmup")
def test_cosine_scheduler_params(mock_get_cosine, optimizer, training_params):
    training_params.lr_scheduler_type = SchedulerType.COSINE
    num_training_steps = 1000
    last_epoch = 10
    num_cycles = 2
    build_lr_scheduler(
        optimizer, training_params, num_training_steps, last_epoch, num_cycles
    )
    mock_get_cosine.assert_called_once_with(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps,
        last_epoch=last_epoch,
        num_cycles=num_cycles,
    )
