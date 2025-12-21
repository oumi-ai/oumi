import os

import pytest

from oumi.utils.distributed_utils import (
    is_under_distributed_launcher,
    is_using_accelerate,
    is_using_accelerate_fsdp,
    is_using_torchrun,
)


def test_is_using_accelerate():
    for var in [
        "ACCELERATE_DYNAMO_BACKEND",
        "ACCELERATE_DYNAMO_MODE",
        "ACCELERATE_DYNAMO_USE_FULLGRAPH",
        "ACCELERATE_DYNAMO_USE_DYNAMIC",
    ]:
        if var in os.environ:
            del os.environ[var]
    assert not is_using_accelerate()

    os.environ["ACCELERATE_DYNAMO_BACKEND"] = "some_value"
    assert is_using_accelerate()
    del os.environ["ACCELERATE_DYNAMO_BACKEND"]

    os.environ["ACCELERATE_DYNAMO_MODE"] = "some_value"
    assert is_using_accelerate()
    del os.environ["ACCELERATE_DYNAMO_MODE"]

    os.environ["ACCELERATE_DYNAMO_USE_FULLGRAPH"] = "some_value"
    assert is_using_accelerate()
    del os.environ["ACCELERATE_DYNAMO_USE_FULLGRAPH"]

    os.environ["ACCELERATE_DYNAMO_USE_DYNAMIC"] = "some_value"
    assert is_using_accelerate()
    del os.environ["ACCELERATE_DYNAMO_USE_DYNAMIC"]

    os.environ["ACCELERATE_DYNAMO_BACKEND"] = "some_value"
    os.environ["ACCELERATE_DYNAMO_MODE"] = "some_value"
    os.environ["ACCELERATE_DYNAMO_USE_FULLGRAPH"] = "some_value"
    os.environ["ACCELERATE_DYNAMO_USE_DYNAMIC"] = "some_value"
    assert is_using_accelerate()


def test_is_using_accelerate_fsdp():
    if "ACCELERATE_USE_FSDP" in os.environ:
        del os.environ["ACCELERATE_USE_FSDP"]
    assert not is_using_accelerate_fsdp()

    os.environ["ACCELERATE_USE_FSDP"] = "false"
    assert not is_using_accelerate_fsdp()

    os.environ["ACCELERATE_USE_FSDP"] = "true"
    assert is_using_accelerate_fsdp()

    os.environ["ACCELERATE_USE_FSDP"] = "invalid_value"
    with pytest.raises(ValueError, match="Cannot convert 'invalid_value' to boolean."):
        is_using_accelerate_fsdp()


def test_is_using_torchrun():
    """Test detection of torchrun launcher via TORCHELASTIC_RUN_ID."""
    # Clean up any existing env var
    if "TORCHELASTIC_RUN_ID" in os.environ:
        del os.environ["TORCHELASTIC_RUN_ID"]

    # Not using torchrun
    assert not is_using_torchrun()

    # Using torchrun
    os.environ["TORCHELASTIC_RUN_ID"] = "some_run_id"
    assert is_using_torchrun()

    # Cleanup
    del os.environ["TORCHELASTIC_RUN_ID"]


def test_is_under_distributed_launcher_torchrun():
    """Test is_under_distributed_launcher detects torchrun."""
    # Clean up env vars
    for var in [
        "TORCHELASTIC_RUN_ID",
        "ACCELERATE_DYNAMO_BACKEND",
        "ACCELERATE_DYNAMO_MODE",
        "ACCELERATE_DYNAMO_USE_FULLGRAPH",
        "ACCELERATE_DYNAMO_USE_DYNAMIC",
    ]:
        if var in os.environ:
            del os.environ[var]

    # Not under any launcher
    assert not is_under_distributed_launcher()

    # Under torchrun
    os.environ["TORCHELASTIC_RUN_ID"] = "some_run_id"
    assert is_under_distributed_launcher()

    # Cleanup
    del os.environ["TORCHELASTIC_RUN_ID"]


def test_is_under_distributed_launcher_accelerate():
    """Test is_under_distributed_launcher detects accelerate."""
    # Clean up env vars
    for var in [
        "TORCHELASTIC_RUN_ID",
        "ACCELERATE_DYNAMO_BACKEND",
        "ACCELERATE_DYNAMO_MODE",
        "ACCELERATE_DYNAMO_USE_FULLGRAPH",
        "ACCELERATE_DYNAMO_USE_DYNAMIC",
    ]:
        if var in os.environ:
            del os.environ[var]

    # Not under any launcher
    assert not is_under_distributed_launcher()

    # Under accelerate
    os.environ["ACCELERATE_DYNAMO_BACKEND"] = "inductor"
    assert is_under_distributed_launcher()

    # Cleanup
    del os.environ["ACCELERATE_DYNAMO_BACKEND"]


def test_is_under_distributed_launcher_both():
    """Test is_under_distributed_launcher when both are set."""
    # Clean up env vars
    for var in [
        "TORCHELASTIC_RUN_ID",
        "ACCELERATE_DYNAMO_BACKEND",
    ]:
        if var in os.environ:
            del os.environ[var]

    # Under both (edge case)
    os.environ["TORCHELASTIC_RUN_ID"] = "some_run_id"
    os.environ["ACCELERATE_DYNAMO_BACKEND"] = "inductor"
    assert is_under_distributed_launcher()

    # Cleanup
    del os.environ["TORCHELASTIC_RUN_ID"]
    del os.environ["ACCELERATE_DYNAMO_BACKEND"]
