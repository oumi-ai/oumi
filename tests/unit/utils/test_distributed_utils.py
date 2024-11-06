import os

import pytest

from oumi.utils.distributed_utils import is_using_accelerate, is_using_accelerate_fsdp


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
