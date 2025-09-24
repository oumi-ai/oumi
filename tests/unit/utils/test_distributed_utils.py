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

"""Tests for distributed utility helpers."""

import pytest

from oumi.utils.distributed_utils import is_using_accelerate, is_using_accelerate_fsdp


@pytest.fixture(autouse=True)
def _reset_accelerate_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ACCELERATE_DYNAMO_BACKEND", raising=False)
    monkeypatch.delenv("ACCELERATE_DYNAMO_MODE", raising=False)
    monkeypatch.delenv("ACCELERATE_DYNAMO_USE_FULLGRAPH", raising=False)
    monkeypatch.delenv("ACCELERATE_DYNAMO_USE_DYNAMIC", raising=False)
    monkeypatch.delenv("ACCELERATE_USE_FSDP", raising=False)


def test_is_using_accelerate(monkeypatch: pytest.MonkeyPatch):
    assert not is_using_accelerate()

    monkeypatch.setenv("ACCELERATE_DYNAMO_BACKEND", "some_value")
    assert is_using_accelerate()
    monkeypatch.delenv("ACCELERATE_DYNAMO_BACKEND")

    monkeypatch.setenv("ACCELERATE_DYNAMO_MODE", "some_value")
    assert is_using_accelerate()
    monkeypatch.delenv("ACCELERATE_DYNAMO_MODE")

    monkeypatch.setenv("ACCELERATE_DYNAMO_USE_FULLGRAPH", "some_value")
    assert is_using_accelerate()
    monkeypatch.delenv("ACCELERATE_DYNAMO_USE_FULLGRAPH")

    monkeypatch.setenv("ACCELERATE_DYNAMO_USE_DYNAMIC", "some_value")
    assert is_using_accelerate()
    monkeypatch.delenv("ACCELERATE_DYNAMO_USE_DYNAMIC")

    monkeypatch.setenv("ACCELERATE_DYNAMO_BACKEND", "some_value")
    monkeypatch.setenv("ACCELERATE_DYNAMO_MODE", "some_value")
    monkeypatch.setenv("ACCELERATE_DYNAMO_USE_FULLGRAPH", "some_value")
    monkeypatch.setenv("ACCELERATE_DYNAMO_USE_DYNAMIC", "some_value")
    assert is_using_accelerate()


def test_is_using_accelerate_fsdp(monkeypatch: pytest.MonkeyPatch):
    assert not is_using_accelerate_fsdp()

    monkeypatch.setenv("ACCELERATE_USE_FSDP", "false")
    assert not is_using_accelerate_fsdp()

    monkeypatch.setenv("ACCELERATE_USE_FSDP", "true")
    assert is_using_accelerate_fsdp()

    monkeypatch.setenv("ACCELERATE_USE_FSDP", "invalid_value")
    with pytest.raises(ValueError, match="Cannot convert 'invalid_value' to boolean."):
        is_using_accelerate_fsdp()

    monkeypatch.delenv("ACCELERATE_USE_FSDP")
