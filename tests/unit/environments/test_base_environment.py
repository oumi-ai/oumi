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

import dataclasses

import pytest

from oumi.environments.base_environment import BaseEnvironment


def test_base_environment_is_not_a_dataclass():
    assert not dataclasses.is_dataclass(BaseEnvironment)


def test_base_environment_cannot_be_instantiated_directly():
    with pytest.raises(TypeError, match=r"abstract|instantiate"):
        BaseEnvironment()  # type: ignore[abstract]


def test_subclass_without_step_cannot_instantiate():
    class Incomplete(BaseEnvironment):
        pass

    with pytest.raises(TypeError, match=r"abstract|instantiate"):
        Incomplete()  # type: ignore[abstract]
