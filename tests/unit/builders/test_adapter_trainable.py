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

from unittest.mock import MagicMock, patch

from oumi.builders.models import _apply_pretrained_adapter
from oumi.core.configs import ModelParams


def test_adapter_loads_frozen_by_default():
    params = ModelParams(model_name="base", adapter_model="/adapters/sft")
    base = MagicMock()
    with patch("oumi.builders.models.PeftModel") as peft:
        _apply_pretrained_adapter(base, params)
    peft.from_pretrained.assert_called_once_with(
        base, "/adapters/sft", is_trainable=False
    )


def test_adapter_trainable_forwards():
    params = ModelParams(
        model_name="base", adapter_model="/adapters/sft", adapter_trainable=True
    )
    base = MagicMock()
    with patch("oumi.builders.models.PeftModel") as peft:
        _apply_pretrained_adapter(base, params)
    peft.from_pretrained.assert_called_once_with(
        base, "/adapters/sft", is_trainable=True
    )
