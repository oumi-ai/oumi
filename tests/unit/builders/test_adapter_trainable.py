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

import pytest

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
        peft.from_pretrained.return_value.load_adapter.return_value.missing_keys = []
        _apply_pretrained_adapter(base, params)
    peft.from_pretrained.assert_called_once_with(
        base, "/adapters/sft", is_trainable=True
    )


def test_trainable_adapter_with_missing_lora_keys_fails():
    params = ModelParams(
        model_name="base", adapter_model="/adapters/sft", adapter_trainable=True
    )
    with patch("oumi.builders.models.PeftModel") as peft:
        peft.from_pretrained.return_value.load_adapter.return_value.missing_keys = [
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
        ]
        with pytest.raises(ValueError, match="does not match the model"):
            _apply_pretrained_adapter(MagicMock(), params)
