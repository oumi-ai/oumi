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

from unittest.mock import patch

from oumi.core.configs.params.gold_params import GoldParams


def _to_kwargs_with_trl(is_v1_5_or_later: bool, **params_kwargs) -> dict:
    params = GoldParams(teacher_model_name_or_path="teacher", **params_kwargs)
    with patch(
        "oumi.core.configs.params.gold_params.is_trl_v1_5_or_later",
        return_value=is_v1_5_or_later,
    ):
        return params.to_hf_trainer_kwargs()["teacher_model_init_kwargs"]


def test_default_emits_torch_dtype_for_trl_below_1_5():
    """trl < 1.5 GOLDTrainer reads teacher_model_init_kwargs['torch_dtype']."""
    init_kwargs = _to_kwargs_with_trl(False)
    assert init_kwargs.get("torch_dtype") == "auto"
    assert "dtype" not in init_kwargs


def test_default_emits_dtype_for_trl_1_5_or_later():
    """trl >= 1.5 renamed the key to 'dtype' and dropped the alias."""
    init_kwargs = _to_kwargs_with_trl(True)
    assert init_kwargs.get("dtype") == "auto"
    assert "torch_dtype" not in init_kwargs


def test_user_torch_dtype_normalized_to_dtype_on_trl_1_5():
    """A user-supplied torch_dtype is renamed to dtype under trl >= 1.5."""
    init_kwargs = _to_kwargs_with_trl(
        True, teacher_model_init_kwargs={"torch_dtype": "bfloat16"}
    )
    assert init_kwargs.get("dtype") == "bfloat16"
    assert "torch_dtype" not in init_kwargs


def test_user_dtype_normalized_to_torch_dtype_below_trl_1_5():
    """A user-supplied dtype is renamed to torch_dtype under trl < 1.5."""
    init_kwargs = _to_kwargs_with_trl(
        False, teacher_model_init_kwargs={"dtype": "bfloat16"}
    )
    assert init_kwargs.get("torch_dtype") == "bfloat16"
    assert "dtype" not in init_kwargs


def test_extra_init_kwargs_preserved():
    """Non-dtype kwargs pass through untouched."""
    init_kwargs = _to_kwargs_with_trl(
        True,
        teacher_model_init_kwargs={"attn_implementation": "sdpa"},
    )
    assert init_kwargs["attn_implementation"] == "sdpa"
    assert init_kwargs.get("dtype") == "auto"


def test_does_not_mutate_original_params():
    """to_hf_trainer_kwargs must not mutate the caller's dict."""
    original = {"torch_dtype": "bfloat16"}
    params = GoldParams(
        teacher_model_name_or_path="teacher",
        teacher_model_init_kwargs=original,
    )
    with patch(
        "oumi.core.configs.params.gold_params.is_trl_v1_5_or_later",
        return_value=True,
    ):
        params.to_hf_trainer_kwargs()
    assert original == {"torch_dtype": "bfloat16"}
