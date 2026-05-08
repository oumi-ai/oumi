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

from oumi.core.configs.params.training_params import TrainingParams


@patch("oumi.utils.packaging.is_transformers_v5", return_value=True)
def test_get_token_tracking_kwargs_v5_returns_only_num_input_tokens_seen(_mock):
    """In transformers v5, only include_num_input_tokens_seen should be returned."""
    params = TrainingParams(include_performance_metrics=True)

    result = params._get_token_tracking_kwargs()

    assert result == {"include_num_input_tokens_seen": True}
    assert "include_tokens_per_second" not in result


@patch("oumi.utils.packaging.is_transformers_v5", return_value=True)
def test_get_token_tracking_kwargs_v5_with_metrics_disabled(_mock):
    """In transformers v5, disabled metrics should return False."""
    params = TrainingParams(include_performance_metrics=False)

    result = params._get_token_tracking_kwargs()

    assert result == {"include_num_input_tokens_seen": False}


@patch("oumi.utils.packaging.is_transformers_v5", return_value=False)
def test_get_token_tracking_kwargs_v4_returns_both_params(_mock):
    """In transformers v4, both token tracking params should be returned."""
    params = TrainingParams(include_performance_metrics=True)

    result = params._get_token_tracking_kwargs()

    assert result == {
        "include_tokens_per_second": True,
        "include_num_input_tokens_seen": True,
    }


@patch("oumi.utils.packaging.is_transformers_v5", return_value=False)
def test_get_token_tracking_kwargs_v4_with_metrics_disabled(_mock):
    """In transformers v4, disabled metrics should return False for both."""
    params = TrainingParams(include_performance_metrics=False)

    result = params._get_token_tracking_kwargs()

    assert result == {
        "include_tokens_per_second": False,
        "include_num_input_tokens_seen": False,
    }


# Tests for _get_warmup_kwargs


@patch("oumi.utils.packaging.is_transformers_v5", return_value=True)
def test_get_warmup_kwargs_v5_with_warmup_ratio(_mock):
    """In transformers v5, warmup_ratio should be converted to warmup_steps."""
    params = TrainingParams(warmup_ratio=0.1)

    result = params._get_warmup_kwargs()

    assert result == {"warmup_steps": 0.1}
    assert "warmup_ratio" not in result


@patch("oumi.utils.packaging.is_transformers_v5", return_value=True)
def test_get_warmup_kwargs_v5_with_warmup_steps(_mock):
    """In transformers v5, warmup_steps should be passed as-is."""
    params = TrainingParams(warmup_steps=100)

    result = params._get_warmup_kwargs()

    assert result == {"warmup_steps": 100}


@patch("oumi.utils.packaging.is_transformers_v5", return_value=True)
def test_get_warmup_kwargs_v5_warmup_steps_takes_precedence(_mock):
    """In transformers v5, warmup_steps should take precedence over warmup_ratio."""
    params = TrainingParams(warmup_ratio=0.1, warmup_steps=50)

    result = params._get_warmup_kwargs()

    assert result == {"warmup_steps": 50}


@patch("oumi.utils.packaging.is_transformers_v5", return_value=True)
def test_get_warmup_kwargs_v5_defaults_to_zero(_mock):
    """In transformers v5, default should be warmup_steps=0."""
    params = TrainingParams()

    result = params._get_warmup_kwargs()

    assert result == {"warmup_steps": 0}


@patch("oumi.utils.packaging.is_transformers_v5", return_value=False)
def test_get_warmup_kwargs_v4_returns_both_params(_mock):
    """In transformers v4, both warmup_ratio and warmup_steps should be returned."""
    params = TrainingParams(warmup_ratio=0.1, warmup_steps=50)

    result = params._get_warmup_kwargs()

    assert result == {"warmup_ratio": 0.1, "warmup_steps": 50}


@patch("oumi.utils.packaging.is_transformers_v5", return_value=False)
def test_get_warmup_kwargs_v4_defaults(_mock):
    """In transformers v4, defaults should be warmup_ratio=0.0, warmup_steps=0."""
    params = TrainingParams()

    result = params._get_warmup_kwargs()

    assert result == {"warmup_ratio": 0.0, "warmup_steps": 0}
