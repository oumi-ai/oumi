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

import pytest

from oumi.core.configs.params.gkd_params import GkdParams


class TestGkdParams:
    """Tests for GkdParams validation and conversion."""

    def test_default_params(self):
        """Test GkdParams with default values."""
        params = GkdParams()
        assert params.teacher_model_name_or_path is None
        assert params.teacher_model_init_kwargs == {}
        assert params.temperature == 0.9
        assert params.lmbda == 0.5
        assert params.beta == 0.5
        assert params.max_new_tokens == 128
        assert params.disable_dropout is True
        assert params.seq_kd is False

    def test_valid_teacher_model_name(self):
        """Test valid teacher model name."""
        params = GkdParams(teacher_model_name_or_path="meta-llama/Llama-3.1-8B")
        assert params.teacher_model_name_or_path == "meta-llama/Llama-3.1-8B"

    def test_empty_teacher_model_name(self):
        """Test that empty teacher model name raises error."""
        with pytest.raises(ValueError, match="cannot be empty or whitespace"):
            GkdParams(teacher_model_name_or_path="")

    def test_whitespace_teacher_model_name(self):
        """Test that whitespace-only teacher model name raises error."""
        with pytest.raises(ValueError, match="cannot be empty or whitespace"):
            GkdParams(teacher_model_name_or_path="   ")

    def test_invalid_teacher_model_name_type(self):
        """Test that non-string teacher model name raises error."""
        with pytest.raises(TypeError, match="must be a string"):
            GkdParams(teacher_model_name_or_path=123)

    def test_temperature_valid_range(self):
        """Test valid temperature values."""
        params = GkdParams(temperature=0.5)
        assert params.temperature == 0.5

        params = GkdParams(temperature=1.0)
        assert params.temperature == 1.0

        params = GkdParams(temperature=0.1)
        assert params.temperature == 0.1

    def test_temperature_zero(self):
        """Test that temperature=0 raises error (must be > 0)."""
        with pytest.raises(ValueError, match="temperature must be in range"):
            GkdParams(temperature=0.0)

    def test_temperature_negative(self):
        """Test that negative temperature raises error."""
        with pytest.raises(ValueError, match="temperature must be in range"):
            GkdParams(temperature=-0.1)

    def test_temperature_too_high(self):
        """Test that temperature > 1.0 raises error."""
        with pytest.raises(ValueError, match="temperature must be in range"):
            GkdParams(temperature=1.1)

    def test_lmbda_valid_range(self):
        """Test valid lmbda values."""
        params = GkdParams(lmbda=0.0)
        assert params.lmbda == 0.0

        params = GkdParams(lmbda=0.5)
        assert params.lmbda == 0.5

        params = GkdParams(lmbda=1.0)
        assert params.lmbda == 1.0

    def test_lmbda_negative(self):
        """Test that negative lmbda raises error."""
        with pytest.raises(ValueError, match="lmbda must be in range"):
            GkdParams(lmbda=-0.1)

    def test_lmbda_too_high(self):
        """Test that lmbda > 1.0 raises error."""
        with pytest.raises(ValueError, match="lmbda must be in range"):
            GkdParams(lmbda=1.1)

    def test_beta_valid_range(self):
        """Test valid beta values."""
        params = GkdParams(beta=0.0)
        assert params.beta == 0.0

        params = GkdParams(beta=0.5)
        assert params.beta == 0.5

        params = GkdParams(beta=1.0)
        assert params.beta == 1.0

    def test_beta_negative(self):
        """Test that negative beta raises error."""
        with pytest.raises(ValueError, match="beta must be in range"):
            GkdParams(beta=-0.1)

    def test_beta_too_high(self):
        """Test that beta > 1.0 raises error."""
        with pytest.raises(ValueError, match="beta must be in range"):
            GkdParams(beta=1.1)

    def test_max_new_tokens_positive(self):
        """Test valid max_new_tokens values."""
        params = GkdParams(max_new_tokens=256)
        assert params.max_new_tokens == 256

        params = GkdParams(max_new_tokens=1)
        assert params.max_new_tokens == 1

    def test_max_new_tokens_zero(self):
        """Test that max_new_tokens=0 raises error."""
        with pytest.raises(ValueError, match="max_new_tokens must be positive"):
            GkdParams(max_new_tokens=0)

    def test_max_new_tokens_negative(self):
        """Test that negative max_new_tokens raises error."""
        with pytest.raises(ValueError, match="max_new_tokens must be positive"):
            GkdParams(max_new_tokens=-10)

    def test_to_hf_trainer_kwargs_minimal(self):
        """Test conversion to HF trainer kwargs with minimal config."""
        params = GkdParams()
        kwargs = params.to_hf_trainer_kwargs()

        assert kwargs["temperature"] == 0.9
        assert kwargs["lmbda"] == 0.5
        assert kwargs["beta"] == 0.5
        assert kwargs["max_new_tokens"] == 128
        assert kwargs["disable_dropout"] is True
        assert kwargs["seq_kd"] is False
        assert "teacher_model_name_or_path" not in kwargs  # None by default

    def test_to_hf_trainer_kwargs_with_teacher(self):
        """Test conversion to HF trainer kwargs with teacher model."""
        params = GkdParams(
            teacher_model_name_or_path="teacher-model",
            teacher_model_init_kwargs={"torch_dtype": "bfloat16"},
        )
        kwargs = params.to_hf_trainer_kwargs()

        # teacher_model_name_or_path is NOT in config (passed to trainer constructor)
        assert "teacher_model_name_or_path" not in kwargs
        # teacher_model_init_kwargs IS included in config
        assert kwargs["teacher_model_init_kwargs"] == {"torch_dtype": "bfloat16"}

    def test_to_hf_trainer_kwargs_full(self):
        """Test conversion to HF trainer kwargs with all parameters."""
        params = GkdParams(
            teacher_model_name_or_path="meta-llama/Llama-3.1-8B",
            teacher_model_init_kwargs={"torch_dtype": "int8", "device_map": "auto"},
            temperature=0.7,
            lmbda=0.8,
            beta=0.3,
            max_new_tokens=512,
            disable_dropout=False,
            seq_kd=True,
        )
        kwargs = params.to_hf_trainer_kwargs()

        # teacher_model_name_or_path is NOT in config (passed to trainer constructor)
        assert "teacher_model_name_or_path" not in kwargs
        # teacher_model_init_kwargs IS included in config
        assert kwargs["teacher_model_init_kwargs"] == {
            "torch_dtype": "int8",
            "device_map": "auto",
        }
        assert kwargs["temperature"] == 0.7
        assert kwargs["lmbda"] == 0.8
        assert kwargs["beta"] == 0.3
        assert kwargs["max_new_tokens"] == 512
        assert kwargs["disable_dropout"] is False
        assert kwargs["seq_kd"] is True

    def test_to_hf_trainer_kwargs_empty_teacher_init_kwargs(self):
        """Test that empty teacher_model_init_kwargs is not included."""
        params = GkdParams(
            teacher_model_name_or_path="teacher-model",
            teacher_model_init_kwargs={},
        )
        kwargs = params.to_hf_trainer_kwargs()

        assert "teacher_model_init_kwargs" not in kwargs

    def test_custom_all_params(self):
        """Test GkdParams with all custom values."""
        params = GkdParams(
            teacher_model_name_or_path="custom/teacher",
            teacher_model_init_kwargs={"custom": "value"},
            temperature=0.8,
            lmbda=0.6,
            beta=0.4,
            max_new_tokens=200,
            disable_dropout=False,
            seq_kd=True,
        )

        assert params.teacher_model_name_or_path == "custom/teacher"
        assert params.teacher_model_init_kwargs == {"custom": "value"}
        assert params.temperature == 0.8
        assert params.lmbda == 0.6
        assert params.beta == 0.4
        assert params.max_new_tokens == 200
        assert params.disable_dropout is False
        assert params.seq_kd is True
