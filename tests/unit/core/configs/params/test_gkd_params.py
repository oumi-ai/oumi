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


from oumi.core.configs.params.gkd_params import GkdParams


def test_teacher_model_init_kwargs_uses_dtype_not_torch_dtype():
    """Test that default uses 'dtype' instead of 'torch_dtype'."""
    params = GkdParams()

    # Check default has 'dtype'
    assert "dtype" in params.teacher_model_init_kwargs
    assert params.teacher_model_init_kwargs["dtype"] == "auto"


def test_to_hf_trainer_kwargs_with_default_teacher_init_kwargs():
    """Test to_hf_trainer_kwargs with default empty teacher_model_init_kwargs."""
    params = GkdParams(teacher_model_init_kwargs={})
    kwargs = params.to_hf_trainer_kwargs()

    # Should initialize empty dict and add dtype
    assert "teacher_model_init_kwargs" in kwargs
    assert kwargs["teacher_model_init_kwargs"] == {"dtype": "auto"}


def test_to_hf_trainer_kwargs_preserves_dtype():
    """Test that existing 'dtype' in teacher_model_init_kwargs is preserved."""
    params = GkdParams(
        teacher_model_init_kwargs={
            "dtype": "float16",
            "device_map": "auto",
        }
    )
    kwargs = params.to_hf_trainer_kwargs()

    # Should preserve the custom dtype
    assert kwargs["teacher_model_init_kwargs"]["dtype"] == "float16"
    assert kwargs["teacher_model_init_kwargs"]["device_map"] == "auto"


def test_to_hf_trainer_kwargs_adds_dtype_when_missing():
    """Test that 'dtype' is added when not present in teacher_model_init_kwargs."""
    params = GkdParams(
        teacher_model_init_kwargs={
            "device_map": "auto",
            "attn_implementation": "flash_attention_2",
        }
    )
    kwargs = params.to_hf_trainer_kwargs()

    # Should add dtype = "auto" when not present
    assert kwargs["teacher_model_init_kwargs"]["dtype"] == "auto"
    assert kwargs["teacher_model_init_kwargs"]["device_map"] == "auto"
    assert (
        kwargs["teacher_model_init_kwargs"]["attn_implementation"]
        == "flash_attention_2"
    )


def test_to_hf_trainer_kwargs_with_nonempty_teacher_init_kwargs():
    """Test to_hf_trainer_kwargs when teacher_model_init_kwargs is not empty."""
    params = GkdParams(
        teacher_model_init_kwargs={
            "device_map": "cuda:0",
        }
    )
    kwargs = params.to_hf_trainer_kwargs()

    assert "teacher_model_init_kwargs" in kwargs
    assert kwargs["teacher_model_init_kwargs"]["device_map"] == "cuda:0"
    # Should add dtype since it's missing
    assert kwargs["teacher_model_init_kwargs"]["dtype"] == "auto"


def test_to_hf_trainer_kwargs_includes_all_fields():
    """Test that to_hf_trainer_kwargs includes all expected fields."""
    params = GkdParams(
        temperature=0.8,
        lmbda=0.6,
        beta=0.7,
        max_new_tokens=256,
        disable_dropout=False,
        seq_kd=True,
    )
    kwargs = params.to_hf_trainer_kwargs()

    assert kwargs["temperature"] == 0.8
    assert kwargs["lmbda"] == 0.6
    assert kwargs["beta"] == 0.7
    assert kwargs["max_new_tokens"] == 256
    assert kwargs["disable_dropout"] is False
    assert kwargs["seq_kd"] is True
    assert "teacher_model_init_kwargs" in kwargs
