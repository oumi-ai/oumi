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

import math
from dataclasses import dataclass, field, fields
from typing import Any, Optional

from oumi.core.configs.params.base_params import BaseParams


@dataclass
class GrpoParams(BaseParams):
    model_init_kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments for `AutoModelForCausalLM.from_pretrained(...)`"""

    max_prompt_length: Optional[int] = None
    """Maximum length of the prompt.

    If the prompt is longer than this value, it will be truncated left.
    If unspecified (`None`), defaults to 512.
    """

    max_completion_length: Optional[int] = None
    """Maximum length of the generated completion.

    If unspecified (`None`), defaults to 256.
    """

    num_generations: Optional[int] = None
    """Number of generations per prompt to sample.

    The global batch size (num_processes * per_device_batch_size) must be divisible
    by this value. If unspecified (`None`), defaults to 8.
    """

    temperature: float = 0.9
    """Temperature for sampling.

    The higher the temperature, the more random the completions.
    """

    remove_unused_columns: bool = False
    """Whether to only keep the column `"prompt"` in the dataset.

    If you use a custom reward function that requires any column other than `"prompts"`
    and `"completions"`, you should set it to `False`.
    """

    repetition_penalty: Optional[float] = 1.0
    """Float that penalizes new tokens if they appear in the prompt/response so far.

    Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage
    the model to repeat tokens.
    """

    use_vllm: bool = False
    """Whether to use vLLM for generating completions.

    If set to `True`, ensure that a GPU is kept unused for training,
    as vLLM will require one for generation.
    """

    vllm_device: Optional[str] = None
    """Device where vLLM generation will run.

    For example, "cuda:1". If set to `None`, the system will
    automatically select the next available GPU after the last one used for training.
    This assumes that training has not already occupied all available GPUs.
    If only one device is available, the device will be shared between both training
    and vLLM.
    """

    vllm_gpu_memory_utilization: float = 0.9
    """Ratio (between 0 and 1) of GPU memory to reserve.

    Fraction of VRAM reserved  for the model weights, activations, and KV cache on
    the device dedicated to generation powered by vLLM. Higher values will increase
    the KV cache size and thus improve the model's throughput.
    However, if the value is too high, it may cause out-of-memory (OOM) errors
    during initialization.
    """

    vllm_dtype: Optional[str] = None
    """Data type to use for vLLM generation.

    If set to `None`, the data type will be automatically determined based on
    the model configuration. Find the supported values in the vLLM documentation.
    """

    vllm_max_model_len: Optional[int] = None
    """The `max_model_len` to use for vLLM.

    This could be useful when running with reduced
    `vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM
    will use the model context size, which might be much larger than the KV cache,
    leading to inefficiencies.
    """

    epsilon: float = 0.2
    """Epsilon value for clipping the relative probability in the loss.

    For example, if epsilon is 0.2, then the new probability can only differ from
    the old probability by a factor of x0.8-1.2."""

    log_completions: bool = False
    """Whether to log prompt and completion pairs every `logging_steps` steps."""

    def __post_init__(self):
        """Verifies params."""
        if self.max_prompt_length is not None and self.max_prompt_length <= 0:
            raise ValueError(
                "GrpoParams.max_prompt_length must be positive. "
                f"Actual: {self.max_prompt_length}"
            )
        if self.max_completion_length is not None and self.max_completion_length <= 0:
            raise ValueError(
                "GrpoParams.max_completion_length must be positive. "
                f"Actual: {self.max_completion_length}"
            )
        if self.num_generations is not None and self.num_generations <= 0:
            raise ValueError(
                "GrpoParams.num_generations must be positive. "
                f"Actual: {self.num_generations}"
            )
        if not (
            math.isfinite(self.temperature)
            and self.temperature >= 0.0
            and self.temperature <= 1.0
        ):
            raise ValueError(
                "GrpoParams.temperature must be within [0.0, 1.0] range. "
                f"Actual: {self.temperature}"
            )

    def to_hf_trainer_kwargs(self) -> dict[str, Any]:
        """Converts GrpoParams to TRL's GRPOConfig kwargs."""
        result = {}
        if len(self.model_init_kwargs) > 0:
            result["model_init_kwargs"] = self.model_init_kwargs
        if self.max_prompt_length is not None:
            result["max_prompt_length"] = self.max_prompt_length
        if self.max_completion_length is not None:
            result["max_completion_length"] = self.max_completion_length
        if self.num_generations is not None:
            result["num_generations"] = self.num_generations

        already_processed_keys: set[str] = set(
            {
                "model_init_kwargs",
                "max_prompt_length",
                "max_completion_length",
                "num_generations",
            }
        )

        # Copy the majority of fields that aren't special-cased.
        for param in fields(self):
            if param.name.startswith("vllm_") or param.name in already_processed_keys:
                continue
            result[param.name] = getattr(self, param.name)

        if self.use_vllm:  # Return vLLM params only if vLLM is enabled.
            if self.vllm_device is not None:
                result["vllm_device"] = self.vllm_device
            result["vllm_gpu_memory_utilization"] = self.vllm_gpu_memory_utilization
            if self.vllm_dtype is not None:
                result["vllm_dtype"] = self.vllm_dtype
            if self.vllm_max_model_len is not None:
                result["vllm_max_model_len"] = self.vllm_max_model_len
        return result
