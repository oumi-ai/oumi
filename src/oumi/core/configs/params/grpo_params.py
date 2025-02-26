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

from dataclasses import dataclass
from typing import Any, Optional

from oumi.core.configs.params.base_params import BaseParams


@dataclass
class GrpoParams(BaseParams):
    model_init_kwargs: Optional[dict[str, Any]] = None
    """Keyword arguments for `AutoModelForCausalLM.from_pretrained(...)`"""

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
