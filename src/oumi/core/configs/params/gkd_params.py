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
from dataclasses import dataclass, field
from typing import Any, Optional

from oumi.core.configs.params.base_params import BaseParams


@dataclass
class GkdParams(BaseParams):
    """Parameters for Generalized Knowledge Distillation (GKD) training.

    GKD implements on-policy distillation where the student model generates outputs
    and learns from teacher corrections in real-time during training.

    Based on "On-Policy Distillation of Language Models: Learning from Self-Generated
    Mistakes" (https://arxiv.org/abs/2306.13649).

    Warning:
        GKDTrainer is experimental and may be changed or removed in future versions.
    """

    teacher_model_name_or_path: Optional[str] = None
    """Path or identifier of the teacher model.

    This is required for GKD training. Can be a HuggingFace model ID or local path.
    """

    teacher_model_init_kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments for loading the teacher model.

    Passed to `AutoModelForCausalLM.from_pretrained(...)` when loading the teacher.
    Common kwargs include `torch_dtype`, `device_map`, `attn_implementation`, etc.
    """

    temperature: float = 0.9
    """Temperature for sampling during generation.

    Higher values (e.g., 1.0) produce more diverse outputs, while lower values
    (e.g., 0.5) produce more focused outputs. Must be in range (0.0, 1.0].
    """

    lmbda: float = 0.5
    """Student data fraction (lambda parameter).

    Controls the mix between on-policy (student-generated) and off-policy
    (dataset) examples. Value of 0.5 means 50% on-policy, 50% off-policy.
    Must be in range [0.0, 1.0].
    """

    beta: float = 0.5
    """Jensen-Shannon Divergence interpolation coefficient.

    Controls the balance in the JSD loss function:
    - beta = 0.0: Uses KL divergence (teacher → student)
    - beta = 0.5: Uses symmetric JSD
    - beta = 1.0: Uses reverse KL divergence (student → teacher)

    Must be in range [0.0, 1.0].
    """

    max_new_tokens: int = 128
    """Maximum number of tokens to generate per prompt.

    This controls how long the student model's completions can be during
    on-policy generation.
    """

    disable_dropout: bool = True
    """Whether to disable dropout in the student model during training.

    Recommended to keep as `True` for more stable distillation.
    """

    seq_kd: bool = False
    """Whether to use sequence-level knowledge distillation.

    If `True`, uses sequence-level KD where the loss is computed at the sequence
    level. If `False`, uses token-level KD (default and recommended).
    """

    def __post_init__(self):
        """Validates GKD parameters."""
        if self.teacher_model_name_or_path is not None:
            if not isinstance(self.teacher_model_name_or_path, str):
                raise TypeError(
                    "GkdParams.teacher_model_name_or_path must be a string. "
                    f"Actual type: {type(self.teacher_model_name_or_path)}"
                )
            if not self.teacher_model_name_or_path.strip():
                raise ValueError(
                    "GkdParams.teacher_model_name_or_path cannot be empty."
                )

        if not (
            math.isfinite(self.temperature)
            and self.temperature > 0.0
            and self.temperature <= 1.0
        ):
            raise ValueError(
                "GkdParams.temperature must be in range (0.0, 1.0]. "
                f"Actual: {self.temperature}"
            )

        if not (math.isfinite(self.lmbda) and 0.0 <= self.lmbda <= 1.0):
            raise ValueError(
                f"GkdParams.lmbda must be in range [0.0, 1.0]. Actual: {self.lmbda}"
            )

        if not (math.isfinite(self.beta) and 0.0 <= self.beta <= 1.0):
            raise ValueError(
                f"GkdParams.beta must be in range [0.0, 1.0]. Actual: {self.beta}"
            )

        if self.max_new_tokens <= 0:
            raise ValueError(
                "GkdParams.max_new_tokens must be positive. "
                f"Actual: {self.max_new_tokens}"
            )

    def to_hf_trainer_kwargs(self) -> dict[str, Any]:
        """Converts GkdParams to TRL's GKDConfig kwargs.

        Note:
            The teacher_model_name_or_path is NOT passed to GKDConfig.
            Instead, it's passed to the GKDTrainer constructor via train.py.

            The teacher_model_init_kwargs goes into GKDConfig for TRL to use when
            loading the teacher model.

        Returns:
            Dictionary of kwargs to pass to TRL's GKDConfig.
        """
        result = {
            "temperature": self.temperature,
            "lmbda": self.lmbda,
            "beta": self.beta,
            "max_new_tokens": self.max_new_tokens,
            "disable_dropout": self.disable_dropout,
            "seq_kd": self.seq_kd,
        }

        if len(self.teacher_model_init_kwargs) > 0:
            result["teacher_model_init_kwargs"] = self.teacher_model_init_kwargs

        return result
