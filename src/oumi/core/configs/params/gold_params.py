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
class GoldParams(BaseParams):
    """Parameters for GOLD (General Online Logit Distillation) training.

    GOLD extends GKD to support cross-tokenizer knowledge distillation through
    Universal Logit Distillation (ULD), enabling distillation between models with
    different tokenizers (e.g., Llama → Qwen).

    Based on "Unlocking On-Policy Distillation for Any Model Family"
    (https://arxiv.org/abs/2501.xxxxx).

    Warning:
        GOLDTrainer is experimental and may be changed or removed in future versions.
    """

    teacher_model_name_or_path: Optional[str] = None
    """Path or identifier of the teacher model.

    This is required for GOLD training. Can be a HuggingFace model ID or local path.
    """

    teacher_model_init_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"torch_dtype": "auto"}
    )
    """Keyword arguments for loading the teacher model.

    Passed to `AutoModelForCausalLM.from_pretrained(...)` when loading the teacher.
    Common kwargs include `device_map`, `attn_implementation`, etc.

    Defaults to {"torch_dtype": "auto"} to allow the model to use the default dtype
    of the teacher model.
    """

    teacher_tokenizer_name_or_path: Optional[str] = None
    """Tokenizer name or path for the teacher model.

    Required when using ULD loss for cross-tokenizer distillation. If None when
    using ULD loss, will use the same tokenizer as the student model (not
    recommended for cross-tokenizer distillation).
    """

    temperature: float = 0.9
    """Temperature for sampling during generation.

    Higher values (e.g., 1.0) produce more diverse outputs, while lower values
    (e.g., 0.5) produce more focused outputs. Must be in range (0.0, 1.0].
    """

    top_p: float = 0.95
    """Nucleus sampling parameter.

    Only the smallest set of most probable tokens with probabilities that add up
    to `top_p` or higher are kept for generation. Must be in range (0.0, 1.0].
    """

    top_k: int = 0
    """Top-k sampling parameter.

    The number of highest probability vocabulary tokens to keep for top-k-filtering.
    If 0, top-k filtering is disabled.
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

    max_completion_length: int = 128
    """Maximum number of tokens to generate per completion.

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

    use_uld_loss: bool = False
    """Whether to use Universal Logit Distillation (ULD) loss.

    When True, enables cross-tokenizer distillation through token alignment.
    When False, uses Generalized Jensen-Shannon Divergence loss (same as GKD).

    Set to True when student and teacher use different tokenizers.
    """

    use_extended_uld: bool = True
    """Whether to enable extended ULD alignment.

    Uses tokenizers to align and merge token probabilities across student and
    teacher tokenizations. When True, computes token mappings and merges
    probabilities for split tokens. When False, uses simple positional truncation
    like in the original ULD paper.

    Only relevant when use_uld_loss=True.
    """

    uld_use_hybrid_loss: bool = False
    """Whether to use hybrid ULD + JSD loss.

    When True, combines JSD loss for matched tokens and ULD loss for unmatched
    tokens. This can provide better distillation when there's partial tokenizer
    overlap.

    Only relevant when use_uld_loss=True.
    """

    uld_hybrid_matched_weight: Optional[float] = None
    """Weight for matched token loss in hybrid mode.

    Scales the JSD loss for tokens with direct student-teacher mappings.
    If None, uses adaptive weighting based on vocabulary overlap.
    Must be set together with uld_hybrid_unmatched_weight.

    Only relevant when use_uld_loss=True and uld_use_hybrid_loss=True.
    """

    uld_hybrid_unmatched_weight: Optional[float] = None
    """Weight for unmatched token loss in hybrid mode.

    Scales the ULD loss for tokens without direct student-teacher mappings.
    If None, uses adaptive weighting based on vocabulary overlap.
    Must be set together with uld_hybrid_matched_weight.

    Only relevant when use_uld_loss=True and uld_use_hybrid_loss=True.
    """

    uld_crossentropy_weight: float = 0.0
    """Weight for cross-entropy loss component in ULD.

    If 0, only ULD distillation loss is used. Higher values add supervised
    learning signal from ground-truth labels.

    Only relevant when use_uld_loss=True.
    """

    uld_distillation_weight: float = 1.0
    """Weight for distillation loss component in ULD.

    Controls the relative importance of distillation vs cross-entropy loss.

    Only relevant when use_uld_loss=True.
    """

    uld_student_temperature: float = 1.0
    """Temperature for student logits in ULD loss.

    Higher temperatures make the probability distribution softer.
    Must be positive.

    Only relevant when use_uld_loss=True.
    """

    uld_teacher_temperature: float = 1.0
    """Temperature for teacher logits in ULD loss.

    Higher temperatures make the probability distribution softer.
    Must be positive.

    Only relevant when use_uld_loss=True.
    """

    uld_skip_student_eos: bool = True
    """Whether to skip EOS token for student in ULD loss.

    Only relevant when use_uld_loss=True.
    """

    uld_skip_teacher_eos: bool = True
    """Whether to skip EOS token for teacher in ULD loss.

    Only relevant when use_uld_loss=True.
    """

    use_transformers_paged: bool = False
    """Whether to use transformers paged attention for generation.

    If True, uses paged implementation instead of default padded implementation.
    Can improve memory efficiency for generation.
    """

    use_vllm: bool = False
    """Whether to use vLLM for generating completions.

    Requires `vllm` to be installed. Can significantly speed up generation.
    """

    vllm_mode: str = "server"
    """Mode for vLLM integration.

    Either "server" (connect to running TRL vLLM server) or "colocate"
    (run vLLM in same process).

    Only relevant when use_vllm=True.
    """

    vllm_server_host: str = "0.0.0.0"
    """Host of the vLLM server.

    Only relevant when use_vllm=True and vllm_mode="server".
    """

    vllm_server_port: int = 8001
    """Port of the vLLM server.

    Only relevant when use_vllm=True and vllm_mode="server".
    """

    vllm_server_timeout: float = 240.0
    """Timeout for connecting to vLLM server (seconds).

    Only relevant when use_vllm=True and vllm_mode="server".
    """

    vllm_gpu_memory_utilization: float = 0.9
    """GPU memory utilization for colocated vLLM engine.

    Recommended to set lower if student and teacher share GPU.

    Only relevant when use_vllm=True and vllm_mode="colocate".
    """

    vllm_tensor_parallel_size: int = 1
    """Tensor parallel size for colocated vLLM engine.

    Only relevant when use_vllm=True and vllm_mode="colocate".
    """

    vllm_guided_decoding_regex: Optional[str] = None
    """Regex pattern for vLLM guided decoding.

    Only relevant when use_vllm=True.
    """

    vllm_sync_frequency: int = 1
    """Frequency (in steps) to sync model weights to vLLM engine.

    Set to 1 to sync after every step.

    Only relevant when use_vllm=True.
    """

    vllm_enable_sleep_mode: bool = False
    """Enable vLLM sleep mode to offload weights during optimizer step.

    Keeps GPU memory low but adds latency when waking engine.

    Only relevant when use_vllm=True and vllm_mode="colocate".
    """

    def __post_init__(self):
        """Validates GOLD parameters."""
        if self.teacher_model_name_or_path is not None:
            if not isinstance(self.teacher_model_name_or_path, str):
                raise TypeError(
                    "GoldParams.teacher_model_name_or_path must be a string. "
                    f"Actual type: {type(self.teacher_model_name_or_path)}"
                )
            if not self.teacher_model_name_or_path.strip():
                raise ValueError(
                    "GoldParams.teacher_model_name_or_path cannot be empty."
                )

        if not (
            math.isfinite(self.temperature)
            and self.temperature > 0.0
            and self.temperature <= 1.0
        ):
            raise ValueError(
                "GoldParams.temperature must be in range (0.0, 1.0]. "
                f"Actual: {self.temperature}"
            )

        if not (math.isfinite(self.top_p) and 0.0 < self.top_p <= 1.0):
            raise ValueError(
                f"GoldParams.top_p must be in range (0.0, 1.0]. Actual: {self.top_p}"
            )

        if self.top_k < 0:
            raise ValueError(
                f"GoldParams.top_k must be non-negative. Actual: {self.top_k}"
            )

        if not (math.isfinite(self.lmbda) and 0.0 <= self.lmbda <= 1.0):
            raise ValueError(
                f"GoldParams.lmbda must be in range [0.0, 1.0]. Actual: {self.lmbda}"
            )

        if not (math.isfinite(self.beta) and 0.0 <= self.beta <= 1.0):
            raise ValueError(
                f"GoldParams.beta must be in range [0.0, 1.0]. Actual: {self.beta}"
            )

        if self.max_completion_length <= 0:
            raise ValueError(
                "GoldParams.max_completion_length must be positive. "
                f"Actual: {self.max_completion_length}"
            )

        # Validate ULD parameters
        if self.use_uld_loss:
            if self.uld_crossentropy_weight < 0.0:
                raise ValueError(
                    "GoldParams.uld_crossentropy_weight must be non-negative. "
                    f"Actual: {self.uld_crossentropy_weight}"
                )

            if self.uld_distillation_weight < 0.0:
                raise ValueError(
                    "GoldParams.uld_distillation_weight must be non-negative. "
                    f"Actual: {self.uld_distillation_weight}"
                )

            if self.uld_student_temperature <= 0.0:
                raise ValueError(
                    "GoldParams.uld_student_temperature must be positive. "
                    f"Actual: {self.uld_student_temperature}"
                )

            if self.uld_teacher_temperature <= 0.0:
                raise ValueError(
                    "GoldParams.uld_teacher_temperature must be positive. "
                    f"Actual: {self.uld_teacher_temperature}"
                )

            # Validate hybrid loss weights
            if self.uld_use_hybrid_loss:
                if (self.uld_hybrid_matched_weight is None) != (
                    self.uld_hybrid_unmatched_weight is None
                ):
                    raise ValueError(
                        "GoldParams.uld_hybrid_matched_weight and "
                        "uld_hybrid_unmatched_weight must both be None (for adaptive "
                        "weighting) or both be set to numeric values. "
                        f"Got uld_hybrid_matched_weight="
                        f"{self.uld_hybrid_matched_weight} and "
                        f"uld_hybrid_unmatched_weight="
                        f"{self.uld_hybrid_unmatched_weight}."
                    )

                if (
                    self.uld_hybrid_matched_weight is not None
                    and self.uld_hybrid_unmatched_weight is not None
                ):
                    if self.uld_hybrid_matched_weight < 0.0:
                        raise ValueError(
                            "GoldParams.uld_hybrid_matched_weight must be "
                            f"non-negative. Actual: {self.uld_hybrid_matched_weight}"
                        )
                    if self.uld_hybrid_unmatched_weight < 0.0:
                        raise ValueError(
                            "GoldParams.uld_hybrid_unmatched_weight must be "
                            f"non-negative. Actual: {self.uld_hybrid_unmatched_weight}"
                        )

        if self.vllm_mode not in ("server", "colocate"):
            raise ValueError(
                f"GoldParams.vllm_mode must be 'server' or 'colocate'. "
                f"Actual: {self.vllm_mode}"
            )

    def to_hf_trainer_kwargs(self) -> dict[str, Any]:
        """Converts GoldParams to TRL's GOLDConfig kwargs.

        Note:
            The teacher_model_name_or_path is NOT passed to GOLDConfig.
            Instead, it's passed to the GOLDTrainer constructor via train.py.

            The teacher_model_init_kwargs goes into GOLDConfig for TRL to use when
            loading the teacher model.

        Returns:
            Dictionary of kwargs to pass to TRL's GOLDConfig.
        """
        result = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "lmbda": self.lmbda,
            "beta": self.beta,
            "max_completion_length": self.max_completion_length,
            "disable_dropout": self.disable_dropout,
            "seq_kd": self.seq_kd,
            "use_uld_loss": self.use_uld_loss,
            "use_extended_uld": self.use_extended_uld,
            "uld_use_hybrid_loss": self.uld_use_hybrid_loss,
            "uld_crossentropy_weight": self.uld_crossentropy_weight,
            "uld_distillation_weight": self.uld_distillation_weight,
            "uld_student_temperature": self.uld_student_temperature,
            "uld_teacher_temperature": self.uld_teacher_temperature,
            "uld_skip_student_eos": self.uld_skip_student_eos,
            "uld_skip_teacher_eos": self.uld_skip_teacher_eos,
            "use_transformers_paged": self.use_transformers_paged,
            "use_vllm": self.use_vllm,
            "vllm_mode": self.vllm_mode,
            "vllm_server_host": self.vllm_server_host,
            "vllm_server_port": self.vllm_server_port,
            "vllm_server_timeout": self.vllm_server_timeout,
            "vllm_gpu_memory_utilization": self.vllm_gpu_memory_utilization,
            "vllm_tensor_parallel_size": self.vllm_tensor_parallel_size,
            "vllm_sync_frequency": self.vllm_sync_frequency,
            "vllm_enable_sleep_mode": self.vllm_enable_sleep_mode,
        }

        # Add optional parameters
        if self.teacher_tokenizer_name_or_path:
            result["teacher_tokenizer_name_or_path"] = (
                self.teacher_tokenizer_name_or_path
            )

        if self.uld_hybrid_matched_weight is not None:
            result["uld_hybrid_matched_weight"] = self.uld_hybrid_matched_weight

        if self.uld_hybrid_unmatched_weight is not None:
            result["uld_hybrid_unmatched_weight"] = self.uld_hybrid_unmatched_weight

        if self.vllm_guided_decoding_regex:
            result["vllm_guided_decoding_regex"] = self.vllm_guided_decoding_regex

        if len(self.teacher_model_init_kwargs) > 0:
            result["teacher_model_init_kwargs"] = self.teacher_model_init_kwargs
        else:
            result["teacher_model_init_kwargs"] = {}

        if "torch_dtype" not in result["teacher_model_init_kwargs"]:
            result["teacher_model_init_kwargs"]["torch_dtype"] = "auto"

        return result
