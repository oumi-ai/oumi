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

"""Config metadata types for config discoverability and documentation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ConfigType(str, Enum):
    """Type of configuration file."""

    TRAINING = "training"
    """Training configuration for fine-tuning or pretraining models."""

    INFERENCE = "inference"
    """Inference configuration for text generation."""

    EVALUATION = "evaluation"
    """Evaluation configuration for benchmarking models."""

    JOB = "job"
    """Job configuration for remote execution."""

    JUDGE = "judge"
    """Judge configuration for LLM-as-judge evaluation."""

    QUANTIZE = "quantize"
    """Quantization configuration."""


class TrainingMethod(str, Enum):
    """Training method/algorithm."""

    SFT = "sft"
    """Supervised Fine-Tuning."""

    DPO = "dpo"
    """Direct Preference Optimization."""

    GRPO = "grpo"
    """Group Relative Policy Optimization."""

    KTO = "kto"
    """Kahneman-Tversky Optimization."""

    GKD = "gkd"
    """Generalized Knowledge Distillation."""

    PRETRAINING = "pretraining"
    """Pretraining from scratch or continued pretraining."""


class FinetuningType(str, Enum):
    """Type of fine-tuning approach."""

    FULL = "full"
    """Full fine-tuning (all parameters trainable)."""

    LORA = "lora"
    """Low-Rank Adaptation (LoRA)."""

    QLORA = "qlora"
    """Quantized Low-Rank Adaptation (QLoRA)."""


@dataclass
class ConfigMetadata:
    """Metadata for config discoverability and documentation.

    This class holds metadata about a configuration file that can be used
    for CLI filtering, documentation generation, and config discovery.

    Attributes:
        config_type: The type of configuration (training, inference, etc.).
        model_family: The model family (llama, qwen, gemma, etc.).
        model_size_billions: The model size in billions of parameters.
        training_method: The training method (sft, dpo, grpo, etc.).
        finetuning_type: The fine-tuning approach (full, lora, qlora).
        min_vram_gb: Estimated minimum VRAM required in GB.
        recommended_gpus: Recommended number of GPUs.
        is_vision_model: Whether this is a vision-language model.
        tags: Optional tags for additional categorization.
        description: Optional human-readable description.
    """

    config_type: ConfigType
    model_family: Optional[str] = None
    model_size_billions: Optional[float] = None
    training_method: Optional[TrainingMethod] = None
    finetuning_type: Optional[FinetuningType] = None
    min_vram_gb: Optional[float] = None
    recommended_gpus: Optional[int] = None
    is_vision_model: bool = False
    tags: list[str] = field(default_factory=list)
    description: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert metadata to a dictionary for JSON serialization."""
        return {
            "config_type": self.config_type.value if self.config_type else None,
            "model_family": self.model_family,
            "model_size_billions": self.model_size_billions,
            "training_method": (
                self.training_method.value if self.training_method else None
            ),
            "finetuning_type": (
                self.finetuning_type.value if self.finetuning_type else None
            ),
            "min_vram_gb": self.min_vram_gb,
            "recommended_gpus": self.recommended_gpus,
            "is_vision_model": self.is_vision_model,
            "tags": self.tags,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConfigMetadata":
        """Create ConfigMetadata from a dictionary."""
        return cls(
            config_type=ConfigType(data["config_type"]) if data.get("config_type") else ConfigType.TRAINING,
            model_family=data.get("model_family"),
            model_size_billions=data.get("model_size_billions"),
            training_method=(
                TrainingMethod(data["training_method"])
                if data.get("training_method")
                else None
            ),
            finetuning_type=(
                FinetuningType(data["finetuning_type"])
                if data.get("finetuning_type")
                else None
            ),
            min_vram_gb=data.get("min_vram_gb"),
            recommended_gpus=data.get("recommended_gpus"),
            is_vision_model=data.get("is_vision_model", False),
            tags=data.get("tags", []),
            description=data.get("description"),
        )
