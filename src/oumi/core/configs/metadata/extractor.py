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

"""Metadata extractor for config files."""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from oumi.core.configs.metadata.comment_parser import (
    parse_metadata_comments,
    parse_tags,
)
from oumi.core.configs.metadata.config_metadata import (
    ConfigMetadata,
    ConfigType,
    FinetuningType,
    TrainingMethod,
)

if TYPE_CHECKING:
    from oumi.core.configs.base_config import BaseConfig


# Mapping from TrainerType enum values to TrainingMethod
_TRAINER_TYPE_TO_METHOD = {
    "trl_sft": TrainingMethod.SFT,
    "trl_dpo": TrainingMethod.DPO,
    "trl_kto": TrainingMethod.KTO,
    "trl_grpo": TrainingMethod.GRPO,
    "verl_grpo": TrainingMethod.GRPO,
    "trl_gkd": TrainingMethod.GKD,
    "trl_gold": TrainingMethod.GKD,  # GOLD is a variant of GKD
    "hf": TrainingMethod.SFT,  # HF trainer is typically used for SFT
    "oumi": TrainingMethod.SFT,  # Oumi trainer is typically used for SFT
}

# Model family patterns (case-insensitive)
_MODEL_FAMILY_PATTERNS = [
    (r"llama", "llama"),
    (r"qwen", "qwen"),
    (r"gemma", "gemma"),
    (r"phi", "phi"),
    (r"mistral", "mistral"),
    (r"falcon", "falcon"),
    (r"olmo", "olmo"),
    (r"gpt-?2", "gpt2"),
    (r"pythia", "pythia"),
    (r"smol", "smollm"),
    (r"llava", "llava"),
    (r"internvl", "internvl"),
    (r"molmo", "molmo"),
    (r"claude", "anthropic"),
    (r"gpt-?4|gpt-?5|o1|o3", "openai"),
    (r"gemini", "google"),
    (r"deepseek", "deepseek"),
]

# Patterns to extract model size in billions
_SIZE_PATTERNS = [
    r"(\d+(?:\.\d+)?)[bB]",  # 8B, 70B, 1.5B
    r"-(\d+(?:\.\d+)?)-",  # -8- in name
    r"_(\d+(?:\.\d+)?)_",  # _8_ in name
]

# Vision model indicators
_VISION_INDICATORS = [
    "vision",
    "vl",
    "vlm",
    "llava",
    "qwen2-vl",
    "qwen2.5-vl",
    "internvl",
    "molmo",
    "smolvlm",
]


def _parse_model_family(model_name: Optional[str]) -> Optional[str]:
    """Parse model family from model name.

    Args:
        model_name: The HuggingFace model name or path.

    Returns:
        The model family identifier or None if not recognized.
    """
    if not model_name:
        return None

    model_name_lower = model_name.lower()
    for pattern, family in _MODEL_FAMILY_PATTERNS:
        if re.search(pattern, model_name_lower):
            return family

    return None


def _parse_model_size(model_name: Optional[str]) -> Optional[float]:
    """Parse model size in billions from model name.

    Args:
        model_name: The HuggingFace model name or path.

    Returns:
        The model size in billions of parameters or None if not found.
    """
    if not model_name:
        return None

    for pattern in _SIZE_PATTERNS:
        match = re.search(pattern, model_name, re.IGNORECASE)
        if match:
            return float(match.group(1))

    return None


def _is_vision_model(model_name: Optional[str]) -> bool:
    """Check if the model is a vision-language model.

    Args:
        model_name: The HuggingFace model name or path.

    Returns:
        True if the model appears to be a vision-language model.
    """
    if not model_name:
        return False

    model_name_lower = model_name.lower()
    return any(indicator in model_name_lower for indicator in _VISION_INDICATORS)


def _get_config_type(config: "BaseConfig") -> ConfigType:
    """Determine the config type from the config class.

    Args:
        config: The config object.

    Returns:
        The ConfigType enum value.
    """
    # Import here to avoid circular imports
    from oumi.core.configs.evaluation_config import EvaluationConfig
    from oumi.core.configs.inference_config import InferenceConfig
    from oumi.core.configs.job_config import JobConfig
    from oumi.core.configs.judge_config import JudgeConfig
    from oumi.core.configs.training_config import TrainingConfig

    if isinstance(config, TrainingConfig):
        return ConfigType.TRAINING
    elif isinstance(config, InferenceConfig):
        return ConfigType.INFERENCE
    elif isinstance(config, EvaluationConfig):
        return ConfigType.EVALUATION
    elif isinstance(config, JobConfig):
        return ConfigType.JOB
    elif isinstance(config, JudgeConfig):
        return ConfigType.JUDGE
    else:
        # Default to training for unknown types
        return ConfigType.TRAINING


def _get_training_method(config: "BaseConfig") -> Optional[TrainingMethod]:
    """Extract training method from config.

    Args:
        config: The config object (should be TrainingConfig).

    Returns:
        The TrainingMethod enum value or None.
    """
    from oumi.core.configs.training_config import TrainingConfig

    if not isinstance(config, TrainingConfig):
        return None

    trainer_type = config.training.trainer_type
    if trainer_type is None:
        return None

    trainer_value = trainer_type.value.lower()
    return _TRAINER_TYPE_TO_METHOD.get(trainer_value)


def _get_finetuning_type(config: "BaseConfig") -> Optional[FinetuningType]:
    """Extract finetuning type from config.

    Args:
        config: The config object (should be TrainingConfig).

    Returns:
        The FinetuningType enum value or None.
    """
    from oumi.core.configs.training_config import TrainingConfig

    if not isinstance(config, TrainingConfig):
        return None

    peft = config.peft

    # Check for QLoRA first (QLoRA implies LoRA + quantization)
    if peft.q_lora:
        return FinetuningType.QLORA

    # Check for LoRA (has target modules or non-zero rank configured)
    if peft.lora_target_modules is not None or peft.lora_r > 0:
        # Check if PEFT is actually being used via training params
        if hasattr(config.training, "use_peft") and config.training.use_peft:
            return FinetuningType.LORA
        # Also check if LoRA modules are explicitly set
        if peft.lora_target_modules:
            return FinetuningType.LORA

    # Default to full fine-tuning
    return FinetuningType.FULL


class MetadataExtractor:
    """Extract metadata from config files.

    This class combines derived metadata from config fields with
    explicit metadata from @meta comments in the config file.
    """

    @staticmethod
    def extract(
        config: "BaseConfig",
        config_path: Union[str, Path],
        vram_estimator: Optional[callable] = None,
    ) -> ConfigMetadata:
        """Extract metadata from a config and its file.

        Args:
            config: The parsed config object.
            config_path: Path to the config file.
            vram_estimator: Optional function to estimate VRAM requirements.
                Should take (config, model_size, finetuning_type) and return float.

        Returns:
            ConfigMetadata with derived and explicit values merged.
        """
        # Parse explicit metadata from comments
        explicit = parse_metadata_comments(config_path)

        # Derive metadata from config fields
        config_type = _get_config_type(config)
        model_name = getattr(getattr(config, "model", None), "model_name", None)
        model_family = _parse_model_family(model_name)
        model_size = _parse_model_size(model_name)
        is_vision = _is_vision_model(model_name)
        training_method = _get_training_method(config)
        finetuning_type = _get_finetuning_type(config)

        # Estimate VRAM if estimator provided
        min_vram_gb = None
        if vram_estimator and model_size:
            min_vram_gb = vram_estimator(config, model_size, finetuning_type)

        # Build metadata, explicit values take precedence
        return ConfigMetadata(
            config_type=(
                ConfigType(explicit["config_type"])
                if "config_type" in explicit
                else config_type
            ),
            model_family=explicit.get("model_family") or model_family,
            model_size_billions=(
                float(explicit["model_size"])
                if "model_size" in explicit
                else model_size
            ),
            training_method=(
                TrainingMethod(explicit["training_method"])
                if "training_method" in explicit
                else training_method
            ),
            finetuning_type=(
                FinetuningType(explicit["finetuning_type"])
                if "finetuning_type" in explicit
                else finetuning_type
            ),
            min_vram_gb=(
                float(explicit["min_vram_gb"])
                if "min_vram_gb" in explicit
                else min_vram_gb
            ),
            recommended_gpus=(
                int(explicit["recommended_gpus"])
                if "recommended_gpus" in explicit
                else None
            ),
            is_vision_model=(
                explicit.get("is_vision_model", "").lower() == "true"
                if "is_vision_model" in explicit
                else is_vision
            ),
            tags=parse_tags(explicit.get("tags", "")),
            description=explicit.get("description"),
        )

    @staticmethod
    def extract_from_path(
        config_path: Union[str, Path],
        config_type_hint: Optional[ConfigType] = None,
    ) -> ConfigMetadata:
        """Extract metadata from a config file path without loading the full config.

        This is a lighter-weight extraction that only reads the file header
        and infers what it can from the path and explicit metadata.

        Args:
            config_path: Path to the config file.
            config_type_hint: Optional hint about the config type.

        Returns:
            ConfigMetadata with available values.
        """
        explicit = parse_metadata_comments(config_path)
        path_str = str(config_path).lower()

        # Infer config type from path or explicit
        config_type = ConfigType.TRAINING
        if "config_type" in explicit:
            config_type = ConfigType(explicit["config_type"])
        elif config_type_hint:
            config_type = config_type_hint
        elif "infer" in path_str:
            config_type = ConfigType.INFERENCE
        elif "eval" in path_str:
            config_type = ConfigType.EVALUATION
        elif "job" in path_str:
            config_type = ConfigType.JOB

        # Infer training method from path
        training_method = None
        if "training_method" in explicit:
            training_method = TrainingMethod(explicit["training_method"])
        elif "dpo" in path_str:
            training_method = TrainingMethod.DPO
        elif "grpo" in path_str:
            training_method = TrainingMethod.GRPO
        elif "kto" in path_str:
            training_method = TrainingMethod.KTO
        elif "sft" in path_str or "train" in path_str:
            training_method = TrainingMethod.SFT
        elif "pretrain" in path_str:
            training_method = TrainingMethod.PRETRAINING

        # Infer finetuning type from path
        finetuning_type = None
        if "finetuning_type" in explicit:
            finetuning_type = FinetuningType(explicit["finetuning_type"])
        elif "qlora" in path_str:
            finetuning_type = FinetuningType.QLORA
        elif "lora" in path_str:
            finetuning_type = FinetuningType.LORA
        elif "full" in path_str or "fft" in path_str:
            finetuning_type = FinetuningType.FULL

        # Infer model family from path
        model_family = explicit.get("model_family")
        if not model_family:
            model_family = _parse_model_family(path_str)

        # Infer model size from path
        model_size = None
        if "model_size" in explicit:
            model_size = float(explicit["model_size"])
        else:
            model_size = _parse_model_size(path_str)

        # Infer vision model from path
        is_vision = explicit.get("is_vision_model", "").lower() == "true"
        if not is_vision:
            is_vision = _is_vision_model(path_str)

        return ConfigMetadata(
            config_type=config_type,
            model_family=model_family,
            model_size_billions=model_size,
            training_method=training_method,
            finetuning_type=finetuning_type,
            min_vram_gb=(
                float(explicit["min_vram_gb"]) if "min_vram_gb" in explicit else None
            ),
            recommended_gpus=(
                int(explicit["recommended_gpus"])
                if "recommended_gpus" in explicit
                else None
            ),
            is_vision_model=is_vision,
            tags=parse_tags(explicit.get("tags", "")),
            description=explicit.get("description"),
        )
