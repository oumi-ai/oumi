from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar

import torch
import transformers
from omegaconf import MISSING, OmegaConf
from peft.utils.peft_types import TaskType


#
# Training Params
#
class TrainerType(Enum):
    """Enum representing the supported trainers."""

    TRL_SFT = "trl_sft"
    "Supervised fine-tuning trainer from `trl` library."

    TRL_DPO = "trl_dpo"
    "Direct preference optimization trainer from `trl` library."

    HF = "hf"
    "Generic HuggingFace trainer from `transformers` library."


@dataclass
class TrainingParams:
    optimizer: str = "adamw_torch"
    use_peft: bool = False
    trainer_type: TrainerType = TrainerType.TRL_SFT
    enable_gradient_checkpointing: bool = False
    output_dir: str = "output"
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_steps: int = -1

    def to_hf(self):
        """Convert LeMa config to HuggingFace's TrainingArguments."""
        return transformers.TrainingArguments(
            optim=self.optimizer,
            output_dir=self.output_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_steps=self.max_steps,
        )


@dataclass
class DataParams:
    dataset_name: str = MISSING

    preprocessing_function_name: Optional[str] = None

    trainer_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelParams:
    model_name: str = MISSING
    trust_remote_code: bool = False
    torch_dtype_str: str = "float32"

    def torch_dtype(self):
        """Convert string dtype to torch.dtype."""
        if self.torch_dtype_str in ["f64", "float64", "double"]:
            return torch.float64
        elif self.torch_dtype_str in ["f32", "float32", "float"]:
            return torch.float32
        elif self.torch_dtype_str in ["bf16", "bfloat16"]:
            return torch.bfloat16
        else:
            raise ValueError(f"Unsupported data type: {self.torch_dtype_str}")


@dataclass
class PeftParams:
    # Lora Params
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    lora_bias: str = "none"
    lora_task_type: TaskType = TaskType.CAUSAL_LM

    # Q-Lora Params
    q_lora: bool = False
    q_lora_bits: int = 4


#
# Configs
#
T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    def to_yaml(self, path: str) -> None:
        """Save the configuration to a YAML file."""
        OmegaConf.save(config=self, f=path)

    @classmethod
    def from_yaml(cls: Type[T], path: str) -> Type[T]:
        """Load a configuration from a YAML file.

        Args:
            path: The path to the YAML file.

        Returns:
            BaseConfig: The merged configuration object.
        """
        schema = OmegaConf.structured(cls)
        file_config = OmegaConf.load(path)
        config = OmegaConf.merge(schema, file_config)
        return config


@dataclass
class TrainingConfig(BaseConfig):
    data: DataParams = field(default_factory=DataParams)
    model: ModelParams = field(default_factory=ModelParams)
    training: TrainingParams = field(default_factory=TrainingParams)
    peft: PeftParams = field(default_factory=PeftParams)


@dataclass
class EvaluationConfig(BaseConfig):
    data: DataParams
    model: ModelParams
