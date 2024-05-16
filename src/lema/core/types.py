from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

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

    def to_hf(self):
        """Convert LeMa config to HuggingFace's TrainingArguments."""
        return transformers.TrainingArguments(
            optim=self.optimizer, output_dir=self.output_dir
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
@dataclass
class BaseConfig:
    def to_yaml(self, path: str) -> None:
        """Save the configuration to a YAML file."""
        OmegaConf.save(config=self, f=path)


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
