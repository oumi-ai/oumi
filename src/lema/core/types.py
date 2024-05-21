from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar

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

    logger = "wandb", "tensorboard"
    run_name: str = "default"

    log_level: str = "info"
    dep_log_level: str = "warning"

    logging_strategy: Literal["no", "epoch", "steps"] = "steps"
    logging_dir = "output/runs"
    logging_steps: int = 50

    def to_hf(self):
        """Convert LeMa config to HuggingFace's TrainingArguments."""
        return transformers.TrainingArguments(
            log_level=self.dep_log_level,
            logging_dir=self.logging_dir,
            logging_nan_inf_filter=True,
            logging_steps=self.logging_steps,
            logging_strategy=self.logging_strategy,
            optim=self.optimizer,
            output_dir=self.output_dir,
            push_to_hub=False,
            run_name=self.run_name,
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
