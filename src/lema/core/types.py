from dataclasses import dataclass, field
from typing import List, Optional

import transformers
from peft import TaskType


#
# Training Params
#
@dataclass
class TrainingParams(transformers.TrainingArguments):
    optim: str = "adamw_torch"
    use_peft: bool = False
    trainer_name: str = "trl_sft"

    enable_gradient_checkpointing: bool = False


@dataclass
class DataParams:
    dataset_name: Optional[str] = None

    preprocessing_function_name: Optional[str] = None

    trainer_kwargs: Optional[dict] = field(
        default_factory=lambda: {
            "dataset_text_field": "prompt"
        }  # TODO: remove this default
    )


@dataclass
class ModelParams:
    model_name: str
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
    pass


@dataclass
class TrainingConfig(BaseConfig):
    data_params: DataParams
    model_params: ModelParams
    training_params: TrainingParams
    peft_params: PeftParams


@dataclass
class EvaluationConfig(BaseConfig):
    data_params: DataParams
    model_params: ModelParams
