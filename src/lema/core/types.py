from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar, cast

import torch
import transformers
from omegaconf import MISSING, OmegaConf
from peft.utils.peft_types import TaskType

from lema.logging import logger


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
    trainer_type: TrainerType = TrainerType.HF
    enable_gradient_checkpointing: bool = False
    output_dir: str = "output"
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_steps: int = -1
    num_train_epochs: int = 3

    run_name: str = "default"

    log_level: str = "info"
    dep_log_level: str = "warning"

    enable_wandb: bool = False
    enable_tensorboard: bool = True

    logging_strategy: str = "steps"  # possible values: "steps", "epoch", "no"
    logging_dir: str = "output/runs"
    logging_steps: int = 50

    learning_rate: float = 5e-05
    lr_scheduler_type: str = "cosine"  # TODO Update by enumerating *more* options
    warmup_ratio: float = 0.0

    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-08

    gradient_checkpointing_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Whether to include performance metrics e.g., tokens stats
    include_performance_metrics: Optional[bool] = None

    def to_hf(self):
        """Convert LeMa config to HuggingFace's TrainingArguments."""
        return transformers.TrainingArguments(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            log_level=self.dep_log_level,
            logging_dir=self.logging_dir,
            logging_nan_inf_filter=True,
            logging_steps=self.logging_steps,
            logging_strategy=self.logging_strategy,
            max_steps=self.max_steps,
            optim=self.optimizer,
            output_dir=self.output_dir,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            per_device_train_batch_size=self.per_device_train_batch_size,
            push_to_hub=False,
            report_to=self._get_hf_report_to(),
            run_name=self.run_name,
            learning_rate=self.learning_rate,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            adam_beta1=self.adam_beta1,
            adam_beta2=self.adam_beta2,
            adam_epsilon=self.adam_epsilon,
            gradient_checkpointing_kwargs=self.gradient_checkpointing_kwargs,
            include_tokens_per_second=self.include_performance_metrics,
            include_num_input_tokens_seen=self.include_performance_metrics,
        )

    def _get_hf_report_to(self) -> List[str]:
        """Get the list of reporting tools enabled for the current instance.

        Returns:
            list: A list of reporting tools enabled.
                Possible values are "wandb", "tensorboard", or "none".
        """
        report_to = []
        if self.enable_wandb:
            report_to.append("wandb")
        if self.enable_tensorboard:
            report_to.append("tensorboard")
        if len(report_to) == 0:
            report_to.append("none")
        return report_to


@dataclass
class DataParams:
    # Parameters for `datasets.load_dataset()`
    dataset_name: str = MISSING
    dataset_config: Optional[str] = None
    split: str = "train"
    stream: bool = False

    # Whether to pack the text into constant-length chunks,
    # each the size of the model's max input length.
    # This will stream the dataset, and tokenize on the fly
    # if the dataset isn't already tokenized (i.e. has an `input_ids` column).
    # Requires `stream` to be set to True.
    pack: bool = False

    @staticmethod
    def _default_factory_preprocessing_kwargs() -> dict:
        """Create default param values the data preprocessing mapping (.map) function.

        Returns:
        dict: contains the default set params.
        """
        defaults = dict()
        defaults["batched"] = True  # Note the default of huggingface is False.
        return defaults

    # The dataset column name containing the text to train on. Required for SFTTrainer.
    text_col: Optional[str] = None
    preprocessing_function_name: Optional[str] = None
    preprocessing_function_kwargs: Dict[str, Any] = field(
        default_factory=_default_factory_preprocessing_kwargs
    )
    trainer_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Verify params."""
        if self.pack:
            if not self.stream:
                raise ValueError("`stream` must be enabled if `pack` is enabled.")
            if not self.text_col:
                raise ValueError("`text_col` must be specified if `pack` is enabled.")


@dataclass
class ModelParams:
    model_name: str = MISSING
    tokenizer_name: Optional[str] = None
    model_max_length: Optional[int] = None
    trust_remote_code: bool = False
    torch_dtype_str: str = "float32"
    chat_template: Optional[str] = None

    def torch_dtype(self):
        """Convert string dtype to torch.dtype."""
        if self.torch_dtype_str in ["f64", "float64", "double"]:
            return torch.float64
        elif self.torch_dtype_str in ["f32", "float32", "float"]:
            return torch.float32
        elif self.torch_dtype_str in ["bf16", "bfloat16"]:
            return torch.bfloat16
        elif self.torch_dtype_str in ["f16", "float16", "half"]:
            return torch.float16
        else:
            raise ValueError(f"Unsupported data type: {self.torch_dtype_str}")


@dataclass
class PeftParams:
    # Lora Params
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA R value."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout."},
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "LoRA target modules."},
    )
    lora_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Model layers to unfreeze and train."},
    )
    lora_bias: str = field(
        default="none",
        metadata={
            "help": (
                "Bias type for Lora. Can be 'none', 'all' or 'lora_only'. "
                "If 'all' or 'lora_only', the corresponding biases will "
                "be updated during training. Be aware that this means that, "
                "even when disabling the adapters, the model will not "
                "produce the same output as the base model would have "
                "without adaptation."
                "NOTE: see: "
                "https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py"
                "for more details."
            )
        },
    )

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
    def from_yaml(cls: Type[T], path: str) -> T:
        """Load a configuration from a YAML file.

        Args:
            path: The path to the YAML file.

        Returns:
            BaseConfig: The merged configuration object.
        """
        schema = OmegaConf.structured(cls)
        file_config = OmegaConf.load(path)
        config = OmegaConf.to_object(OmegaConf.merge(schema, file_config))
        if not isinstance(config, cls):
            raise TypeError(f"config is not {cls}")
        return cast(cls, config)


@dataclass
class TrainingConfig(BaseConfig):
    data: DataParams = field(default_factory=DataParams)
    model: ModelParams = field(default_factory=ModelParams)
    training: TrainingParams = field(default_factory=TrainingParams)
    peft: PeftParams = field(default_factory=PeftParams)

    def __post_init__(self):
        """Verify/populate params."""
        if self.training.trainer_type == TrainerType.TRL_SFT:
            if not self.data.text_col:
                raise ValueError("`text_col` must be specified for TRL_SFT Trainer.")

            # Set `dataset_text_field` in `trainer_kwargs` since it's requried for
            # `SFTTrainer`, and warn users if their value will be overridden.
            existing_dataset_text_field = self.data.trainer_kwargs.get(
                "dataset_text_field"
            )
            if (
                existing_dataset_text_field is not None
            ) and existing_dataset_text_field != self.data.text_col:
                logger.warning(
                    "Overriding existing `dataset_text_field` value "
                    f'"{existing_dataset_text_field}" with "{self.data.text_col}"'
                )
            self.data.trainer_kwargs["dataset_text_field"] = self.data.text_col


@dataclass
class EvaluationConfig(BaseConfig):
    data: DataParams
    model: ModelParams


@dataclass
class GenerationConfig(BaseConfig):
    # TODO: Add more parameters to control text generation.
    max_new_tokens: int = 256


@dataclass
class InferenceConfig(BaseConfig):
    model: ModelParams = field(default_factory=ModelParams)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
