import dataclasses
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    cast,
)

import torch
import transformers
from omegaconf import MISSING, OmegaConf
from peft.utils.peft_types import TaskType
from transformers.utils import is_flash_attn_2_available

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
    save_steps: int = 100

    run_name: str = "default"

    log_level: str = "info"
    dep_log_level: str = "warning"

    enable_wandb: bool = False
    enable_tensorboard: bool = True

    logging_strategy: str = "steps"  # possible values: "steps", "epoch", "no"
    logging_dir: str = "output/runs"
    logging_steps: int = 50

    # TODO consider using this with our logger too
    logging_first_step: bool = field(
        default=False,
        metadata={"help": "Whether to log and evaluate the first global_step or not."},
    )

    learning_rate: float = 5e-05
    lr_scheduler_type: str = "cosine"  # TODO Update by enumerating *more* options
    warmup_ratio: float = 0.0

    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-08

    gradient_checkpointing_kwargs: Dict[str, Any] = field(default_factory=dict)

    fp16: bool = False  # 16-bit (mixed) precision training instead of 32-bit training
    bf16: bool = False  # Whether to use bf16 16-bit (mixed) precision training instead
    # of 32-bit training. Requires Ampere or higher NVIDIA architecture
    # or using CPU or Ascend NPU.

    # Whether to include performance metrics e.g., tokens stats
    include_performance_metrics: Optional[bool] = None

    # Whether to print model summary e.g., layer names, for informational purposes.
    log_model_summary: bool = False

    # Whether to resume training by loading first the pointed model from this folder.
    resume_from_checkpoint: Optional[str] = None

    # If True, try to find the last checkpoint in "output_dir".
    # If present, training will resume from the model/optimizer/scheduler states loaded
    # here. Otherwise (if checkpoint is not present), then training will continue
    # w/o loading any intermediate checkpoints.
    # NOTE: if `resume_from_checkpoint` is specified and contains a non-empty path,
    # then this parameter has no effect.
    try_resume_from_last_checkpoint: bool = False

    def to_hf(self):
        """Converts LeMa config to HuggingFace's TrainingArguments."""
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
            fp16=self.fp16,
            bf16=self.bf16,
            save_steps=self.save_steps,
            logging_first_step=self.logging_first_step,
            resume_from_checkpoint=self.resume_from_checkpoint,
        )

    def _get_hf_report_to(self) -> List[str]:
        """Gets the list of reporting tools enabled for the current instance.

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
        """Creates default param values for the data preprocessing .map function.

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
        """Verifies params."""
        if self.pack:
            if not self.stream:
                raise ValueError("`stream` must be enabled if `pack` is enabled.")
            if not self.text_col:
                raise ValueError("`text_col` must be specified if `pack` is enabled.")


@dataclass
class ModelParams:
    model_name: str = MISSING
    adapter_model: Optional[str] = None
    tokenizer_name: Optional[str] = None
    model_max_length: Optional[int] = None
    trust_remote_code: bool = False
    torch_dtype_str: str = "float32"
    chat_template: Optional[str] = None
    attn_implementation: Optional[str] = None

    def torch_dtype(self):
        """Converts string dtype to torch.dtype."""
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

    def __post_init__(self):
        """Verifies params."""
        # check if flash-attention-2 is requested and supported
        if (self.attn_implementation == "flash_attention_2") and (
            not is_flash_attn_2_available()
        ):
            raise ValueError(
                "Flash attention 2 was requested but it is not "
                "supported. Confirm that your hardware is compatible and then "
                "consider installing it: pip install -U flash-attn --no-build-isolation"
            )

        # check if flash-attention-2 is requested with half-precision
        if (self.attn_implementation == "flash_attention_2") and (
            self.torch_dtype() not in [torch.bfloat16, torch.float16]
        ):
            logger.warning(
                "Cannot use flash_attention_2 with a full-precision "
                f"({self.torch_dtype()}) model. Ignoring request for using "
                "flash_attention_2 by setting attn_implementation system's default."
            )
            self.attn_implementation = None

    @property
    def should_use_flash_attention_2(self) -> bool:
        """Checks if flash-attention-2 was requested.

        Note: Flash attention 2 paper https://arxiv.org/abs/2307.08691
        TODO add flash-attention-2 in optional dependecies if we want to
        use it frequently (.toml).
        """
        return self.attn_implementation == "flash_attention_2"


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
    q_lora: bool = field(default=False, metadata={"help": "Use model quantization."})
    q_lora_bits: int = field(
        default=4, metadata={"help": "Quantization (precision) bits."}
    )
    # FIXME the names below use the bnb short for bits-and bytes
    # If we consider wrapping more quantization libraries a better
    # naming convention should be applied.
    bnb_4bit_quant_type: str = field(
        default="fp4", metadata={"help": "4-bit quantization type (fp4 or nf4)."}
    )
    use_bnb_nested_quant: bool = field(
        default=False, metadata={"help": "Use nested quantization."}
    )
    bnb_4bit_quant_storage: str = field(
        default="uint8",
        metadata={"help": "Storage type to pack the quanitzed 4-bit prarams."},
    )

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """An iterator over field names and values."""
        for param in dataclasses.fields(self):
            yield param.name, getattr(self, param.name)


#
# Configs
#
T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    def to_yaml(self, config_path: str) -> None:
        """Saves the configuration to a YAML file."""
        OmegaConf.save(config=self, f=config_path)

    @classmethod
    def from_yaml(cls: Type[T], config_path: str) -> T:
        """Loads a configuration from a YAML file.

        Args:
            config_path: The path to the YAML file.

        Returns:
            BaseConfig: The merged configuration object.
        """
        schema = OmegaConf.structured(cls)
        file_config = OmegaConf.load(config_path)
        config = OmegaConf.to_object(OmegaConf.merge(schema, file_config))
        if not isinstance(config, cls):
            raise TypeError(f"config is not {cls}")
        return cast(cls, config)

    @classmethod
    def from_yaml_and_arg_list(
        cls: Type[T],
        config_path: Optional[str],
        arg_list: List[str],
        logger: Optional[logging.Logger] = None,
    ) -> T:
        """Loads a configuration from various sources.

        If both YAML and arguments list are provided, then
        parameters specified in `arg_list` have higher precedence.

        Args:
            config_path: The path to the YAML file.
            arg_list: Command line arguments list.
            logger: (optional) Logger.

        Returns:
            BaseConfig: The merged configuration object.
        """
        # Start with an empty typed config. This forces OmegaConf to validate
        # that all other configs are of this structured type as well.
        all_configs = [OmegaConf.structured(cls)]

        # Override with configuration file if provided.
        if config_path is not None:
            all_configs.append(cls.from_yaml(config_path))

        # Override with CLI arguments.
        all_configs.append(OmegaConf.from_cli(arg_list))
        try:
            # Merge and validate configs
            config = OmegaConf.merge(*all_configs)
        except Exception:
            if logger:
                logger.exception(f"Failed to merge Omega configs: {all_configs}")
            raise

        config = OmegaConf.to_object(config)
        if not isinstance(config, cls):
            raise TypeError(f"config {type(config)} is not {type(cls)}")

        return cast(cls, config)


@dataclass
class TrainingConfig(BaseConfig):
    data: DataParams = field(default_factory=DataParams)
    model: ModelParams = field(default_factory=ModelParams)
    training: TrainingParams = field(default_factory=TrainingParams)
    peft: PeftParams = field(default_factory=PeftParams)

    def __post_init__(self):
        """Verifies/populates params."""
        if self.training.trainer_type == TrainerType.TRL_SFT:
            if not self.data.text_col:
                raise ValueError("`text_col` must be specified for TRL_SFT Trainer.")

            # Set `dataset_text_field` in `trainer_kwargs` since it's requried for
            # `SFTTrainer`, and warn users if their value will be overridden.
            existing_dataset_text_field = self.data.trainer_kwargs.get(
                "dataset_text_field"
            )
            if (existing_dataset_text_field is not None) and (
                existing_dataset_text_field != self.data.text_col
            ):
                logger.warning(
                    "Overriding existing `dataset_text_field` value "
                    f"'{existing_dataset_text_field}' with '{self.data.text_col}'"
                )
            self.data.trainer_kwargs["dataset_text_field"] = self.data.text_col

        if self.model.model_max_length and self.model.model_max_length > 0:
            max_seq_length_value = int(self.model.model_max_length)
            max_seq_length_key = None
            if self.training.trainer_type == TrainerType.TRL_SFT:
                max_seq_length_key = "max_seq_length"
            elif self.training.trainer_type == TrainerType.TRL_DPO:
                max_seq_length_key = "max_length"
                # TODO: DPOTrainer also defines "max_prompt_length" and
                # "max_target_length". How to handle them?
            else:
                logger.warning(
                    f"Ignored model.model_max_length={max_seq_length_value} config "
                    f"parameter for trainer {self.training.trainer_type}."
                )

            if max_seq_length_key:
                existing_max_seq_length = self.data.trainer_kwargs.get(
                    max_seq_length_key
                )
                if (existing_max_seq_length is not None) and (
                    existing_max_seq_length != max_seq_length_value
                ):
                    logger.warning(
                        f"Overriding existing '{max_seq_length_key}' value "
                        f"'{existing_max_seq_length}' with '{max_seq_length_value}'"
                    )
                self.data.trainer_kwargs[max_seq_length_key] = max_seq_length_value


@dataclass
class GenerationConfig(BaseConfig):
    # TODO: Add more parameters to control text generation.
    max_new_tokens: int = 256


@dataclass
class InferenceConfig(BaseConfig):
    model: ModelParams = field(default_factory=ModelParams)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


@dataclass
class EvaluationConfig(BaseConfig):
    data: DataParams = field(default_factory=DataParams)
    model: ModelParams = field(default_factory=ModelParams)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    batch_size: int = 8
