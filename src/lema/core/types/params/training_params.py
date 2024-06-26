from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import transformers


class TrainerType(Enum):
    """Enum representing the supported trainers."""

    TRL_SFT = "trl_sft"
    "Supervised fine-tuning trainer from `trl` library."

    TRL_DPO = "trl_dpo"

    HF = "hf"
    "Generic HuggingFace trainer from `transformers` library."


@dataclass
class TrainingParams:
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

    # The name of the metrics function in the LeMa registry to use for evaluation
    # during training. The method must accept as input a HuggingFace EvalPrediction and
    # return a dictionary of metrics, with string keys mapping to metric values. A
    # single metrics_function may compute multiple metrics.
    metrics_function: Optional[str] = None

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

    eval_strategy: str = "no"  # possible values: "steps", "epoch", "no"
    eval_steps: int = 50

    # Learning rate schedule.
    learning_rate: float = 5e-05
    # See possible scheduler types here:
    # https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_utils.py#L408-L418
    lr_scheduler_type: str = "cosine"
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    warmup_ratio: float = 0.0
    warmup_steps: int = 0

    # Optimizer params.
    optimizer: str = "adamw_torch"
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

    trainer_kwargs: Dict[str, Any] = field(default_factory=dict)

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
            num_train_epochs=self.num_train_epochs,
            output_dir=self.output_dir,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            per_device_train_batch_size=self.per_device_train_batch_size,
            push_to_hub=False,
            report_to=self._get_hf_report_to(),
            run_name=self.run_name,
            optim=self.optimizer,
            learning_rate=self.learning_rate,
            lr_scheduler_type=self.lr_scheduler_type,
            lr_scheduler_kwargs=self.lr_scheduler_kwargs,
            warmup_ratio=self.warmup_ratio,
            warmup_steps=self.warmup_steps,
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
            eval_strategy=self.eval_strategy,
            eval_steps=self.eval_steps,
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
