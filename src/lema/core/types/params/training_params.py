from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import transformers
import trl

from lema.core.types.params.base_params import BaseParams
from lema.core.types.params.profiler_params import ProfilerParams
from lema.utils.str_utils import sanitize_run_name


class TrainerType(Enum):
    """Enum representing the supported trainers."""

    TRL_SFT = "trl_sft"
    "Supervised fine-tuning trainer from `trl` library."

    TRL_DPO = "trl_dpo"

    HF = "hf"
    "Generic HuggingFace trainer from `transformers` library."

    LEMA = "lema"
    "Custom generic trainer implementation."


class SchedulerType(str, Enum):
    """Enum representing the supported learning rate schedulers.

    For optional args for each scheduler, see src/lema/builders/lr_schedules.py.
    """

    LINEAR = "linear"
    "Linear scheduler."

    COSINE = "cosine"
    "Cosine scheduler."

    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    "Cosine with restarts scheduler."

    COSINE_WITH_MIN_LR = "cosine_with_min_lr"
    "Cosine with a minimum learning rate scheduler."

    CONSTANT = "constant"
    "Constant scheduler."


class MixedPrecisionDtype(str, Enum):
    """Enum representing the dtype used for mixed precision training.

    For more details on mixed-precision training, see:
    https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
    """

    NONE = "none"
    "No mixed precision. Uses `ModelParams.torch_dtype` as the dtype for all tensors "
    "(model weights, optimizer state, activations, etc.)."

    FP16 = "fp16"
    "fp16 mixed precision. Requires `ModelParams.torch_dtype` (the dtype of the model "
    "weights) to be fp32. The model weights and optimizer state are fp32, but some ops "
    "will run in fp16 to improve training speed."

    BF16 = "bf16"
    "Same as above, but with bf16 instead. This requires Ampere or higher NVIDIA "
    "architecture, or using CPU or Ascend NPU."


@dataclass
class TrainingParams(BaseParams):
    use_peft: bool = False
    trainer_type: TrainerType = TrainerType.HF
    enable_gradient_checkpointing: bool = False
    output_dir: str = "output"
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_steps: int = -1
    num_train_epochs: int = 3
    #: Save a checkpoint at the end of every epoch.
    save_epoch: bool = False
    #: Save a checkpoint every `save_steps`. If both `save_steps` and
    #: `save_epoch` are set, then `save_steps` takes precedence.
    #: To disable saving checkpoints during training,
    #: set `save_steps` to `0` and `save_epoch` to `False`.
    save_steps: int = 100
    #: Whether to save model at the end of training. Should normally be `True`
    #: but in some cases you may want to disable it e.g., if saving a large model
    #: takes a long time, and you want to quickly test training speed/metrics.
    save_final_model: bool = True
    #: Random seed, passed to the trainer and to all downstream dependencies
    seed: int = 42

    run_name: str = "default"

    #: The name of the metrics function in the LeMa registry to use for evaluation
    #: during training. The method must accept as input a HuggingFace EvalPrediction and
    #: return a dictionary of metrics, with string keys mapping to metric values. A
    #: single metrics_function may compute multiple metrics.
    metrics_function: Optional[str] = None

    log_level: str = "info"
    dep_log_level: str = "warning"

    enable_wandb: bool = False
    enable_tensorboard: bool = True

    logging_strategy: str = "steps"  #: possible values: "steps", "epoch", "no"
    logging_dir: str = "output/runs"
    logging_steps: int = 50

    # TODO consider using this with our logger too
    logging_first_step: bool = field(
        default=False,
        metadata={"help": "Whether to log and evaluate the first global_step or not."},
    )

    eval_strategy: str = "no"  #: possible values: "steps", "epoch", "no"
    eval_steps: int = 50

    #: Learning rate schedule.
    learning_rate: float = 5e-05
    #: See possible scheduler types here:
    #: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_utils.py#L408-L418
    lr_scheduler_type: str = "cosine"
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    warmup_ratio: Optional[float] = None
    warmup_steps: Optional[int] = None

    #: Optimizer params.
    optimizer: str = "adamw_torch"
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-08
    sgd_momentum: float = 0.9

    #: `use_reentrant` is a required parameter and is recommended to be set to False.
    #: See: https://pytorch.org/docs/stable/checkpoint.html
    gradient_checkpointing_kwargs: Dict[str, Any] = field(default_factory=dict)

    mixed_precision_dtype: MixedPrecisionDtype = MixedPrecisionDtype.NONE

    #: Whether to JIT compile the model. This param should be used instead of
    #: `ModelParams.compile` for training.
    compile: bool = False

    #: Whether to include performance metrics e.g., tokens stats
    include_performance_metrics: bool = False

    #: Whether to report alternative MFU metrics e.g., based on HuggingFace
    #: `total_flos`. This option is enabled only if `include_performance_metrics`
    #: is `True`.
    include_alternative_mfu_metrics: bool = False

    #: Whether to print model summary e.g., layer names, for informational purposes.
    log_model_summary: bool = False

    #: Whether to resume training by loading first the pointed model from this folder.
    resume_from_checkpoint: Optional[str] = None

    #: If True, try to find the last checkpoint in "output_dir".
    #: If present, training will resume from the model/optimizer/scheduler states loaded
    #: here. Otherwise (if checkpoint is not present), then training will continue
    #: w/o loading any intermediate checkpoints.
    #: NOTE: if `resume_from_checkpoint` is specified and contains a non-empty path,
    #: then this parameter has no effect.
    try_resume_from_last_checkpoint: bool = False

    #: Number of subprocesses to use for data loading (PyTorch only).
    #: 0 means that the data will be loaded in the main process.
    #:
    #: You can also use the special value "auto" to select the number
    #: of dataloader workers using a simple heuristic based on the number of CPU-s and
    #: GPU-s per node. Note that the accurate estimation of workers is difficult and
    #: depends on many factors (the properties of a model, dataset, VM, network, etc)
    #: so you can start with "auto" then experimentally tune the exact number to make it
    #: more optimal for your specific case. If "auto" is requested,
    #: then at minumum 1 worker is guaranteed to be assigned.
    dataloader_num_workers: Union[int, str] = 0

    #: Number of batches loaded in advance by each worker. 2 means there will be
    #: a total of 2 * num_workers batches prefetched across all workers.
    #: Can only be set if dataloader_num_workers >= 1.
    dataloader_prefetch_factor: Optional[int] = None

    #: If set to `True`, the dataloader is only iterated through on the main process
    #: (rank 0), then the batches are split and broadcast to each process.
    #: This can reduce the number of requests to the dataset, and helps ensure
    #: that each example is seen by max one GPU per epoch, but may become a performance
    #: bottleneck if a large number of GPUs is used.
    #: If set to `False`, the dataloader is iterated through on each
    #: GPU process.
    #: If set to `None` (*default*), then `True` or `False` is auto-selected based on
    #: heuristics (properties of dataset, the number of nodes and/or GPUs, etc).
    #: NOTE: We recommend to benchmark your setup, and configure `True` or `False`.
    dataloader_main_process_only: Optional[bool] = None

    #: When using distributed training, the value of the flag `find_unused_parameters`
    #: passed to `DistributedDataParallel`. Will default to `False` if gradient
    #: checkpointing is used, `True` otherwise.
    ddp_find_unused_parameters: Optional[bool] = None

    #: Maximum gradient norm (for gradient clipping) to avoid exploding gradients which
    #: can destabilize training.
    max_grad_norm: float = 1.0

    trainer_kwargs: Dict[str, Any] = field(default_factory=dict)

    #: Parameters for performance profiling.
    profiler: ProfilerParams = field(default_factory=ProfilerParams)

    def to_hf(self):
        """Converts LeMa config to HuggingFace's TrainingArguments."""
        save_strategy: str = "no"
        if self.save_epoch:
            save_strategy = "epoch"
        if self.save_steps > 0:
            save_strategy = "steps"

        dataloader_num_workers = 0
        if isinstance(self.dataloader_num_workers, int):
            dataloader_num_workers = self.dataloader_num_workers
        else:
            raise ValueError(
                "Unexpected type of dataloader_num_workers: "
                f"{type(self.dataloader_num_workers)} "
                f"({self.dataloader_num_workers}). Must be `int`."
            )

        if self.trainer_type == TrainerType.TRL_SFT:
            config_class = trl.SFTConfig
        elif self.trainer_type == TrainerType.TRL_DPO:
            config_class = trl.DPOConfig
        else:
            config_class = transformers.TrainingArguments

        dispatch_batches = self.dataloader_main_process_only

        result = config_class(
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
            warmup_ratio=self.warmup_ratio or 0.0,  # same default as transformers
            warmup_steps=self.warmup_steps or 0,  # same default as transformers
            weight_decay=self.weight_decay,
            adam_beta1=self.adam_beta1,
            adam_beta2=self.adam_beta2,
            adam_epsilon=self.adam_epsilon,
            gradient_checkpointing_kwargs=self.gradient_checkpointing_kwargs,
            include_tokens_per_second=self.include_performance_metrics,
            include_num_input_tokens_seen=self.include_performance_metrics,
            fp16=self.mixed_precision_dtype == MixedPrecisionDtype.FP16,
            bf16=self.mixed_precision_dtype == MixedPrecisionDtype.BF16,
            torch_compile=self.compile,
            save_steps=self.save_steps,
            save_strategy=save_strategy,
            logging_first_step=self.logging_first_step,
            resume_from_checkpoint=self.resume_from_checkpoint,
            eval_strategy=self.eval_strategy,
            eval_steps=self.eval_steps,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_prefetch_factor=(
                self.dataloader_prefetch_factor if dataloader_num_workers > 0 else None
            ),
            dataloader_pin_memory=True,  # Set it to True to be explicit.
            ddp_find_unused_parameters=self.ddp_find_unused_parameters,
            max_grad_norm=self.max_grad_norm,
            dispatch_batches=dispatch_batches,
            # TODO Switch to `accelerator_config` for `dispatch_batches`
            # accelerator_config={  # accelerator config for multi-device training
            #    "split_batches": False,
            #    "dispatch_batches": dispatch_batches,
            #    "even_batches": True,
            #    "use_seedable_sampler": True,
            # },
            seed=self.seed,
            # TODO Re-enable `data_seed`. Should it depend on RANK?
            # data_seed=self.seed,
            **self.trainer_kwargs,
        )
        assert isinstance(result, transformers.TrainingArguments)
        return result

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

    def __post_init__(self):
        """Verifies params."""
        self.run_name = sanitize_run_name(self.run_name)

        if isinstance(self.dataloader_num_workers, str) and not (
            self.dataloader_num_workers == "auto"
        ):
            raise ValueError(
                "Unknown value of "
                f"dataloader_num_workers: {self.dataloader_num_workers}"
            )

        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1.")
