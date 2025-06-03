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

import copy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import transformers
import trl

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.grpo_params import GrpoParams
from oumi.core.configs.params.profiler_params import ProfilerParams
from oumi.core.configs.params.telemetry_params import TelemetryParams
from oumi.utils.str_utils import sanitize_run_name


class TrainerType(Enum):
    """Enum representing the supported trainers."""

    TRL_SFT = "trl_sft"
    """Supervised fine-tuning trainer from `trl` library.

    This trainer is specifically designed for supervised fine-tuning tasks
    using the TRL (Transformer Reinforcement Learning) library.
    """

    TRL_DPO = "trl_dpo"
    """Direct Preference Optimization trainer from `trl` library.

    This trainer implements the Direct Preference Optimization algorithm
    for fine-tuning language models based on human preferences.
    """

    TRL_GRPO = "trl_grpo"
    """Group Relative Policy Optimization trainer from `trl` library.

    This trainer implements the Group Relative Policy Optimization algorithm
    introduced in the paper https://arxiv.org/pdf/2402.03300
    for fine-tuning language models.
    Optionally, supports user-defined reward functions.
    """

    HF = "hf"
    """Generic HuggingFace trainer from `transformers` library.

    This is the standard trainer provided by the Hugging Face Transformers
    library, suitable for a wide range of training tasks.
    """

    OUMI = "oumi"
    """Custom generic trainer implementation.

    This is a custom trainer implementation specific to the Oumi project,
    designed to provide additional flexibility and features.
    """

    VERL_GRPO = "verl_grpo"
    """Group Relative Policy Optimization trainer from `verl` library.

    This trainer implements the Group Relative Policy Optimization algorithm
    introduced in the paper https://arxiv.org/pdf/2402.03300
    for fine-tuning language models.
    Optionally, supports user-defined reward functions.
    """


class SchedulerType(str, Enum):
    """Enum representing the supported learning rate schedulers.

    For optional args for each scheduler, see src/oumi/builders/lr_schedules.py.
    """

    LINEAR = "linear"
    """Linear scheduler.

    Decreases the learning rate linearly from the initial value to 0 over the course
    of training.
    """

    COSINE = "cosine"
    """Cosine scheduler.

    Decays the learning rate following the decreasing part of a cosine curve.
    """

    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    """Cosine with restarts scheduler.

    Decays the learning rate following a cosine curve with periodic restarts.
    """

    COSINE_WITH_MIN_LR = "cosine_with_min_lr"
    """Cosine with a minimum learning rate scheduler.

    Similar to cosine scheduler, but maintains a minimum learning rate at the end.
    """

    CONSTANT = "constant"
    """Constant scheduler.

    Keeps the learning rate constant throughout training.
    """


class MixedPrecisionDtype(str, Enum):
    """Enum representing the dtype used for mixed precision training.

    For more details on mixed-precision training, see:
    https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
    """

    NONE = "none"
    """No mixed precision.

    Uses `ModelParams.torch_dtype` as the dtype for all tensors
    (model weights, optimizer state, activations, etc.).
    """

    FP16 = "fp16"
    """fp16 mixed precision.

    Requires `ModelParams.torch_dtype` (the dtype of the model
    weights) to be fp32. The model weights and optimizer state are fp32, but some ops
    will run in fp16 to improve training speed.
    """

    BF16 = "bf16"
    """Similar to fp16 mixed precision, but with bf16 instead.

    This requires Ampere or higher NVIDIA architecture, or using CPU or Ascend NPU.
    """


@dataclass
class TrainingParams(BaseParams):
    use_peft: bool = False
    """Whether to use Parameter-Efficient Fine-Tuning (PEFT) techniques.

    PEFT methods allow for efficient adaptation of pre-trained language models
    to specific tasks by only updating a small number of (extra) model parameters.
    This can significantly reduce memory usage and training time.
    """

    trainer_type: TrainerType = TrainerType.HF
    """The type of trainer to use for the training process.

    Options are defined in the `TrainerType` enum and include:
    - HF: HuggingFace's Trainer
    - TRL_SFT: TRL's SFT Trainer
    - TRL_DPO: TRL's DPO Trainer
    - TRL_GRPO: TRL's GRPO Trainer
    - OUMI: Custom generic trainer implementation
    - VERL_GRPO: verl's GRPO Trainer
    """

    enable_gradient_checkpointing: bool = False
    """Whether to enable gradient checkpointing to save memory at the expense of speed.

    Gradient checkpointing works by trading compute for memory. Rather than storing
    all intermediate activations of the entire computation graph for computing
    backward pass, it recomputes these activations during the backward pass.
    This can make the training slower, but it can also significantly reduce memory
    usage.
    """

    gradient_checkpointing_kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments for gradient checkpointing.

    The `use_reentrant` parameter is required and is recommended to be set to False.
    For more details, see: https://pytorch.org/docs/stable/checkpoint.html
    """

    output_dir: str = "output"
    """Directory where the output files will be saved.

    This includes checkpoints, evaluation results, and any other artifacts
    produced during the training process.
    """

    per_device_train_batch_size: int = 8
    """Number of samples per batch on each device during training.

    This parameter directly affects memory usage and training speed. Larger batch
    sizes generally lead to better utilization of GPU compute capabilities but
    require more memory.
    """

    per_device_eval_batch_size: int = 8
    """Number of samples per batch on each device during evaluation.

    Similar to `per_device_train_batch_size`, but used during evaluation phases.
    Can often be set higher than the train batch size as no gradients are stored.
    """

    gradient_accumulation_steps: int = 1
    """Number of update steps to accumulate before performing a backward/update pass.

    This technique allows for effectively larger batch sizes and is especially
    useful when such batch sizes would not fit in memory. This is achieved by
    accumulating gradients from multiple forward passes before performing
    a single optimization step. Setting this to >1 can increase however
    memory usage for training setups without existing gradient accumulation buffers
    (ex. 1-GPU training).
    """

    max_steps: int = -1
    """If set to a positive number, the total number of training steps to perform.

    This parameter overrides `num_train_epochs`. If set to -1 (default),
    the number of training steps is determined by `num_train_epochs`.
    """

    num_train_epochs: int = 3
    """Total number of training epochs to perform (if `max_steps` is not specified).

    An epoch is one complete pass through the entire training dataset. This parameter
    is ignored if `max_steps` is set to a positive number.
    """

    save_epoch: bool = False
    """Save a checkpoint at the end of every epoch.

    When set to True, this ensures that a model checkpoint is saved after
    each complete pass through the training data. This can be useful for
    tracking model progress over time and for resuming training from a
    specific epoch if needed.

    If both `save_steps` and `save_epoch` are set, then `save_steps` takes precedence.
    """

    save_steps: int = 500
    """Save a checkpoint every `save_steps` training steps.

    This parameter determines the frequency of saving checkpoints during
    training based on the number of steps. If both `save_steps` and
    `save_epoch` are set, then `save_steps` takes precedence.

    To disable saving checkpoints during training, set `save_steps` to `0`
    and `save_epoch` to `False`. If enabled, a checkpoint will be saved at the end of
    training if there's any residual steps left.
    """

    save_final_model: bool = True
    """Whether to save the model at the end of training.

    For different options for saving PEFT models, see `PeftParams.peft_save_mode`.
    This should normally be set to `True` to ensure the final trained model
    is saved. However, in some cases, you may want to disable it, for example:
    - If saving a large model which takes a long time
    - When quickly testing training speed or metrics
    - During debugging or experimentation phases
    """

    seed: int = 42
    """Random seed used for initialization.

    This seed is passed to the trainer and to all downstream dependencies
    to ensure reproducibility of results. It affects random number generation
    in various parts of the training process, including data shuffling,
    weight initialization, and any stochastic operations.
    """

    data_seed: int = 42
    """Random data_seed used for initialization.
    The seed to use for the underlying generator when using
    use_seedable_sampler. If None, the generator will use
    the current default seed from torch.
    Used only by the HuggingFace trainers.
    """

    use_deterministic: bool = False
    """Whether to use deterministic algorithms for reproducibility.
    If set to True, this will only allow those CuDNN algorithms
    that are (believed to be) deterministic. Please refer to
    https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    for more details. If using distributed training,
    this will override ddp_find_unused_parameters to False and will
    also use ddp_broadcast_buffers, and disable gradient checkpointing.
    Note that this will not guarantee full reproducibility,
    but will help to reduce the variance between runs.
    """

    full_determinism: bool = False
    """If True, enable_full_determinism() is called instead of set_seed()
    to ensure reproducible results in distributed training. This will only
    affect HF trainers. Important: this will negatively impact performance,
    so only use it for debugging.
    """

    run_name: Optional[str] = None
    """A unique identifier for the current training run.

    This name is used to identify the run in logging outputs, saved model
    checkpoints, and experiment tracking tools like Weights & Biases or
    TensorBoard. It's particularly useful when running multiple experiments
    or when you want to easily distinguish between different training sessions.
    """

    metrics_function: Optional[str] = None
    """The name of the metrics function in the Oumi registry to use for evaluation
    during training.

    The method must accept as input a HuggingFace EvalPrediction and
    return a dictionary of metrics, with string keys mapping to metric values. A
    single metrics_function may compute multiple metrics.
    """

    reward_functions: Optional[list[str]] = None
    """The names of the reward function in the Oumi registry to use for reinforcement
    learning.

    Only supported with the TRL_GRPO and VERL_GRPO trainers. Currently,
    VERL_GRPO only supports specifying a single reward function.

    For TRL_GRPO, refer to https://huggingface.co/docs/trl/main/en/grpo_trainer
    for documentation about the function signature.

    For VERL_GRPO, refer to
    https://verl.readthedocs.io/en/latest/preparation/reward_function.html
    for documentation about the function signature.
    """

    grpo: GrpoParams = field(default_factory=GrpoParams)
    """Parameters for GRPO training."""

    log_level: str = "info"
    """The logging level for the main Oumi logger.

    Possible values are "debug", "info", "warning", "error", "critical".
    """

    dep_log_level: str = "warning"
    """The logging level for dependency loggers (e.g., HuggingFace, PyTorch).

    Possible values are "debug", "info", "warning", "error", "critical".
    """

    enable_wandb: bool = False
    """Whether to enable Weights & Biases (wandb) logging.

    If True, wandb will be used for experiment tracking and visualization.
    Wandb will also log a summary of the training run, including hyperparameters,
    metrics, and other relevant information at the end of training.

    After enabling, you must set the `WANDB_API_KEY` environment variable.
    Alternatively, you can use the `wandb login` command to authenticate.
    """

    enable_mlflow: bool = False
    """Whether to enable MLflow logging.

    If True, MLflow will be used for experiment tracking and visualization.
    If you want to use MLflow, you must set the `MLFLOW_TRACKING_URI` environment
    variable to specify the tracking server URI and the `MLFLOW_EXPERIMENT_ID` or
    `MLFLOW_EXPERIMENT_NAME` environment variable to specify the experiment to report
    the run to.
    """

    enable_tensorboard: bool = True
    """Whether to enable TensorBoard logging.

    If True, TensorBoard will be used for logging metrics and visualizations.
    """

    logging_strategy: str = "steps"
    """The strategy to use for logging during training.

    Possible values are:
    - "steps": Log every `logging_steps` steps.
    - "epoch": Log at the end of each epoch.
    - "no": Disable logging.
    """

    logging_dir: Optional[str] = None
    """The directory where training logs will be saved.

    This includes TensorBoard logs and other training-related output.
    """

    logging_steps: int = 50
    """Number of update steps between two logs if logging_strategy="steps".

    Ignored if logging_strategy is not "steps".
    """

    logging_first_step: bool = field(
        default=False,
        metadata={"help": "Whether to log and evaluate the first global_step or not."},
    )
    """Whether to log and evaluate the first global step.

    If True, metrics will be logged and evaluation will be performed at the very
    beginning of training. Skipping the first step can be useful to avoid logging
    and evaluation of the initial random model.

    The first step is usually not representative of the model's performance, as it
    includes model compilation, optimizer initialization, and other setup steps.
    """

    eval_strategy: str = "no"
    """The strategy to use for evaluation during training.

    Possible values:
    - "no": No evaluation is done during training.
    - "steps": Evaluation is done every `eval_steps`.
    - "epoch": Evaluation is done at the end of each epoch.
    """

    eval_steps: int = 500
    """Number of update steps between two evaluations if eval_strategy="steps".

    Ignored if eval_strategy is not "steps".
    """

    learning_rate: float = 5e-05
    """The initial learning rate for the optimizer.

    This value can be adjusted by the learning rate scheduler during training.
    """

    lr_scheduler_type: str = "linear"
    """The type of learning rate scheduler to use.

    Possible values include "linear", "cosine", "cosine_with_restarts",
      "cosine_with_min_lr" and "constant".

    See `src/oumi/builders/lr_schedules.py` for more details on each scheduler.
    """

    lr_scheduler_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments to pass to the learning rate scheduler.

    These arguments can be used to fine-tune the behavior of the chosen scheduler.
    """

    warmup_ratio: Optional[float] = None
    """The ratio of total training steps used for a linear warmup from 0 to the
    learning rate.

    If set along with `warmup_steps`, this value will be ignored.
    """

    warmup_steps: Optional[int] = None
    """The number of steps for the warmup phase of the learning rate scheduler.

    If set, will override the value of `warmup_ratio`.
    """

    # ---------------------
    # Optimizer params.
    # ---------------------
    optimizer: str = "adamw_torch"
    """The optimizer to use for training.

    See pytorch documentation for more information on available optimizers:
    https://pytorch.org/docs/stable/optim.html

    Default is "adamw_torch" (AdamW implemented by PyTorch).
    """

    weight_decay: float = 0.0
    """Weight decay (L2 penalty) to apply to the model's parameters.

    In the HF trainers and the OUMI trainer, this is automatically applied to only
    weight tensors, and skips biases/layernorms.

    Default is 0.0 (no weight decay).
    """

    adam_beta1: float = 0.9
    """The beta1 parameter for Adam-based optimizers.

    Exponential decay rate for the first moment estimates.
    Default is 0.9.
    """

    adam_beta2: float = 0.999
    """The beta2 parameter for Adam-based optimizers.

    Exponential decay rate for the second moment estimates.
    Default is 0.999.
    """

    adam_epsilon: float = 1e-08
    """Epsilon parameter for Adam-based optimizers.

    Small constant for numerical stability.
    Default is 1e-08.
    """

    sgd_momentum: float = 0.0
    """Momentum factor for SGD optimizer.

    Only used when optimizer is set to "sgd", and when `trainer_type` is set to OUMI.
    Default is 0.0.
    """

    mixed_precision_dtype: MixedPrecisionDtype = MixedPrecisionDtype.NONE
    """The data type to use for mixed precision training.

    Default is NONE, which means no mixed precision is used.
    """

    compile: bool = False
    """Whether to JIT compile the model.

    This parameter should be used instead of `ModelParams.compile` for training.
    """

    include_performance_metrics: bool = False
    """Whether to include performance metrics such as token statistics."""

    include_alternative_mfu_metrics: bool = False
    """Whether to report alternative MFU (Model FLOPs Utilization) metrics.

    These metrics are based on HuggingFace's `total_flos`.
    This option is only used if `include_performance_metrics` is `True`.
    """

    log_model_summary: bool = False
    """Whether to print a model summary, including layer names."""

    resume_from_checkpoint: Optional[str] = None
    """Path to a checkpoint folder from which to resume training.

    If specified, training will resume by first loading the model from this folder.
    """

    try_resume_from_last_checkpoint: bool = False
    """If True, attempt to resume from the last checkpoint in "output_dir".

    If a checkpoint is found, training will resume from the model/optimizer/scheduler
    states loaded from this checkpoint. If no checkpoint is found, training will
    continue without loading any intermediate checkpoints.

    Note: If `resume_from_checkpoint` is specified and contains a non-empty path,
    this parameter has no effect.
    """

    dataloader_num_workers: Union[int, str] = 0
    """Number of subprocesses to use for data loading (PyTorch only).
    0 means that the data will be loaded in the main process.

    You can also use the special value "auto" to select the number
    of dataloader workers using a simple heuristic based on the number of CPU-s and
    GPU-s per node. Note that the accurate estimation of workers is difficult and
    depends on many factors (the properties of a model, dataset, VM, network, etc)
    so you can start with "auto" then experimentally tune the exact number to make it
    more optimal for your specific case. If "auto" is requested,
    then at minimum 1 worker is guaranteed to be assigned.
    """

    dataloader_persistent_workers: bool = False
    """Whether to use persistent workers for data loading (HF Trainers only).
    If True, the data loader will not shut down the worker processes after
    a dataset has been consumed once. This allows to maintain the workers
    Dataset instances alive. Can potentially speed up training, but will
    increase RAM usage. Will default to False.
    """

    dataloader_prefetch_factor: Optional[int] = None
    """Number of batches loaded in advance by each worker.

    2 means there will be a total of 2 * num_workers batches prefetched across
    all workers.

    This is only used if dataloader_num_workers >= 1.
    """

    dataloader_main_process_only: Optional[bool] = None
    """Controls whether the dataloader is iterated through on the main process only.

    If set to `True`, the dataloader is only iterated through on the main process
    (rank 0), then the batches are split and broadcast to each process.
    This can reduce the number of requests to the dataset, and helps ensure
    that each example is seen by max one GPU per epoch, but may become a performance
    bottleneck if a large number of GPUs is used.

    If set to `False`, the dataloader is iterated through on each GPU process.

    If set to `None` (default), then `True` or `False` is auto-selected based on
    heuristics (properties of dataset, the number of nodes and/or GPUs, etc).

    NOTE: We recommend to benchmark your setup, and configure `True` or `False`.
    """

    ddp_find_unused_parameters: Optional[bool] = None
    """When using PyTorch's DistributedDataParallel training, the value of this flag is
    passed to `find_unused_parameters`.

    Will default to `False` if gradient checkpointing is used, `True` otherwise.
    """

    max_grad_norm: Optional[float] = 1.0
    """Maximum gradient norm (for gradient clipping) to avoid exploding gradients which
    can destabilize training.

    Defaults to 1.0. When set to 0.0 or None gradient clipping will not be applied.
    """

    trainer_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments to pass to the HF/TRL Trainer.

    This allows for customization of the Trainer beyond the standard parameters
    defined in this class. Any key-value pairs added here will be passed directly
    to the Trainer's constructor. Note that this field is only used for
    HuggingFace and TRL trainers (TRL_SFT, TRL_DPO, TRL_GRPO, HF).
    """

    verl_config_overrides: dict[str, Any] = field(default_factory=dict)
    """Values to override in the verl config.

    This field is only used for the `VERL_GRPO` trainer.
    To see supported params in verl, see:
    https://verl.readthedocs.io/en/latest/examples/config.html

    The verl config is a nested dict, so the kwargs should be structured accordingly.
    For example, to set `actor_rollout_ref.actor.use_kl_loss` to `True`, you can use:
    `{"actor_rollout_ref": {"actor": {"use_kl_loss": True}}}`.

    The priority of setting verl config params, from highest to lowest, is:
    1. Values specified by this field.
    2. Values automatically set by Oumi in
        `src/oumi/core/trainers/verl_grpo_trainer.py:_create_config()` for verl params
        which have corresponding Oumi params. For example,
        Oumi's `training.output_dir` -> verl's `trainer.default_local_dir`
    3. Default verl config values in `src/oumi/core/trainers/verl_trainer_config.yaml`.
    """

    profiler: ProfilerParams = field(default_factory=ProfilerParams)
    """Parameters for performance profiling.

    This field contains configuration options for the profiler, which can be used
    to analyze the performance of the training process. It uses the ProfilerParams
    class to define specific profiling settings.
    """

    telemetry: TelemetryParams = field(default_factory=TelemetryParams)
    """Parameters for telemetry.

    This field contains telemetry configuration options.
    """

    empty_device_cache_steps: Optional[int] = None
    """Number of steps to wait before calling `torch.<device>.empty_cache()`.

    This parameter determines how frequently the GPU cache should be cleared during
    training. If set, it will trigger cache clearing every `empty_device_cache_steps`.
    If left as None, the cache will not be emptied automatically.

    Setting this can help manage GPU memory usage, especially for large models or
    long training runs, but may impact performance if set too low.
    """

    nccl_default_timeout_minutes: Optional[float] = None
    """Default timeout for NCCL operations in minutes.

    See: https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group

    If unset, will use the default value of `torch.distributed.init_process_group`
    which is 10min.
    """

    label_ignore_index: Optional[int] = None
    """Tokens with this label value don't contribute to the loss computation.
    For example, this can be `PAD`, or image tokens. `-100` is the PyTorch convention.
    Refer to the `ignore_index` parameter of `torch.nn.CrossEntropyLoss()`
    for more details.

    If unspecified (`None`), then the default model-specific preferences
    configured in Oumi may be used.

    Users should only set `label_ignore_index` if the default behavior is
    not satisfactory, or for new models not yet fully-integrated by Oumi.
    """

    def to_hf(self):
        """Converts Oumi config to HuggingFace's TrainingArguments."""
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

        dispatch_batches = self.dataloader_main_process_only

        if self.use_deterministic:
            self.enable_gradient_checkpointing = (
                False  # Fails ddp_broadcast_buffers=True
            )
            dispatch_batches = False  # Prevents dynamic batch redistribution
            self.ddp_find_unused_parameters = False  # Helps with determinism in DDP
            ddp_broadcast_buffers = True  # Ensures consistent buffer states
        else:
            ddp_broadcast_buffers = None

        if self.trainer_type == TrainerType.TRL_SFT:
            config_class = trl.SFTConfig
        elif self.trainer_type == TrainerType.TRL_DPO:
            config_class = trl.DPOConfig
        elif self.trainer_type == TrainerType.TRL_GRPO:
            config_class = trl.GRPOConfig
        else:
            config_class = transformers.TrainingArguments

        trainer_kwargs = copy.deepcopy(self.trainer_kwargs)
        if self.trainer_type == TrainerType.TRL_GRPO:
            grpo_kwargs = self.grpo.to_hf_trainer_kwargs()
            conflicting_keys = set(trainer_kwargs.keys()).intersection(
                grpo_kwargs.keys()
            )
            if len(conflicting_keys) > 0:
                raise ValueError(
                    "trainer_kwargs attempt to override the following "
                    f"GRPO kwargs: {conflicting_keys}. "
                    "Use properties of GrpoParams instead."
                )
            trainer_kwargs.update(grpo_kwargs)

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
            gradient_checkpointing=self.enable_gradient_checkpointing,
            gradient_checkpointing_kwargs=self.gradient_checkpointing_kwargs,
            include_tokens_per_second=self.include_performance_metrics,
            include_num_input_tokens_seen=self.include_performance_metrics,
            fp16=self.mixed_precision_dtype == MixedPrecisionDtype.FP16,
            bf16=self.mixed_precision_dtype == MixedPrecisionDtype.BF16,
            torch_compile=self.compile,
            save_steps=self.save_steps,
            save_strategy=save_strategy,
            logging_first_step=self.logging_first_step,
            torch_empty_cache_steps=self.empty_device_cache_steps,
            resume_from_checkpoint=self.resume_from_checkpoint,
            eval_strategy=self.eval_strategy,
            eval_steps=self.eval_steps,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_prefetch_factor=(
                self.dataloader_prefetch_factor if dataloader_num_workers > 0 else None
            ),
            full_determinism=self.full_determinism,
            ddp_broadcast_buffers=ddp_broadcast_buffers,
            dataloader_persistent_workers=self.dataloader_persistent_workers,
            dataloader_pin_memory=True,  # Set it to True to be explicit.
            ddp_find_unused_parameters=self.ddp_find_unused_parameters,
            max_grad_norm=self.max_grad_norm,  # type: ignore
            accelerator_config={  # accelerator config for multi-device training
                "dispatch_batches": dispatch_batches,
                # The params below are set to their default values.
                "split_batches": False,
                "even_batches": True,
                "use_seedable_sampler": True,
            },
            seed=self.seed,
            data_seed=self.data_seed,
            **trainer_kwargs,
        )
        assert isinstance(result, transformers.TrainingArguments)
        return result

    def _get_hf_report_to(self) -> list[str]:
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
        if self.enable_mlflow:
            report_to.append("mlflow")
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

        if self.max_grad_norm is not None and self.max_grad_norm < 0:
            raise ValueError("max_grad_norm must be >= 0.")

        if not (self.max_steps > 0 or self.num_train_epochs > 0):
            raise ValueError(
                "At least one of max_steps and num_train_epochs must be positive. "
                f"Actual: max_steps: {self.max_steps}, "
                f"num_train_epochs: {self.num_train_epochs}."
            )

        if (
            self.trainer_type != TrainerType.TRL_GRPO
            and self.trainer_type != TrainerType.VERL_GRPO
            and self.reward_functions is not None
        ):
            function_names = [name for name in self.reward_functions if name]
            if len(function_names) > 0:
                raise ValueError(
                    "reward_functions may only be defined for the TRL_GRPO or VERL_GRPO"
                    f"trainers. Actual: {self.trainer_type}"
                )
            if self.trainer_type == TrainerType.VERL_GRPO:
                if len(function_names) > 1:
                    raise ValueError(
                        "VERL_GRPO only supports a single reward function. "
                        f"Actual: {function_names}"
                    )

        # TODO: #1540 - Remove when TRL bug is fixed.
        if (
            self.trainer_type == TrainerType.TRL_GRPO
            and self.include_performance_metrics
        ):
            raise ValueError(
                "`include_performance_metrics` is not supported for TRL_GRPO trainer."
            )

    @property
    def telemetry_dir(self) -> Optional[Path]:
        """Returns the telemetry stats output directory."""
        result: Optional[Path] = None
        if self.telemetry.telemetry_dir:
            result = Path(self.telemetry.telemetry_dir)

        if self.output_dir:
            output_dir = Path(self.output_dir)
            # If `telemetry.telemetry_dir` is relative, then treat it
            # as a sub-directory of `output_dir`.
            if result and not result.is_absolute():
                result = output_dir / result

        return result
