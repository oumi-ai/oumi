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

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from oumi.core.configs.params.base_params import BaseParams


class ZeRORuntimeStage(str, Enum):
    """DeepSpeed ZeRO optimization stages.

    See DeepSpeed ZeRO documentation: https://www.deepspeed.ai/tutorials/zero/
    """

    ZERO_0 = "0"
    """Disabled ZeRO optimization. Standard data parallelism only."""

    ZERO_1 = "1"
    """Optimizer state sharding only."""

    ZERO_2 = "2"
    """Optimizer state and gradient sharding."""

    ZERO_3 = "3"
    """Full sharding: optimizer state, gradients, and model parameters."""


class DeepSpeedPrecision(str, Enum):
    """Mixed precision options for DeepSpeed."""

    FP16 = "fp16"
    """Half precision floating point."""

    BF16 = "bf16"
    """Brain floating point 16-bit format."""


class DeepSpeedOffloadDevice(str, Enum):
    """Offload device options for DeepSpeed."""

    CPU = "cpu"
    """Offload to CPU memory."""

    NVME = "nvme"
    """Offload to NVMe storage."""


@dataclass
class OffloadConfig:
    """Configuration for DeepSpeed parameter/optimizer offloading."""

    device: DeepSpeedOffloadDevice = DeepSpeedOffloadDevice.CPU
    """Device to offload to."""

    pin_memory: bool = True
    """Whether to use pinned memory for faster transfers."""

    buffer_count: int = 4
    """Number of buffers for overlapping transfers."""

    buffer_count_nvme: int = 2
    """Number of NVMe buffers for overlapping transfers."""

    block_size: int = 1048576
    """Buffer block size for NVMe offloading."""

    queue_depth: int = 8
    """NVMe queue depth."""

    single_submit: bool = False
    """Whether to use single submission for NVMe operations."""

    overlap_events: bool = True
    """Whether to overlap NVMe events."""

    thread_count: int = 1
    """Number of threads for NVMe operations."""


@dataclass
class DeepSpeedParams(BaseParams):
    """Configuration options for DeepSpeed distributed training.

    Based on DeepSpeed configuration schema and best practices from
    LLaMA-Factory implementation.

    For detailed ZeRO configuration options, see:
    https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/zero/config.py
    """

    enable_deepspeed: bool = False
    """If True, enables DeepSpeed distributed training.

    When False, uses standard PyTorch distributed training (DDP/FSDP).
    When True, initializes DeepSpeed engine with ZeRO optimizations based on zero_stage.
    Allows training larger models using ZeRO optimization stages
    for memory-efficient distributed training.
    """

    deepspeed_config_path: Optional[Union[str, Path]] = None
    """Path to a DeepSpeed JSON configuration file.

    If this parameter is not None, all following fields in this class are ignored.
    Unspecified parameters will use DeepSpeed's internal defaults (not Oumi's defaults).
    """

    zero_stage: ZeRORuntimeStage = ZeRORuntimeStage.ZERO_0
    """ZeRO optimization stage.

    Controls the level of memory optimization:
    - Stage 0: Disabled ZeRO optimization, but DeepSpeed engine still active - DEFAULT
    - Stage 1: Optimizer state sharding
    - Stage 2: Optimizer + gradient sharding
    - Stage 3: Full sharding (optimizer + gradients + parameters)

    Stage 0 is the default to match DeepSpeed behavior. Unlike enable_deepspeed=False,
    Stage 0 still uses DeepSpeed's runtime features without ZeRO memory optimizations.
    Enable higher stages for memory efficiency with large models.
    """

    offload_optimizer: Optional[OffloadConfig] = None
    """Configuration for optimizer state offloading.

    When enabled, optimizer states are offloaded to CPU/NVMe
    to save GPU memory. Available for ZeRO stages 1, 2, and 3.
    """

    offload_param: Optional[OffloadConfig] = None
    """Configuration for parameter offloading.

    When enabled, model parameters are offloaded to CPU/NVMe
    when not actively used. Only available for ZeRO stage 3.
    """

    precision: Optional[DeepSpeedPrecision] = None
    """Mixed precision training mode.

    Options:
    - fp16: Half precision (faster on older GPUs)
    - bf16: Brain float 16 (better numerical stability on newer GPUs)
    - None: Use model's native precision
    """

    # Communication optimization parameters (general)
    overlap_comm: bool = False
    """Whether to overlap communication with computation.

    DeepSpeed default is False. Automatically enabled for ZeRO Stage 3.
    """

    contiguous_gradients: bool = True
    """Whether to ensure gradient memory is contiguous."""

    reduce_bucket_size: Union[int, str] = int(5e8)
    """Bucket size for gradient reduction.

    Can be an integer or "auto" for automatic sizing.
    """

    # ZeRO-1 and ZeRO-2 specific communication parameters
    allgather_bucket_size: int = int(5e8)
    """Bucket size for allgather operations (ZeRO stages 1 and 2)."""

    allgather_partitions: bool = True
    """Enable allgather partitions for ZeRO-2."""

    reduce_scatter: bool = True
    """Enable reduce scatter for ZeRO-2."""

    round_robin_gradients: bool = False
    """Enable round robin gradients (ZeRO stages 1 and 2)."""

    # ZeRO-3 specific parameters
    stage3_prefetch_bucket_size: Union[int, str] = int(5e7)
    """Bucket size for parameter prefetching in ZeRO-3."""

    stage3_param_persistence_threshold: Union[int, str] = int(1e5)
    """Parameter persistence threshold in ZeRO-3."""

    stage3_max_live_parameters: int = int(1e9)
    """Maximum number of live parameters in ZeRO-3."""

    stage3_max_reuse_distance: int = int(1e9)
    """Maximum reuse distance for parameters in ZeRO-3."""

    stage3_gather_16bit_weights_on_model_save: bool = False
    """Whether to gather 16-bit weights during model saving in ZeRO-3.

    DeepSpeed default is False. Set to True if you need full precision weights
    in saved checkpoints.
    """

    sub_group_size: int = int(1e9)
    """Sub-group size for ZeRO-3 parameter sharding."""

    # Training parameters (auto-configured by HuggingFace Transformers)
    train_batch_size: Union[int, str] = "auto"
    """Total training batch size across all GPUs.

    Can be an integer, or "auto" for automatic configuration by HuggingFace.
    When using TRL trainers, this should remain "auto" to allow proper batch size
    management.
    """

    train_micro_batch_size_per_gpu: Union[int, str] = "auto"
    """Micro batch size per GPU.

    Can be an integer, or "auto" for automatic configuration by HuggingFace.
    When using TRL trainers, this should remain "auto" to allow proper batch size
    management.
    """

    gradient_accumulation_steps: Union[int, str] = "auto"
    """Number of gradient accumulation steps.

    Can be an integer, or "auto" for automatic configuration by HuggingFace.
    """

    gradient_clipping: Union[int, str] = "auto"
    """Gradient clipping value.

    Can be an integer, or "auto" for automatic configuration by HuggingFace.
    """

    # Advanced options
    zero_allow_untested_optimizer: bool = True
    """Allow optimizers not explicitly tested with ZeRO."""

    zero_force_ds_cpu_optimizer: bool = True
    """Force DeepSpeed CPU optimizer when CPU offloading is enabled."""

    # Activation checkpointing
    activation_checkpointing: dict[str, Any] = field(default_factory=dict)
    """Configuration for activation checkpointing to save memory.

    DeepSpeed activation checkpointing trades computation for memory by recomputing
    activations during backward pass. Available parameters:

    - partition_activations (bool, default=False): Partition activation checkpoints
      across model parallel GPUs
    - checkpoint_in_cpu (bool, default=False): Move activation checkpoints to CPU.
      Only works when partition_activations=True
    - contiguous_checkpointing (bool, default=False): Copy checkpoints to contiguous
      memory buffer. Requires num_checkpoints to be set
    - num_checkpoints (int, optional): Number of activation checkpoints stored
      during forward propagation. Required for contiguous_checkpointing
    - synchronize (bool, default=False): Perform device synchronization at
      checkpoint boundaries
    - profile (bool, default=False): Log forward/backward time for each checkpoint

    Example: {"partition_activations": True, "checkpoint_in_cpu": True,
              "profile": False}

    For detailed configuration options and API reference, see:
    https://deepspeed.readthedocs.io/en/latest/activation-checkpointing.html
    """

    # Memory optimization
    memory_efficient_linear: bool = False
    """Enable memory-efficient linear layers."""

    # Logging and monitoring
    steps_per_print: int = 10
    """Number of steps between DeepSpeed log prints."""

    wall_clock_breakdown: bool = False
    """Enable detailed wall clock time breakdown logging."""

    def __post_init__(self) -> None:
        """Validate DeepSpeed configuration parameters."""
        # Validate offloading configurations
        if (
            self.offload_param is not None
            and self.zero_stage != ZeRORuntimeStage.ZERO_3
        ):
            raise ValueError(
                "Parameter offloading is only supported with ZeRO stage 3. "
                f"Current stage: {self.zero_stage}"
            )

        if self.offload_optimizer is not None and self.zero_stage not in [
            ZeRORuntimeStage.ZERO_1,
            ZeRORuntimeStage.ZERO_2,
            ZeRORuntimeStage.ZERO_3,
        ]:
            raise ValueError(
                "Optimizer offloading requires ZeRO stage 1, 2, or 3. "
                f"Current stage: {self.zero_stage}"
            )

    def to_deepspeed(self) -> dict[str, Any]:
        """Generate DeepSpeed configuration dictionary.

        Creates a DeepSpeed configuration dict based on the current parameters,
        following the format expected by DeepSpeed's configuration system.

        Returns:
            Dictionary containing DeepSpeed configuration parameters.
        """
        config = {
            "train_batch_size": self.train_batch_size,
            "train_micro_batch_size_per_gpu": self.train_micro_batch_size_per_gpu,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_clipping": self.gradient_clipping,
            "zero_allow_untested_optimizer": self.zero_allow_untested_optimizer,
            "zero_force_ds_cpu_optimizer": self.zero_force_ds_cpu_optimizer,
            "steps_per_print": self.steps_per_print,
            "wall_clock_breakdown": self.wall_clock_breakdown,
        }

        # Add mixed precision configuration
        if self.precision == DeepSpeedPrecision.FP16:
            config["fp16"] = {
                "enabled": "auto",
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1,
            }
        elif self.precision == DeepSpeedPrecision.BF16:
            config["bf16"] = {"enabled": "auto"}

        # Add ZeRO optimization configuration
        zero_config: dict[str, Any] = {
            "stage": int(self.zero_stage.value),
        }

        # Add stage-specific parameters
        if self.zero_stage in [
            ZeRORuntimeStage.ZERO_0,
            ZeRORuntimeStage.ZERO_1,
            ZeRORuntimeStage.ZERO_2,
        ]:
            zero_config.update(
                {
                    "allgather_partitions": self.allgather_partitions,
                    "allgather_bucket_size": self.allgather_bucket_size,
                    "overlap_comm": self.overlap_comm,
                    "reduce_scatter": self.reduce_scatter,
                    "reduce_bucket_size": self.reduce_bucket_size,
                    "contiguous_gradients": self.contiguous_gradients,
                    "round_robin_gradients": self.round_robin_gradients,
                }
            )

        if self.zero_stage == ZeRORuntimeStage.ZERO_3:
            zero_config.update(
                {
                    "overlap_comm": self.overlap_comm,
                    "contiguous_gradients": self.contiguous_gradients,
                    "sub_group_size": self.sub_group_size,
                    "reduce_bucket_size": self.reduce_bucket_size,
                    "stage3_prefetch_bucket_size": self.stage3_prefetch_bucket_size,
                    "stage3_param_persistence_threshold": (
                        self.stage3_param_persistence_threshold
                    ),
                    "stage3_max_live_parameters": self.stage3_max_live_parameters,
                    "stage3_max_reuse_distance": self.stage3_max_reuse_distance,
                    "stage3_gather_16bit_weights_on_model_save": (
                        self.stage3_gather_16bit_weights_on_model_save
                    ),
                }
            )

        # Add offloading configurations
        if self.offload_optimizer is not None:
            offload_config: dict[str, Any] = {
                "device": self.offload_optimizer.device.value,
                "pin_memory": self.offload_optimizer.pin_memory,
            }
            if self.offload_optimizer.device == DeepSpeedOffloadDevice.NVME:
                offload_config.update(
                    {
                        "buffer_count": self.offload_optimizer.buffer_count_nvme,
                        "block_size": self.offload_optimizer.block_size,
                        "queue_depth": self.offload_optimizer.queue_depth,
                        "single_submit": self.offload_optimizer.single_submit,
                        "overlap_events": self.offload_optimizer.overlap_events,
                        "thread_count": self.offload_optimizer.thread_count,
                    }
                )
            else:
                offload_config["buffer_count"] = self.offload_optimizer.buffer_count

            zero_config["offload_optimizer"] = offload_config

        if self.offload_param is not None:
            offload_config: dict[str, Any] = {
                "device": self.offload_param.device.value,
                "pin_memory": self.offload_param.pin_memory,
            }
            if self.offload_param.device == DeepSpeedOffloadDevice.NVME:
                offload_config.update(
                    {
                        "buffer_count": self.offload_param.buffer_count_nvme,
                        "block_size": self.offload_param.block_size,
                        "queue_depth": self.offload_param.queue_depth,
                        "single_submit": self.offload_param.single_submit,
                        "overlap_events": self.offload_param.overlap_events,
                        "thread_count": self.offload_param.thread_count,
                    }
                )
            else:
                offload_config["buffer_count"] = self.offload_param.buffer_count

            zero_config["offload_param"] = offload_config

        config["zero_optimization"] = zero_config

        # Add activation checkpointing if configured
        if self.activation_checkpointing:
            config["activation_checkpointing"] = self.activation_checkpointing

        return config

    def is_zero3_enabled(self) -> bool:
        """Check if ZeRO stage 3 is enabled."""
        return self.enable_deepspeed and self.zero_stage == ZeRORuntimeStage.ZERO_3

    def requires_param_offloading(self) -> bool:
        """Check if parameter offloading is configured."""
        return self.offload_param is not None

    def requires_optimizer_offloading(self) -> bool:
        """Check if optimizer offloading is configured."""
        return self.offload_optimizer is not None
