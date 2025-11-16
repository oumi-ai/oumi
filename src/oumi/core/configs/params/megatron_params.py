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
from typing import Any, Optional

from oumi.core.configs.params.base_params import BaseParams


@dataclass
class MegatronDDPConfig(BaseParams):
    """Configuration for Megatron's DistributedDataParallel.

    Maps to Megatron's DistributedDataParallelConfig object.
    See: https://github.com/NVIDIA/Megatron-LM/blob/core_r0.13.0/megatron/core/distributed/distributed_data_parallel_config.py
    """

    grad_reduce_in_fp32: bool = True
    """Whether to reduce gradients in FP32 precision for better numerical stability."""

    overlap_grad_reduce: bool = False
    """Whether to overlap gradient reduction with backward computation."""

    overlap_param_gather: bool = False
    """Whether to overlap parameter gathering with forward computation."""

    average_in_collective: bool = True
    """Whether to average gradients in the collective operation."""


@dataclass
class MegatronOptimizerConfig(BaseParams):
    """Configuration for Megatron optimizer options.

    Maps to Megatron's OptimizerConfig object.
    See: https://github.com/NVIDIA/Megatron-LM/blob/core_r0.13.0/megatron/core/optimizer/optimizer_config.py
    """

    overlap_cpu_optimizer_d2h_h2d: bool = False
    """Whether to overlap device-to-host and host-to-device data transfers for CPU optimizer."""

    use_precision_aware_optimizer: bool = False
    """Whether to use precision-aware optimizer updates."""

    optimizer_cpu_offload: bool = False
    """Whether to offload optimizer states to CPU memory."""

    optimizer_offload_fraction: float = 0.0
    """Fraction of optimizer states to offload to CPU (0.0-1.0). Use 1.0 for full offload."""


@dataclass
class MegatronTransformerConfig(BaseParams):
    """Configuration for Megatron transformer-specific options.

    Maps to Megatron's TransformerConfig object.
    See: https://github.com/NVIDIA/Megatron-LM/blob/core_r0.13.0/megatron/core/transformer/transformer_config.py
    """

    recompute_granularity: Optional[str] = None
    """Granularity for gradient/activation checkpointing ('full', 'selective', or None)."""

    recompute_method: Optional[str] = None
    """Method for recomputation ('uniform', 'block', or None)."""

    recompute_num_layers: Optional[int] = None
    """Number of transformer layers to apply recomputation to."""

    recompute_modules: list[str] = field(default_factory=lambda: ["core_attn"])
    """List of module names to recompute during backward pass."""


@dataclass
class MegatronCheckpointConfig(BaseParams):
    """Configuration for Megatron checkpointing strategy."""

    use_async_checkpoint: bool = False
    """Whether to use asynchronous checkpointing for better performance."""

    use_fully_parallel_strategy: bool = True
    """Whether to use fully parallel load/save strategy for distributed checkpoints."""

    async_persistent: bool = True
    """Whether to use persistent background workers for async checkpointing."""


@dataclass
class MegatronParams(BaseParams):
    """Parameters for Megatron-based distributed training.

    This configuration enables advanced model parallelism strategies for training
    very large models (70B+) that require tensor, pipeline, context, and expert parallelism.
    """

    # Core parallelism dimensions
    tensor_model_parallel_size: int = 1
    """Number of GPUs to use for tensor model parallelism (TP).

    Splits model layers across multiple GPUs. Recommended for large models (70B+).
    Must evenly divide the number of attention heads.
    """

    pipeline_model_parallel_size: int = 1
    """Number of GPUs to use for pipeline model parallelism (PP).

    Splits model vertically across multiple GPUs. Recommended for very large models.
    Must evenly divide the number of transformer layers.
    """

    context_parallel_size: int = 1
    """Number of GPUs to use for context parallelism (CP).

    Splits sequence length across multiple GPUs for long-context training.
    Useful for sequences > 8K tokens.
    """

    expert_model_parallel_size: int = 1
    """Number of GPUs to use for expert model parallelism (EP) in MoE models.

    Splits MoE experts across multiple GPUs. Only applicable for MoE architectures.
    """

    expert_tensor_parallel_size: Optional[int] = None
    """Tensor parallelism size within each expert in MoE models.

    If None, defaults to tensor_model_parallel_size.
    """

    virtual_pipeline_model_parallel_size: Optional[int] = None
    """Number of virtual pipeline stages (for interleaved pipeline schedules).

    Enables more efficient pipeline schedules by interleaving microbatches.
    Must be divisible by pipeline_model_parallel_size.
    """

    # Training configuration
    micro_batch_size: int = 1
    """Micro-batch size for pipeline parallelism.

    Actual batch size per GPU step. Global batch size = micro_batch_size *
    gradient_accumulation_steps * data_parallel_size.
    """

    global_batch_size: Optional[int] = None
    """Global batch size across all GPUs.

    If specified, overrides micro_batch_size calculation. Must be divisible by
    (DP size * gradient_accumulation_steps).
    """

    enable_sequence_packing: bool = False
    """Whether to enable sequence packing for RL training.

    Packs multiple sequences into a single batch for better GPU utilization.
    Recommended for RL workloads with variable-length sequences.
    """

    # Sub-configurations
    ddp_config: MegatronDDPConfig = field(default_factory=MegatronDDPConfig)
    """DDP configuration for data parallelism."""

    optimizer_config: MegatronOptimizerConfig = field(
        default_factory=MegatronOptimizerConfig
    )
    """Optimizer-specific configuration."""

    transformer_config: MegatronTransformerConfig = field(
        default_factory=MegatronTransformerConfig
    )
    """Transformer architecture configuration."""

    checkpoint_config: MegatronCheckpointConfig = field(
        default_factory=MegatronCheckpointConfig
    )
    """Checkpointing strategy configuration."""

    # Model configuration overrides
    model_config_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional kwargs to pass to the HuggingFace model config."""

    # Weight synchronization
    enable_weight_sync: bool = True
    """Whether to enable weight synchronization between training and inference.

    For disaggregated actor-rollout setups, enables efficient weight updates
    via NCCL/gloo or checkpoint-and-reload.
    """

    weight_sync_method: str = "nccl"
    """Method for weight synchronization ('nccl', 'gloo', or 'checkpoint').

    - 'nccl': Fast GPU-to-GPU sync (requires NCCL)
    - 'gloo': CPU-based sync (more flexible but slower)
    - 'checkpoint': Save/load checkpoints (most flexible, slowest)
    """

    # Inference backend
    inference_backend: str = "vllm"
    """Inference backend for generation ('vllm', 'sglang', or 'megatron').

    - 'vllm': Recommended for most use cases (requires refit from Megatron weights)
    - 'sglang': Alternative high-performance backend
    - 'megatron': Native Megatron inference (no conversion needed)
    """

    def __post_init__(self):
        """Validates Megatron configuration parameters."""
        # Validate parallelism dimensions
        if self.tensor_model_parallel_size < 1:
            raise ValueError(
                f"tensor_model_parallel_size must be >= 1, got {self.tensor_model_parallel_size}"
            )

        if self.pipeline_model_parallel_size < 1:
            raise ValueError(
                f"pipeline_model_parallel_size must be >= 1, got {self.pipeline_model_parallel_size}"
            )

        if self.context_parallel_size < 1:
            raise ValueError(
                f"context_parallel_size must be >= 1, got {self.context_parallel_size}"
            )

        if self.expert_model_parallel_size < 1:
            raise ValueError(
                f"expert_model_parallel_size must be >= 1, got {self.expert_model_parallel_size}"
            )

        # Validate virtual pipeline parallelism
        if self.virtual_pipeline_model_parallel_size is not None:
            if self.virtual_pipeline_model_parallel_size < 1:
                raise ValueError(
                    "virtual_pipeline_model_parallel_size must be >= 1 or None, "
                    f"got {self.virtual_pipeline_model_parallel_size}"
                )
            if self.pipeline_model_parallel_size > 1:
                if (
                    self.virtual_pipeline_model_parallel_size
                    % self.pipeline_model_parallel_size
                    != 0
                ):
                    raise ValueError(
                        "virtual_pipeline_model_parallel_size must be divisible by "
                        f"pipeline_model_parallel_size. Got {self.virtual_pipeline_model_parallel_size} "
                        f"and {self.pipeline_model_parallel_size}"
                    )

        # Validate batch sizes
        if self.micro_batch_size < 1:
            raise ValueError(
                f"micro_batch_size must be >= 1, got {self.micro_batch_size}"
            )

        if self.global_batch_size is not None and self.global_batch_size < 1:
            raise ValueError(
                f"global_batch_size must be >= 1 or None, got {self.global_batch_size}"
            )

        # Validate weight sync method
        valid_sync_methods = {"nccl", "gloo", "checkpoint"}
        if self.weight_sync_method not in valid_sync_methods:
            raise ValueError(
                f"weight_sync_method must be one of {valid_sync_methods}, "
                f"got {self.weight_sync_method}"
            )

        # Validate inference backend
        valid_backends = {"vllm", "sglang", "megatron"}
        if self.inference_backend not in valid_backends:
            raise ValueError(
                f"inference_backend must be one of {valid_backends}, "
                f"got {self.inference_backend}"
            )

        # Validate optimizer CPU offload fraction
        if not (0.0 <= self.optimizer_config.optimizer_offload_fraction <= 1.0):
            raise ValueError(
                "optimizer_offload_fraction must be between 0.0 and 1.0, "
                f"got {self.optimizer_config.optimizer_offload_fraction}"
            )

    def get_data_parallel_size(self, world_size: int) -> int:
        """Calculate the data parallel size given the world size and parallelism config.

        Args:
            world_size: Total number of GPUs available.

        Returns:
            The data parallel size (number of data parallel replicas).
        """
        model_parallel_size = (
            self.tensor_model_parallel_size
            * self.pipeline_model_parallel_size
            * self.context_parallel_size
            * self.expert_model_parallel_size
        )

        if world_size % model_parallel_size != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by the product of all "
                f"model parallel sizes ({model_parallel_size} = "
                f"{self.tensor_model_parallel_size} * {self.pipeline_model_parallel_size} * "
                f"{self.context_parallel_size} * {self.expert_model_parallel_size})"
            )

        return world_size // model_parallel_size

    def to_megatron_config_dict(self) -> dict[str, Any]:
        """Convert to a dictionary compatible with Megatron configuration.

        Returns:
            Dictionary of Megatron configuration parameters.
        """
        config = {
            "tensor_model_parallel_size": self.tensor_model_parallel_size,
            "pipeline_model_parallel_size": self.pipeline_model_parallel_size,
            "context_parallel_size": self.context_parallel_size,
            "expert_model_parallel_size": self.expert_model_parallel_size,
            "expert_tensor_parallel_size": self.expert_tensor_parallel_size,
            "virtual_pipeline_model_parallel_size": self.virtual_pipeline_model_parallel_size,
            "micro_batch_size": self.micro_batch_size,
            "global_batch_size": self.global_batch_size,
            "enable_sequence_packing": self.enable_sequence_packing,
        }

        # Add DDP config
        config["ddp_config"] = {
            "grad_reduce_in_fp32": self.ddp_config.grad_reduce_in_fp32,
            "overlap_grad_reduce": self.ddp_config.overlap_grad_reduce,
            "overlap_param_gather": self.ddp_config.overlap_param_gather,
            "average_in_collective": self.ddp_config.average_in_collective,
        }

        # Add optimizer config
        config["optimizer_config_kwargs"] = {
            "overlap_cpu_optimizer_d2h_h2d": self.optimizer_config.overlap_cpu_optimizer_d2h_h2d,
            "use_precision_aware_optimizer": self.optimizer_config.use_precision_aware_optimizer,
            "optimizer_cpu_offload": self.optimizer_config.optimizer_cpu_offload,
            "optimizer_offload_fraction": self.optimizer_config.optimizer_offload_fraction,
        }

        # Add transformer config
        config["transformer_config_kwargs"] = {
            "recompute_granularity": self.transformer_config.recompute_granularity,
            "recompute_method": self.transformer_config.recompute_method,
            "recompute_num_layers": self.transformer_config.recompute_num_layers,
            "recompute_modules": self.transformer_config.recompute_modules,
        }

        # Add model config overrides
        if self.model_config_kwargs:
            config["model_config_kwargs"] = self.model_config_kwargs

        return config
