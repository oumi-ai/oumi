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

"""Utility functions for integrating Megatron-Bridge with Oumi training."""

import os
from pathlib import Path
from typing import Optional

import torch

try:
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.model_provider import get_model
    from megatron.bridge.training.checkpointing import (
        init_checkpointing_context,
        load_checkpoint,
    )
    from megatron.bridge.training.config import (
        CheckpointConfig,
        ConfigContainer,
        DistributedDataParallelConfig,
        OptimizerConfig,
        SchedulerConfig,
        TokenizerConfig,
        TrainingConfig as MegatronTrainingConfig,
    )
    from megatron.bridge.training.initialize import (
        initialize_megatron,
        set_jit_fusion_options,
    )
    from megatron.bridge.training.optim import setup_optimizer
    from megatron.bridge.training.state import GlobalState

    MEGATRON_BRIDGE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    MEGATRON_BRIDGE_AVAILABLE = False
    # Define placeholder types for type hints
    AutoBridge = None  # type: ignore
    ConfigContainer = None  # type: ignore
    GlobalState = None  # type: ignore

from oumi.core.configs import TrainingConfig
from oumi.core.configs.params.megatron_params import MegatronParams
from oumi.utils.logging import logger


def check_megatron_bridge_available():
    """Check if Megatron-Bridge is available and raise error if not."""
    if not MEGATRON_BRIDGE_AVAILABLE:
        raise ImportError(
            "Megatron-Bridge is not available. Please install it via:\n"
            "pip install megatron-bridge\n"
            "or from source:\n"
            "git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git\n"
            "cd Megatron-Bridge && pip install -e ."
        )


def import_hf_to_megatron(
    hf_model_id: str,
    megatron_checkpoint_path: str,
    megatron_params: MegatronParams,
    trust_remote_code: bool = False,
    hf_token: Optional[str] = None,
) -> None:
    """Convert a HuggingFace model to Megatron checkpoint format.

    This is typically a one-time conversion step before training.

    Args:
        hf_model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3.2-1B")
        megatron_checkpoint_path: Path to save the Megatron checkpoint
        megatron_params: Megatron parallelism configuration
        trust_remote_code: Whether to trust remote code from HF
        hf_token: HuggingFace API token for gated models

    Example:
        >>> import_hf_to_megatron(
        ...     "meta-llama/Llama-3.2-1B",
        ...     "/tmp/megatron_ckpt/llama32_1b",
        ...     megatron_params
        ... )
    """
    check_megatron_bridge_available()

    logger.info(f"Converting HF model {hf_model_id} to Megatron format...")
    logger.info(f"Target checkpoint path: {megatron_checkpoint_path}")

    # Set HF token if provided
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # Load model from HF using AutoBridge
    bridge = AutoBridge.from_hf_pretrained(
        hf_model_id, trust_remote_code=trust_remote_code
    )

    # Create Megatron provider with parallelism settings
    provider = bridge.to_megatron_provider(load_weights=True)

    # Configure parallelism for conversion
    provider.tensor_model_parallel_size = megatron_params.tensor_model_parallel_size
    provider.pipeline_model_parallel_size = (
        megatron_params.pipeline_model_parallel_size
    )
    provider.expert_model_parallel_size = megatron_params.expert_model_parallel_size

    if megatron_params.expert_tensor_parallel_size is not None:
        provider.expert_tensor_parallel_size = (
            megatron_params.expert_tensor_parallel_size
        )

    provider.finalize()

    # Create distributed model
    megatron_model = provider.provide_distributed_model(wrap_with_ddp=False)

    # Save as Megatron checkpoint
    logger.info(f"Saving Megatron checkpoint to {megatron_checkpoint_path}...")
    bridge.save_megatron_model(megatron_model, megatron_checkpoint_path)

    logger.info("HF → Megatron conversion complete!")


def build_megatron_config(
    training_config: TrainingConfig,
    megatron_checkpoint_path: str,
    train_checkpoint_dir: str,
) -> "ConfigContainer":
    """Build Megatron ConfigContainer from Oumi TrainingConfig.

    Args:
        training_config: Oumi training configuration
        megatron_checkpoint_path: Path to pretrained Megatron checkpoint
        train_checkpoint_dir: Directory to save training checkpoints

    Returns:
        ConfigContainer for Megatron-Core initialization
    """
    check_megatron_bridge_available()

    megatron_params = training_config.megatron
    training_params = training_config.training
    model_params = training_config.model

    # Build model config from megatron params
    model_cfg = megatron_params.to_megatron_config_dict()

    # Add precision settings
    if model_params.torch_dtype == torch.bfloat16:
        model_cfg["bf16"] = True
        model_cfg["fp16"] = False
    elif model_params.torch_dtype == torch.float16:
        model_cfg["bf16"] = False
        model_cfg["fp16"] = True
    else:
        model_cfg["bf16"] = False
        model_cfg["fp16"] = False

    # Build checkpoint config
    checkpoint = CheckpointConfig(
        save_interval=training_params.save_steps or 100,
        save=train_checkpoint_dir,
        load=train_checkpoint_dir,
        pretrained_checkpoint=megatron_checkpoint_path,
        async_save=megatron_params.checkpoint_config.use_async_checkpoint,
        fully_parallel_save=megatron_params.checkpoint_config.use_fully_parallel_strategy,
        fully_parallel_load=megatron_params.checkpoint_config.use_fully_parallel_strategy,
        load_rng=False,  # Don't load RNG state for reproducibility
    )

    # Build DDP config
    ddp = DistributedDataParallelConfig(
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=megatron_params.ddp_config.grad_reduce_in_fp32,
        overlap_grad_reduce=megatron_params.ddp_config.overlap_grad_reduce,
        overlap_param_gather=megatron_params.ddp_config.overlap_param_gather,
        average_in_collective=megatron_params.ddp_config.average_in_collective,
        use_distributed_optimizer=True,  # Always use distributed optimizer
    )

    # Build optimizer config
    opt = OptimizerConfig(
        lr=training_params.learning_rate,
        weight_decay=training_params.weight_decay,
        adam_beta1=training_params.adam_beta1,
        adam_beta2=training_params.adam_beta2,
        adam_eps=training_params.adam_epsilon,
        clip_grad=training_params.max_grad_norm,
        use_distributed_optimizer=True,
        overlap_param_gather_with_optimizer_step=(
            megatron_params.optimizer_config.overlap_cpu_optimizer_d2h_h2d
        ),
    )

    # Build scheduler config
    # Map Oumi scheduler types to Megatron scheduler types
    scheduler_type_map = {
        "linear": "LinearWarmupDecay",
        "cosine": "CosineAnnealing",
        "constant": "ConstantLR",
        "constant_with_warmup": "ConstantLR",  # Megatron handles warmup separately
    }

    megatron_scheduler_type = scheduler_type_map.get(
        training_params.lr_scheduler_type, "LinearWarmupDecay"
    )

    warmup_steps = training_params.warmup_steps or 0
    if training_params.warmup_ratio and training_params.warmup_ratio > 0:
        # Calculate warmup steps from ratio if max_steps is known
        if training_params.max_steps:
            warmup_steps = int(training_params.max_steps * training_params.warmup_ratio)

    sch = SchedulerConfig(
        lr_decay_style=megatron_scheduler_type,
        lr_warmup_iters=warmup_steps,
        lr_decay_iters=training_params.max_steps or 1000,
        min_lr=training_params.learning_rate * 0.1,  # 10% of initial LR
    )

    # Build training config
    train = MegatronTrainingConfig(
        micro_batch_size=megatron_params.micro_batch_size,
        global_batch_size=megatron_params.global_batch_size or training_params.per_device_train_batch_size,
        train_iters=training_params.max_steps or 1000,
    )

    # Build tokenizer config
    tokenizer = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=model_params.tokenizer_name or model_params.model_name,
    )

    return ConfigContainer(
        model=model_cfg,
        checkpoint=checkpoint,
        logger=None,
        train=train,
        optimizer=opt,
        ddp=ddp,
        scheduler=sch,
        dataset=None,
        tokenizer=tokenizer,
    )


def initialize_megatron_model(
    megatron_config: ConfigContainer,
    load_pretrained: bool = True,
):
    """Initialize Megatron model, optimizer, and scheduler.

    Args:
        megatron_config: Megatron configuration container
        load_pretrained: Whether to load pretrained checkpoint

    Returns:
        Tuple of (model, optimizer, scheduler, state)

    Example:
        >>> config = build_megatron_config(training_config, ckpt_path, train_dir)
        >>> model, optimizer, scheduler, state = initialize_megatron_model(config)
    """
    check_megatron_bridge_available()

    logger.info("Initializing Megatron-Core...")

    # Create global state
    state = GlobalState()
    state.cfg = megatron_config

    # Initialize Megatron distributed environment
    initialize_megatron(cfg=megatron_config)
    set_jit_fusion_options(
        megatron_config.model, megatron_config.train.micro_batch_size
    )

    # Initialize checkpointing context
    ckpt_ctx = init_checkpointing_context(megatron_config.checkpoint)

    # Create model
    logger.info("Creating Megatron model...")
    model_list = get_model(
        megatron_config.model,
        megatron_config.ddp,
        use_torch_fsdp2=False,  # Not using FSDP2 with Megatron
        overlap_param_gather_with_optimizer_step=(
            megatron_config.optimizer.overlap_param_gather_with_optimizer_step
        ),
        data_parallel_random_init=False,
    )

    # Setup optimizer and scheduler
    logger.info("Setting up optimizer and scheduler...")
    optimizer, scheduler = setup_optimizer(
        optimizer_config=megatron_config.optimizer,
        scheduler_config=megatron_config.scheduler,
        model=model_list,
        use_gloo_process_groups=False,
    )

    # Load pretrained checkpoint if requested
    if load_pretrained and megatron_config.checkpoint.pretrained_checkpoint:
        logger.info(
            f"Loading pretrained checkpoint from {megatron_config.checkpoint.pretrained_checkpoint}..."
        )
        load_checkpoint(
            state,
            model_list,
            optimizer,
            scheduler,
            checkpointing_context=ckpt_ctx,
            skip_load_to_model_and_opt=False,
        )

    model = model_list[0]  # Get first model (no virtual pipeline parallelism)

    logger.info("Megatron-Core initialization complete!")

    return model, optimizer, scheduler, state


def export_megatron_to_hf(
    megatron_model,
    bridge,  # AutoBridge type
    output_path: str,
    model_name: str,
) -> None:
    """Export trained Megatron model back to HuggingFace format.

    Args:
        megatron_model: Trained Megatron model
        bridge: AutoBridge instance used for conversion
        output_path: Path to save HF checkpoint
        model_name: Name for the HF model config
    """
    check_megatron_bridge_available()

    logger.info(f"Exporting Megatron model to HuggingFace format at {output_path}...")

    # Create output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Stream weights back to HF format
    hf_state_dict = {}
    for name, weight in bridge.export_hf_weights(megatron_model, cpu=True):
        hf_state_dict[name] = weight

    # Save HF checkpoint
    torch.save(hf_state_dict, Path(output_path) / "pytorch_model.bin")

    logger.info(f"Megatron → HF export complete! Saved to {output_path}")
