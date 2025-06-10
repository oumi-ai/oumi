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

import functools
import logging
import os
import random
from contextlib import contextmanager
from datetime import timedelta
from typing import NamedTuple, Optional, TypeVar, cast

import numpy as np
import torch
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.nn.parallel import DistributedDataParallel

from oumi.core.configs.params.fsdp_params import AutoWrapPolicy
from oumi.core.configs.training_config import TrainingConfig
from oumi.utils.logging import logger
from oumi.utils.torch_naming_heuristics import (
    resolve_transformer_layer_cls_string_as_module_set,
    simplify_transformer_layer_cls_string,
)


#
# Types
#
class DeviceRankInfo(NamedTuple):
    world_size: int
    rank: int
    local_world_size: int
    local_rank: int


def _get_use_orig_params(config: TrainingConfig) -> bool:
    """Returns whether to use the PyTorch Module's original parameters for FSDP.

    If the user specified a value, return that. Else, infer its value based on other
    config values (compilation, FSDP, PEFT).
    """
    if config.fsdp.use_orig_params is not None:
        return config.fsdp.use_orig_params
    # use_orig_params must be true for model compilation.
    if not config.training.compile:
        # use_orig_params should be false for FSDP PEFT training to realize GPU memory
        # savings.
        # https://huggingface.co/docs/peft/main/en/accelerate/fsdp#the-important-parts
        if config.training.use_peft and config.fsdp.enable_fsdp:
            return False
    return True


#
# Process Info
#
def _parse_rank(rank: Optional[str]) -> int:
    """Parse the rank from the environment variable."""
    if not rank:
        return 0

    # -1 is a special value that means "not set".
    # It's used by the Accelerate launcher.
    # Defaulting to 0.
    if rank.strip() == "-1":
        return 0

    if not rank.isdigit():
        raise ValueError(f"Rank must be a number. Actual: {rank}.")

    return int(rank)


@functools.cache  # same as @cache added in Python 3.9
def get_device_rank_info() -> DeviceRankInfo:
    """Returns device rank and world size."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive. Actual: {world_size}.")
    rank = _parse_rank(os.environ.get("RANK"))
    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"RANK must be within this range [0, {world_size}). Actual: {rank}."
        )
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    if local_world_size <= 0 or local_world_size > world_size:
        raise ValueError(
            f"LOCAL_WORLD_SIZE must be within this range [1, {world_size}]. "
            f"Actual: {local_world_size}."
        )
    # Per https://pytorch.org/docs/stable/elastic/run.html
    # NEVER hard code any assumptions about the stable-ness of ranks or
    # some correlation between RANK and LOCAL_RANK.
    local_rank = _parse_rank(os.environ.get("LOCAL_RANK"))
    if local_rank < 0 or local_rank >= local_world_size:
        raise ValueError(
            f"LOCAL_RANK must be within this range [0, {local_world_size}). "
            f"Actual: {local_rank}."
        )
    return DeviceRankInfo(
        world_size=world_size,
        rank=rank,
        local_world_size=local_world_size,
        local_rank=local_rank,
    )


def verify_torch_distributed_initialized_if_needed() -> None:
    """Checks if torch.dist is initialized if WORLD_SIZE> 1."""
    device_rank_info: DeviceRankInfo = get_device_rank_info()
    world_size = device_rank_info.world_size
    if world_size > 1 and not (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    ):
        raise RuntimeError(
            f"World size {world_size} is greater than 1, "
            "while distributed torch isn't available/initialized ("
            f"available: {torch.distributed.is_available()}, "
            f"initialized: {torch.distributed.is_initialized()}, "
            f"{device_rank_info}"
            ")"
        )


def is_world_process_zero() -> bool:
    """Whether or not this process is the global main process.

    When training in a distributed fashion on several machines
    this is only going to be `True` for one process.
    """
    device_rank_info: DeviceRankInfo = get_device_rank_info()
    return device_rank_info.rank == 0


def is_local_process_zero() -> bool:
    """Whether or not this process is the local main process.

    When training in a distributed fashion on several machines
    this is only going to be `True` for one process per node.
    """
    device_rank_info: DeviceRankInfo = get_device_rank_info()
    return device_rank_info.local_rank == 0


def is_distributed() -> bool:
    """Whether or not the training is distributed.

    Returns:
        bool: True if the training is distributed, False otherwise.
    """
    device_rank_info: DeviceRankInfo = get_device_rank_info()
    return device_rank_info.world_size > 1


#
# Distributed Operations
#
def barrier(
    group: Optional[torch.distributed.ProcessGroup] = None, monitored: bool = False
) -> None:
    """Barrier synchronization among all processes in the group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if monitored:
            torch.distributed.monitored_barrier(group=group)
        else:
            torch.distributed.barrier(group=group)
        return

    return


T = TypeVar("T")


def all_gather_object(
    obj: T, group: Optional[torch.distributed.ProcessGroup] = None
) -> list[T]:
    """Gathers picklable objects from the whole group into a list."""
    verify_torch_distributed_initialized_if_needed()
    if is_distributed():
        device_rank_info: DeviceRankInfo = get_device_rank_info()
        # Placeholder array to gather results from all workers.
        object_list = [None] * device_rank_info.world_size
        torch.distributed.all_gather_object(object_list, obj, group=group)
    else:
        object_list = [obj]

    # We have to cast because the inferred type is `List[Optional[T]])`
    # while `None` must never happen here.
    return cast(list[T], object_list)


def local_leader_only(*barrier_args, **barrier_kwargs):
    """Decorator for local leaders only operations."""

    def decorator(user_function):
        @functools.wraps(user_function)
        def wrapper(*args, **kwargs):
            if is_local_process_zero():
                # Execute the user function
                result = user_function(*args, **kwargs)

                # Sync back with all processed before resuming
                barrier(*barrier_args, **barrier_kwargs)
                return result
            else:
                # User function is not called
                # Wait for the local leader to finish
                barrier(*barrier_args, **barrier_kwargs)
                return None

        return wrapper

    return decorator


@contextmanager
def local_leader_first(*args, **kwargs):
    """Context manager for local leader first operations."""
    if is_local_process_zero():
        yield
        barrier(*args, **kwargs)
    else:
        barrier(*args, **kwargs)
        yield


def global_leader_only(*args, **kwargs):
    """Decorator for global leader only operations."""

    def decorator(user_function):
        @functools.wraps(user_function)
        def wrapper(*user_fn_args, **user_fn_kwargs):
            if is_world_process_zero():
                # Execute the user function
                result = user_function(*user_fn_args, **user_fn_kwargs)

                # Sync back with all processed before resuming
                barrier(*args, **kwargs)
                return result
            else:
                # User function is not called
                # Wait for the global leader to finish
                barrier(*args, **kwargs)
                return None

        return wrapper

    return decorator


@contextmanager
def global_leader_first(*args, **kwargs):
    """Context manager for global leader first operations."""
    if is_world_process_zero():
        yield
        barrier(*args, **kwargs)
    else:
        barrier(*args, **kwargs)
        yield


#
# Distributed Initialization
#
def init_distributed(
    backend: str = "nccl", timeout_minutes: Optional[float] = None
) -> None:
    """Initialize the distributed environment."""
    device_rank_info: DeviceRankInfo = get_device_rank_info()
    timeout = (
        timedelta(minutes=timeout_minutes) if timeout_minutes is not None else None
    )
    torch.cuda.set_device(int(device_rank_info.local_rank))
    torch.distributed.init_process_group(
        backend=backend,
        rank=device_rank_info.rank,
        world_size=device_rank_info.world_size,
        device_id=torch.device(int(device_rank_info.local_rank)),
        timeout=timeout,
    )
    initialized = torch.distributed.is_initialized()
    logger.info(f"Initialized distributed ({initialized}): {device_rank_info}")


def cleanup_distributed():
    """Clean up the distributed environment."""
    torch.distributed.destroy_process_group()


#
# FSDP and DDP
#
def prepare_model_for_distributed(
    model: torch.nn.Module,
    config: TrainingConfig,
    ddp_find_unused_parameters: Optional[bool] = None,
) -> torch.nn.Module:
    """Wrap the model for distributed training (DDP or FSDP).

    Args:
        model: The model to be wrapped.
        config: The training config.
        ddp_find_unused_parameters: Whether to traverse the autograd graph from all
            tensors contained in the return value of the wrapped module's `forward`
            function. Parameters that don't receive gradients as part of this
            graph are preemptively marked as being ready to be reduced. In addition,
            parameters that may have been used in the wrapped module's ``forward``
            function but were not part of loss computation and thus would also
            not receive gradients are preemptively marked as ready to be reduced.

    Returns:
        torch.nn.Module: The wrapped model for distributed training.
    """
    logger = logging.getLogger("oumi")

    device_rank_info = get_device_rank_info()
    fsdp_params = config.fsdp

    if fsdp_params is None or not fsdp_params.enable_fsdp:
        logger.info("Using DistributedDataParallel (DDP) for distributed training.")
        model = DistributedDataParallel(
            model,
            device_ids=[device_rank_info.local_rank],
            find_unused_parameters=(ddp_find_unused_parameters or False),
        )
        return model

    logger.info("Using FullyShardedDataParallel (FSDP) for distributed training.")

    # Sharding Strategy
    sharding_strategy = fsdp_params.sharding_strategy.to_torch()

    # Wrapping Policy
    if fsdp_params.auto_wrap_policy == AutoWrapPolicy.TRANSFORMER_BASED_WRAP:
        from oumi.utils.torch_naming_heuristics import (
            guess_transformer_layer_cls,
        )

        transformer_layer_classes = set()
        if fsdp_params.transformer_layer_cls is None:
            transformer_layer_cls = guess_transformer_layer_cls(model)
            logger.info(
                "Automatically inferred transformer layer class to wrap: "
                f"{transformer_layer_cls}"
            )
            transformer_layer_classes.add(transformer_layer_cls)
        else:
            logger.info(
                "Using transformer layer class to wrap: "
                f"{fsdp_params.transformer_layer_cls}"
            )
            transformer_layer_classes = (
                resolve_transformer_layer_cls_string_as_module_set(
                    fsdp_params.transformer_layer_cls
                )
            )

        wrapping_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_layer_classes,
            recurse=True,
            nonwrapped_numel=0,
        )
    elif fsdp_params.auto_wrap_policy == AutoWrapPolicy.SIZE_BASED_WRAP:
        wrapping_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=fsdp_params.min_num_params,
            recurse=True,
            nonwrapped_numel=0,
        )

    else:
        wrapping_policy = None

    # Mixed Precision
    mixed_precision = None
    if fsdp_params.mixed_precision:
        if fsdp_params.mixed_precision == "bf16":
            dtype = torch.bfloat16
        elif fsdp_params.mixed_precision == "fp16":
            dtype = torch.float16
        else:
            raise ValueError(
                f"Unsupported mixed precision type: {fsdp_params.mixed_precision}"
            )
        mixed_precision = MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
            buffer_dtype=dtype,
        )

    # CPU Offload
    cpu_offload = CPUOffload(offload_params=fsdp_params.cpu_offload)

    # Backward Prefetch
    backward_prefetch = fsdp_params.backward_prefetch.to_torch()

    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        cpu_offload=cpu_offload,
        backward_prefetch=backward_prefetch,
        mixed_precision=mixed_precision,
        auto_wrap_policy=wrapping_policy,
        device_id=torch.cuda.current_device(),
        sync_module_states=fsdp_params.sync_module_states,
        forward_prefetch=fsdp_params.forward_prefetch,
        use_orig_params=_get_use_orig_params(config),
        # Leaving these to their default values for now
        # but we may want to make them configurable later
        limit_all_gathers=True,
        param_init_fn=None,
        ignored_modules=None,
    )

    return model


def get_accelerate_env_vars(config: TrainingConfig) -> dict[str, str]:
    """Gets environment vars for FSDP Accelerate corresponding to Oumi training params.

    This mimics the environment variables set here:
    https://github.com/huggingface/accelerate/blob/bf4572b6ce0a534a9d73537485a0edf1d68144b8/src/accelerate/utils/launch.py#L260-L285
    Note how they lowercase all boolean values, except for
    `ACCELERATE_DYNAMO_USE_FULLGRAPH` and `ACCELERATE_DYNAMO_USE_DYNAMIC`, which we
    also do. It's worth pointing out that `ACCELERATE_USE_FSDP` must be lowercase:
    https://github.com/huggingface/accelerate/blob/bf4572b6ce0a534a9d73537485a0edf1d68144b8/src/accelerate/accelerator.py#L341

    Returns:
        dict[str, str]: The environment variables and values to set for HF Accelerate.
    """
    env_vars = {}
    # These environment variables are set by default in HF Accelerate.
    env_vars["ACCELERATE_DYNAMO_BACKEND"] = "NO"
    env_vars["ACCELERATE_DYNAMO_MODE"] = "default"
    env_vars["ACCELERATE_DYNAMO_USE_FULLGRAPH"] = "False"
    env_vars["ACCELERATE_DYNAMO_USE_DYNAMIC"] = "False"

    # We haven't seen a need to make this configurable yet.
    # https://github.com/huggingface/transformers/blob/33868a057c02f0368ba63bd1edb746be38fe3d90/src/transformers/modeling_utils.py#L146
    env_vars["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"

    env_vars["FSDP_USE_ORIG_PARAMS"] = str(_get_use_orig_params(config)).lower()
    # These env vars are set based on FSDPParams.
    env_vars["ACCELERATE_USE_FSDP"] = str(config.fsdp.enable_fsdp).lower()
    env_vars["FSDP_SHARDING_STRATEGY"] = config.fsdp.sharding_strategy.value
    env_vars["FSDP_OFFLOAD_PARAMS"] = str(config.fsdp.cpu_offload).lower()
    if config.fsdp.mixed_precision:
        env_vars["ACCELERATE_MIXED_PRECISION"] = config.fsdp.mixed_precision
    env_vars["FSDP_BACKWARD_PREFETCH"] = config.fsdp.backward_prefetch.value
    env_vars["FSDP_FORWARD_PREFETCH"] = str(config.fsdp.forward_prefetch).lower()
    env_vars["FSDP_STATE_DICT_TYPE"] = config.fsdp.state_dict_type.value
    env_vars["FSDP_AUTO_WRAP_POLICY"] = config.fsdp.auto_wrap_policy.value
    env_vars["FSDP_MIN_NUM_PARAMS"] = str(config.fsdp.min_num_params)
    if config.fsdp.transformer_layer_cls:
        simplified_value = simplify_transformer_layer_cls_string(
            config.fsdp.transformer_layer_cls
        )
        if simplified_value != config.fsdp.transformer_layer_cls:
            logger.info(
                f"'FSDP_TRANSFORMER_CLS_TO_WRAP' is set to '{simplified_value}' "
                f"based on '{config.fsdp.transformer_layer_cls}'."
            )
        env_vars["FSDP_TRANSFORMER_CLS_TO_WRAP"] = simplified_value
    env_vars["FSDP_SYNC_MODULE_STATES"] = str(config.fsdp.sync_module_states).lower()

    # This is set from TrainingParams.
    env_vars["FSDP_ACTIVATION_CHECKPOINTING"] = str(
        config.training.enable_gradient_checkpointing
    ).lower()
    return env_vars


def prepare_accelerate_fsdp_run(config: TrainingConfig) -> dict[str, str]:
    """Prepares our FSDP training job to run with the HuggingFace Accelerate library.

    This function should be run if we didn't invoke the current training job from the
    Accelerate launcher, but still want to use FSDP with Accelerate. The motivation for
    this is to remove the need for the Accelerate config, centralize all config values
    under the Oumi `TrainingConfig`, and make it easier to switch between HF and Oumi
    trainers. For more information, see PR#803.

    `training.enable_gradient_checkpointing` is also disabled, as FSDP gradient
    checkpointing is handled by Accelerate.

    Args:
        config: The training configuration.

    Returns:
        dict[str, str]: The environment variables set to prepare for Accelerate.
    """
    env_vars = get_accelerate_env_vars(config)
    # Disable Oumi's gradient checkpointing param, as Accelerate should handle it.
    config.training.enable_gradient_checkpointing = False

    for name, value in env_vars.items():
        if name in os.environ:
            logger.warning(
                f"Environment variable `{name}` has existing value "
                f"`{os.environ[name]}`, overriding to new value `{value}`."
            )
        os.environ[name] = value
    return env_vars


def estimate_dataloader_num_workers(
    gpus_per_node: Optional[int] = None, cpu_count: Optional[int] = None
) -> int:
    """Estimates the number of dataloader workers.

    Uses a simple heuristic based on the number of GPU-s and CPU-s per node.

    Args:
        gpus_per_node: The number of GPU-s per node.
        cpu_count: The number of CPU cores.

    Returns:
        The estimated number of dataloader workers (a non-zero positive number).
    """
    # Limit the maximum number of dataloader workers.
    _MAX_WORKERS = 8

    # Scale the number of workers with the number of GPUs (the more GPU-s the more data)
    if gpus_per_node is None:
        gpus_per_node = get_device_rank_info().local_world_size
    result = min(2 * gpus_per_node, _MAX_WORKERS)

    # Limit the maximum number of CPU cores used for dataloaders
    # to leave enough CPU-s for computation. This condition is expected to
    # kick-in rarely, only for unusual machine configurations when a weak VM
    # with small number of CPU cores has many GPU-s assigned.
    # For example, Polaris has 64 CPU cores and 4 GPU-s per node.
    _MAX_FRACTION_OF_CPUS_FOR_DATALOADERS = 0.25
    if cpu_count is None:
        cpu_count = os.cpu_count() or 1
    result = min(result, int(cpu_count * _MAX_FRACTION_OF_CPUS_FOR_DATALOADERS))

    # Make sure it's a positive number (>=1).
    result = max(result, 1)
    return result


def set_random_seeds(seed: int = 42, set_deterministic: bool = False) -> None:
    """Set random seeds for reproducibility.

    Each worker will have a different seed to ensure that each worker
    starts with a different random state.

    Args:
        seed: The seed value to set for random number generators.
        set_deterministic: Whether to set deterministic mode for CUDA operations.
    """
    device_info = get_device_rank_info()

    local_seed = seed + device_info.rank

    logger.info(f"Setting random seed to {local_seed} on rank {device_info.rank}.")
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed(local_seed)

    if set_deterministic:
        logger.info("Setting deterministic mode for CUDA operations.")
        torch.backends.cudnn.deterministic = True
