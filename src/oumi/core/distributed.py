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

from oumi.core.configs.params.fsdp_params import AutoWrapPolicy, FSDPParams
from oumi.utils.logging import logger
from oumi.utils.torch_naming_heuristics import get_module_class_from_name


#
# Types
#
class DeviceRankInfo(NamedTuple):
    world_size: int
    rank: int
    local_world_size: int
    local_rank: int


#
# Process Info
#
@functools.cache  # same as @cache added in Python 3.9
def get_device_rank_info() -> DeviceRankInfo:
    """Returns device rank and world size."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive. Actual: {world_size}.")
    rank = int(os.environ.get("RANK", 0))
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
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
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
    torch.distributed.init_process_group(backend=backend, timeout=timeout)
    torch.cuda.set_device(int(device_rank_info.local_rank))


def cleanup_distributed():
    """Clean up the distributed environment."""
    torch.distributed.destroy_process_group()


#
# FSDP and DDP
#
def prepare_model_for_distributed(
    model: torch.nn.Module,
    fsdp_params: Optional[FSDPParams] = None,
) -> torch.nn.Module:
    """Wrap the model for distributed training (DDP or FSDP).

    Args:
        model: The model to be wrapped.
        use_fsdp: Whether to use FSDP for distributed training.
        fsdp_params: Configuration options for FSDP. Defaults to None.

    Returns:
        torch.nn.Module: The wrapped model for distributed training.
    """
    logger = logging.getLogger("oumi")

    device_rank_info = get_device_rank_info()

    if fsdp_params is None or not fsdp_params.enable_fsdp:
        logger.info("Using DistributedDataParallel (DDP) for distributed training.")
        model = DistributedDataParallel(
            model,
            device_ids=[device_rank_info.local_rank],
        )
        return model

    logger.info("Using FullyShardedDataParallel (FSDP) for distributed training.")

    # Sharding Strategy
    sharding_strategy = fsdp_params.sharding_strategy.to_torch()

    # Wrapping Policy
    if fsdp_params.auto_wrap_policy == AutoWrapPolicy.TRANSFORMER_BASED:
        from oumi.utils.torch_naming_heuristics import (
            guess_transformer_layer_cls,
        )

        if fsdp_params.transformer_layer_cls is None:
            transformer_layer_cls = guess_transformer_layer_cls(model)
            logger.info(
                "Automatically inferred transformer layer class to wrap: "
                f"{transformer_layer_cls}"
            )
        else:
            logger.info(
                "Using transformer layer class to wrap: "
                f"{fsdp_params.transformer_layer_cls}"
            )
            transformer_layer_cls = get_module_class_from_name(
                fsdp_params.transformer_layer_cls
            )

        wrapping_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={transformer_layer_cls},
            recurse=True,
            nonwrapped_numel=0,
        )
    elif fsdp_params.auto_wrap_policy == AutoWrapPolicy.SIZE_BASED:
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
        # Leaving these to their default values for now
        # but we may want to make them configurable later
        use_orig_params=True,  # This needs to be True for torch.compile to work
        limit_all_gathers=True,
        param_init_fn=None,
        ignored_modules=None,
    )

    return model


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
