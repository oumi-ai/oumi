import functools
import os
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Dict, NamedTuple, Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel

from lema.utils.str_utils import str_to_bool


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
@functools.lru_cache(maxsize=None)  # same as @cache added in Python 3.9
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
    if world_size > 1 and not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError(
            f"World size {world_size} is greater than 1, "
            "while distributed torch isn't available/initialized ("
            f"available: {dist.is_available()}, "
            f"initialized: {dist.is_initialized()}, "
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
def barrier(group: Optional[dist.ProcessGroup] = None, monitored: bool = False) -> None:
    """Barrier synchronization among all processes in the group."""
    if dist.is_available() and dist.is_initialized():
        if monitored:
            dist.monitored_barrier(group=group)
        else:
            dist.barrier(group=group)
        return

    return


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
    dist.init_process_group(backend=backend, timeout=timeout)
    torch.cuda.set_device(int(device_rank_info.local_rank))


def cleanup_distributed():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


#
# FSDP and DDP
#
def get_default_fsdp_wrapping_policy(model: torch.nn.Module):
    """Get the FSDP wrapping policy based on the model size.

    Note: this is a naive policy that wraps layers if they have
    more than 100k parameters.

    Args:
        model: The PyTorch model.

    Returns:
        The FSDP wrapping policy.

    """
    return size_based_auto_wrap_policy(
        model, min_num_params=100000, recurse=True, nonwrapped_numel=0
    )


def get_default_fsdp_mixed_precision():
    """Get the FSDP mixed precision settings.

    Returns:
        MixedPrecision: An object containing the default FSDP mixed precision settings.
    """
    return MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )


def prepare_model_for_distributed(
    model: torch.nn.Module, use_fsdp: bool, fsdp_config: Optional[Dict[str, Any]] = None
) -> torch.nn.Module:
    """Wrap the model for distributed training (DDP or FSDP).

    Args:
        model (torch.nn.Module): The model to be wrapped.
        use_fsdp (bool): Whether to use FSDP for distributed training.
        fsdp_config (Optional[Dict[str, Any]], optional):
            Configuration options for FSDP. Defaults to None.

    Returns:
        torch.nn.Module: The wrapped model for distributed training.
    """
    device_rank_info = get_device_rank_info()

    if use_fsdp:
        fsdp_config = fsdp_config or {}
        wrapping_policy = fsdp_config.get(
            "wrapping_policy", get_default_fsdp_wrapping_policy(model)
        )
        mixed_precision = fsdp_config.get(
            "mixed_precision", get_default_fsdp_mixed_precision()
        )

        model = FSDP(
            model,
            auto_wrap_policy=wrapping_policy,
            mixed_precision=mixed_precision,
            device_id=torch.cuda.current_device(),
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            limit_all_gathers=True,
        )
    else:
        model = DistributedDataParallel(
            model,
            device_ids=[device_rank_info.local_rank],
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


#
# Accelerate
#
def is_using_accelerate_fsdp() -> bool:
    """Checks if the training is using Accelerate's FSDP implementation.

    Returns:
        bool: True if Accelerate's FSDP is being used, False otherwise.
    """
    env_var = os.environ.get("ACCELERATE_USE_FSDP", "false")

    return str_to_bool(env_var)
