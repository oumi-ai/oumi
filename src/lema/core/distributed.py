import functools
import os
from contextlib import contextmanager
from typing import NamedTuple, Optional

import torch.distributed as dist


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
    if world_size > 1 and not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError

    return DeviceRankInfo(
        world_size=world_size,
        rank=rank,
        local_world_size=local_world_size,
        local_rank=local_rank,
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
