import statistics
import time
from contextlib import ContextDecorator
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, cast

import pydantic
import torch

from lema.utils.logging import get_logger

LOGGER = get_logger("lema.telemetry")


class TelemetryState(pydantic.BaseModel):
    measurements: Dict[str, List[float]] = pydantic.Field(default_factory=dict)
    # TODO: OPE-226 - implement async timers
    cuda_measurements: Dict[str, List[float]] = pydantic.Field(default_factory=dict)
    gpu_memory: List[Dict[str, float]] = pydantic.Field(default_factory=list)
    start_time: float = pydantic.Field(default_factory=time.perf_counter)


class TimerContext(ContextDecorator):
    """A context manager and decorator for timing CPU code execution."""

    def __init__(self, name: str, measurements: Optional[List[float]] = None):
        """Initializes a TimerContext object.

        Args:
            name: The name of the timer.
            measurements: A list to store the timing measurements.
        """
        self.name = name
        self.measurements = measurements if measurements is not None else []
        self.start_time: Optional[float] = None

        # Enable to accurately time the duration of ops on CUDA.
        # This should only be used for debuggings since it may increase latency.
        self.cuda_synchronize: bool = False

    def __enter__(self) -> "TimerContext":
        """Starts the timer."""
        if self.cuda_synchronize:
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        """Stops the timer and records the elapsed time."""
        if self.start_time is not None:
            if self.cuda_synchronize:
                torch.cuda.synchronize()
            elapsed_time = time.perf_counter() - self.start_time
            self.measurements.append(elapsed_time)
            self.start_time = None
        return False


class CudaTimerContext(ContextDecorator):
    """A context manager and decorator for timing CUDA operations."""

    def __init__(self, name: str, measurements: Optional[List[float]] = None):
        """Initializes a CudaTimerContext object.

        Args:
            name: The name of the timer.
            measurements: A list to store the timing measurements.
        """
        self.name = name
        self.measurements = measurements if measurements is not None else []
        self.start_event = self._get_new_cuda_event()
        self.end_event = self._get_new_cuda_event()

        # Debugging flags
        self.pre_synchronize: bool = False

    def _get_new_cuda_event(self) -> torch.cuda.Event:
        """Returns a CUDA event."""
        return cast(torch.cuda.Event, torch.cuda.Event(enable_timing=True))

    def __enter__(self) -> "CudaTimerContext":
        """Starts the CUDA timer."""
        if not torch.cuda.is_available():
            LOGGER.debug("CUDA is not available. Skipping CUDA benchmark.")
            return self

        if self.pre_synchronize:
            torch.cuda.synchronize()

        self.start_event.record()
        return self

    def __exit__(self, *exc) -> bool:
        """Stops the CUDA timer and records the elapsed time."""
        if not torch.cuda.is_available():
            LOGGER.debug("CUDA is not available. Skipping CUDA benchmark.")
            return False

        assert self.end_event is not None
        self.end_event.record()

        # TODO: OPE-226 - implement async timers
        # We need to sync here as we read the elapsed time soon after.
        torch.cuda.synchronize()

        elapsed_time = (
            self.start_event.elapsed_time(self.end_event) / 1000
        )  # Convert to seconds

        self.measurements.append(elapsed_time)
        return False


def gpu_memory_logger(user_function: Callable, synchronize: bool = True) -> Callable:
    """Decorator function that logs the GPU memory usage of a given function.

    Args:
        user_function: The function to be decorated.
        synchronize: Flag indicating whether to synchronize
          GPU operations before measuring memory usage. Defaults to True.

    Returns:
        The decorated function.
    """

    @wraps(user_function)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            LOGGER.debug("CUDA is not available. GPU memory usage cannot be logged.")
            return user_function(*args, **kwargs)

        if synchronize:
            torch.cuda.synchronize()

        start_memory = torch.cuda.memory_allocated()

        result = user_function(*args, **kwargs)

        if synchronize:
            torch.cuda.synchronize()

        end_memory = torch.cuda.memory_allocated()
        memory_diff = end_memory - start_memory
        LOGGER.debug(
            f"{user_function.__name__} used {memory_diff / 1024**2:.2f} MiB "
            "of GPU memory."
        )

        return result

    return wrapper


class TelemetryTracker:
    """A class for tracking various telemetry metrics."""

    def __init__(self):
        """Initializes the TelemetryTracker object."""
        self.state = TelemetryState()

    #
    # Context Managers
    #
    def timer(self, name: str) -> TimerContext:
        """Creates a timer with the given name.

        Args:
            name: The name of the timer.

        Returns:
            A TimerContext object.
        """
        if name not in self.state.measurements:
            self.state.measurements[name] = []
        return TimerContext(name, self.state.measurements[name])

    def cuda_timer(self, name: str) -> CudaTimerContext:
        """Creates a CUDA benchmark with the given name.

        Args:
            name: The name of the benchmark.

        Returns:
            A CudaTimerContext object.
        """
        if name not in self.state.cuda_measurements:
            self.state.cuda_measurements[name] = []
        return CudaTimerContext(name, self.state.cuda_measurements[name])

    def log_gpu_memory(self, custom_logger: Optional[Callable] = None) -> None:
        """Logs the GPU memory usage.

        Args:
            custom_logger: A custom logging function. If None, store in self.gpu_memory.
        """
        if not torch.cuda.is_available():
            LOGGER.debug("CUDA is not available. GPU memory usage cannot be logged.")
            return

        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MiB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2  # Convert to MiB
        memory_info = {"allocated": memory_allocated, "reserved": memory_reserved}

        if custom_logger:
            custom_logger(memory_info)
        else:
            self.state.gpu_memory.append(memory_info)

    #
    # Summary
    #
    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of the telemetry statistics.

        Returns:
            A dictionary containing the summary statistics.
        """
        total_time = time.perf_counter() - self.state.start_time

        summary = {
            "total_time": total_time,
            "timers": {},
            "cuda_timers": {},
            "gpu_memory": self.state.gpu_memory,
        }

        for name, measurements in self.state.measurements.items():
            summary["timers"][name] = self._calculate_stats(measurements, total_time)

        for name, measurements in self.state.cuda_measurements.items():
            summary["cuda_timers"][name] = self._calculate_stats(measurements)

        return summary

    def print_summary(self) -> None:
        """Prints a summary of the telemetry statistics."""
        summary = self.get_summary()
        LOGGER.info("Telemetry Summary:")
        LOGGER.info(f"Total time: {summary['total_time']:.2f} seconds")

        if summary["timers"]:
            LOGGER.info("\nCPU Timers:")
            for name, stats in summary["timers"].items():
                self._log_timer_stats(name, stats)

        if summary["cuda_timers"]:
            LOGGER.info("\nCUDA Timers:")
            for name, stats in summary["cuda_timers"].items():
                self._log_timer_stats(name, stats)

        if summary["gpu_memory"]:
            max_memory = max(usage["allocated"] for usage in summary["gpu_memory"])
            LOGGER.info(f"\nPeak GPU memory usage: {max_memory:.2f} MiB")

    #
    # State Management
    #
    def state_dict(self) -> dict:
        """Returns the TelemetryState as a dict."""
        return self.state.model_dump()

    def load_state_dict(self, state_dict: dict) -> None:
        """Loads TelemetryState from state_dict."""
        self.state = TelemetryState.model_validate(state_dict, strict=True)

    #
    # Helper Methods
    #
    def _calculate_stats(
        self, measurements: List[float], total_time: Optional[float] = None
    ) -> Dict[str, float]:
        stats = {
            "total": sum(measurements),
            "mean": statistics.mean(measurements),
            "median": statistics.median(measurements),
            "std_dev": statistics.stdev(measurements) if len(measurements) > 1 else 0,
            "min": min(measurements),
            "max": max(measurements),
            "count": len(measurements),
        }
        if total_time:
            stats["percentage"] = (stats["total"] / total_time) * 100
        return stats

    def _log_timer_stats(
        self, name: str, stats: Dict[str, float], is_cuda: bool = False
    ) -> None:
        LOGGER.info(f"\t{name}:")
        LOGGER.info(f"\t\tTotal: {stats['total']:.6f} seconds")
        LOGGER.info(f"\t\tMean: {stats['mean']:.6f} seconds")
        LOGGER.info(f"\t\tMedian: {stats['median']:.6f} seconds")
        LOGGER.info(f"\t\tStd Dev: {stats['std_dev']:.6f} seconds")
        LOGGER.info(f"\t\tMin: {stats['min']:.6f} seconds")
        LOGGER.info(f"\t\tMax: {stats['max']:.6f} seconds")
        LOGGER.info(f"\t\tCount: {stats['count']}")
        LOGGER.info(f"\t\tPercentage of total time: {stats['percentage']:.2f}%")
