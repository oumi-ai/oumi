import collections
import statistics
import time
from contextlib import ContextDecorator
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, cast

import pydantic
import torch

from lema.utils.debugging_utils import get_nvidia_gpu_temperature
from lema.utils.logging import get_logger

LOGGER = get_logger("lema.telemetry")


class TelemetryState(pydantic.BaseModel):
    measurements: Dict[str, List[float]] = pydantic.Field(default_factory=dict)
    # TODO: OPE-226 - implement async timers
    cuda_measurements: Dict[str, List[float]] = pydantic.Field(default_factory=dict)
    gpu_memory: List[Dict[str, float]] = pydantic.Field(default_factory=list)
    gpu_temperature: List[float] = pydantic.Field(default_factory=list)
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

    def record_gpu_temperature(self) -> float:
        """Records the current GPU temperature.

        Returns:
           GPU temperature, in degrees Celsius.
        """
        if not torch.cuda.is_available():
            LOGGER.debug("CUDA is not available. GPU temperature cannot be logged.")
            return 0.0

        temperature = get_nvidia_gpu_temperature()
        self.state.gpu_temperature.append(temperature)
        return temperature

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
            "gpu_temperature": {},
        }

        for name, measurements in self.state.measurements.items():
            summary["timers"][name] = self._calculate_timer_stats(
                measurements, total_time
            )

        for name, measurements in self.state.cuda_measurements.items():
            summary["cuda_timers"][name] = self._calculate_timer_stats(measurements)

        if self.state.gpu_temperature:
            summary["gpu_temperature"] = self._calculate_basic_stats(
                self.state.gpu_temperature
            )

        return summary

    def print_summary(self) -> None:
        """Prints a summary of the telemetry statistics."""
        summary = self.get_summary()
        log_lines: List[str] = [
            "Telemetry Summary:",
            f"Total time: {summary['total_time']:.2f} seconds",
        ]

        if summary["timers"]:
            log_lines.append("\nCPU Timers:")
            for name, stats in summary["timers"].items():
                log_lines.extend(self._format_timer_stats_as_lines(name, stats))

        if summary["cuda_timers"]:
            log_lines.append("\nCUDA Timers:")
            for name, stats in summary["cuda_timers"].items():
                log_lines.extend(self._format_timer_stats_as_lines(name, stats))

        if summary["gpu_memory"]:
            max_memory = max(usage["allocated"] for usage in summary["gpu_memory"])
            log_lines.append(f"\nPeak GPU memory usage: {max_memory:.2f} MiB")

        if summary["gpu_temperature"]:
            min_temperature = summary["gpu_temperature"]["min"]
            max_temperature = summary["gpu_temperature"]["max"]
            log_lines.append(
                f"\nGPU temperature: max: {max_temperature}C "
                f"min: {min_temperature}C"
            )

        # Log everything as a single value to ensure that stats from different
        # ranks aren't interleaved confusingly.
        LOGGER.info("\n".join(log_lines))

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
    def _calculate_basic_stats(self, measurements: List[float]) -> Dict[str, float]:
        count = len(measurements)
        # Use `defaultdict()` to make `_format_timer_stats_as_lines()` and
        # other functions usable even if `count` is zero, which can happen
        # for example for epochs timer if logging is called in the middle
        # of the first epoch.
        stats: Dict[str, float] = collections.defaultdict(float)
        stats["count"] = float(count)
        if count > 0:
            stats["mean"] = statistics.mean(measurements)
            stats["median"] = statistics.median(measurements)
            stats["std_dev"] = statistics.stdev(measurements) if count > 1 else 0
            stats["min"] = min(measurements)
            stats["max"] = max(measurements)
        return stats

    def _calculate_timer_stats(
        self, measurements: List[float], total_time: Optional[float] = None
    ) -> Dict[str, float]:
        """Same as above but also computes `total` and `percentage`."""
        stats: Dict[str, float] = self._calculate_basic_stats(measurements)

        count = len(measurements)
        if count > 0:
            stats["total"] = sum(measurements)
            if total_time:
                stats["percentage"] = (stats["total"] / total_time) * 100
        return stats

    def _format_timer_stats_as_lines(
        self, name: str, stats: Dict[str, float], is_cuda: bool = False
    ) -> List[str]:
        return [
            f"\t{name}:",
            f"\t\tTotal: {stats['total']:.6f} seconds",
            f"\t\tMean: {stats['mean']:.6f} seconds",
            f"\t\tMedian: {stats['median']:.6f} seconds",
            f"\t\tStd Dev: {stats['std_dev']:.6f} seconds",
            f"\t\tMin: {stats['min']:.6f} seconds",
            f"\t\tMax: {stats['max']:.6f} seconds",
            f"\t\tCount: {stats['count']}",
            f"\t\tPercentage of total time: {stats['percentage']:.2f}%",
        ]
