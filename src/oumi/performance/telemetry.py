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

import collections
import socket
import statistics
import time
from contextlib import ContextDecorator
from functools import wraps
from pprint import pformat
from typing import Any, Callable, Optional, Union, cast

import numpy as np
import pydantic
import torch

from oumi.core.distributed import (
    DeviceRankInfo,
    all_gather_object,
    get_device_rank_info,
)
from oumi.utils.device_utils import (
    get_nvidia_gpu_runtime_info,
    get_nvidia_gpu_temperature,
)
from oumi.utils.logging import get_logger

LOGGER = get_logger("oumi.telemetry")


class TelemetryState(pydantic.BaseModel):
    start_time: float = pydantic.Field(default_factory=time.perf_counter)
    hostname: str = pydantic.Field(default_factory=socket.gethostname)
    measurements: dict[str, list[float]] = pydantic.Field(default_factory=dict)
    # TODO: OPE-226 - implement async timers
    cuda_measurements: dict[str, list[float]] = pydantic.Field(default_factory=dict)
    gpu_memory: list[dict[str, float]] = pydantic.Field(default_factory=list)
    gpu_temperature: list[float] = pydantic.Field(default_factory=list)


class TimerContext(ContextDecorator):
    """A context manager and decorator for timing CPU code execution."""

    def __init__(self, name: str, measurements: Optional[list[float]] = None):
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

    def __init__(self, name: str, measurements: Optional[list[float]] = None):
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


_SUMMARY_KEY_HOSTNAME = "hostname"
_SUMMARY_KEY_TOTAL_TIME = "total_time"
_SUMMARY_KEY_TIMERS = "timers"
_SUMMARY_KEY_CUDA_TIMERS = "cuda_timers"
_SUMMARY_KEY_GPU_MEMORY = "gpu_memory"
_SUMMARY_KEY_GPU_TEMPERATURE = "gpu_temperature"


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

        device_rank_info: DeviceRankInfo = get_device_rank_info()
        temperature = get_nvidia_gpu_temperature(device_rank_info.local_rank)
        # Log extra info when we see an increase in max temperature above 78C.
        if temperature >= 78:
            max_temperature = (
                np.max(self.state.gpu_temperature)
                if len(self.state.gpu_temperature) > 0
                else 0
            )
            if temperature > max_temperature:
                info = get_nvidia_gpu_runtime_info(device_rank_info.local_rank)
                LOGGER.info(
                    f"Highest temperature {temperature}C observed! "
                    f"{pformat(info)} | {pformat(device_rank_info)}"
                )

        self.state.gpu_temperature.append(temperature)

        return temperature

    #
    # Summary
    #
    def get_summary(self) -> dict[str, Any]:
        """Returns a summary of the telemetry statistics.

        Returns:
            A dictionary containing the summary statistics.
        """
        total_time = time.perf_counter() - self.state.start_time

        summary = {
            _SUMMARY_KEY_HOSTNAME: self.state.hostname,
            _SUMMARY_KEY_TOTAL_TIME: total_time,
            _SUMMARY_KEY_TIMERS: {},
            _SUMMARY_KEY_CUDA_TIMERS: {},
            _SUMMARY_KEY_GPU_MEMORY: self.state.gpu_memory,
            _SUMMARY_KEY_GPU_TEMPERATURE: {},
        }

        for name, measurements in self.state.measurements.items():
            summary[_SUMMARY_KEY_TIMERS][name] = self._calculate_timer_stats(
                measurements, total_time
            )

        for name, measurements in self.state.cuda_measurements.items():
            summary[_SUMMARY_KEY_CUDA_TIMERS][name] = self._calculate_timer_stats(
                measurements
            )

        if self.state.gpu_temperature:
            summary[_SUMMARY_KEY_GPU_TEMPERATURE] = self._calculate_basic_stats(
                self.state.gpu_temperature
            )

        return summary

    def print_summary(self) -> None:
        """Prints a summary of the telemetry statistics."""
        summary = self.get_summary()
        log_lines: list[str] = [
            f"Telemetry Summary ({summary[_SUMMARY_KEY_HOSTNAME]}):",
            f"Total time: {summary['total_time']:.2f} seconds",
        ]

        if summary[_SUMMARY_KEY_TIMERS]:
            log_lines.append("\nCPU Timers:")
            for name, stats in summary[_SUMMARY_KEY_TIMERS].items():
                log_lines.extend(self._format_timer_stats_as_lines(name, stats))

        if summary[_SUMMARY_KEY_CUDA_TIMERS]:
            log_lines.append("\nCUDA Timers:")
            for name, stats in summary[_SUMMARY_KEY_CUDA_TIMERS].items():
                log_lines.extend(self._format_timer_stats_as_lines(name, stats))

        if summary[_SUMMARY_KEY_GPU_MEMORY]:
            max_memory = max(
                usage["allocated"] for usage in summary[_SUMMARY_KEY_GPU_MEMORY]
            )
            log_lines.append(f"\nPeak GPU memory usage: {max_memory:.2f} MiB")

        if summary[_SUMMARY_KEY_GPU_TEMPERATURE]:
            log_lines.extend(
                self._format_gpu_temperature_stats_as_lines(
                    summary[_SUMMARY_KEY_GPU_TEMPERATURE]
                )
            )

        # Log everything as a single value to ensure that stats from different
        # ranks aren't interleaved confusingly.
        LOGGER.info("\n".join(log_lines))

    def get_summaries_from_all_ranks(self) -> list[dict[str, Any]]:
        """Returns an array of telemetry summaries from all ranks.

        To work correctly in distributed environment, the method must be called
        by all ranks. If distributed training is not used then returns
        an array with 1 element (the current rank's summary).

        Returns:
            A list of telemetry summaries indexed by rank.
        """
        return all_gather_object(self.get_summary())

    def compute_cross_rank_summaries(
        self,
        rank_summaries: list[dict[str, Any]],
        *,
        measurement_names: Union[set[str], dict[str, Any]],
    ) -> dict[str, Any]:
        """Computes a cross-rank summary from summaries produced by individual ranks.

        For example, it can be used to compute distribution
        of `{"gpu_temperature": {"max"}}` over ranks.

        Args:
            rank_summaries: An array of summaries indexed by rank e.g.,
                returned by the `get_summaries_from_all_ranks()` method.
            measurement_names: A hierarchy of measurment names of interest,
                which must match the hierarchical naming structure in `rank_summaries`.

                For example:

                - 1 level:  ``{"total_time"}``
                - 2 levels: ``{"gpu_temperature": {"max", "median"}}``
                - 3 levels: ``{"timers": { "compile": {"mean"},
                  "forward": {"max", "min"}}}``

        Returns:
            A dictionary containing the statistics specified in `measurement_names`,
            and aggregated across ranks.
            The returned object can be nested (e.g., a dictionary of dictionaries)
            with potentially multiple levels of nesting, forming a tree whose
            structure mimics the structure of `measurement_names` with one additional
            layer containing cross-rank stats.

            For example, if input `measurement_names` is
            ``{"gpu_temperature": {"max", "median"}}`` then the returned value will look
            as follows:

            .. code-block:: python

                {
                    "gpu_temperature":{
                        "max": { "count": 7, "max": 75, ... },
                        "median": { "count": 7, "max": 68, ... }
                    }
                }
        """
        if not measurement_names:
            return {}

        result = {}

        def _aggregate_cross_rank_stats(
            key: str, rank_summaries: list[dict[str, Any]]
        ) -> Optional[dict[str, float]]:
            measurements = []
            for rank_summary in rank_summaries:
                if key in rank_summary and isinstance(rank_summary[key], (float, int)):
                    measurements.append(rank_summary[key])
            if not measurements:
                return None
            return self._calculate_basic_stats(measurements, include_index=True)

        if isinstance(measurement_names, dict):
            for key in measurement_names:
                if isinstance(measurement_names[key], (dict, set)):
                    # If a value associated with this `key` is a dictionary or a set
                    # then recurse (support hierarchical naming).
                    next_level_summaries = []
                    for rank_summary in rank_summaries:
                        if key in rank_summary and isinstance(rank_summary[key], dict):
                            next_level_summaries.append(rank_summary[key])
                    if next_level_summaries:
                        result[key] = self.compute_cross_rank_summaries(
                            next_level_summaries,
                            measurement_names=measurement_names[key],
                        )
                else:
                    # If a value associated with this `key` is not a dictionary or a set
                    # then we've reached the last layer. Let's compute stats.
                    stats = _aggregate_cross_rank_stats(key, rank_summaries)
                    if stats is not None:
                        result[key] = stats
        else:
            # If `measurement_names` is a set then iterate over its elements
            # and compute stats for each measurement.
            assert isinstance(measurement_names, set)
            for key in measurement_names:
                stats = _aggregate_cross_rank_stats(key, rank_summaries)
                if stats is not None:
                    result[key] = stats

        return result

    #
    # State Management
    #
    def state_dict(self) -> dict:
        """Returns the TelemetryState as a dict."""
        return self.state.model_dump()

    def load_state_dict(self, state_dict: dict) -> None:
        """Loads TelemetryState from state_dict."""
        self.state = TelemetryState.model_validate(state_dict, strict=True)

    def get_state_dicts_from_all_ranks(self) -> list[dict]:
        """Returns an array of `state_dict`-s from all ranks.

        To work correctly in distributed environment, the method must be called
        by all ranks. If distributed training is not used then returns
        an array with 1 element (the current rank's `state_dict`).

        Returns:
            A list of `state_dict`-s indexed by rank.
        """
        return all_gather_object(self.state_dict())

    #
    # Helper Methods
    #
    def _calculate_basic_stats(
        self, measurements: list[float], include_index: bool = False
    ) -> dict[str, float]:
        count = len(measurements)
        # Use `defaultdict()` to make `_format_timer_stats_as_lines()` and
        # other functions usable even if `count` is zero, which can happen
        # for example for epochs timer if logging is called in the middle
        # of the first epoch.
        stats: dict[str, float] = collections.defaultdict(float)
        stats["count"] = float(count)
        if count > 0:
            stats["mean"] = statistics.mean(measurements)
            stats["median"] = statistics.median(measurements)
            stats["std_dev"] = statistics.stdev(measurements) if count > 1 else 0

            min_index = np.argmin(measurements)
            stats["min"] = measurements[min_index]
            if include_index:
                stats["min_index"] = float(min_index)

            max_index = np.argmax(measurements)
            stats["max"] = measurements[max_index]
            if include_index:
                stats["max_index"] = float(max_index)
        return stats

    def _calculate_timer_stats(
        self, measurements: list[float], total_time: Optional[float] = None
    ) -> dict[str, float]:
        """Same as above but also computes `total` and `percentage`."""
        stats: dict[str, float] = self._calculate_basic_stats(measurements)

        count = len(measurements)
        if count > 0:
            stats["total"] = sum(measurements)
            if total_time:
                stats["percentage"] = (stats["total"] / total_time) * 100
        return stats

    def _format_timer_stats_as_lines(
        self, name: str, stats: dict[str, float], is_cuda: bool = False
    ) -> list[str]:
        return [
            f"\t{name}:",
            (
                f"\t\tTotal: {stats['total']:.4f}s "
                f"Mean: {stats['mean']:.4f}s Median: {stats['median']:.4f}s"
            ),
            (
                f"\t\tMin: {stats['min']:.4f}s "
                f"Max: {stats['max']:.4f}s "
                f"StdDev: {stats['std_dev']:.4f}s"
            ),
            (
                f"\t\tCount: {stats['count']} "
                f"Percentage of total time: {stats['percentage']:.2f}%"
            ),
        ]

    def _format_gpu_temperature_stats_as_lines(
        self, stats: dict[str, float]
    ) -> list[str]:
        return [
            "\tGPU temperature:",
            (
                f"\t\tMean: {stats['mean']:.2f}C "
                f"Median: {stats['median']:.2f}C "
                f"StdDev: {stats['std_dev']:.4f}C"
            ),
            (
                f"\t\tMin: {stats['min']:.2f}C "
                f"Max: {stats['max']:.2f}C "
                f"Count: {stats['count']}"
            ),
        ]
