import statistics
import time
from contextlib import ContextDecorator
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, cast

import torch


class TimerContext(ContextDecorator):
    """A context manager and decorator for timing code execution."""

    def __init__(self, name: str, measurements: List[float]):
        """Initialize a TimerContext object.

        Args:
            name: The name of the timer.
            measurements: A list to store the timing measurements.
        """
        self.name = name
        self.measurements = measurements
        self.start_time: Optional[float] = None

    def __enter__(self) -> "TimerContext":
        """Start the timer."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        """Stop the timer and record the elapsed time."""
        if self.start_time is not None:
            elapsed_time = time.perf_counter() - self.start_time
            self.measurements.append(elapsed_time)
            self.start_time = None
        return False


class CUDATimerContext(ContextDecorator):
    """A context manager and decorator for benchmarking CUDA operations."""

    def __init__(
        self,
        name: str,
        measurements: List[float],
    ):
        """Initialize a CUDATimerContext object.

        Args:
            name: The name of the timer.
            measurements: A list to store the timing measurements.
        """
        self.name = name
        self.measurements = measurements
        self.start_event = cast(torch.cuda.Event, torch.cuda.Event(enable_timing=True))
        self.end_event = cast(torch.cuda.Event, torch.cuda.Event(enable_timing=True))

        # Debugging flags
        self.pre_synchronize: bool = False
        self.post_synchronize: bool = False

    def __enter__(self) -> "CUDATimerContext":
        """Start the CUDA benchmark."""
        if not torch.cuda.is_available():
            return self

        if self.pre_synchronize:
            torch.cuda.synchronize()

        self.start_event.record()
        return self

    def __exit__(self, *exc) -> bool:
        """Stops the CUDA timer and record the elapsed time."""
        assert self.end_event is not None
        self.end_event.record()

        if self.post_synchronize:
            torch.cuda.synchronize()

        elapsed_time = (
            self.start_event.elapsed_time(self.end_event) / 1000
        )  # Convert to seconds

        self.measurements.append(elapsed_time)
        return False


def gpu_memory_logger(func: Callable) -> Callable:
    """Decorator function that logs the GPU memory usage of a given function.

    Args:
        func: The function to be decorated.

    Returns:
        The decorated function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            print("CUDA is not available. GPU memory usage cannot be logged.")
            return func(*args, **kwargs)

        torch.cuda.synchronize()
        start_memory = torch.cuda.memory_allocated()

        result = func(*args, **kwargs)

        torch.cuda.synchronize()
        end_memory = torch.cuda.memory_allocated()
        memory_diff = end_memory - start_memory
        print(f"{func.__name__} used {memory_diff / 1024**2:.2f} MB of GPU memory")

        return result

    return wrapper


class TelemetryTracker:
    """A class for tracking various telemetry metrics."""

    def __init__(self):
        """Initializes the TelemetryTracker object."""
        self.measurements: Dict[str, List[float]] = {}
        self.cuda_measurements: Dict[str, List[float]] = {}
        self.gpu_memory: List[Dict[str, float]] = []
        self.start_time = time.perf_counter()

    def timer(self, name: str) -> TimerContext:
        """Create a timer with the given name.

        Args:
            name: The name of the timer.

        Returns:
            A TimerContext object.
        """
        if name not in self.measurements:
            self.measurements[name] = []
        return TimerContext(name, self.measurements[name])

    def cuda_benchmark(
        self,
        name: str,
    ) -> CUDATimerContext:
        """Creates a CUDA benchmark with the given name.

        Args:
            name: The name of the benchmark.

        Returns:
            A CUDATimerContext object.
        """
        if name not in self.cuda_measurements:
            self.cuda_measurements[name] = []
        return CUDATimerContext(
            name,
            self.cuda_measurements[name],
        )

    def log_gpu_memory(self, custom_logger: Optional[Callable] = None) -> None:
        """Logs the GPU memory usage.

        Args:
            custom_logger: A custom logging function. If None, store in self.gpu_memory.
        """
        if not torch.cuda.is_available():
            print("CUDA is not available. GPU memory usage cannot be logged.")
            return

        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2  # Convert to MB
        memory_info = {"allocated": memory_allocated, "reserved": memory_reserved}

        if custom_logger:
            custom_logger(memory_info)
        else:
            self.gpu_memory.append(memory_info)

    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of the telemetry statistics.

        Returns:
            A dictionary containing the summary statistics.
        """
        total_time = time.perf_counter() - self.start_time
        summary = {
            "total_time": total_time,
            "timers": {},
            "cuda_benchmarks": {},
            "gpu_memory": self.gpu_memory,
        }

        for name, measurements in self.measurements.items():
            summary["timers"][name] = {
                "total": sum(measurements),
                "mean": statistics.mean(measurements),
                "median": statistics.median(measurements),
                "std_dev": statistics.stdev(measurements)
                if len(measurements) > 1
                else 0,
                "min": min(measurements),
                "max": max(measurements),
                "count": len(measurements),
            }
            summary["timers"][name]["percentage"] = (
                summary["timers"][name]["total"] / total_time * 100
            )

        for name, measurements in self.cuda_measurements.items():
            summary["cuda_benchmarks"][name] = {
                "mean": statistics.mean(measurements),
                "median": statistics.median(measurements),
                "std_dev": statistics.stdev(measurements)
                if len(measurements) > 1
                else 0,
                "min": min(measurements),
                "max": max(measurements),
                "count": len(measurements),
            }

        return summary

    def print_summary(self) -> None:
        """Prints a summary of the telemetry statistics."""
        summary = self.get_summary()
        print("Telemetry Summary:")
        print(f"Total time: {summary['total_time']:.2f} seconds")

        if summary["timers"]:
            print("\nCPU Timers:")
            for name, stats in summary["timers"].items():
                print(f"  {name}:")
                print(f"    Total: {stats['total']:.6f} seconds")
                print(f"    Mean: {stats['mean']:.6f} seconds")
                print(f"    Median: {stats['median']:.6f} seconds")
                print(f"    Std Dev: {stats['std_dev']:.6f} seconds")
                print(f"    Min: {stats['min']:.6f} seconds")
                print(f"    Max: {stats['max']:.6f} seconds")
                print(f"    Count: {stats['count']}")
                print(f"    Percentage of total time: {stats['percentage']:.2f}%")

        if summary["cuda_benchmarks"]:
            print("\nCUDA Benchmarks:")
            for name, stats in summary["cuda_benchmarks"].items():
                print(f"  {name}:")
                print(f"    Mean: {stats['mean']:.6f} seconds")
                print(f"    Median: {stats['median']:.6f} seconds")
                print(f"    Std Dev: {stats['std_dev']:.6f} seconds")
                print(f"    Min: {stats['min']:.6f} seconds")
                print(f"    Max: {stats['max']:.6f} seconds")
                print(f"    Count: {stats['count']}")

        if summary["gpu_memory"]:
            max_memory = max(usage["allocated"] for usage in summary["gpu_memory"])
            print(f"\nPeak GPU memory usage: {max_memory:.2f} MB")
