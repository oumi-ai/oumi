import math
import time

import torch

from oumi.performance.telemetry import (
    CudaTimerContext,
    TelemetryTracker,
    TimerContext,
)
from tests.markers import requires_gpus


#
# Timer
#
def test_timer_context():
    measurements = []
    with TimerContext("test_timer", measurements):
        time.sleep(0.1)

    assert len(measurements) == 1
    assert math.isclose(0.1, measurements[0], rel_tol=0.1)


def test_timer_context_as_decorator():
    measurements = []

    @TimerContext("test_decorator", measurements)
    def sample_function():
        time.sleep(0.1)

    sample_function()

    assert len(measurements) == 1
    assert math.isclose(0.1, measurements[0], rel_tol=0.1)


#
# Cuda Timer
#
@requires_gpus()
def test_cuda_timer_context():
    measurements = []
    with CudaTimerContext("test_cuda_timer", measurements):
        time.sleep(0.1)

    assert len(measurements) == 1
    assert measurements[0] > 0


@requires_gpus()
def test_cuda_timer_context_as_decorator():
    measurements = []

    @CudaTimerContext("test_cuda_decorator", measurements)
    def sample_cuda_function():
        time.sleep(0.1)

    sample_cuda_function()

    assert len(measurements) == 1
    assert measurements[0] > 0


#
# Telemetry Tracker
#
def test_telemetry_tracker_timer():
    tracker = TelemetryTracker()

    with tracker.timer("test_operation"):
        time.sleep(0.1)

    summary = tracker.get_summary()
    assert "test_operation" in summary["timers"]
    assert math.isclose(0.1, summary["timers"]["test_operation"]["total"], rel_tol=0.1)

    tracker.print_summary()


@requires_gpus()
def test_telemetry_tracker_cuda_timer():
    tracker = TelemetryTracker()

    with tracker.cuda_timer("test_cuda_operation"):
        torch.cuda.synchronize()
        time.sleep(0.1)

    summary = tracker.get_summary()
    assert "test_cuda_operation" in summary["cuda_timers"]
    assert summary["cuda_timers"]["test_cuda_operation"]["mean"] > 0

    tracker.print_summary()


@requires_gpus()
def test_telemetry_tracker_log_gpu_memory():
    tracker = TelemetryTracker()

    tracker.log_gpu_memory()

    summary = tracker.get_summary()
    assert len(summary["gpu_memory"]) == 1
    assert "allocated" in summary["gpu_memory"][0]
    assert "reserved" in summary["gpu_memory"][0]

    tracker.print_summary()


@requires_gpus()
def test_telemetry_tracker_record_gpu_temperature():
    tracker = TelemetryTracker()

    tracker.record_gpu_temperature()
    tracker.record_gpu_temperature()
    tracker.record_gpu_temperature()

    summary = tracker.get_summary()
    assert "gpu_temperature" in summary
    assert len(summary["gpu_temperature"]) == 6
    assert summary["gpu_temperature"]["count"] == 3

    assert (
        summary["gpu_temperature"]["min"] > 0
        and summary["gpu_temperature"]["min"] < 100
    )
    assert (
        summary["gpu_temperature"]["max"] > 0
        and summary["gpu_temperature"]["max"] < 100
    )
    assert (
        summary["gpu_temperature"]["mean"] > 0
        and summary["gpu_temperature"]["mean"] < 100
    )
    assert (
        summary["gpu_temperature"]["median"] > 0
        and summary["gpu_temperature"]["median"] < 100
    )
    assert summary["gpu_temperature"]["std_dev"] >= 0

    tracker.print_summary()


def test_telemetry_tracker_get_summary():
    tracker = TelemetryTracker()

    with tracker.timer("operation1"):
        time.sleep(0.1)

    with tracker.timer("operation2"):
        time.sleep(0.2)

    summary = tracker.get_summary()
    assert "total_time" in summary
    assert "timers" in summary
    assert "operation1" in summary["timers"]
    assert "operation2" in summary["timers"]
    assert (
        summary["timers"]["operation2"]["total"]
        > summary["timers"]["operation1"]["total"]
    )

    all_summaries = tracker.get_summaries_from_all_ranks()
    assert len(all_summaries) == 1
    assert "total_time" in all_summaries[0]
    assert "timers" in all_summaries[0]
    assert "operation1" in all_summaries[0]["timers"]
    assert "operation2" in all_summaries[0]["timers"]
    assert (
        all_summaries[0]["timers"]["operation2"]["total"]
        > all_summaries[0]["timers"]["operation1"]["total"]
    )

    info = tracker.compute_cross_rank_summaries(
        all_summaries, measurement_names={"total_time"}
    )
    assert "total_time" in info
    assert len(info) == 1

    info = info["total_time"]
    assert len(info) == 8
    assert "count" in info and info["count"] == 1.0
    assert "max" in info and info["max"] > 0.0
    assert "max_index" in info and info["max_index"] == 0
    assert "mean" in info and info["mean"] > 0.0 and info["mean"] == info["max"]
    assert "median" in info and info["median"] > 0.0 and info["median"] == info["max"]
    assert "min" in info and info["min"] > 0.0 and info["min"] == info["max"]
    assert "min_index" in info and info["min_index"] == 0
    assert "std_dev" in info and info["std_dev"] == 0

    info = tracker.compute_cross_rank_summaries(
        all_summaries,
        measurement_names={
            "timers": {
                "operation1": {"median"},
                "operation2": {"total", "max"},
            },
        },
    )
    assert "timers" in info
    assert isinstance(info["timers"], dict)
    assert len(info["timers"]) == 2

    assert "operation1" in info["timers"]
    assert isinstance(info["timers"]["operation1"], dict)
    assert len(info["timers"]["operation1"]) == 1
    assert "median" in info["timers"]["operation1"]
    assert isinstance(info["timers"]["operation1"]["median"], dict)
    assert len(info["timers"]["operation1"]["median"]) == 8

    assert "operation2" in info["timers"]
    assert "total" in info["timers"]["operation2"]
    assert isinstance(info["timers"]["operation2"]["total"], dict)
    assert len(info["timers"]["operation2"]["total"]) == 8
    assert "max" in info["timers"]["operation2"]
    assert isinstance(info["timers"]["operation2"]["max"], dict)
    assert len(info["timers"]["operation2"]["max"]) == 8


@requires_gpus()
def test_telemetry_tracker_get_summary_with_gpu_temperature():
    tracker = TelemetryTracker()

    with tracker.timer("operation1"):
        time.sleep(0.1)

    with tracker.timer("operation2"):
        time.sleep(0.2)

    tracker.record_gpu_temperature()
    tracker.record_gpu_temperature()
    tracker.record_gpu_temperature()

    summary = tracker.get_summary()
    assert "total_time" in summary
    assert "gpu_temperature" in summary
    assert len(summary["gpu_temperature"]) == 6
    assert "timers" in summary
    assert "operation1" in summary["timers"]
    assert "operation2" in summary["timers"]
    assert (
        summary["timers"]["operation2"]["total"]
        > summary["timers"]["operation1"]["total"]
    )

    all_summaries = tracker.get_summaries_from_all_ranks()
    assert len(all_summaries) == 1
    assert "total_time" in all_summaries[0]
    assert "gpu_temperature" in all_summaries[0]
    assert "timers" in all_summaries[0]
    assert "operation1" in all_summaries[0]["timers"]
    assert "operation2" in all_summaries[0]["timers"]
    assert (
        all_summaries[0]["timers"]["operation2"]["total"]
        > all_summaries[0]["timers"]["operation1"]["total"]
    )

    info = tracker.compute_cross_rank_summaries(
        all_summaries, measurement_names={"total_time"}
    )
    assert "total_time" in info
    assert len(info) == 1

    info = info["total_time"]
    assert len(info) == 8
    assert "count" in info and info["count"] == 1.0
    assert "max" in info and info["max"] > 0.0
    assert "max_index" in info and info["max_index"] == 0
    assert "mean" in info and info["mean"] > 0.0 and info["mean"] == info["max"]
    assert "median" in info and info["median"] > 0.0 and info["median"] == info["max"]
    assert "min" in info and info["min"] > 0.0 and info["min"] == info["max"]
    assert "min_index" in info and info["min_index"] == 0
    assert "std_dev" in info and info["std_dev"] == 0

    info = tracker.compute_cross_rank_summaries(
        all_summaries,
        measurement_names={
            "timers": {
                "operation1": {"median"},
                "operation2": {"total", "max"},
            },
            "gpu_temperature": {"max"},
        },
    )
    assert "timers" in info
    assert isinstance(info["timers"], dict)
    assert len(info["timers"]) == 2
    assert "gpu_temperature" in info
    assert isinstance(info["gpu_temperature"], dict)
    assert len(info["gpu_temperature"]) == 1

    assert "max" in info["gpu_temperature"]
    assert isinstance(info["gpu_temperature"]["max"], dict)
    assert len(info["gpu_temperature"]["max"]) == 8

    assert "count" in info["gpu_temperature"]["max"]
    assert "mean" in info["gpu_temperature"]["max"]
    assert "median" in info["gpu_temperature"]["max"]
    assert "std_dev" in info["gpu_temperature"]["max"]
    assert "min" in info["gpu_temperature"]["max"]
    assert "min_index" in info["gpu_temperature"]["max"]
    assert "max" in info["gpu_temperature"]["max"]
    assert "max_index" in info["gpu_temperature"]["max"]

    assert "operation1" in info["timers"]
    assert isinstance(info["timers"]["operation1"], dict)
    assert len(info["timers"]["operation1"]) == 1
    assert "median" in info["timers"]["operation1"]
    assert isinstance(info["timers"]["operation1"]["median"], dict)
    assert len(info["timers"]["operation1"]["median"]) == 8

    assert "operation2" in info["timers"]
    assert "total" in info["timers"]["operation2"]
    assert isinstance(info["timers"]["operation2"]["total"], dict)
    assert len(info["timers"]["operation2"]["total"]) == 8
    assert "max" in info["timers"]["operation2"]
    assert isinstance(info["timers"]["operation2"]["max"], dict)
    assert len(info["timers"]["operation2"]["max"]) == 8
