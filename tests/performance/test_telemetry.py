import math
import time

import pytest
import torch

from lema.performance.telemetry import (
    CudaTimerContext,
    TelemetryTracker,
    TimerContext,
)


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
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_timer_context():
    measurements = []
    with CudaTimerContext("test_cuda_timer", measurements):
        time.sleep(0.1)

    assert len(measurements) == 1
    assert measurements[0] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_telemetry_tracker_cuda_timer():
    tracker = TelemetryTracker()

    with tracker.cuda_timer("test_cuda_operation"):
        torch.cuda.synchronize()
        time.sleep(0.1)

    summary = tracker.get_summary()
    assert "test_cuda_operation" in summary["cuda_timers"]
    assert summary["cuda_timers"]["test_cuda_operation"]["mean"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_telemetry_tracker_log_gpu_memory():
    tracker = TelemetryTracker()

    tracker.log_gpu_memory()

    summary = tracker.get_summary()
    assert len(summary["gpu_memory"]) == 1
    assert "allocated" in summary["gpu_memory"][0]
    assert "reserved" in summary["gpu_memory"][0]


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
