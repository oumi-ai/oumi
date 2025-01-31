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

from dataclasses import dataclass, field
from typing import Optional

from oumi.core.configs.params.base_params import BaseParams


@dataclass
class ProfilerScheduleParams(BaseParams):
    """Parameters that define what subset of training steps to profile.

    Keeping profiling enabled for all training steps may be impractical
    as it may result in out-of-memory errors, extremely large trace files,
    and may interfere with regular training performance. This config can be used
    to enable PyTorch profiler only for a small number of training steps,
    which is not affected by such issues, and may still provide a useful signal
    for performance analysis.
    """

    enable_schedule: bool = False
    """Whether profiling schedule is enabled.

    If `False`, then profiling is enabled for the entire process
    duration, and all schedule parameters below will be ignored.
    """

    wait: int = 0
    """The number of training steps to skip at the beginning of
    each profiling cycle (`ProfilerAction.NONE`).
    Each cycle includes `wait + warmup + active` steps.
    """

    warmup: int = 1
    """The number of training steps to do profiling warmup (`ProfilerAction.WARMUP`)
    in each profiling cycle.
    """

    active: int = 3
    """The number of training steps to do active recording (`ProfilerAction.RECORD`)
    in each profiling cycle.
    """

    repeat: int = 1
    """The optional number of profiling cycles.

    Each cycle includes `wait + warmup + active` steps. The zero value means that
    the cycles will continue until the profiling is finished.
    """

    skip_first: int = 1
    """The number of initial training steps to skip at the beginning of profiling
    session (`ProfilerAction.NONE`).
    """

    def __post_init__(self):
        """Verifies params."""
        if not (
            self.wait >= 0
            and self.warmup >= 0
            and self.active > 0
            and self.repeat >= 0
            and self.skip_first >= 0
        ):
            raise ValueError(
                "Invalid profiler schedule arguments. The parameters "
                "wait: {self.wait}, warmup: {self.warmup}, repeat: {self.repeat}"
                "skip_first: {self.skip_first} must be non-negative."
            )
        if not (self.active > 0):
            raise ValueError(
                "Invalid profiler schedule arguments. The parameter "
                "active: {self.active} must be positive."
            )


@dataclass
class ProfilerParams(BaseParams):
    save_dir: Optional[str] = None
    """Directory where the profiling data will be saved to.

    If not specified and profiling is enabled, then the `profiler` sub-dir will be
    used under `output_dir`.
    """

    enable_cpu_profiling: bool = False
    """Whether to profile CPU activity.

    Corresponds to `torch.profiler.ProfilerActivity.CPU`.
    """

    enable_cuda_profiling: bool = False
    """Whether to profile CUDA.

    Corresponds to `torch.profiler.ProfilerActivity.CUDA`.
    """

    record_shapes: bool = False
    """Save information about operator’s input shapes."""

    profile_memory: bool = False
    """Track tensor memory allocation/deallocation."""

    with_stack: bool = False
    """Record source information (file and line number) for the ops."""

    with_flops: bool = False
    """Record module hierarchy (including function names) corresponding to
    the callstack of the op.
    """

    with_modules: bool = False
    """Use formula to estimate the FLOPs (floating point operations) of
    specific operators (matrix multiplication and 2D convolution).
    """

    row_limit: int = 50
    """Max number of rows to include into profiling report tables.

    Set to -1 to make it unlimited.
    """

    schedule: ProfilerScheduleParams = field(default_factory=ProfilerScheduleParams)
    """Parameters that define what subset of training steps to profile."""
