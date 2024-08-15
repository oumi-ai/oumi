from dataclasses import dataclass, field
from typing import Optional

from lema.core.types.params.base_params import BaseParams


@dataclass
class ProfilerScheduleParams(BaseParams):
    #: Whether profiling schedule is enabled.
    #: If `False`, then profiling is enabled for the entire process
    #: duration, and all schedule parameters below will be ignored.
    enable_schedule: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether profiling schedule is enabled. "
                "If `False`, then profiling is enabled for the entire process "
                "duration, and all schedule parameters below will be ignored. "
            )
        },
    )

    #: The number of training steps to skip at the beginning of
    #: each profiling cycle (`ProfilerAction.NONE`).
    #: Each cycle includes `wait + warmup + active` steps.
    wait: int = field(
        default=0,
        metadata={
            "help": (
                "The number of training steps to skip at the beginning of "
                "each profiling cycle (`ProfilerAction.NONE`). "
                "Each cycle includes `wait + warmup + active` steps."
            )
        },
    )

    #: The number of training steps to do profiling warmup
    #: (`ProfilerAction.WARMUP`) in each profiling cycle.
    warmup: int = field(
        default=1,
        metadata={
            "help": (
                "The number of training steps to do profiling warmup "
                "(`ProfilerAction.WARMUP`) in each profiling cycle. "
            )
        },
    )

    #: The number of training steps to do active recording
    #: (`ProfilerAction.RECORD`) in each profiling cycle.
    active: int = field(
        default=3,
        metadata={
            "help": (
                "The number of training steps to do active recording "
                "(`ProfilerAction.RECORD`) in each profiling cycle. "
            )
        },
    )

    #: The optional number of profiling cycles.
    #: Each cycle includes `wait + warmup + active` steps.
    #: The zero value means that the cycles will continue
    #: until the profiling is finished.
    repeat: int = field(
        default=1,
        metadata={
            "help": (
                "The optional number of profiling cycles. "
                "Each cycle includes `wait + warmup + active` steps."
                "The zero value means that the cycles will continue "
                "until the profiling is finished."
            )
        },
    )

    #: The number of initial training steps to skip at the beginning of
    #: profiling session (`ProfilerAction.NONE`).
    skip_first: int = field(
        default=1,
        metadata={
            "help": (
                "The number of initial training steps to skip at the beginning of "
                "profiling session (`ProfilerAction.NONE`)."
            )
        },
    )

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
    #: Directory where the profiling data will be saved to.
    #: If not specified and profiling is enabled, then
    #: the `profiler` sub-dir will be used under `output_dir`.
    save_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Directory where the profiling data will be saved to. "
                "If not specified and profiling is enabled, then "
                "the `profiler` sub-dir will be used under `output_dir`."
            )
        },
    )

    #: Whether to profile CPU activity.
    #: Corresponds to `torch.profiler.ProfilerActivity.CPU`.
    enable_cpu_profiling: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to profile CPU activity. "
                "Corresponds to `torch.profiler.ProfilerActivity.CPU`."
            )
        },
    )

    #: Whether to profile CUDA.
    #: Corresponds to `torch.profiler.ProfilerActivity.CUDA`.
    enable_cuda_profiling: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to profile CUDA. "
                "Corresponds to `torch.profiler.ProfilerActivity.CUDA`."
            )
        },
    )
    # TODO: Add schedule params
    record_shapes: bool = field(
        default=False,
        metadata={"help": "Save information about operator’s input shapes."},
    )
    profile_memory: bool = field(
        default=False,
        metadata={"help": "Track tensor memory allocation/deallocation."},
    )

    #: Record source information (file and line number) for the ops.
    with_stack: bool = field(
        default=False,
        metadata={
            "help": "Record source information (file and line number) for the ops."
        },
    )

    #: Record module hierarchy (including function names) corresponding to
    #: the callstack of the op.
    with_flops: bool = field(
        default=False,
        metadata={
            "help": (
                "Record module hierarchy (including function names) corresponding to "
                "the callstack of the op."
            )
        },
    )

    #: Use formula to estimate the FLOPs (floating point operations) of
    #: specific operators (matrix multiplication and 2D convolution).
    with_modules: bool = field(
        default=False,
        metadata={
            "help": (
                "Use formula to estimate the FLOPs (floating point operations) of "
                "specific operators (matrix multiplication and 2D convolution)."
            )
        },
    )

    #: Max number of rows to include into profiling report tables.
    #: Set to -1 to make it unlimited.
    row_limit: int = field(
        default=50,
        metadata={
            "help": (
                "Max number of rows to include into profiling report tables."
                "Set to -1 to make it unlimited."
            )
        },
    )

    schedule: ProfilerScheduleParams = field(default_factory=ProfilerScheduleParams)
