from dataclasses import dataclass, field
from typing import Optional

from lema.core.types.params.base_params import BaseParams


@dataclass
class ProfilerParams(BaseParams):
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
    enable_cpu_profiling: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to profile CPU activity. "
                "Corresponds to `torch.profiler.ProfilerActivity.CPU`."
            )
        },
    )
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
    with_stack: bool = field(
        default=False,
        metadata={
            "help": "Record source information (file and line number) for the ops."
        },
    )
    with_flops: bool = field(
        default=False,
        metadata={
            "help": (
                "Record module hierarchy (including function names) corresponding to "
                "the callstack of the op."
            )
        },
    )
    with_modules: bool = field(
        default=False,
        metadata={
            "help": (
                "Use formula to estimate the FLOPs (floating point operations) of "
                "specific operators (matrix multiplication and 2D convolution)."
            )
        },
    )
    row_limit: int = field(
        default=50,
        metadata={
            "help": (
                "Max number of rows to include into profiling report tables."
                "Set to -1 to make it unlimited."
            )
        },
    )
