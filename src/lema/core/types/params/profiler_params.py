from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProfilerParams:
    save_dir: str = field(
        default="",
        metadata={"help": "Directory where the profile data will be saved to."},
    )
    enable_cpu_profiling: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to profile CPU activity. "
                "Corresponds to `torch.profiler.ProfilerActivity.CPU`."
            )
        },
    )
    enable_cuda_profiling: bool = field(
        default=True,
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
        metadata={"help": ("Save information about operator’s input shapes.")},
    )
    profile_memory: bool = field(
        default=False,
        metadata={"help": ("Track tensor memory allocation/deallocation.")},
    )
    with_stack: bool = field(
        default=False,
        metadata={
            "help": ("Record source information (file and line number) for the ops.")
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
        default=20,
        metadata={
            "help": ("Max number of rows to include into profiling report tables.")
        },
    )
