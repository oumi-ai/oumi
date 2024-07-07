import pathlib
from contextlib import contextmanager
from typing import Optional

import torch

from lema.core.types.params.profiler_params import ProfilerParams
from lema.logging import logger

_PROFILER_PREFIX = "PROF:"


@contextmanager
def torch_profile(
    params: ProfilerParams,
    save_dir: Optional[str] = None,
    out_prefix: str = "",
    record_function_name: str = "lema.train",
):
    """Initializes Profiler context."""
    profile_activities = []
    if params.enable_cpu_profiling:
        profile_activities.append(torch.profiler.ProfilerActivity.CPU)
    if params.enable_cuda_profiling:
        profile_activities.append(torch.profiler.ProfilerActivity.CUDA)

    if not profile_activities:
        # Nothing to profile. Return noop/null context.
        logger.info(f"{_PROFILER_PREFIX} profiler disabled!")
        yield
        return

    save_dir = save_dir or params.save_dir

    logger.info(f"{_PROFILER_PREFIX} Starting profiling...")
    logger.info(f"{_PROFILER_PREFIX} Save dir: {save_dir}")
    logger.info(f"{_PROFILER_PREFIX} Output prefix: {out_prefix}")
    logger.info(f"{_PROFILER_PREFIX} Function: {record_function_name}")
    logger.info(f"{_PROFILER_PREFIX} Params: {params}")

    with torch.profiler.profile(
        activities=profile_activities,
        # schedule=schedule,
        on_trace_ready=(
            torch.profiler.tensorboard_trace_handler(save_dir) if save_dir else None
        ),
        record_shapes=params.record_shapes,
        profile_memory=params.profile_memory,
        with_stack=params.with_stack,
        with_flops=params.with_flops,
        with_modules=params.with_modules,
    ) as prof:
        try:
            with torch.profiler.record_function(record_function_name):
                yield
        except Exception as e:
            # The inner function raised an error
            import traceback

            logger.error(
                _PROFILER_PREFIX
                + "".join(traceback.format_exception(None, e, e.__traceback__))
            )
            raise

    save_dir_path: Optional[pathlib.Path] = pathlib.Path(save_dir) if save_dir else None
    if save_dir_path:
        save_dir_path.mkdir(parents=True, exist_ok=True)

    sort_by = []
    if params.enable_cpu_profiling:
        sort_by.extend(
            [
                "cpu_time_total",
                "self_cpu_time_total",
                "cpu_memory_usage",
                "self_cpu_memory_usage",
            ]
        )
    if params.enable_cuda_profiling:
        sort_by.extend(
            [
                "cuda_time_total",
                "self_cuda_time_total",
                "cuda_memory_usage",
                "self_cuda_memory_usage",
            ]
        )
    for group_by_input_shape in (False, True):
        group_by_shape_tag = "_by_shape" if group_by_input_shape else ""
        prof_avgs = prof.key_averages(group_by_input_shape=group_by_input_shape)
        for sort_key in sort_by:
            prof_table = prof_avgs.table(sort_by=sort_by, row_limit=params.row_limit)
            logger.info(
                f"PROFILER: {sort_key}[group_by_input_shape={group_by_input_shape}]"
                f"\n{prof_table}\n"
            )
            if save_dir_path:
                file_path: pathlib.Path = (
                    save_dir_path / f"{out_prefix}{sort_key}{group_by_shape_tag}.txt"
                )
                with file_path.open("w") as f:
                    f.write(prof_table)

    if save_dir_path:
        file_name: pathlib.Path = save_dir_path / f"{out_prefix}trace.json"
        logger.info(f"Exporting profiler Chrome trace to {file_name} ...")
        prof.export_chrome_trace(file_name.as_posix())

    return
