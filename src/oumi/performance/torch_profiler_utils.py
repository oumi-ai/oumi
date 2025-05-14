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

import functools
import pathlib
from contextlib import contextmanager
from typing import Optional

import torch

from oumi.core.configs.params.profiler_params import ProfilerParams
from oumi.core.distributed import DeviceRankInfo, get_device_rank_info
from oumi.utils.logging import logger

_PROFILER_LOG_PREFIX = "PROF:"
_PROFILER_DEFAULT_SUB_DIR = "profiler"


def _configure_torch_profile_save_dir(
    params: ProfilerParams, training_output_dir: Optional[str]
) -> ProfilerParams:
    """Auto-generates ProfilerParams.saved_dir if not specified explicitly."""
    if not params.save_dir and training_output_dir:
        params.save_dir = str(
            pathlib.Path(training_output_dir) / _PROFILER_DEFAULT_SUB_DIR
        )
    return params


def _on_trace_ready(
    prof,
    *,
    out_prefix: str,
    enable_cpu_profiling: bool,
    enable_cuda_profiling: bool,
    params: ProfilerParams,
    save_dir_path: Optional[pathlib.Path],
) -> None:
    logger.info(f"{_PROFILER_LOG_PREFIX} on_trace_ready(out_prefix={out_prefix})")
    sort_by = []
    if enable_cpu_profiling:
        sort_by.extend(
            [
                "cpu_time_total",
                "self_cpu_time_total",
            ]
        )
        if params.profile_memory:
            sort_by.extend(
                [
                    "cpu_memory_usage",
                    "self_cpu_memory_usage",
                ]
            )

    if enable_cuda_profiling:
        sort_by.extend(
            [
                "cuda_time_total",
                "self_cuda_time_total",
            ]
        )
        if params.profile_memory:
            sort_by.extend(
                [
                    "cuda_memory_usage",
                    "self_cuda_memory_usage",
                ]
            )
    # if `params.record_shapes` is True, then also generate reports with breakdowns
    # by tensor shapes. Otherwise (the default), only produce profiling reports
    # without shape breakdowns (less verbose).
    for group_by_input_shape in [False] + ([True] if params.record_shapes else []):
        group_by_shape_tag = "_by_shape" if group_by_input_shape else ""
        prof_avgs = prof.key_averages(group_by_input_shape=group_by_input_shape)
        for sort_key in sort_by:
            prof_table = prof_avgs.table(sort_by=sort_key, row_limit=params.row_limit)
            logger.info(
                f"{_PROFILER_LOG_PREFIX} {sort_key} "
                f"[group_by_shape={group_by_input_shape}]"
                f"\n{prof_table}\n"
            )
            if save_dir_path:
                file_path: pathlib.Path = (
                    save_dir_path / f"{out_prefix}_{sort_key}{group_by_shape_tag}.txt"
                )
                with file_path.open("w") as f:
                    f.write(prof_table)

    if save_dir_path:
        # Save gzip-ed trace (JSON traces compress extremely well).
        # Open traces on https://ui.perfetto.dev/ or `chrome://tracing`
        file_name: pathlib.Path = save_dir_path / f"{out_prefix}_pt_trace.json.gz"
        logger.info(f"Exporting profiler Chrome trace to {file_name} ...")
        prof.export_chrome_trace(str(file_name))


@contextmanager
def torch_profile(
    params: ProfilerParams,
    *,
    training_output_dir: Optional[str],
    record_function_name: str = "oumi.train",
):
    """Creates PyTorch Profiler context manager.

    Args:
        params: Profiler config.
        training_output_dir: If `ProfilerParams.save_dir` is not specified, then
            a "profiler" sub-directory will be created under `training_output_dir`,
            and used to save profiler traces.
        record_function_name: The name to use with `torch.profiler.record_function()`
            for top-level `train()` operation.

    Yields:
        torch.profiler.profile or None: The newly-created Profiler object if profiling
            is enabled, or `None` otherwise.

    Example:
        To profile a training loop::

            with torch_profile(params, record_function_name="oumi.train") as prof:
                for i in range(n):
                    training_step()
                    if prof is not None:
                        prof.step()
    """
    params = _configure_torch_profile_save_dir(params, training_output_dir)

    device_rank_info: DeviceRankInfo = get_device_rank_info()
    out_prefix: str = (
        f"prof_{device_rank_info.rank:03}_local_{device_rank_info.local_rank:02}"
    )

    profile_activities = []
    enable_cpu_profiling = params.enable_cpu_profiling
    if enable_cpu_profiling:
        profile_activities.append(torch.profiler.ProfilerActivity.CPU)

    enable_cuda_profiling = False
    if params.enable_cuda_profiling:
        enable_cuda_profiling = torch.cuda.is_available()
        if enable_cuda_profiling:
            profile_activities.append(torch.profiler.ProfilerActivity.CUDA)
        else:
            logger.warning(
                f"{_PROFILER_LOG_PREFIX} CUDA profiling is requested "
                "while CUDA is not available!"
            )

    if not profile_activities:
        # Nothing to profile. Return noop/null context.
        logger.info(f"{_PROFILER_LOG_PREFIX} Torch Profiler disabled!")
        yield None
        return

    logger.info(f"{_PROFILER_LOG_PREFIX} Starting profiling...")
    logger.info(f"{_PROFILER_LOG_PREFIX} Save dir: {params.save_dir}")
    save_dir_path: Optional[pathlib.Path] = (
        pathlib.Path(params.save_dir) if params.save_dir else None
    )
    if save_dir_path:
        save_dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"{_PROFILER_LOG_PREFIX} Save dir: {save_dir_path}")
    logger.info(f"{_PROFILER_LOG_PREFIX} Output prefix: {out_prefix}")
    logger.info(f"{_PROFILER_LOG_PREFIX} Function: {record_function_name}")
    logger.info(f"{_PROFILER_LOG_PREFIX} Params: {params}")

    # See also torch.profiler.tensorboard_trace_handler
    trace_handler = functools.partial(
        _on_trace_ready,
        out_prefix=out_prefix,
        enable_cpu_profiling=enable_cpu_profiling,
        enable_cuda_profiling=enable_cuda_profiling,
        params=params,
        save_dir_path=save_dir_path,
    )

    schedule = None
    if params.schedule.enable_schedule:
        schedule = torch.profiler.schedule(
            skip_first=params.schedule.skip_first,
            wait=params.schedule.wait,
            warmup=params.schedule.warmup,
            active=params.schedule.active,
            repeat=params.schedule.repeat,
        )

    profiler = torch.profiler.profile(
        activities=profile_activities,
        on_trace_ready=trace_handler,
        record_shapes=params.record_shapes,
        profile_memory=params.profile_memory,
        with_stack=params.with_stack,
        with_flops=params.with_flops,
        with_modules=params.with_modules,
        schedule=schedule,
    )
    with profiler:
        try:
            with torch.profiler.record_function(record_function_name):
                yield profiler
        except Exception as e:
            # The inner function raised an error
            import traceback

            logger.error(
                f"{_PROFILER_LOG_PREFIX}"
                + "".join(traceback.format_exception(None, e, e.__traceback__))
            )
            raise

    logger.info(f"{_PROFILER_LOG_PREFIX} Finished post-processing!")
    return
