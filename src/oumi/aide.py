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

"""AIDE agentic code optimization orchestration.

This module parallels :mod:`oumi.tune` — it provides the top-level ``aide()``
function that orchestrates the full AIDE optimization lifecycle: directory
creation, logging setup, telemetry, search loop, result extraction, and
distributed cleanup.
"""

import time
from pathlib import Path
from pprint import pformat

from oumi.builders.agentic import build_agentic_optimizer
from oumi.core.agentic.aide_optimizer import _build_oumi_task_desc
from oumi.core.agentic.base_agentic_optimizer import AideResult
from oumi.core.configs import AideConfig
from oumi.core.distributed import (
    barrier,
    cleanup_distributed,
    get_device_rank_info,
    is_distributed,
    is_local_process_zero,
    is_world_process_zero,
)
from oumi.train import _ensure_dir_exists, _log_feedback_request
from oumi.utils.device_utils import log_nvidia_gpu_runtime_info
from oumi.utils.git_utils import get_git_revision_hash, get_git_tag
from oumi.utils.io_utils import save_json
from oumi.utils.logging import configure_logger, logger
from oumi.utils.torch_utils import (
    device_cleanup,
    log_devices_info,
    log_peak_gpu_memory,
    log_versioning_info,
)
from oumi.utils.version_utils import get_oumi_version, is_dev_build


def _create_aide_dirs(config: AideConfig) -> None:
    """Creates misc directories referenced in config."""
    _ensure_dir_exists(config.aide.output_dir, "aide.output_dir")
    _ensure_dir_exists(config.aide.workspace_dir, "aide.workspace_dir")
    telemetry_dir = config.aide.telemetry_dir
    if telemetry_dir:
        _ensure_dir_exists(telemetry_dir, "aide.telemetry_dir")


def _log_aide_info(config: AideConfig) -> None:
    """Logs misc infos about AIDE config/devices/etc. Writes to files."""
    telemetry_dir = config.aide.telemetry_dir
    if telemetry_dir and is_world_process_zero():
        device_rank_info = get_device_rank_info()
        save_json(
            {
                "LOCAL_WORLD_SIZE": device_rank_info.local_world_size,
                "WORLD_SIZE": device_rank_info.world_size,
            },
            telemetry_dir / "world_size.json",
        )

    if is_local_process_zero():
        log_versioning_info()
        log_devices_info(
            (telemetry_dir / "devices_info.txt")
            if telemetry_dir and is_world_process_zero()
            else None
        )
        logger.info(f"Oumi version: {get_oumi_version()}")
        if is_dev_build():
            logger.info(f"Git revision hash: {get_git_revision_hash()}")
            logger.info(f"Git tag: {get_git_tag()}")


def aide(
    config: AideConfig,
    verbose: bool = False,
) -> AideResult:
    """Run AIDE agentic code optimization.

    This function orchestrates the complete AIDE optimization lifecycle,
    paralleling :func:`oumi.tune.tune` for Optuna-based tuning:

    1. Create output/workspace directories
    2. Set up logging and telemetry
    3. Build the AIDE optimizer (with try/except for optional dependency)
    4. Run the tree-search loop (draft/debug/improve)
    5. Extract and return the best solution
    6. Clean up distributed resources

    Args:
        config: The AIDE configuration.
        verbose: Enable verbose logging with additional debug information.

    Returns:
        AideResult containing the best solution found, its metric value,
        and paths to the journal and solution files.
    """
    _START_TIME = time.time()

    _create_aide_dirs(config)
    _log_aide_info(config)

    # Configure logging to file
    log_dir = Path(config.aide.output_dir) / "logs"
    for logger_name in ("oumi", "oumi.telemetry"):
        configure_logger(
            logger_name,
            level=config.aide.log_level,
            log_dir=log_dir,
        )

    telemetry_dir = config.aide.telemetry_dir
    if is_local_process_zero():
        if verbose:
            logger.info(f"AideConfig:\n{pformat(config)}")
        if telemetry_dir and is_world_process_zero():
            config_path = telemetry_dir / "aide_config.yaml"
            config.to_yaml(str(config_path))
            logger.info(f"AIDE config saved to {config_path}")

    # Build task description from config
    task_desc = _build_oumi_task_desc(
        goal=config.goal,
        surface=config.aide.surface,
        target_metric=config.aide.target_metric,
        target_direction=config.aide.target_direction,
        base_config_yaml=config.base_training_config or "",
        mutable_paths=config.mutable_config_paths,
    )

    # Build optimizer
    workspace_dir = Path(config.aide.workspace_dir)
    optimizer = build_agentic_optimizer(
        aide_params=config.aide,
        task_desc=task_desc,
        workspace_dir=workspace_dir,
        base_training_config=config.base_training_config,
    )

    logger.info(f"AIDE init time: {time.time() - _START_TIME:.3f}s")
    logger.info(
        f"Starting AIDE agentic optimization: "
        f"{config.aide.steps} steps, surface={config.aide.surface.value}, "
        f"metric={config.aide.target_metric} ({config.aide.target_direction})"
    )

    # Run the search loop
    for step_idx in range(config.aide.steps):
        logger.info(f"AIDE step {step_idx + 1}/{config.aide.steps}")
        optimizer.step()

    # Extract best result
    result = optimizer.get_best_solution()
    optimizer.cleanup()

    # Log summary
    summary = optimizer.get_search_summary()
    logger.info(
        f"AIDE optimization complete. "
        f"Best metric: {result.best_metric}, "
        f"Good: {summary['good_nodes']}, Buggy: {summary['buggy_nodes']}"
    )
    logger.info(f"Best solution saved to: {result.best_solution_path}")

    # Cleanup (mirrors tune.py pattern)
    device_cleanup()
    log_nvidia_gpu_runtime_info(log_prefix="GPU Metrics After AIDE:")
    log_peak_gpu_memory()
    barrier()
    if is_distributed():
        cleanup_distributed()

    _log_feedback_request()

    return result
