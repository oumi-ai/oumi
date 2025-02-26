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

import copy
from pathlib import Path
from typing import Any, Optional

from oumi.core.configs import EvaluationConfig, EvaluationTaskParams
from oumi.core.evaluation.evaluation_result import EvaluationResult
from oumi.utils.serialization_utils import json_serializer
from oumi.utils.version_utils import get_python_package_versions

# Output filenames for saving evaluation results and reproducibility information.
OUTPUT_FILENAME_TASK_RESULT = "task_result.json"
OUTPUT_FILENAME_TASK_PARAMS = "task_params.json"
OUTPUT_FILENAME_BACKEND_CONFIG = "backend_config.json"
OUTPUT_FILENAME_MODEL_PARAMS = "model_params.json"
OUTPUT_FILENAME_GENERATION_PARAMS = "generation_params.json"
OUTPUT_FILENAME_INFERENCE_PARAMS = "inference_params.json"
OUTPUT_FILENAME_PACKAGE_VERSIONS = "package_versions.json"


def _save_to_file(output_path: Path, data: Any) -> None:
    """Serialize and save `data` to `output_path`."""
    with open(output_path, "w") as file_out:
        file_out.write(json_serializer(data))


def _find_non_existing_output_dir_from_base_dir(base_dir: Path) -> Path:
    """Finds a new output directory, if the provided `base_dir` already exists.

    Why is this function useful?
        Users may repeatedly run the same evaluation script, which will overwrite the
        results of the existing output directory. When this happens, we could fail, to
        avoid corrupting previous evaluation results. However, for a more user-friendly
        experience, we can automatically create a new output directory with a unique
        name. This function does this by appending an index to the base directory that
        was provided (`base_dir`), as follows: `<base_dir>_<index>`.

    Args:
        base_dir: The base directory where the evaluation results would be saved.

    Returns:
        A new output directory (does not exist yet), if `base_dir` already exists,
        or the original `base_dir` if it does not exist.
    """
    counter = 1
    while base_dir.exists():
        counter += 1
        base_dir = Path(f"{base_dir}_{counter}")
    return base_dir


def save_evaluation_output(
    backend_name: str,
    task_params: EvaluationTaskParams,
    evaluation_result: EvaluationResult,
    base_output_dir: Optional[str],
    config: Optional[EvaluationConfig],
    start_time_str: Optional[str],
    elapsed_time_sec: Optional[int],
) -> None:
    """Writes configuration settings and evaluation outputs to files.

    Args:
        backend_name: The name of the evaluation backend used (e.g., "lm_harness").
        task_params: Oumi task parameters used for this evaluation.
        evaluation_result: The evaluation results to save.
        base_output_dir: The directory where the evaluation results will be saved.
            A subdirectory with the name `<base_output_dir> / <platform>_<time>`
            will be created to retain all files related to this evaluation. If there is
            an existing directory with the same name, a new directory with a unique
            index will be created: `<base_output_dir> / <platform>_<time>_<index>`.
        config: Oumi evaluation configuration settings used for the evaluation.
        start_time_str: User-friendly string with the start date/time of the evaluation.
        elapsed_time_sec: The duration of the evaluation (in seconds).
    """
    # Ensure the evaluation backend and output directory are valid.
    if not backend_name:
        raise ValueError("The evaluation backend name must be provided.")
    base_output_dir = base_output_dir or "."

    # Create the output directory: `<base_output_dir> / <platform>_<time>`.
    start_time_in_path = f"_{start_time_str}" if start_time_str else ""
    output_dir = Path(base_output_dir) / f"{backend_name}{start_time_in_path}"
    if output_dir.exists():
        output_dir = _find_non_existing_output_dir_from_base_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    # Save all evaluation metrics, start date/time, and duration.
    if evaluation_result.task_result:
        task_result = copy.deepcopy(evaluation_result.task_result)
    else:
        task_result = {}
    if start_time_str:
        task_result["start_time"] = start_time_str
    if elapsed_time_sec:
        task_result["duration_sec"] = elapsed_time_sec
    if task_result:
        _save_to_file(output_dir / OUTPUT_FILENAME_TASK_RESULT, task_result)

    # Save platform-specific task configuration.
    if evaluation_result.backend_config:
        _save_to_file(
            output_dir / OUTPUT_FILENAME_BACKEND_CONFIG,
            evaluation_result.backend_config,
        )

    # Save Oumi's task parameters/configuration.
    _save_to_file(output_dir / OUTPUT_FILENAME_TASK_PARAMS, task_params)

    # Save all relevant Oumi configurations.
    if config:
        if config.model:
            _save_to_file(output_dir / OUTPUT_FILENAME_MODEL_PARAMS, config.model)
        if config.generation:
            _save_to_file(
                output_dir / OUTPUT_FILENAME_GENERATION_PARAMS, config.generation
            )
        if config.inference_engine or config.inference_remote_params:
            inference_params = {
                "engine": config.inference_engine,
                "remote_params": config.inference_remote_params,
            }
            _save_to_file(
                output_dir / OUTPUT_FILENAME_INFERENCE_PARAMS, inference_params
            )

    # Save python environment (package versions).
    package_versions = get_python_package_versions()
    _save_to_file(output_dir / OUTPUT_FILENAME_PACKAGE_VERSIONS, package_versions)
