from pathlib import Path
from typing import Any, Optional

from oumi.core.configs import (
    EvaluationTaskParams,
    GenerationParams,
    InferenceConfig,
    ModelParams,
)
from oumi.core.configs.params.evaluation_params import EvaluationPlatform
from oumi.utils.serialization_utils import json_serializer
from oumi.utils.version_utils import get_python_package_versions

# Output filenames for saving evaluation results.
OUTPUT_FILENAME_PLATFORM_RESULTS = "platform_results.json"
OUTPUT_FILENAME_PLATFORM_TASK_CONFIG = "platform_task_config.json"
OUTPUT_FILENAME_TASK_PARAMS = "task_params.json"
OUTPUT_FILENAME_MODEL_PARAMS = "model_params.json"
OUTPUT_FILENAME_GENERATION_PARAMS = "generation_params.json"
OUTPUT_FILENAME_INFERENCE_CONFIG = "inference_config.json"
OUTPUT_FILENAME_PACKAGE_VERSIONS = "package_versions.json"


def _save_to_file(
    output_filename: str,
    output_dir: str,
    data: Any,
    platform: EvaluationPlatform,
    time: str,
) -> None:
    """Saves `data` under `<output_dir>/<platform>_<time>/<output_filename>`."""
    # Create the output directory.
    full_output_dir = Path(output_dir) / f"{platform.value}_{time}"
    full_output_dir.mkdir(parents=True, exist_ok=True)

    # Serialize and save `data` to `output_filename`.
    output_path = full_output_dir / output_filename
    with open(output_path, "w") as file_out:
        file_out.write(json_serializer(data))


def save_evaluation_output(
    output_dir: str,
    platform: EvaluationPlatform,
    platform_results: dict[str, Any],
    platform_task_config: dict[str, Any],
    task_params: EvaluationTaskParams,
    start_time_str: str,
    elapsed_time_sec: float,
    model_params: ModelParams,
    generation_params: GenerationParams,
    inference_config: Optional[InferenceConfig] = None,
) -> None:
    """Writes configuration settings and evaluations outputs to files.

    Args:
        output_dir: The directory where the evaluation results will be saved.
        platform: The evaluation platform used (e.g., "lm_harness", "alpaca_eval").
        platform_results: The evaluation results (metrics and their values) to save.
        platform_task_config: The platform-specific task configuration to save.
        task_params: The Oumi task parameters that were used for this evaluation.
        start_time_str: A string containing the start date/time of this evaluation.
        elapsed_time_sec: The duration of the evaluation (in seconds).
        model_params: The model parameters that were used in the evaluation.
        generation_params: The generation parameters that were used in the evaluation.
        inference_config: The inference configuration used in the evaluation
        (if inference is required for the corresponding evaluation platform).
    """
    # Save all evaluation metrics, start date/time, and duration.
    platform_results["duration_sec"] = elapsed_time_sec
    platform_results["start_time"] = start_time_str
    _save_to_file(
        output_filename=OUTPUT_FILENAME_PLATFORM_RESULTS,
        output_dir=output_dir,
        data=platform_results,
        platform=platform,
        time=start_time_str,
    )

    # Save platform-specific task configuration.
    _save_to_file(
        output_filename=OUTPUT_FILENAME_PLATFORM_TASK_CONFIG,
        output_dir=output_dir,
        data=platform_task_config,
        platform=platform,
        time=start_time_str,
    )

    # Save Oumi's task parameters/configuration.
    _save_to_file(
        output_filename=OUTPUT_FILENAME_TASK_PARAMS,
        output_dir=output_dir,
        data=task_params,
        platform=platform,
        time=start_time_str,
    )

    # Save all relevant Oumi configurations.
    _save_to_file(
        output_filename=OUTPUT_FILENAME_MODEL_PARAMS,
        output_dir=output_dir,
        data=model_params,
        platform=platform,
        time=start_time_str,
    )
    _save_to_file(
        output_filename=OUTPUT_FILENAME_GENERATION_PARAMS,
        output_dir=output_dir,
        data=generation_params,
        platform=platform,
        time=start_time_str,
    )
    if inference_config:
        _save_to_file(
            output_filename=OUTPUT_FILENAME_INFERENCE_CONFIG,
            output_dir=output_dir,
            data=inference_config,
            platform=platform,
            time=start_time_str,
        )

    # Save python environment (package versions).
    package_versions = get_python_package_versions()
    _save_to_file(
        output_filename=OUTPUT_FILENAME_PACKAGE_VERSIONS,
        output_dir=output_dir,
        data=package_versions,
        platform=platform,
        time=start_time_str,
    )
