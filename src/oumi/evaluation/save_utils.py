from pathlib import Path
from typing import Any, Optional, Union

from oumi.core.configs import (
    AlpacaEvalTaskParams,
    GenerationParams,
    InferenceConfig,
    LMHarnessTaskParams,
    ModelParams,
)
from oumi.core.configs.params.evaluation_params import EvaluationPlatform
from oumi.utils.serialization_utils import json_serializer
from oumi.utils.version_utils import get_python_package_versions

OUTPUT_FILENAME_PLATFORM_RESULTS = "{platform}_{time}_platform_results.json"
OUTPUT_FILENAME_PLATFORM_TASK_CONFIG = "{platform}_{time}_platform_task_config.json"
OUTPUT_FILENAME_TASK_PARAMS = "{platform}_{time}_task_params.json"
OUTPUT_FILENAME_MODEL_PARAMS = "{platform}_{time}_model_params.json"
OUTPUT_FILENAME_GENERATION_PARAMS = "{platform}_{time}_generation_params.json"
OUTPUT_FILENAME_INFERENCE_CONFIG = "{platform}_{time}_inference_config.json"
OUTPUT_FILENAME_PACKAGE_VERSIONS = "{platform}_{time}_package_versions.json"


def save_evaluation_output(
    output_dir: str,
    platform: EvaluationPlatform,
    platform_results: dict[str, Any],
    platform_task_config: dict[str, Any],
    task_params: Union[LMHarnessTaskParams, AlpacaEvalTaskParams],
    start_time_str: str,
    elapsed_time_sec: float,
    model_params: ModelParams,
    generation_params: GenerationParams,
    inference_config: Optional[InferenceConfig] = None,
) -> None:
    """Writes configuration settings and evaluations outputs to files."""
    # Make sure the output folder exists.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Save all evaluation metrics, start date/time, and duration ---
    output_file_platform_results = OUTPUT_FILENAME_PLATFORM_RESULTS.format(
        platform=platform.value, time=start_time_str
    )
    platform_results["duration_sec"] = elapsed_time_sec
    platform_results["start_time"] = start_time_str
    with open(output_path / output_file_platform_results, "w") as file_out:
        file_out.write(json_serializer(platform_results))

    # --- Save platform-specific task configuration ---
    output_file_platform_task_config = OUTPUT_FILENAME_PLATFORM_TASK_CONFIG.format(
        platform=platform.value, time=start_time_str
    )
    with open(output_path / output_file_platform_task_config, "w") as file_out:
        file_out.write(json_serializer(platform_task_config))

    # --- Save Oumi's task parameters / configuration ---
    output_file_task_params = OUTPUT_FILENAME_TASK_PARAMS.format(
        platform=platform.value, time=start_time_str
    )
    with open(output_path / output_file_task_params, "w") as file_out:
        file_out.write(json_serializer(task_params))

    # --- Save all relevant Oumi configurations ---
    output_file_model_params = OUTPUT_FILENAME_MODEL_PARAMS.format(
        platform=platform.value, time=start_time_str
    )
    with open(output_path / output_file_model_params, "w") as file_out:
        file_out.write(json_serializer(model_params))

    output_file_generation_params = OUTPUT_FILENAME_GENERATION_PARAMS.format(
        platform=platform.value, time=start_time_str
    )
    with open(output_path / output_file_generation_params, "w") as file_out:
        file_out.write(json_serializer(generation_params))

    if inference_config:
        output_file_inference_config = OUTPUT_FILENAME_INFERENCE_CONFIG.format(
            platform=platform.value, time=start_time_str
        )
        with open(output_path / output_file_inference_config, "w") as file_out:
            file_out.write(json_serializer(inference_config))

    # --- Save python environment (package versions) ---
    output_file_pkg_versions = OUTPUT_FILENAME_PACKAGE_VERSIONS.format(
        platform=platform.value, time=start_time_str
    )
    package_versions = get_python_package_versions()
    with open(output_path / output_file_pkg_versions, "w") as file_out:
        file_out.write(json_serializer(package_versions))
