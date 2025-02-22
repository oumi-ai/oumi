import json
import os
import tempfile
from typing import Any

import pytest

from oumi import evaluate
from oumi.core.configs import (
    EvaluationConfig,
    EvaluationTaskParams,
    GenerationParams,
    ModelParams,
)
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.evaluation.save_utils import OUTPUT_FILENAME_PLATFORM_RESULTS
from tests.markers import requires_gpus


def _get_evaluation_config(input_config: dict) -> EvaluationConfig:
    evaluation_task_params = EvaluationTaskParams(
        evaluation_platform="lm_harness",
        task_name=input_config["task_name"],
        num_samples=input_config["num_samples"],
    )
    model_params = ModelParams(
        model_name=input_config["model_name"],
        model_max_length=input_config["model_max_length"],
        trust_remote_code=True,
    )
    return EvaluationConfig(
        output_dir=input_config["output_dir"],
        tasks=[evaluation_task_params],
        model=model_params,
        generation=GenerationParams(),
        run_name=input_config["run_name"],
        inference_engine=input_config["inference_engine"],
    )


def _validate_results_returned(
    expected_results: dict[str, Any],
    actual_results: dict[str, Any],
    task_name: str,
) -> None:
    # Results nested under the task name.
    actual_results = actual_results[task_name]

    # Validate the results.
    for expected_key in expected_results:
        if expected_key not in actual_results:
            raise ValueError(
                f"Key `{expected_key}` was not found in the results: `{actual_results}`"
            )
        expected_value = expected_results[expected_key]["value"]
        round_digits = expected_results[expected_key]["round_digits"]
        actual_value = actual_results[expected_key]
        if round(actual_value, round_digits) != expected_value:
            raise ValueError(
                f"Expected value for key `{expected_key}` should be `{expected_value}` "
                f"(rounded to `{round_digits}` digits), but instead the actual value "
                f"that was returned is `{actual_value}`."
            )


def _validate_results_in_file(
    expected_results: dict[str, Any],
    output_dir: str,
    task_name: str,
) -> None:
    # Identify the relevant `output_path` for the evaluation test:
    # <output_dir> / <platform>_<timestamp> / platform_results.json
    subfolders = [f for f in os.listdir(output_dir) if f.startswith("lm_harness_")]
    assert len(subfolders) == 1
    output_path = os.path.join(
        output_dir, subfolders[0], OUTPUT_FILENAME_PLATFORM_RESULTS
    )
    assert os.path.exists(output_path)

    # Read the results from the evaluation test's output file.
    with open(output_path, encoding="utf-8") as file_ptr:
        results_dict = json.load(file_ptr)["results"]

    _validate_results_returned(
        expected_results=expected_results,
        actual_results=results_dict,
        task_name=task_name,
    )


@pytest.mark.parametrize(
    "input_config, expected_results",
    [
        (
            {
                "task_name": "mmlu_abstract_algebra",
                "num_samples": 10,
                "model_name": "openai-community/gpt2",
                "model_max_length": 128,
                "run_name": "test_lm_harness",
                "inference_engine": InferenceEngineType.NATIVE,
            },
            {
                "acc,none": {"value": 0.2, "round_digits": 3},
                "acc_stderr,none": {"value": 0.133, "round_digits": 3},
            },
        ),
        (
            {
                "task_name": "mmlu_abstract_algebra",
                "num_samples": 10,
                "model_name": "openai-community/gpt2",
                "model_max_length": 128,
                "run_name": "test_lm_harness",
                "inference_engine": InferenceEngineType.VLLM,
            },
            {
                "acc,none": {"value": 0.2, "round_digits": 3},
                "acc_stderr,none": {"value": 0.133, "round_digits": 3},
            },
        ),
    ],
    ids=[
        "lm_harness_test_native",
        "lm_harness_test_vllm",
    ],
)
@requires_gpus()
def test_evaluate_lm_harness(input_config, expected_results):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        nested_output_dir = os.path.join(output_temp_dir, "nested", "dir")
        input_config = {**input_config, "output_dir": nested_output_dir}
        evaluation_config = _get_evaluation_config(input_config)

        results_list = evaluate(evaluation_config)
        assert len(results_list) == 1  # 1 task was evaluated.
        results_dict = results_list[0]["results"]

        _validate_results_returned(
            expected_results=expected_results,
            actual_results=results_dict,
            task_name=input_config["task_name"],
        )
        _validate_results_in_file(
            expected_results=expected_results,
            output_dir=nested_output_dir,
            task_name=input_config["task_name"],
        )
