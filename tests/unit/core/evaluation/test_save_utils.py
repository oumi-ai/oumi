import json
import os
import tempfile

import pytest

from oumi.core.configs import EvaluationConfig, EvaluationTaskParams
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.configs.params.generation_params import GenerationParams
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.configs.params.remote_params import RemoteParams
from oumi.core.evaluation.evaluation_result import EvaluationResult
from oumi.core.evaluation.utils.save_utils import (
    OUTPUT_FILENAME_BACKEND_CONFIG,
    OUTPUT_FILENAME_GENERATION_PARAMS,
    OUTPUT_FILENAME_INFERENCE_PARAMS,
    OUTPUT_FILENAME_MODEL_PARAMS,
    OUTPUT_FILENAME_PACKAGE_VERSIONS,
    OUTPUT_FILENAME_TASK_PARAMS,
    OUTPUT_FILENAME_TASK_RESULT,
    save_evaluation_output,
)


def test_save_evaluation_output_happy_path():
    # Set input parameters.
    task_params = EvaluationTaskParams(
        evaluation_backend="lm_harness",
        task_name="mmlu",
        num_samples=5,
        log_samples=True,
        eval_kwargs={"num_fewshot": 5},
    )
    result = EvaluationResult(
        task_name="NOT_USED",
        task_result={"my_metric": 10},
        backend_config={"config_param": 11},
        start_time="20250303_094257",
        elapsed_time_sec=15,
    )
    config = EvaluationConfig(
        tasks=[],
        model=ModelParams(model_name="my_model"),
        generation=GenerationParams(max_new_tokens=8),
        inference_engine=InferenceEngineType.VLLM,
        inference_remote_params=RemoteParams(api_url="my_api_url"),
    )

    with tempfile.TemporaryDirectory() as output_temp_dir:
        my_base_output_dir = os.path.join(output_temp_dir, "my_base_output_dir")

        # Run the function to be tested.
        save_evaluation_output(
            backend_name="mock_backend_name",
            task_params=task_params,
            evaluation_result=result,
            base_output_dir=my_base_output_dir,
            config=config,
        )

        # Ensure the output path was created.
        output_path = os.path.join(
            my_base_output_dir, f"mock_backend_name_{result.start_time}"
        )
        assert os.path.exists(output_path)

        # Validate the task result from the test's output file.
        task_result_path = os.path.join(output_path, OUTPUT_FILENAME_TASK_RESULT)
        with open(task_result_path, encoding="utf-8") as file_ptr:
            task_result_dict = json.load(file_ptr)
        assert task_result_dict == {
            "my_metric": 10,
            "start_time": "20250303_094257",
            "duration_sec": 15,
        }

        # Validate the backend config from the test's output file.
        backend_config_path = os.path.join(output_path, OUTPUT_FILENAME_BACKEND_CONFIG)
        with open(backend_config_path, encoding="utf-8") as file_ptr:
            backend_config_dict = json.load(file_ptr)
        assert backend_config_dict == {"config_param": 11}

        # Validate the task params from the test's output file.
        task_params_path = os.path.join(output_path, OUTPUT_FILENAME_TASK_PARAMS)
        with open(task_params_path, encoding="utf-8") as file_ptr:
            task_params_dict = json.load(file_ptr)
        if "evaluation_platform" in task_params_dict:
            task_params_dict.pop("evaluation_platform")  # deprecated field.
        assert task_params_dict == {
            "evaluation_backend": "lm_harness",
            "task_name": "mmlu",
            "num_samples": 5,
            "log_samples": True,
            "eval_kwargs": {"num_fewshot": 5},
        }

        # Validate the model params from the test's output file.
        model_params_path = os.path.join(output_path, OUTPUT_FILENAME_MODEL_PARAMS)
        with open(model_params_path, encoding="utf-8") as file_ptr:
            model_params_dict = json.load(file_ptr)
        assert "model_name" in model_params_dict
        assert model_params_dict["model_name"] == "my_model"

        # Validate the generation params from the test's output file.
        generation_params_path = os.path.join(
            output_path, OUTPUT_FILENAME_GENERATION_PARAMS
        )
        with open(generation_params_path, encoding="utf-8") as file_ptr:
            generation_params_dict = json.load(file_ptr)
        assert "max_new_tokens" in generation_params_dict
        assert generation_params_dict["max_new_tokens"] == 8

        # Validate the inference params from the test's output file.
        inference_params_path = os.path.join(
            output_path, OUTPUT_FILENAME_INFERENCE_PARAMS
        )
        with open(inference_params_path, encoding="utf-8") as file_ptr:
            inference_params_dict = json.load(file_ptr)
        assert "engine" in inference_params_dict
        assert inference_params_dict["engine"] == "VLLM"
        assert "remote_params" in inference_params_dict
        assert inference_params_dict["remote_params"]
        assert "api_url='my_api_url'" in inference_params_dict["remote_params"]

        # Validate the package versions from the test's output file.
        package_versions_path = os.path.join(
            output_path, OUTPUT_FILENAME_PACKAGE_VERSIONS
        )
        with open(package_versions_path, encoding="utf-8") as file_ptr:
            package_versions_dict = json.load(file_ptr)
        assert "oumi" in package_versions_dict


def test_save_evaluation_output_empty_results():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        my_base_output_dir = os.path.join(output_temp_dir, "my_base_output_dir")

        # Run the function to be tested.
        save_evaluation_output(
            backend_name="mock_backend_name",
            task_params=EvaluationTaskParams(evaluation_backend="lm_harness"),
            evaluation_result=EvaluationResult(),
            base_output_dir=my_base_output_dir,
            config=EvaluationConfig(),
        )

        # Ensure the output directory was created.
        output_path = os.path.join(my_base_output_dir, "mock_backend_name")
        assert os.path.exists(output_path)
        output_files = list(os.listdir(output_path))

    # Ensure all parames have been saved.
    expected_files = [
        OUTPUT_FILENAME_TASK_PARAMS,
        OUTPUT_FILENAME_MODEL_PARAMS,
        OUTPUT_FILENAME_GENERATION_PARAMS,
        OUTPUT_FILENAME_INFERENCE_PARAMS,
        OUTPUT_FILENAME_PACKAGE_VERSIONS,
    ]
    for file in expected_files:
        assert file in output_files

    # Ensure that the task result and backend config files were NOT created.
    assert OUTPUT_FILENAME_TASK_RESULT not in output_files
    assert OUTPUT_FILENAME_BACKEND_CONFIG not in output_files


def test_save_evaluation_output_missing_backend():
    with pytest.raises(
        ValueError,
        match="The evaluation backend name must be provided.",
    ):
        save_evaluation_output(
            backend_name="",
            task_params=EvaluationTaskParams(evaluation_backend="lm_harness"),
            evaluation_result=EvaluationResult(),
            base_output_dir="",
            config=EvaluationConfig(),
        )


@pytest.mark.parametrize(
    "backend_dir, preexisting_dirs, expected_dir",
    [
        ("backend_dir", ["backend_dir"], "backend_dir_1"),
        ("backend_dir", ["backend_dir", "backend_dir_1"], "backend_dir_2"),
        (
            "backend_dir",
            ["backend_dir", "backend_dir_1", "backend_dir_2"],
            "backend_dir_3",
        ),
    ],
    ids=[
        "test_save_evaluation_output_with_a_preexisting_dir",
        "test_save_evaluation_output_with_2_preexisting_dirs",
        "test_save_evaluation_output_with_3_preexisting_dirs",
    ],
)
def test_save_evaluation_output_with_preexisting_path(
    backend_dir: str,
    preexisting_dirs: list[str],
    expected_dir: str,
) -> None:
    with tempfile.TemporaryDirectory() as output_temp_dir:
        # Create directories that the test requires to pre-exist.
        for preexisting_dir in preexisting_dirs:
            preexisting_path = os.path.join(output_temp_dir, preexisting_dir)
            os.makedirs(preexisting_path)

        # Run the save function to create a new dictionary.
        save_evaluation_output(
            backend_name=backend_dir,
            task_params=EvaluationTaskParams(evaluation_backend="lm_harness"),
            evaluation_result=EvaluationResult(),
            base_output_dir=output_temp_dir,
            config=EvaluationConfig(),
        )

        # Find the generated dictionary name and ensure it's correct.
        paths = list(os.listdir(output_temp_dir))
        assert len(paths) == len(preexisting_dirs) + 1
        new_dir = list(set(paths) - set(preexisting_dirs))
        assert len(new_dir) == 1
        new_dir = new_dir[0]
        assert new_dir == expected_dir
