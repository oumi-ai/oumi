from unittest.mock import MagicMock, patch

import pytest
import torch
from lm_eval.api.group import ConfigurableGroup
from lm_eval.api.task import ConfigurableTask

from oumi.core.configs import (
    GenerationParams,
    InferenceEngineType,
    LMHarnessTaskParams,
    ModelParams,
    RemoteParams,
)
from oumi.core.configs.params.evaluation_params import EvaluationPlatform
from oumi.evaluation.lm_harness import (
    _generate_lm_harness_model_args,
    _get_task_dict,
    evaluate,
)


@pytest.mark.parametrize(
    "lm_harness_model, is_multimodal, device, model_params, generation_params, "
    "inference_engine_type, inference_remote_params, expected_model_args",
    [
        (
            "hf",
            False,
            "mps",
            ModelParams(model_name="text_model"),
            GenerationParams(batch_size=None),
            InferenceEngineType.NATIVE,
            None,
            {
                "trust_remote_code": False,
                "pretrained": "text_model",
                "dtype": torch.float32,
                "max_length": None,
                "batch_size": "auto",
                "max_batch_size": None,
                "device": "mps",
                "parallelize": False,
                "device_map": "auto",
            },
        ),
        (
            "hf-multimodal",
            True,
            "cuda:0",
            ModelParams(model_name="vision_model", model_max_length=128),
            GenerationParams(),
            InferenceEngineType.NATIVE,
            None,
            {
                "trust_remote_code": False,
                "pretrained": "vision_model",
                "dtype": torch.float32,
                "max_length": 128,
                "batch_size": 1,
                "max_batch_size": None,
                "device": "cuda:0",
                "parallelize": False,
                "device_map": "auto",
                "max_images": 1,
                "interleave": True,
                "convert_img_format": True,
                "image_string": "my_image_token",
                "image_token_id": 1111,
            },
        ),
        (
            "vllm",
            False,
            "cuda:0",
            ModelParams(model_name="text_model", model_max_length=128),
            GenerationParams(batch_size=1),
            InferenceEngineType.VLLM,
            None,
            {
                "trust_remote_code": False,
                "pretrained": "text_model",
                "dtype": torch.float32,
                "max_length": 128,
                "batch_size": 1,
                "max_batch_size": None,
                "device": "cuda:0",
            },
        ),
        (
            "vllm-vlm",
            True,
            "cuda:0",
            ModelParams(
                model_name="vision_model", model_max_length=128, trust_remote_code=True
            ),
            GenerationParams(batch_size=8),
            InferenceEngineType.VLLM,
            None,
            {
                "trust_remote_code": True,
                "pretrained": "vision_model",
                "dtype": torch.float32,
                "max_length": 128,
                "batch_size": 8,
                "max_batch_size": None,
                "device": "cuda:0",
                "max_images": 1,
                "interleave": True,
            },
        ),
        (
            "local-completions",
            False,
            "cpu",
            ModelParams(model_name="some_model"),
            GenerationParams(),
            InferenceEngineType.REMOTE,
            RemoteParams(
                api_url="http://localhost:6864/v1/completions",
                num_workers=16,
                max_retries=3,
                connection_timeout=120,
            ),
            {
                "trust_remote_code": False,
                "pretrained": "some_model",
                "dtype": torch.float32,
                "max_length": None,
                "batch_size": 1,
                "max_batch_size": None,
                "device": "cpu",
                "base_url": "http://localhost:6864/v1/completions",
                "num_concurrent": 16,
                "max_retries": 3,
                "timeout": 120,
            },
        ),
    ],
    ids=[
        "model_args_hf_native",
        "model_args_hf-multimodal_native",
        "model_args_vllm",
        "model_args_vllm-vlm",
        "model_args_local-completions",
    ],
)
@patch("oumi.evaluation.lm_harness.build_tokenizer")
@patch("oumi.evaluation.lm_harness.build_processor")
def test_generate_lm_harness_model_args(
    mock_build_processor,
    mock_build_tokenizer,
    lm_harness_model,
    is_multimodal,
    device,
    model_params,
    generation_params,
    inference_engine_type,
    inference_remote_params,
    expected_model_args,
):
    mock_build_tokenizer.return_value = MagicMock()
    mock_build_processor.return_value = MagicMock(
        image_token="my_image_token", image_token_id=1111
    )

    model_args = _generate_lm_harness_model_args(
        lm_harness_model,
        is_multimodal,
        device,
        model_params,
        generation_params,
        inference_engine_type,
        inference_remote_params,
    )

    assert model_args == expected_model_args


def test_get_task_dict_for_configurable_task():
    task_params = LMHarnessTaskParams(
        evaluation_platform="lm_harness",
        task_name="mmlu_college_computer_science",
        num_fewshot=33,
    )

    task_dict = _get_task_dict(task_params)

    assert len(task_dict) == 1
    assert "mmlu_college_computer_science" in task_dict
    task: ConfigurableTask = task_dict["mmlu_college_computer_science"]  # type: ignore

    assert task.config.task == "mmlu_college_computer_science"
    assert task.config.num_fewshot == 33
    assert len(task.eval_docs) == 100
    assert task.OUTPUT_TYPE == "multiple_choice"


@pytest.mark.skip(reason="Temporarily disabled because it times out.")
def test_get_task_dict_for_configurable_group():
    task_params = LMHarnessTaskParams(
        evaluation_platform="lm_harness", task_name="mmmu_val", num_fewshot=222
    )

    task_dict = _get_task_dict(task_params)

    # Top Level: A single ConfigurableGroup with 6 subgroups
    assert len(task_dict) == 1
    conf_group_key = next(iter(task_dict))
    assert isinstance(conf_group_key, ConfigurableGroup)
    assert conf_group_key.group == "mmmu_val"
    conf_group_dict = task_dict[conf_group_key]
    assert isinstance(conf_group_dict, dict)
    assert len(conf_group_dict) == 6

    # Subgroup level: ConfigurableGroups consisting of multiple tasks
    for subgroup_key, subgroup_dict in conf_group_dict.items():
        assert isinstance(subgroup_key, ConfigurableGroup)
        assert isinstance(subgroup_dict, dict)

        # Task level: ensure `num_fewshot` has propagated to all tasks.
        for task_key, task in subgroup_dict.items():
            assert isinstance(task_key, str)
            assert task_key.startswith("mmmu_val")
            assert isinstance(task, ConfigurableTask)
            assert task.config.num_fewshot == 222


@patch("oumi.evaluation.lm_harness.save_evaluation_output")
@patch("oumi.evaluation.lm_harness.is_world_process_zero")
@patch("oumi.evaluation.lm_harness.lm_harness_evaluate")
@patch("oumi.evaluation.lm_harness.lm_harness_get_model_class")
@patch("oumi.evaluation.lm_harness._generate_lm_harness_model_args")
@patch("oumi.evaluation.lm_harness._get_task_dict")
@patch("oumi.evaluation.lm_harness.is_image_text_llm_using_model_name")
@patch("oumi.evaluation.lm_harness._set_random_seeds")
@patch("torch.cuda.is_available")
def test_evaluate(
    mock_cuda_is_available,
    mock_set_random_seeds,
    mock_is_image_text_llm_using_model_name,
    mock_get_task_dict,
    mock_generate_lm_harness_model_args,
    mock_lm_harness_get_model_class,
    mock_lm_harness_evaluate,
    mock_is_world_process_zero,
    mock_save_evaluation_output,
):
    # Set the inputs of evaluate() function.
    task_params = LMHarnessTaskParams(
        evaluation_platform="lm_harness", task_name="mmlu", num_samples=222
    )
    output_dir = "test_output"
    model_params = ModelParams(model_name="gpt2")
    generation_params = GenerationParams()
    enable_wandb = False
    inference_engine_type = InferenceEngineType.NATIVE
    inference_remote_params = None
    run_name = "run_name"
    random_seed = 123
    numpy_random_seed = 1234
    torch_random_seed = 12345

    # Mock the outputs of functions that evaluate() calls.
    mock_task_dict = {"mmlu": MagicMock(spec=ConfigurableTask)}
    mock_lm_harness_model_args = {"pretrained": "gpt2"}
    mock_results = {"results": {"mmlu": {"acc": 0.77}}, "configs": {}}

    # Mock functions that evaluate() calls.
    mock_cuda_is_available.return_value = True
    mock_is_image_text_llm_using_model_name.return_value = False
    mock_get_task_dict.return_value = mock_task_dict
    mock_generate_lm_harness_model_args.return_value = mock_lm_harness_model_args
    mock_lm_harness_get_model_class.return_value = MagicMock()
    mock_lm_harness_evaluate.return_value = mock_results
    mock_is_world_process_zero.return_value = True

    evaluate(
        task_params,
        output_dir,
        model_params,
        generation_params,
        enable_wandb,
        inference_engine_type,
        inference_remote_params,
        run_name,
        random_seed,
        numpy_random_seed,
        torch_random_seed,
    )

    # Assertions
    mock_set_random_seeds.assert_called_once_with(
        random_seed=random_seed,
        numpy_random_seed=numpy_random_seed,
        torch_random_seed=torch_random_seed,
    )
    mock_is_image_text_llm_using_model_name.assert_called_once_with(
        model_name=model_params.model_name,
        trust_remote_code=model_params.trust_remote_code,
    )
    mock_get_task_dict.assert_called_once_with(task_params)
    mock_generate_lm_harness_model_args.assert_called_once_with(
        lm_harness_model="hf",
        is_multimodal=False,
        device="cuda:0",
        model_params=model_params,
        generation_params=generation_params,
        inference_engine_type=inference_engine_type,
        inference_remote_params=inference_remote_params,
    )
    mock_lm_harness_get_model_class.assert_called_once_with("hf")

    mock_lm_harness_evaluate.assert_called_once()
    _, kwargs = mock_lm_harness_evaluate.call_args
    assert kwargs["task_dict"] == mock_task_dict
    assert kwargs["limit"] == 222
    assert not kwargs["apply_chat_template"]

    mock_save_evaluation_output.assert_called_once()
    _, kwargs = mock_save_evaluation_output.call_args
    assert kwargs["base_output_dir"] == "test_output"
    assert kwargs["platform"] == EvaluationPlatform.LM_HARNESS
    assert "results" in kwargs["platform_results"]
    assert kwargs["platform_results"]["results"] == {"mmlu": {"acc": 0.77}}
    assert "config" in kwargs["platform_task_config"]
    assert "configs" in kwargs["platform_task_config"]
    assert kwargs["platform_task_config"]["config"]["model"] == "hf"
    assert kwargs["task_params"] == task_params
    assert kwargs["model_params"] == model_params
    assert kwargs["generation_params"] == generation_params
    assert kwargs["inference_config"] is None
