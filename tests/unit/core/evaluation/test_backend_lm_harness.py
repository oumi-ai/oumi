import copy
from unittest.mock import MagicMock, patch

import pytest
import torch
from lm_eval.api.task import ConfigurableTask

from oumi.core.configs import (
    EvaluationConfig,
    GenerationParams,
    InferenceEngineType,
    LMHarnessTaskParams,
    ModelParams,
    RemoteParams,
)
from oumi.core.evaluation.backends.lm_harness import _generate_lm_harness_model_args
from oumi.core.evaluation.backends.lm_harness import evaluate as evaluate_lm_harness


@pytest.fixture
def mock_patches_for_evaluate():
    with (
        patch(
            "oumi.core.evaluation.backends.lm_harness.lm_harness_log_utils"
        ) as mock_lm_harness_log_utils,
        patch(
            "oumi.core.evaluation.backends.lm_harness.WandbLogger"
        ) as mock_WandbLogger,
        patch(
            "oumi.core.evaluation.backends.lm_harness.is_world_process_zero"
        ) as mock_is_world_process_zero,
        patch(
            "oumi.core.evaluation.backends.lm_harness.lm_harness_evaluate"
        ) as mock_lm_harness_evaluate,
        patch(
            "oumi.core.evaluation.backends.lm_harness.lm_harness_get_model_class"
        ) as mock_lm_harness_get_model_class,
        patch(
            "oumi.core.evaluation.backends.lm_harness._generate_lm_harness_model_args"
        ) as mock_generate_lm_harness_model_args,
        patch(
            "oumi.core.evaluation.backends.lm_harness._get_task_dict"
        ) as mock_get_task_dict,
        patch(
            "oumi.core.evaluation.backends.lm_harness.is_image_text_llm_using_model_name"
        ) as mock_is_image_text_llm,
        patch(
            "oumi.core.evaluation.backends.lm_harness._set_random_seeds"
        ) as mock_set_random_seeds,
        patch("torch.cuda.is_available") as mock_cuda_is_available,
    ):
        yield {
            "mock_cuda_is_available": mock_cuda_is_available,
            "mock_set_random_seeds": mock_set_random_seeds,
            "mock_is_image_text_llm": mock_is_image_text_llm,
            "mock_get_task_dict": mock_get_task_dict,
            "mock_generate_lm_harness_model_args": mock_generate_lm_harness_model_args,
            "mock_lm_harness_get_model_class": mock_lm_harness_get_model_class,
            "mock_lm_harness_evaluate": mock_lm_harness_evaluate,
            "mock_is_world_process_zero": mock_is_world_process_zero,
            "mock_WandbLogger": mock_WandbLogger,
            "mock_lm_harness_log_utils": mock_lm_harness_log_utils,
        }


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
                "batch_size": 1,
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
@patch("oumi.core.evaluation.backends.lm_harness.build_tokenizer")
@patch("oumi.core.evaluation.backends.lm_harness.build_processor")
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

    if is_multimodal and inference_engine_type == InferenceEngineType.NATIVE:
        mock_build_tokenizer.assert_called_once_with(model_params)
        mock_build_processor.assert_called_once_with(
            model_params.model_name,
            mock_build_tokenizer.return_value,
            trust_remote_code=model_params.trust_remote_code,
            processor_kwargs={},
        )
    else:
        mock_build_tokenizer.assert_not_called()
        mock_build_processor.assert_not_called()

    assert model_args == expected_model_args


def test_evaluate(mock_patches_for_evaluate):
    # Access the relevant mocks through the fixture.
    mock_cuda_is_available = mock_patches_for_evaluate["mock_cuda_is_available"]
    mock_set_random_seeds = mock_patches_for_evaluate["mock_set_random_seeds"]
    mock_is_image_text_llm = mock_patches_for_evaluate["mock_is_image_text_llm"]
    mock_get_task_dict = mock_patches_for_evaluate["mock_get_task_dict"]
    mock_generate_lm_harness_model_args = mock_patches_for_evaluate[
        "mock_generate_lm_harness_model_args"
    ]
    mock_lm_harness_get_model_class = mock_patches_for_evaluate[
        "mock_lm_harness_get_model_class"
    ]
    mock_lm_harness_evaluate = mock_patches_for_evaluate["mock_lm_harness_evaluate"]
    mock_is_world_process_zero = mock_patches_for_evaluate["mock_is_world_process_zero"]

    # Set the inputs of evaluate() function.
    task_params = LMHarnessTaskParams(
        evaluation_backend="lm_harness",
        task_name="mmlu",
        num_samples=222,
    )
    evaluation_config = EvaluationConfig(
        tasks=[task_params],
        model=ModelParams(model_name="gpt2"),
        generation=GenerationParams(),
        inference_engine=InferenceEngineType.NATIVE,
        inference_remote_params=None,
        run_name="run_name",
        enable_wandb=False,
        output_dir="test_output",
    )
    evaluation_config_without_tasks = copy.deepcopy(evaluation_config)
    evaluation_config_without_tasks.tasks = []
    random_seed = 123
    numpy_random_seed = 1234
    torch_random_seed = 12345

    # Mock the outputs of functions that evaluate() calls.
    mock_task_dict = {"mmlu": MagicMock(spec=ConfigurableTask)}
    mock_lm_harness_model_args = {"pretrained": "gpt2"}
    mock_results = {
        "results": {"mmlu": {"acc": 0.77}},
        "configs": {"my_config": "some_config"},
    }

    # Mock functions that evaluate() calls.
    mock_cuda_is_available.return_value = True
    mock_is_image_text_llm.return_value = False
    mock_get_task_dict.return_value = mock_task_dict
    mock_generate_lm_harness_model_args.return_value = mock_lm_harness_model_args
    mock_lm_harness_get_model_class.return_value = MagicMock()
    mock_lm_harness_evaluate.return_value = copy.deepcopy(mock_results)
    mock_is_world_process_zero.return_value = True

    _ = evaluate_lm_harness(
        task_params=task_params,
        config=evaluation_config,
        random_seed=random_seed,
        numpy_random_seed=numpy_random_seed,
        torch_random_seed=torch_random_seed,
    )

    # Assertions
    mock_set_random_seeds.assert_called_once_with(
        random_seed=random_seed,
        numpy_random_seed=numpy_random_seed,
        torch_random_seed=torch_random_seed,
    )
    mock_is_image_text_llm.assert_called_once_with(
        model_name=evaluation_config.model.model_name,
        trust_remote_code=evaluation_config.model.trust_remote_code,
    )
    mock_get_task_dict.assert_called_once_with(task_params)
    mock_generate_lm_harness_model_args.assert_called_once_with(
        lm_harness_model="hf",
        is_multimodal=False,
        device="cuda:0",
        model_params=evaluation_config.model,
        generation_params=evaluation_config.generation,
        inference_engine_type=evaluation_config.inference_engine,
        inference_remote_params=evaluation_config.inference_remote_params,
    )
    mock_lm_harness_get_model_class.assert_called_once_with("hf")

    mock_lm_harness_evaluate.assert_called_once()
    _, kwargs = mock_lm_harness_evaluate.call_args
    assert kwargs["task_dict"] == mock_task_dict
    assert kwargs["limit"] == 222
    assert not kwargs["apply_chat_template"]


def test_evaluate_failure_vLLM_without_CUDA(mock_patches_for_evaluate):
    # Access the relevant mocks through the fixture.
    mock_cuda_is_available = mock_patches_for_evaluate["mock_cuda_is_available"]
    mock_is_image_text_llm = mock_patches_for_evaluate["mock_is_image_text_llm"]
    mock_get_task_dict = mock_patches_for_evaluate["mock_get_task_dict"]
    mock_generate_lm_harness_model_args = mock_patches_for_evaluate[
        "mock_generate_lm_harness_model_args"
    ]
    mock_lm_harness_get_model_class = mock_patches_for_evaluate[
        "mock_lm_harness_get_model_class"
    ]
    mock_is_world_process_zero = mock_patches_for_evaluate["mock_is_world_process_zero"]

    # This combination should throw (we cannot use VLLM without CUDA).
    inference_engine_type = InferenceEngineType.VLLM
    mock_cuda_is_available.return_value = False

    # Mock functions that evaluate() calls.
    mock_is_image_text_llm.return_value = False
    mock_is_world_process_zero.return_value = True
    mock_get_task_dict.return_value = MagicMock()
    mock_generate_lm_harness_model_args.return_value = MagicMock()
    mock_lm_harness_get_model_class.return_value = MagicMock()

    with pytest.raises(
        ValueError, match="The `VLLM` inference_engine requires a CUDA-enabled GPU."
    ):
        evaluate_lm_harness(
            task_params=LMHarnessTaskParams(
                evaluation_backend="lm_harness",
                task_name="mmlu",
            ),
            config=EvaluationConfig(
                tasks=[],
                inference_engine=inference_engine_type,
            ),
        )


@pytest.mark.parametrize(
    "unsupported_inference_engine_type",
    [
        InferenceEngineType.REMOTE_VLLM,
        InferenceEngineType.SGLANG,
        InferenceEngineType.LLAMACPP,
        InferenceEngineType.ANTHROPIC,
        InferenceEngineType.GOOGLE_VERTEX,
        InferenceEngineType.GOOGLE_GEMINI,
        InferenceEngineType.DEEPSEEK,
        InferenceEngineType.PARASAIL,
        InferenceEngineType.TOGETHER,
        InferenceEngineType.OPENAI,
        InferenceEngineType.SAMBANOVA,
    ],
    ids=[
        "non_supported_engine_remote_vllm",
        "non_supported_engine_sglang",
        "non_supported_engine_llamacpp",
        "non_supported_engine_anthropic",
        "non_supported_engine_google_vertex",
        "non_supported_engine_google_gemini",
        "non_supported_engine_deepseek",
        "non_supported_engine_parasail",
        "non_supported_engine_together",
        "non_supported_engine_openai",
        "non_supported_engine_sambanova",
    ],
)
def test_evaluate_failure_non_supported_engine(
    mock_patches_for_evaluate, unsupported_inference_engine_type
):
    # Access the relevant mocks through the fixture.
    mock_cuda_is_available = mock_patches_for_evaluate["mock_cuda_is_available"]
    mock_is_image_text_llm = mock_patches_for_evaluate["mock_is_image_text_llm"]
    mock_get_task_dict = mock_patches_for_evaluate["mock_get_task_dict"]
    mock_generate_lm_harness_model_args = mock_patches_for_evaluate[
        "mock_generate_lm_harness_model_args"
    ]
    mock_lm_harness_get_model_class = mock_patches_for_evaluate[
        "mock_lm_harness_get_model_class"
    ]
    mock_is_world_process_zero = mock_patches_for_evaluate["mock_is_world_process_zero"]

    # Mock functions that evaluate() calls.
    mock_cuda_is_available.return_value = True
    mock_is_image_text_llm.return_value = False
    mock_is_world_process_zero.return_value = True
    mock_get_task_dict.return_value = MagicMock()
    mock_generate_lm_harness_model_args.return_value = MagicMock()
    mock_lm_harness_get_model_class.return_value = MagicMock()

    with pytest.raises(
        ValueError,
        match=f"Unsupported inference engine type: {unsupported_inference_engine_type}."
        " Our integration with the `lm_harness` evaluation backend supports "
        "the `NATIVE`, `VLLM` and `REMOTE` inference_engine types.",
    ):
        evaluate_lm_harness(
            task_params=LMHarnessTaskParams(
                evaluation_backend="lm_harness",
                task_name="mmlu",
            ),
            config=EvaluationConfig(
                tasks=[],
                inference_engine=unsupported_inference_engine_type,
            ),
        )
