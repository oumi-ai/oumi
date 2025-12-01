import copy
import sys
from unittest.mock import MagicMock, patch

import pytest
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
    """Mock lm_eval dependencies that are imported inside evaluate().

    The lm_eval imports are deferred (local imports inside evaluate()) because
    lm_eval has heavy dependencies (vllm, etc.) that may not always be installed.
    We use sys.modules patching to intercept these imports during testing.
    """
    mock_lm_harness_log_utils = MagicMock()
    mock_evaluator = MagicMock()
    mock_registry = MagicMock()
    mock_loggers = MagicMock()

    modules_to_patch = {
        "lm_eval.loggers.utils": mock_lm_harness_log_utils,
        "lm_eval.loggers": mock_loggers,
        "lm_eval.evaluator": mock_evaluator,
        "lm_eval.api.registry": mock_registry,
    }

    with (
        patch.dict(sys.modules, modules_to_patch),
        patch(
            "oumi.core.evaluation.backends.lm_harness.is_world_process_zero"
        ) as mock_is_world_process_zero,
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
            "mock_lm_harness_get_model_class": mock_registry.get_model,
            "mock_lm_harness_evaluate": mock_evaluator.evaluate,
            "mock_is_world_process_zero": mock_is_world_process_zero,
            "mock_WandbLogger": mock_loggers.WandbLogger,
            "mock_lm_harness_log_utils": mock_lm_harness_log_utils,
        }


@pytest.mark.parametrize(
    "lm_harness_model, is_multimodal, model_params, generation_params, "
    "inference_engine_type, inference_remote_params, expected_model_args",
    [
        (
            "hf",
            False,
            ModelParams(model_name="text_model"),
            GenerationParams(batch_size=None),
            InferenceEngineType.NATIVE,
            None,
            {
                "trust_remote_code": False,
                "pretrained": "text_model",
                "dtype": "auto",
                "max_length": None,
                "batch_size": "auto",
                "device": "cuda:0",
                "parallelize": False,
                "device_map": None,
            },
        ),
        (
            "hf-multimodal",
            True,
            ModelParams(model_name="vision_model", model_max_length=128),
            GenerationParams(),
            InferenceEngineType.NATIVE,
            None,
            {
                "trust_remote_code": False,
                "pretrained": "vision_model",
                "dtype": "auto",
                "max_length": 128,
                "batch_size": 1,
                "device": "cuda:0",
                "parallelize": False,
                "device_map": None,
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
            ModelParams(model_name="text_model", model_max_length=128),
            GenerationParams(batch_size=1),
            InferenceEngineType.VLLM,
            None,
            {
                "trust_remote_code": False,
                "pretrained": "text_model",
                "dtype": "auto",
                "max_length": 128,
                "batch_size": "auto",
                "tensor_parallel_size": 1,
            },
        ),
        (
            "vllm-vlm",
            True,
            ModelParams(
                model_name="vision_model", model_max_length=128, trust_remote_code=True
            ),
            GenerationParams(batch_size=8),
            InferenceEngineType.VLLM,
            None,
            {
                "trust_remote_code": True,
                "pretrained": "vision_model",
                "dtype": "auto",
                "max_length": 128,
                "batch_size": 1,
                "max_images": 1,
                "interleave": True,
                "tensor_parallel_size": 1,
            },
        ),
        (
            "local-completions",
            False,
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
                "dtype": "auto",
                "max_length": None,
                "batch_size": 1,
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
@patch("oumi.core.evaluation.backends.lm_harness.torch.cuda.device_count")
@patch("oumi.core.evaluation.backends.lm_harness.torch.cuda.is_available")
@patch("oumi.core.evaluation.backends.lm_harness.torch.backends.mps.is_available")
@patch("oumi.core.evaluation.backends.lm_harness.get_device_rank_info")
def test_generate_lm_harness_model_args(
    mock_get_device_rank_info,
    mock_mps_available,
    mock_cuda_available,
    mock_device_count,
    mock_build_processor,
    mock_build_tokenizer,
    lm_harness_model,
    is_multimodal,
    model_params,
    generation_params,
    inference_engine_type,
    inference_remote_params,
    expected_model_args,
):
    # Mock device info for CUDA
    mock_device_info = MagicMock()
    mock_device_info.local_rank = 0
    mock_device_info.world_size = 1
    mock_get_device_rank_info.return_value = mock_device_info

    mock_device_count.return_value = 1  # Mock single GPU for tests
    mock_cuda_available.return_value = True  # Mock CUDA availability
    mock_mps_available.return_value = False  # Mock MPS not available

    mock_build_tokenizer.return_value = MagicMock()
    mock_build_processor.return_value = MagicMock(
        image_token="my_image_token", image_token_id=1111
    )

    model_args = _generate_lm_harness_model_args(
        lm_harness_model,
        is_multimodal,
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
        model=ModelParams(model_name="openai-community/gpt2"),
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
    mock_lm_harness_model_args = {"pretrained": "openai-community/gpt2"}
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
        model_params=evaluation_config.model,
        generation_params=evaluation_config.generation,
        inference_engine_type=evaluation_config.inference_engine,
        inference_remote_params=evaluation_config.inference_remote_params,
    )
    mock_lm_harness_get_model_class.assert_called_once_with("hf")

    mock_lm_harness_evaluate.assert_called_once()
    args, kwargs = mock_lm_harness_evaluate.call_args
    assert args[1] == mock_task_dict  # task_dict is now the second positional argument
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
