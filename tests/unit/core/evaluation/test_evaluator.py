from dataclasses import asdict
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from oumi.core.configs import (
    AlpacaEvalTaskParams,
    EvaluationConfig,
    EvaluationTaskParams,
    GenerationParams,
    InferenceEngineType,
    LMHarnessTaskParams,
    ModelParams,
    RemoteParams,
)
from oumi.core.configs.params.evaluation_params import EvaluationBackend
from oumi.core.evaluation.evaluation_result import EvaluationResult
from oumi.core.evaluation.evaluator import Evaluator
from oumi.core.inference import BaseInferenceEngine
from oumi.inference import OpenAIInferenceEngine


@patch("oumi.core.evaluation.evaluator.evaluate_lm_harness")
@patch("oumi.core.evaluation.evaluator.check_prerequisites")
@patch("oumi.core.evaluation.evaluator.save_evaluation_output")
@patch("oumi.core.evaluation.evaluator.build_inference_engine")
def test_evaluate_lm_harness_task(
    mock_build_inference_engine,
    mock_save_evaluation_output,
    mock_check_prerequisites,
    mock_evaluate_lm_harness,
):
    # Inputs.
    task_params = EvaluationTaskParams(
        task_name="test_task",
        evaluation_backend=EvaluationBackend.LM_HARNESS.value,
    )
    evaluation_config = EvaluationConfig(
        tasks=[task_params],
        model=ModelParams(model_name="test_model"),
        generation=GenerationParams(),
        inference_engine=InferenceEngineType.NATIVE,
    )

    # Mocks.
    mock_build_inference_engine.return_value = MagicMock()
    mock_save_evaluation_output.return_value = None
    mock_check_prerequisites.return_value = None
    mock_evaluate_lm_harness.return_value = EvaluationResult(
        task_name="test_task", task_result={"test_metric": 1.0}
    )

    # Run the test.
    evaluator = Evaluator()
    result = evaluator.evaluate(evaluation_config)

    # Check the results.
    mock_build_inference_engine.assert_not_called()
    mock_save_evaluation_output.assert_called_once()
    mock_check_prerequisites.assert_called_once()
    mock_evaluate_lm_harness.assert_called_once()
    _, kwargs = mock_evaluate_lm_harness.call_args

    assert isinstance(kwargs["task_params"], LMHarnessTaskParams)
    assert kwargs["task_params"].task_name == "test_task"
    assert kwargs["task_params"].evaluation_backend == (
        EvaluationBackend.LM_HARNESS.value
    )

    assert isinstance(kwargs["config"], EvaluationConfig)
    assert kwargs["config"].tasks == []
    assert kwargs["config"].model.model_name == "test_model"
    assert kwargs["config"].inference_engine == InferenceEngineType.NATIVE

    assert len(result) == 1
    assert result[0].task_name == "test_task"
    assert result[0].task_result == {"test_metric": 1.0}


@patch("oumi.core.evaluation.evaluator.evaluate_alpaca_eval")
@patch("oumi.core.evaluation.evaluator.check_prerequisites")
@patch("oumi.core.evaluation.evaluator.save_evaluation_output")
@patch("oumi.core.evaluation.evaluator.build_inference_engine")
def test_evaluate_alpaca_eval_task(
    mock_build_inference_engine,
    mock_save_evaluation_output,
    mock_check_prerequisites,
    mock_evaluate_alpaca_eval,
):
    # Inputs.
    task_params = EvaluationTaskParams(
        task_name="test_task",
        evaluation_backend=EvaluationBackend.ALPACA_EVAL.value,
    )
    evaluation_config = EvaluationConfig(
        tasks=[task_params],
        model=ModelParams(model_name="test_model"),
        generation=GenerationParams(),
        inference_engine=InferenceEngineType.VLLM,
    )

    # Mocks.
    mock_build_inference_engine.return_value = MagicMock()
    mock_save_evaluation_output.return_value = None
    mock_check_prerequisites.return_value = None
    mock_evaluate_alpaca_eval.return_value = EvaluationResult(
        task_name="test_task", task_result={"test_metric": 1.0}
    )

    # Run the test.
    evaluator = Evaluator()
    result = evaluator.evaluate(evaluation_config)

    # Check the results.
    mock_build_inference_engine.assert_called_once()
    mock_save_evaluation_output.assert_called_once()
    mock_check_prerequisites.assert_called_once()
    mock_evaluate_alpaca_eval.assert_called_once()
    _, kwargs = mock_evaluate_alpaca_eval.call_args

    assert isinstance(kwargs["task_params"], AlpacaEvalTaskParams)
    assert kwargs["task_params"].task_name == "test_task"
    assert kwargs["task_params"].evaluation_backend == (
        EvaluationBackend.ALPACA_EVAL.value
    )

    assert isinstance(kwargs["config"], EvaluationConfig)
    assert kwargs["config"].tasks == []
    assert kwargs["config"].model.model_name == "test_model"
    assert kwargs["config"].inference_engine == InferenceEngineType.VLLM

    assert len(result) == 1
    assert result[0].task_name == "test_task"
    assert result[0].task_result == {"test_metric": 1.0}


@patch("oumi.core.evaluation.evaluator.REGISTRY.get_evaluation_function")
@patch("oumi.core.evaluation.evaluator.check_prerequisites")
@patch("oumi.core.evaluation.evaluator.save_evaluation_output")
@patch("oumi.core.evaluation.evaluator.build_inference_engine")
def test_evaluate_custom_task(
    mock_build_inference_engine,
    mock_save_evaluation_output,
    mock_check_prerequisites,
    mock_get_evaluation_function,
):
    # Inputs.
    task_params = EvaluationTaskParams(
        task_name="evaluation_fn_reg_name",
        evaluation_backend=EvaluationBackend.CUSTOM.value,
        eval_kwargs={"optional_param_2": "optional_param_2_value"},
    )
    evaluation_config = EvaluationConfig(
        tasks=[task_params],
        model=ModelParams(model_name="test_model"),
        generation=GenerationParams(),
        inference_engine=InferenceEngineType.NATIVE,
    )
    eval_result = {"test_metric": 1.0}

    def evaluation_fn(
        task_params: EvaluationTaskParams,
        config: EvaluationConfig,
        optional_param_1: str,
        optional_param_2: str,
    ) -> dict:
        assert task_params.evaluation_backend == EvaluationBackend.CUSTOM.value
        assert task_params.task_name == "evaluation_fn_reg_name"
        assert optional_param_1 == "optional_param_1_value"
        assert optional_param_2 == "optional_param_2_value"
        return eval_result

    # Mocks.
    mock_build_inference_engine.return_value = MagicMock()
    mock_save_evaluation_output.return_value = None
    mock_check_prerequisites.return_value = None
    mock_get_evaluation_function.return_value = evaluation_fn

    # Run the test.
    evaluator = Evaluator()
    result = evaluator.evaluate(
        evaluation_config, optional_param_1="optional_param_1_value"
    )

    # Check the results.
    mock_build_inference_engine.assert_not_called()
    mock_save_evaluation_output.assert_called_once()
    mock_check_prerequisites.assert_called_once()
    mock_get_evaluation_function.assert_called_once()
    assert len(result) == 1
    assert result[0].task_name == "evaluation_fn_reg_name"
    assert result[0].get_results() == eval_result
    assert result[0].task_result["results"]["evaluation_fn_reg_name"] == eval_result


@patch("oumi.core.evaluation.evaluator.REGISTRY.get_evaluation_function")
@patch("oumi.core.evaluation.evaluator.check_prerequisites")
@patch("oumi.core.evaluation.evaluator.save_evaluation_output")
@patch("oumi.core.evaluation.evaluator.build_inference_engine")
def test_evaluate_custom_task_with_no_evaluation_fn_args(
    mock_build_inference_engine,
    mock_save_evaluation_output,
    mock_check_prerequisites,
    mock_get_evaluation_function,
):
    eval_result = {"test_metric": 1.0}

    # Custom evaluation function with NO input arguments.
    def evaluation_fn() -> dict:
        return eval_result

    # Mocks.
    mock_build_inference_engine.return_value = MagicMock()
    mock_save_evaluation_output.return_value = None
    mock_check_prerequisites.return_value = None
    mock_get_evaluation_function.return_value = evaluation_fn

    # Run the test.
    evaluator = Evaluator()
    result = evaluator.evaluate(
        EvaluationConfig(
            tasks=[
                EvaluationTaskParams(
                    task_name="evaluation_fn_reg_name",
                    evaluation_backend=EvaluationBackend.CUSTOM.value,
                )
            ],
        )
    )

    # Check the results.
    mock_build_inference_engine.assert_not_called()
    mock_save_evaluation_output.assert_called_once()
    mock_check_prerequisites.assert_called_once()
    mock_get_evaluation_function.assert_called_once()
    assert len(result) == 1
    assert result[0].task_name == "evaluation_fn_reg_name"
    assert result[0].get_results() == eval_result
    assert result[0].task_result["results"]["evaluation_fn_reg_name"] == eval_result


@patch("oumi.core.evaluation.evaluator.REGISTRY.get_evaluation_function")
@patch("oumi.core.evaluation.evaluator.check_prerequisites")
@patch("oumi.core.evaluation.evaluator.save_evaluation_output")
def test_evaluate_custom_task_with_inference(
    mock_save_evaluation_output,
    mock_check_prerequisites,
    mock_get_evaluation_function,
):
    # Inputs.
    task_params = EvaluationTaskParams(
        task_name="evaluation_fn_reg_name",
        evaluation_backend=EvaluationBackend.CUSTOM.value,
    )
    evaluation_config = EvaluationConfig(
        tasks=[task_params],
        model=ModelParams(model_name="test_model"),
        generation=GenerationParams(max_new_tokens=8),
        inference_engine=InferenceEngineType.OPENAI,
        inference_remote_params=RemoteParams(api_url="my_api_url"),
    )
    eval_result = {"test_metric": 1.0}

    def evaluation_fn(
        task_params: EvaluationTaskParams,
        config: EvaluationConfig,
        inference_engine: BaseInferenceEngine,
        optional_param: str,
    ) -> dict:
        assert optional_param == "optional_param_value"

        # Validate the `task_params`.
        assert task_params.evaluation_backend == EvaluationBackend.CUSTOM.value
        assert task_params.task_name == "evaluation_fn_reg_name"

        # Validate the `config`.
        expected_config_dict = asdict(evaluation_config)
        expected_config_dict["tasks"] = []
        assert asdict(config) == expected_config_dict

        # Validate the `inference_engine`.
        assert isinstance(inference_engine, OpenAIInferenceEngine)
        open_ai_inference_engine: OpenAIInferenceEngine = inference_engine
        assert open_ai_inference_engine._model_params.model_name == "test_model"
        assert open_ai_inference_engine._generation_params.max_new_tokens == 8

        return eval_result

    # Mocks.
    mock_save_evaluation_output.return_value = None
    mock_check_prerequisites.return_value = None
    mock_get_evaluation_function.return_value = evaluation_fn

    # Run the test.
    evaluator = Evaluator()
    result = evaluator.evaluate(
        evaluation_config, optional_param="optional_param_value"
    )

    # Validate the function calls and results.
    mock_save_evaluation_output.assert_called_once()
    mock_check_prerequisites.assert_called_once()
    mock_get_evaluation_function.assert_called_once()

    assert len(result) == 1
    assert result[0].task_name == "evaluation_fn_reg_name"
    assert result[0].get_results() == eval_result
    assert result[0].task_result["results"]["evaluation_fn_reg_name"] == eval_result


@patch("oumi.core.evaluation.evaluator.REGISTRY.get_evaluation_function")
@patch("oumi.core.evaluation.evaluator.check_prerequisites")
@patch("oumi.core.evaluation.evaluator.save_evaluation_output")
@patch("oumi.core.evaluation.evaluator.build_inference_engine")
def test_evaluate_custom_task_wrong_return_type(
    mock_build_inference_engine,
    mock_save_evaluation_output,
    mock_check_prerequisites,
    mock_get_evaluation_function,
):
    # Inputs.
    task_params = EvaluationTaskParams(
        task_name="evaluation_fn_reg_name",
        evaluation_backend=EvaluationBackend.CUSTOM.value,
    )
    evaluation_config = EvaluationConfig(
        tasks=[task_params],
        model=ModelParams(model_name="test_model"),
        generation=GenerationParams(),
        inference_engine=InferenceEngineType.NATIVE,
    )

    def evaluation_fn():
        return 1.0  # Wrong return type (float, instead of dict).

    # Mocks.
    mock_build_inference_engine.return_value = MagicMock()
    mock_save_evaluation_output.return_value = None
    mock_check_prerequisites.return_value = None
    mock_get_evaluation_function.return_value = evaluation_fn

    # Run the test.
    evaluator = Evaluator()
    with pytest.raises(
        ValueError,
        match=(
            "The custom evaluation function `evaluation_fn_reg_name` must "
            "return either a `dict` or an `EvaluationResult` object, but it is "
            "currently returning an object of type `<class 'float'>`. "
            "Please ensure that the function returns the correct object."
        ),
    ):
        evaluator.evaluate(evaluation_config)

    # Check the results.
    mock_build_inference_engine.assert_not_called()
    mock_save_evaluation_output.assert_not_called()
    mock_check_prerequisites.assert_called_once()
    mock_get_evaluation_function.assert_called_once()


@patch("oumi.core.evaluation.evaluator.REGISTRY.get_evaluation_function")
@patch("oumi.core.evaluation.evaluator.check_prerequisites")
@patch("oumi.core.evaluation.evaluator.save_evaluation_output")
@patch("oumi.core.evaluation.evaluator.build_inference_engine")
def test_evaluate_custom_task_unregistered_fn(
    mock_build_inference_engine,
    mock_save_evaluation_output,
    mock_check_prerequisites,
    mock_get_evaluation_function,
):
    # Inputs.
    task_params = EvaluationTaskParams(
        task_name="evaluation_fn_unregistered",
        evaluation_backend=EvaluationBackend.CUSTOM.value,
    )
    evaluation_config = EvaluationConfig(tasks=[task_params])

    # Mocks.
    mock_build_inference_engine.return_value = MagicMock()
    mock_save_evaluation_output.return_value = None
    mock_check_prerequisites.return_value = None
    mock_get_evaluation_function.return_value = None

    # Run the test.
    evaluator = Evaluator()
    with pytest.raises(
        ValueError,
        match=(
            "Task name `evaluation_fn_unregistered` not found in the "
            "registry. For custom Oumi evaluations, the task name must match "
            "the name of a registered evaluation function. You can register "
            "a new function with the decorator `@register_evaluation_function`."
        ),
    ):
        evaluator.evaluate(evaluation_config)

    # Check the results.
    mock_build_inference_engine.assert_not_called()
    mock_save_evaluation_output.assert_not_called()
    mock_check_prerequisites.assert_called_once()
    mock_get_evaluation_function.assert_called_once()


@patch("oumi.core.evaluation.evaluator.REGISTRY.get_evaluation_function")
@patch("oumi.core.evaluation.evaluator.check_prerequisites")
@patch("oumi.core.evaluation.evaluator.save_evaluation_output")
@patch("oumi.core.evaluation.evaluator.build_inference_engine")
def test_evaluate_custom_task_without_task_name(
    mock_build_inference_engine,
    mock_save_evaluation_output,
    mock_check_prerequisites,
    mock_get_evaluation_function,
):
    # Inputs.
    task_params = EvaluationTaskParams(
        evaluation_backend=EvaluationBackend.CUSTOM.value
    )
    evaluation_config = EvaluationConfig(tasks=[task_params])

    # Mocks.
    mock_build_inference_engine.return_value = MagicMock()
    mock_save_evaluation_output.return_value = None
    mock_check_prerequisites.return_value = None
    mock_get_evaluation_function.return_value = None

    # Run the test.
    evaluator = Evaluator()
    with pytest.raises(
        ValueError,
        match=(
            "Missing `task_name` for custom Oumi evaluation. Please specify the "
            "task name, which should be corresponding to a registered evaluation "
            "function, using the decorator `@register_evaluation_function`."
        ),
    ):
        evaluator.evaluate(evaluation_config)

    # Check the results.
    mock_build_inference_engine.assert_not_called()
    mock_save_evaluation_output.assert_not_called()
    mock_check_prerequisites.assert_called_once()
    mock_get_evaluation_function.assert_not_called()


@pytest.mark.parametrize(
    ("kwargs,eval_kwargs,expected_error_message"),
    [
        (
            {"expected_param_1": 1, "inference_engine": "my_inference_engine"},
            {"expected_param_2": 2},
            "Reserved keys are present when calling `Evaluator.evaluate()`. You are "
            "not allowed to pass the following keyword arguments into the "
            "`evaluation_fn_reg_name` function: ['config', 'inference_engine', "
            "'task_params']. However, you have passed the following reserved keys: "
            "['inference_engine'].",
        ),
        (
            {"expected_param_1": 1},
            {"expected_param_2": 2, "inference_engine": "my_inference_engine"},
            "Reserved keys are present when calling `Evaluator.evaluate()`. You are "
            "not allowed to pass the following keyword arguments into the "
            "`evaluation_fn_reg_name` function: ['config', 'inference_engine', "
            "'task_params']. However, you have passed the following reserved keys: "
            "['inference_engine'].",
        ),
        (
            {
                "expected_param_1": 1,
                "unrecognized_keyword": "unrecognized_keyword_value",
            },
            {"expected_param_2": 2},
            "Unrecognized keyword arguments are present when calling "
            "`Evaluator.evaluate()`. You have passed the following unrecognized "
            "keys: ['unrecognized_keyword'].",
        ),
        (
            {"expected_param_1": 1},
            {
                "expected_param_2": 2,
                "unrecognized_keyword": "unrecognized_keyword_value",
            },
            "Unrecognized keyword arguments are present when calling "
            "`Evaluator.evaluate()`. You have passed the following unrecognized "
            "keys: ['unrecognized_keyword'].",
        ),
        (
            {},
            {"expected_param_2": 2},
            "Missing keyword arguments have been identified when calling "
            "`Evaluator.evaluate()`. You have not passed the following expected keys: "
            "{'expected_param_1'}.",
        ),
        (
            {"expected_param_1": 1},
            {},
            "Missing keyword arguments have been identified when calling "
            "`Evaluator.evaluate()`. You have not passed the following expected keys: "
            "{'expected_param_2'}.",
        ),
    ],
    ids=[
        "kwargs_including_reserved_keyword_inference_engine",
        "eval_kwargs_including_reserved_keyword_inference_engine",
        "kwargs_including_unrecognized_keyword",
        "eval_kwargs_including_unrecognized_keyword",
        "kwargs_missing_required_keyword",
        "eval_kwargs_missing_required_keyword",
    ],
)
@patch("oumi.core.evaluation.evaluator.REGISTRY.get_evaluation_function")
@patch("oumi.core.evaluation.evaluator.check_prerequisites")
@patch("oumi.core.evaluation.evaluator.save_evaluation_output")
@patch("oumi.core.evaluation.evaluator.build_inference_engine")
def test_evaluate_custom_task_incorrect_input_arguments(
    mock_build_inference_engine,
    mock_save_evaluation_output,
    mock_check_prerequisites,
    mock_get_evaluation_function,
    kwargs,
    eval_kwargs,
    expected_error_message,
):
    # Inputs.
    evaluation_config = EvaluationConfig(
        tasks=[
            EvaluationTaskParams(
                task_name="evaluation_fn_reg_name",
                evaluation_backend=EvaluationBackend.CUSTOM.value,
                eval_kwargs=eval_kwargs,
            )
        ],
        model=ModelParams(),
        generation=GenerationParams(),
        inference_engine=InferenceEngineType.NATIVE,
    )

    def evaluation_fn(task_params, config, expected_param_1, expected_param_2):
        raise RuntimeError("This function should not be called.")

    # Mocks.
    mock_build_inference_engine.return_value = MagicMock()
    mock_save_evaluation_output.return_value = None
    mock_check_prerequisites.return_value = None
    mock_get_evaluation_function.return_value = evaluation_fn

    # Run the test.
    evaluator = Evaluator()

    with pytest.raises(RuntimeError) as exc_info:
        _ = evaluator.evaluate(
            evaluation_config,
            **kwargs,
        )

    # Check the error message.
    assert str(exc_info.value).startswith(expected_error_message)

    # Check the results.
    mock_build_inference_engine.assert_not_called()
    mock_save_evaluation_output.assert_not_called()
    mock_check_prerequisites.assert_called_once()
    mock_get_evaluation_function.assert_called_once()


@patch("oumi.core.evaluation.evaluator.REGISTRY.get_evaluation_function")
@patch("oumi.core.evaluation.evaluator.check_prerequisites")
@patch("oumi.core.evaluation.evaluator.save_evaluation_output")
@patch("oumi.core.evaluation.evaluator.build_inference_engine")
def test_evaluate_custom_task_duplicate_optional_param(
    mock_build_inference_engine,
    mock_save_evaluation_output,
    mock_check_prerequisites,
    mock_get_evaluation_function,
):
    # Inputs.
    task_params = EvaluationTaskParams(
        task_name="evaluation_fn_reg_name",
        evaluation_backend=EvaluationBackend.CUSTOM.value,
        eval_kwargs={"optional_param": "value"},
    )
    evaluation_config = EvaluationConfig(tasks=[task_params])

    def evaluation_fn(task_params, config):
        raise RuntimeError("This function should not be called.")

    # Mocks.
    mock_build_inference_engine.return_value = MagicMock()
    mock_save_evaluation_output.return_value = None
    mock_check_prerequisites.return_value = None
    mock_get_evaluation_function.return_value = evaluation_fn

    # Run the test.
    evaluator = Evaluator()

    with pytest.raises(
        ValueError,
        match=(
            r"^The two keyword argument dictionaries contain overlapping keys: "
            "{'optional_param'}."
        ),
    ):
        _ = evaluator.evaluate(
            evaluation_config,
            optional_param="value",  # NOT allowed, already set in `eval_kwargs`.
        )

    # Check the results.
    mock_build_inference_engine.assert_not_called()
    mock_save_evaluation_output.assert_not_called()
    mock_check_prerequisites.assert_called_once()
    mock_get_evaluation_function.assert_called_once()


@patch("oumi.core.evaluation.evaluator.evaluate_lm_harness")
@patch("oumi.core.evaluation.evaluator.evaluate_alpaca_eval")
@patch("oumi.core.evaluation.evaluator.check_prerequisites")
@patch("oumi.core.evaluation.evaluator.save_evaluation_output")
@patch("oumi.core.evaluation.evaluator.build_inference_engine")
def test_evaluate_multiple_tasks(
    mock_build_inference_engine,
    mock_save_evaluation_output,
    mock_check_prerequisites,
    mock_evaluate_alpaca_eval,
    mock_evaluate_lm_harness,
):
    # Inputs.
    task_params_lm_harness_1 = EvaluationTaskParams(
        task_name="test_task_lm_harness_1",
        evaluation_backend=EvaluationBackend.LM_HARNESS.value,
    )
    task_params_alpaca_eval = EvaluationTaskParams(
        task_name="test_task_alpaca_eval",
        evaluation_backend=EvaluationBackend.ALPACA_EVAL.value,
    )
    task_params_lm_harness_2 = EvaluationTaskParams(
        task_name="test_task_lm_harness_2",
        evaluation_backend=EvaluationBackend.LM_HARNESS.value,
    )
    evaluation_config = EvaluationConfig(
        tasks=[
            task_params_lm_harness_1,
            task_params_alpaca_eval,
            task_params_lm_harness_2,
        ],
        model=ModelParams(model_name="test_model"),
        generation=GenerationParams(),
        inference_engine=InferenceEngineType.VLLM,
    )

    # Mocks.
    mock_build_inference_engine.return_value = MagicMock()
    mock_save_evaluation_output.return_value = None
    mock_check_prerequisites.return_value = None
    mock_evaluate_lm_harness.return_value = EvaluationResult(
        task_name="test_task_lm_harness", task_result={"test_metric_lm_harness": 1.0}
    )
    mock_evaluate_alpaca_eval.return_value = EvaluationResult(
        task_name="test_task_alpaca_eval", task_result={"test_metric_alpaca_eval": 2.0}
    )

    # Run the test.
    evaluator = Evaluator()
    result = evaluator.evaluate(evaluation_config)

    # Check the call counts to our mocks.
    assert mock_build_inference_engine.call_count == 1
    assert mock_save_evaluation_output.call_count == 3
    assert mock_check_prerequisites.call_count == 3
    assert mock_evaluate_lm_harness.call_count == 2
    assert mock_evaluate_alpaca_eval.call_count == 1

    # Check the first call to LM Harness.
    _, kwargs = mock_evaluate_lm_harness.call_args_list[0]
    assert isinstance(kwargs["task_params"], LMHarnessTaskParams)
    assert kwargs["task_params"].task_name == "test_task_lm_harness_1"
    assert kwargs["task_params"].evaluation_backend == (
        EvaluationBackend.LM_HARNESS.value
    )
    assert isinstance(kwargs["config"], EvaluationConfig)
    assert kwargs["config"].tasks == []
    assert kwargs["config"].model.model_name == "test_model"
    assert kwargs["config"].inference_engine == InferenceEngineType.VLLM

    # Check the second call to LM Harness.
    _, kwargs = mock_evaluate_lm_harness.call_args_list[1]
    assert isinstance(kwargs["task_params"], LMHarnessTaskParams)
    assert kwargs["task_params"].task_name == "test_task_lm_harness_2"
    assert kwargs["task_params"].evaluation_backend == (
        EvaluationBackend.LM_HARNESS.value
    )
    assert isinstance(kwargs["config"], EvaluationConfig)
    assert kwargs["config"].tasks == []
    assert kwargs["config"].model.model_name == "test_model"
    assert kwargs["config"].inference_engine == InferenceEngineType.VLLM

    # Check the call to Alpaca Eval.
    _, kwargs = mock_evaluate_alpaca_eval.call_args
    assert isinstance(kwargs["task_params"], AlpacaEvalTaskParams)
    assert kwargs["task_params"].task_name == "test_task_alpaca_eval"
    assert kwargs["task_params"].evaluation_backend == (
        EvaluationBackend.ALPACA_EVAL.value
    )
    assert isinstance(kwargs["config"], EvaluationConfig)
    assert kwargs["config"].tasks == []
    assert kwargs["config"].model.model_name == "test_model"
    assert kwargs["config"].inference_engine == InferenceEngineType.VLLM

    # Ensure the 2nd LM Harness call destroyed the inference engine.
    assert evaluator._inference_engine is None

    # Check the result.
    assert len(result) == 3
    assert result[0].task_name == "test_task_lm_harness"
    assert result[0].task_result == {"test_metric_lm_harness": 1.0}
    assert result[1].task_name == "test_task_alpaca_eval"
    assert result[1].task_result == {"test_metric_alpaca_eval": 2.0}
    assert result[2].task_name == "test_task_lm_harness"
    assert result[2].task_result == {"test_metric_lm_harness": 1.0}


@pytest.mark.parametrize(
    (
        "evaluation_backend_str,"
        "evaluation_backend_class,"
        "task_name,"
        "num_samples,"
        "eval_kwargs,"
        "expected_backend_task_params_class,"
        "expected_backend_task_params,"
    ),
    [
        # Alpaca Eval run with no arguments.
        (
            "alpaca_eval",
            EvaluationBackend.ALPACA_EVAL,
            "",
            None,
            {},
            AlpacaEvalTaskParams,
            {
                "evaluation_backend": "alpaca_eval",
                "task_name": "",
                "num_samples": None,
                "eval_kwargs": {},
            },
        ),
        # Alpaca Eval run with arguments.
        (
            "alpaca_eval",
            EvaluationBackend.ALPACA_EVAL,
            "unused_task_name",
            44,
            {"version": 2.0, "eval_param": "eval_param_value"},
            AlpacaEvalTaskParams,
            {
                "evaluation_backend": "alpaca_eval",
                "task_name": "unused_task_name",
                "num_samples": 44,
                "version": 2.0,
                "eval_kwargs": {"eval_param": "eval_param_value"},
            },
        ),
        # LM Harness run with no arguments.
        (
            "lm_harness",
            EvaluationBackend.LM_HARNESS,
            "abstract_algebra",
            None,
            {},
            LMHarnessTaskParams,
            {
                "evaluation_backend": "lm_harness",
                "task_name": "abstract_algebra",
                "num_samples": None,
                "eval_kwargs": {},
            },
        ),
        # LM Harness run with arguments.
        (
            "lm_harness",
            EvaluationBackend.LM_HARNESS,
            "abstract_algebra",
            55,
            {"num_fewshot": 44, "eval_param": "eval_param_value"},
            LMHarnessTaskParams,
            {
                "evaluation_backend": "lm_harness",
                "task_name": "abstract_algebra",
                "num_samples": 55,
                "num_fewshot": 44,
                "eval_kwargs": {"eval_param": "eval_param_value"},
            },
        ),
    ],
    ids=[
        "test_get_backend_task_params_alpaca_eval_no_args",
        "test_get_backend_task_params_alpaca_eval_with_args",
        "test_get_backend_task_params_lm_harness_no_args",
        "test_get_backend_task_params_lm_harness_with_args",
    ],
)
def test_get_backend_task_params(
    evaluation_backend_str: str,
    evaluation_backend_class: type,
    task_name: str,
    num_samples: int,
    eval_kwargs: dict,
    expected_backend_task_params_class: type,
    expected_backend_task_params: dict[str, Any],
):
    task_params = EvaluationTaskParams(
        evaluation_backend=evaluation_backend_str,
        task_name=task_name,
        num_samples=num_samples,
        eval_kwargs=eval_kwargs,
    )

    # Ensure the `EvaluationTaskParams` class members are correct.
    assert task_params.evaluation_backend == evaluation_backend_str
    assert task_params.task_name == task_name
    assert task_params.num_samples == num_samples
    assert task_params.eval_kwargs == eval_kwargs

    # Ensure the correct backend is returned.
    assert task_params.get_evaluation_backend() == evaluation_backend_class

    # Ensure the correct backend class is returned.
    backend_task_params = Evaluator._get_backend_task_params(task_params)
    assert isinstance(backend_task_params, expected_backend_task_params_class)

    # Ensure the backend-specific task parameters are as expected.
    for expected_param, expected_param_value in expected_backend_task_params.items():
        actual_param_value = getattr(backend_task_params, expected_param)
        assert actual_param_value == expected_param_value


def test_get_backend_task_params_error_custom_backend():
    task_params = EvaluationTaskParams(
        evaluation_backend="custom",
        task_name="my_evaluation_fn",
    )

    with pytest.raises(
        ValueError,
        match=(
            r"^The custom evaluation backend is not subclassing EvaluationTaskParams."
        ),
    ):
        Evaluator._get_backend_task_params(task_params)


def test_get_backend_task_params_error_double_definition():
    task_params = EvaluationTaskParams(
        evaluation_backend="lm_harness",
        task_name="some_task",
        eval_kwargs={"task_name": "some_other_task"},
    )

    with pytest.raises(
        ValueError,
        match=(
            "Parameter `task_name` is present twice, in both task parameters "
            "and `eval_kwargs` dictionary. Please remove it from one of them."
        ),
    ):
        Evaluator._get_backend_task_params(task_params)
