from typing import Any
from unittest.mock import patch

import pytest

from oumi.core.configs import (
    AlpacaEvalTaskParams,
    EvaluationConfig,
    EvaluationTaskParams,
    GenerationParams,
    InferenceEngineType,
    LMHarnessTaskParams,
    ModelParams,
)
from oumi.core.configs.params.evaluation_params import EvaluationBackend
from oumi.core.evaluation.evaluation_result import EvaluationResult
from oumi.core.evaluation.evaluator import Evaluator


@patch("oumi.core.evaluation.evaluator.evaluate_lm_harness")
@patch("oumi.core.evaluation.evaluator.check_prerequisites")
@patch("oumi.core.evaluation.evaluator.save_evaluation_output")
def test_evaluate_lm_harness_task(
    mock_save_evaluation_output, mock_check_prerequisites, mock_evaluate_lm_harness
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
    mock_save_evaluation_output.return_value = None
    mock_check_prerequisites.return_value = None
    mock_evaluate_lm_harness.return_value = EvaluationResult(
        task_name="test_task", task_result={"test_metric": 1.0}
    )

    # Run the test.
    evaluator = Evaluator()
    result = evaluator.evaluate(evaluation_config)

    # Check the results.
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
def test_evaluate_alpaca_eval_task(
    mock_save_evaluation_output, mock_check_prerequisites, mock_evaluate_alpaca_eval
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
    mock_save_evaluation_output.return_value = None
    mock_check_prerequisites.return_value = None
    mock_evaluate_alpaca_eval.return_value = EvaluationResult(
        task_name="test_task", task_result={"test_metric": 1.0}
    )

    # Run the test.
    evaluator = Evaluator()
    result = evaluator.evaluate(evaluation_config)

    # Check the results.
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
def test_evaluate_custom_task(
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

    def evaluation_fn(
        task_params: EvaluationTaskParams,
        config: EvaluationConfig,
        optional_param: str,
    ) -> EvaluationResult:
        assert task_params.evaluation_backend == EvaluationBackend.CUSTOM.value
        assert task_params.task_name == "evaluation_fn_reg_name"
        assert optional_param == "optional_param_value"
        return EvaluationResult(
            task_name=task_params.task_name,
            task_result={"test_metric": 1.0},
        )

    # Mocks.
    mock_save_evaluation_output.return_value = None
    mock_check_prerequisites.return_value = None
    mock_get_evaluation_function.return_value = evaluation_fn

    # Run the test.
    evaluator = Evaluator()
    result = evaluator.evaluate(
        evaluation_config, optional_param="optional_param_value"
    )

    # Check the results.
    mock_save_evaluation_output.assert_called_once()
    mock_check_prerequisites.assert_called_once()
    mock_get_evaluation_function.assert_called_once()
    assert len(result) == 1
    assert result[0].task_name == "evaluation_fn_reg_name"
    assert result[0].task_result == {"test_metric": 1.0}


@patch("oumi.core.evaluation.evaluator.REGISTRY.get_evaluation_function")
@patch("oumi.core.evaluation.evaluator.check_prerequisites")
@patch("oumi.core.evaluation.evaluator.save_evaluation_output")
def test_evaluate_custom_task_unregistered_fn(
    mock_save_evaluation_output, mock_check_prerequisites, mock_get_evaluation_function
):
    # Inputs.
    task_params = EvaluationTaskParams(
        task_name="evaluation_fn_unregistered",
        evaluation_backend=EvaluationBackend.CUSTOM.value,
    )
    evaluation_config = EvaluationConfig(tasks=[task_params])

    # Mocks.
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
    mock_save_evaluation_output.assert_not_called()
    mock_check_prerequisites.assert_called_once()
    mock_get_evaluation_function.assert_called_once()


@patch("oumi.core.evaluation.evaluator.REGISTRY.get_evaluation_function")
@patch("oumi.core.evaluation.evaluator.check_prerequisites")
@patch("oumi.core.evaluation.evaluator.save_evaluation_output")
def test_evaluate_custom_task_without_task_name(
    mock_save_evaluation_output, mock_check_prerequisites, mock_get_evaluation_function
):
    # Inputs.
    task_params = EvaluationTaskParams(
        evaluation_backend=EvaluationBackend.CUSTOM.value
    )
    evaluation_config = EvaluationConfig(tasks=[task_params])

    # Mocks.
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
    mock_save_evaluation_output.assert_not_called()
    mock_check_prerequisites.assert_called_once()
    mock_get_evaluation_function.assert_not_called()


@patch("oumi.core.evaluation.evaluator.evaluate_lm_harness")
@patch("oumi.core.evaluation.evaluator.evaluate_alpaca_eval")
@patch("oumi.core.evaluation.evaluator.check_prerequisites")
@patch("oumi.core.evaluation.evaluator.save_evaluation_output")
def test_evaluate_multiple_tasks(
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
