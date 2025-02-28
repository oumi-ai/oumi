from unittest.mock import patch

import pytest

from oumi.core.configs import (
    AlpacaEvalTaskParams,
    CustomTaskParams,
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
def test_evaluate_lm_harness_task(mock_check_prerequisites, mock_evaluate_lm_harness):
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
    mock_check_prerequisites.return_value = None
    mock_evaluate_lm_harness.return_value = EvaluationResult(
        task_name="test_task", task_result={"test_metric": 1.0}
    )

    # Run the test.
    evaluator = Evaluator()
    result = evaluator.evaluate(evaluation_config)

    # Check the results.
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
def test_evaluate_alpaca_eval_task(mock_check_prerequisites, mock_evaluate_alpaca_eval):
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
    mock_check_prerequisites.return_value = None
    mock_evaluate_alpaca_eval.return_value = EvaluationResult(
        task_name="test_task", task_result={"test_metric": 1.0}
    )

    # Run the test.
    evaluator = Evaluator()
    result = evaluator.evaluate(evaluation_config)

    # Check the results.
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
def test_evaluate_custom_task(mock_get_evaluation_function):
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
        task_params: CustomTaskParams,
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
    mock_get_evaluation_function.return_value = evaluation_fn

    # Run the test.
    evaluator = Evaluator()
    result = evaluator.evaluate(
        evaluation_config, optional_param="optional_param_value"
    )

    # Check the results.
    mock_get_evaluation_function.assert_called_once()
    assert len(result) == 1
    assert result[0].task_name == "evaluation_fn_reg_name"
    assert result[0].task_result == {"test_metric": 1.0}


@patch("oumi.core.evaluation.evaluator.REGISTRY.get_evaluation_function")
def test_evaluate_custom_task_unregistered_fn(mock_get_evaluation_function):
    # Inputs.
    task_params = EvaluationTaskParams(
        task_name="evaluation_fn_unregistered",
        evaluation_backend=EvaluationBackend.CUSTOM.value,
    )
    evaluation_config = EvaluationConfig(tasks=[task_params])

    # Mocks.
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


def test_evaluate_custom_task_without_task_name():
    # Inputs.
    task_params = EvaluationTaskParams(
        evaluation_backend=EvaluationBackend.CUSTOM.value
    )
    evaluation_config = EvaluationConfig(tasks=[task_params])

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


@patch("oumi.core.evaluation.evaluator.evaluate_lm_harness")
@patch("oumi.core.evaluation.evaluator.evaluate_alpaca_eval")
@patch("oumi.core.evaluation.evaluator.check_prerequisites")
def test_evaluate_multiple_tasks(
    mock_check_prerequisites, mock_evaluate_alpaca_eval, mock_evaluate_lm_harness
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
