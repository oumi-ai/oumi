from oumi import evaluate
from oumi.core.configs import (
    EvaluationBackend,
    EvaluationConfig,
    EvaluationTaskParams,
)
from oumi.core.evaluation.evaluation_result import EvaluationResult
from oumi.core.registry import register_evaluate_function

EVALUATE_FN_REGISTERED_NAME = "oumi/test_evaluate_fn"
EVAL_KWARGS = {"custom_parameter": "custom_parameter_value"}
TASK_RESULT = {"result": "dummy_result"}


@register_evaluate_function(EVALUATE_FN_REGISTERED_NAME)
def oumi_test_evaluate_fn(
    task_params: EvaluationTaskParams,
    config: EvaluationConfig,
) -> EvaluationResult:
    """Dummy evaluate function for unit testing."""
    # Ensure the task_params are passed correctly.
    assert task_params.evaluation_backend == EvaluationBackend.CUSTOM_OUMI.value
    assert task_params.task_name == EVALUATE_FN_REGISTERED_NAME
    assert task_params.eval_kwargs == EVAL_KWARGS

    return EvaluationResult(
        task_name=task_params.task_name,
        task_result=TASK_RESULT,
        backend_config={"config": "dummy_config"},
    )


def test_evaluate_oumi_custom():
    task_params = EvaluationTaskParams(
        evaluation_backend=EvaluationBackend.CUSTOM_OUMI.value,
        task_name=EVALUATE_FN_REGISTERED_NAME,
        eval_kwargs=EVAL_KWARGS,
    )
    evaluation_config = EvaluationConfig(tasks=[task_params])
    evaluation_result = evaluate(evaluation_config)
    assert len(evaluation_result) == 1
    assert evaluation_result[0] == TASK_RESULT


### FIXME: Add tests for the non-happy paths
