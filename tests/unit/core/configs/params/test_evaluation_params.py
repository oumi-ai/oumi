import pytest

from oumi.core.configs.params.evaluation_params import (
    AlpacaEvalTaskParams,
    EvaluationBackend,
    EvaluationTaskParams,
    LMHarnessTaskParams,
)


@pytest.mark.parametrize(
    (
        "evaluation_backend,"
        "task_name,"
        "num_samples,"
        "eval_kwargs,"
        "expected_backend,"
        "backend_class,"
    ),
    [
        # Alpaca Eval run with no arguments.
        (
            "alpaca_eval",
            "",
            None,
            {},
            EvaluationBackend.ALPACA_EVAL,
            AlpacaEvalTaskParams,
        ),
        # Alpaca Eval run with arguments.
        (
            "alpaca_eval",
            "unused_task_name",
            44,
            {"version": 2.0, "eval_param": "eval_param_value"},
            EvaluationBackend.ALPACA_EVAL,
            AlpacaEvalTaskParams,
        ),
        # LM Harness run with no arguments.
        (
            "lm_harness",
            "abstract_algebra",
            None,
            {},
            EvaluationBackend.LM_HARNESS,
            LMHarnessTaskParams,
        ),
        # LM Harness run with arguments.
        (
            "lm_harness",
            "abstract_algebra",
            55,
            {"num_fewshot": 44, "eval_param": "eval_param_value"},
            EvaluationBackend.LM_HARNESS,
            LMHarnessTaskParams,
        ),
        # Custom run with arguments.
        (
            "custom",
            "my_evaluation_fn",
            66,
            {"eval_param": "eval_param_value"},
            EvaluationBackend.CUSTOM,
            EvaluationTaskParams,
        ),
    ],
    ids=[
        "test_valid_initialization_alpaca_eval_no_args",
        "test_valid_initialization_alpaca_eval_with_args",
        "test_valid_initialization_lm_harness_no_args",
        "test_valid_initialization_lm_harness_with_args",
        "test_valid_initialization_custom_with_args",
    ],
)
def test_valid_initialization(
    evaluation_backend,
    task_name,
    num_samples,
    eval_kwargs,
    expected_backend,
    backend_class,
):
    # Instantiate an `EvaluationTaskParams` object.
    task_params = EvaluationTaskParams(
        evaluation_backend=evaluation_backend,
        task_name=task_name,
        num_samples=num_samples,
        eval_kwargs=eval_kwargs,
    )

    # Ensure the `EvaluationTaskParams` class members are correct.
    assert task_params.evaluation_backend == evaluation_backend
    assert task_params.get_evaluation_backend() == expected_backend
    assert task_params.task_name == task_name
    assert task_params.num_samples == num_samples
    assert task_params.eval_kwargs == eval_kwargs

    # Instantiate an `EvaluationTaskParams` subclass.
    backend_task_params = backend_class(
        evaluation_backend=evaluation_backend,
        task_name=task_name,
        num_samples=num_samples,
        eval_kwargs=eval_kwargs,
    )

    # Ensure the subclass task params are also valid.
    assert backend_task_params.evaluation_backend == evaluation_backend
    assert backend_task_params.get_evaluation_backend() == expected_backend
    assert backend_task_params.task_name == task_name
    assert backend_task_params.num_samples == num_samples
    assert backend_task_params.eval_kwargs == eval_kwargs


def test_invalid_initialization_unknown_backend():
    with pytest.raises(ValueError, match="^Unknown evaluation backend"):
        task_params = EvaluationTaskParams(
            evaluation_backend="non_existing_backend",
            task_name="some_task",
            num_samples=None,
        )
        task_params.get_evaluation_backend()


def test_invalid_initialization_no_backend():
    with pytest.raises(ValueError):
        task_params = EvaluationTaskParams(
            task_name="some_task",
            num_samples=None,
        )
        task_params.get_evaluation_backend()


@pytest.mark.parametrize(
    ("num_samples"),
    [-1, 0],
    ids=[
        "test_invalid_initialization_num_samples_negative",
        "test_invalid_initialization_num_samples_zero",
    ],
)
def test_invalid_initialization_num_samples(num_samples):
    with pytest.raises(ValueError):
        EvaluationTaskParams(
            evaluation_backend="lm_harness",
            task_name="some_task",
            num_samples=num_samples,
        )


def test_lm_harness_invalid_initialization_missing_task():
    with pytest.raises(
        ValueError, match="`task_name` must be a valid LM Harness task."
    ):
        _ = LMHarnessTaskParams(
            evaluation_backend="lm_harness",
            num_samples=10,
        )


def test_lm_harness_invalid_initialization_num_fewshot_negative():
    with pytest.raises(ValueError, match="`num_fewshot` must be non-negative."):
        _ = LMHarnessTaskParams(
            evaluation_backend="lm_harness",
            task_name="some_task",
            num_samples=10,
            num_fewshot=-1,
        )


def test_alpaca_eval_invalid_initialization_version():
    with pytest.raises(ValueError, match="AlpacaEval `version` must be 1.0 or 2.0."):
        _ = AlpacaEvalTaskParams(
            evaluation_backend="alpaca_eval",
            task_name="",
            num_samples=10,
            version=3.0,
        )
