import copy

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
        "expected_task_params_class,"
        "expected_init_kwargs,"
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
            "unused_task_name",
            44,
            {"version": 2.0, "eval_param": "eval_param_value"},
            EvaluationBackend.ALPACA_EVAL,
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
            "abstract_algebra",
            None,
            {},
            EvaluationBackend.LM_HARNESS,
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
            "abstract_algebra",
            55,
            {"num_fewshot": 44, "eval_param": "eval_param_value"},
            EvaluationBackend.LM_HARNESS,
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
        "alpaca_eval_no_args",
        "alpaca_eval_with_args",
        "lm_harness_no_args",
        "lm_harness_with_args",
    ],
)
def test_valid_initialization(
    evaluation_backend,
    task_name,
    num_samples,
    eval_kwargs,
    expected_backend,
    expected_task_params_class,
    expected_init_kwargs,
):
    task_params = EvaluationTaskParams(
        evaluation_backend=evaluation_backend,
        task_name=task_name,
        num_samples=num_samples,
        eval_kwargs=eval_kwargs,
    )

    # Ensure the `EvaluationTaskParams` class members are correct.
    assert task_params.evaluation_backend == evaluation_backend
    assert task_params.task_name == task_name
    assert task_params.num_samples == num_samples
    assert task_params.eval_kwargs == eval_kwargs

    # Ensure the conversion methods produce the expected results.
    assert task_params.get_evaluation_backend() == expected_backend
    backend_task_params = copy.deepcopy(
        task_params
    ).get_evaluation_backend_task_params()
    assert isinstance(backend_task_params, expected_task_params_class)

    # Ensure the backend-specific task parameters are as expected.
    assert expected_init_kwargs == task_params._get_init_kwargs_for_task_params_class(
        expected_task_params_class
    )
    expected_task_params = expected_task_params_class(**expected_init_kwargs)
    assert backend_task_params == expected_task_params


@pytest.mark.parametrize(
    ("evaluation_backend, task_name, num_samples, eval_kwargs"),
    [
        # Missing `EvaluationTaskParams` argument: `evaluation_backend`.
        (
            "",
            "",
            None,
            {},
        ),
        # Incorrect `EvaluationTaskParams` argument: `evaluation_backend`.
        (
            "non_existing_backend",
            "",
            None,
            {},
        ),
        # Incorrect `EvaluationTaskParams` argument: `num_samples` is negative.
        (
            "alpaca_eval",
            "",
            -1,
            {},
        ),
        # Incorrect `EvaluationTaskParams` argument: `num_samples` is zero.
        (
            "alpaca_eval",
            "",
            0,
            {},
        ),
        # Missing `LMHarnessTaskParams` argument: `task_name`.
        (
            "lm_harness",
            "",
            None,
            {},
        ),
    ],
    ids=[
        "no_backend",
        "wrong_backend",
        "num_samples_negative",
        "num_samples_zero",
        "lm_harness_with_no_task_name",
    ],
)
def test_invalid_initialization(
    evaluation_backend,
    task_name,
    num_samples,
    eval_kwargs,
):
    with pytest.raises(ValueError):
        EvaluationTaskParams(
            evaluation_backend=evaluation_backend,
            task_name=task_name,
            num_samples=num_samples,
            eval_kwargs=eval_kwargs,
        )


@pytest.mark.parametrize(
    ("evaluation_backend, task_name, num_samples, eval_kwargs"),
    [
        # Incorrect `AlpacaEvalTaskParams` argument: `version`.
        (
            "alpaca_eval",
            "",
            None,
            {"version": 3.0},
        ),
        # Double definition of variable: `num_samples`.
        (
            "alpaca_eval",
            "44",
            None,
            {"num_samples": 44},
        ),
        # Incorrect `LMHarnessTaskParams` argument: `num_fewshot` negative.
        (
            "lm_harness",
            "abstract_algebra",
            None,
            {"num_fewshot": -1},
        ),
    ],
    ids=[
        "alpaca_eval_wrong_version",
        "alpaca_eval_double_definition",
        "lm_harness_num_fewshot_negative",
    ],
)
def test_backend_task_params_invalid_instantiation(
    evaluation_backend,
    task_name,
    num_samples,
    eval_kwargs,
):
    task_params = EvaluationTaskParams(
        evaluation_backend=evaluation_backend,
        task_name=task_name,
        num_samples=num_samples,
        eval_kwargs=eval_kwargs,
    )
    with pytest.raises(ValueError):
        task_params.get_evaluation_backend_task_params()
