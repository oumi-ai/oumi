import copy

import pytest

from oumi.core.configs.params.evaluation_params import (
    AlpacaEvalTaskParams,
    EvaluationPlatform,
    EvaluationTaskParams,
    LMHarnessTaskParams,
)


@pytest.mark.parametrize(
    (
        "evaluation_platform,"
        "task_name,"
        "num_samples,"
        "eval_kwargs,"
        "expected_platform,"
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
            EvaluationPlatform.ALPACA_EVAL,
            AlpacaEvalTaskParams,
            {
                "evaluation_platform": "alpaca_eval",
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
            EvaluationPlatform.ALPACA_EVAL,
            AlpacaEvalTaskParams,
            {
                "evaluation_platform": "alpaca_eval",
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
            EvaluationPlatform.LM_HARNESS,
            LMHarnessTaskParams,
            {
                "evaluation_platform": "lm_harness",
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
            EvaluationPlatform.LM_HARNESS,
            LMHarnessTaskParams,
            {
                "evaluation_platform": "lm_harness",
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
    evaluation_platform,
    task_name,
    num_samples,
    eval_kwargs,
    expected_platform,
    expected_task_params_class,
    expected_init_kwargs,
):
    task_params = EvaluationTaskParams(
        evaluation_platform=evaluation_platform,
        task_name=task_name,
        num_samples=num_samples,
        eval_kwargs=eval_kwargs,
    )

    # Ensure the `EvaluationTaskParams` class members are correct.
    assert task_params.evaluation_platform == evaluation_platform
    assert task_params.task_name == task_name
    assert task_params.num_samples == num_samples
    assert task_params.eval_kwargs == eval_kwargs

    # Ensure the conversion methods produce the expected results.
    assert task_params.get_evaluation_platform() == expected_platform
    platform_task_params = copy.deepcopy(
        task_params
    ).get_evaluation_platform_task_params()
    assert isinstance(platform_task_params, expected_task_params_class)

    # Ensure the platform-specific task parameters are as expected.
    assert expected_init_kwargs == task_params._get_init_kwargs_for_task_params_class(
        expected_task_params_class
    )
    expected_task_params = expected_task_params_class(**expected_init_kwargs)
    assert platform_task_params == expected_task_params


@pytest.mark.parametrize(
    ("evaluation_platform, task_name, num_samples, eval_kwargs"),
    [
        # Missing `EvaluationTaskParams` argument: `evaluation_platform`.
        (
            "",
            "",
            None,
            {},
        ),
        # Incorrect `EvaluationTaskParams` argument: `evaluation_platform`.
        (
            "non_existing_platform",
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
        "no_platform",
        "wrong_platform",
        "num_samples_negative",
        "num_samples_zero",
        "lm_harness_with_no_task_name",
    ],
)
def test_invalid_initialization(
    evaluation_platform,
    task_name,
    num_samples,
    eval_kwargs,
):
    with pytest.raises(ValueError):
        EvaluationTaskParams(
            evaluation_platform=evaluation_platform,
            task_name=task_name,
            num_samples=num_samples,
            eval_kwargs=eval_kwargs,
        )


@pytest.mark.parametrize(
    ("evaluation_platform, task_name, num_samples, eval_kwargs"),
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
def test_platform_task_params_invalid_instantiation(
    evaluation_platform,
    task_name,
    num_samples,
    eval_kwargs,
):
    task_params = EvaluationTaskParams(
        evaluation_platform=evaluation_platform,
        task_name=task_name,
        num_samples=num_samples,
        eval_kwargs=eval_kwargs,
    )
    with pytest.raises(ValueError):
        task_params.get_evaluation_platform_task_params()
