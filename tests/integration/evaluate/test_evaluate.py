import json
import os
import tempfile
from importlib.util import find_spec
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from oumi import evaluate
from oumi.core.configs import (
    EvaluationConfig,
    EvaluationTaskParams,
    GenerationParams,
    ModelParams,
)
from oumi.evaluation.save_utils import OUTPUT_FILENAME_PLATFORM_RESULTS
from tests.markers import requires_gpus

# Currently supporting two platforms for evaluation: LM Harness and Alpaca Eval.
LM_HARNESS = "lm_harness"
ALPACA_EVAL = "alpaca_eval"

TEST_TASK_PARAMS = {
    LM_HARNESS: {
        # EvaluationTaskParams params.
        "task_name": "mmlu_abstract_algebra",
        "num_samples": 20,
        # ModelParams params.
        "model_name": "openai-community/gpt2",
        "model_max_length": 1,
        # EvaluationConfig (remaining params needed).
        "run_name": "test_lm_harness",
    },
    ALPACA_EVAL: {
        # EvaluationTaskParams params.
        "task_name": "",
        "num_samples": 3,
        # ModelParams params.
        "model_name": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "model_max_length": 10,
        # GenerationParams params (needed because Alpaca Eval runs Oumi inference).
        "max_new_tokens": 10,
        "batch_size": 1,
        # EvaluationConfig (remaining params needed).
        "run_name": "test_alpaca_eval",
    },
}

EXPECTED_RESULTS = {
    LM_HARNESS: {
        ######### actual results returned by the `lm_eval.simple_evaluate` API ########
        # {                                                                           #
        #   "acc,none": 0.2,                                                          #
        #   "acc_stderr,none": 0.09176629354822471,                                   #
        #   "alias": "abstract_algebra"                                               #
        # }                                                                           #
        ###############################################################################
        "acc,none": {"value": 0.2, "round_digits": 3},
        "acc_stderr,none": {"value": 0.092, "round_digits": 3},
    },
    ALPACA_EVAL: {
        ########## actual results returned by the `alpaca_eval.evaluate` API ##########
        # {                                                                           #
        #   "win_rate": 2.1546666665687532e-05,                                       #
        #   "standard_error": 8.230602115518512e-06,                                  #
        #   "n_wins": 0,                                                              #
        #   "n_wins_base": 3,                                                         #
        #   "n_draws": 0,                                                             #
        #   "n_total": 3,                                                             #
        #   "discrete_win_rate": 0.0,                                                 #
        #   "mode": "community",                                                      #
        #   "avg_length": 44,                                                         #
        #   "length_controlled_winrate": 0.05370406224906349,                         #
        #   "lc_standard_error": 0.014120007914531866,                                #
        # }                                                                           #
        ###############################################################################
        "win_rate": {"value": 0.0000215, "round_digits": 7},
        "standard_error": {"value": 0.0000082, "round_digits": 7},
        "length_controlled_winrate": {"value": 0.054, "round_digits": 3},
        "lc_standard_error": {"value": 0.014, "round_digits": 3},
        "avg_length": {"value": 44, "round_digits": 0},
    },
    "alpaca_eval_model_outputs": [
        # Expected responses from `SmolLM2-135M-Instruct` model (with max length = 10).
        "Some of the most famous actors who started their careers",
        "The process of getting a state's name is a",
        "Absolutely, I'd be happy to help you",
    ],
}


def _get_evaluation_config(platform: str, output_dir: str) -> EvaluationConfig:
    evaluation_task_params = EvaluationTaskParams(
        evaluation_platform=platform,
        task_name=TEST_TASK_PARAMS[platform]["task_name"],
        num_samples=TEST_TASK_PARAMS[platform]["num_samples"],
    )
    model_params = ModelParams(
        model_name=TEST_TASK_PARAMS[platform]["model_name"],
        model_max_length=TEST_TASK_PARAMS[platform]["model_max_length"],
        trust_remote_code=True,
    )
    generation_params = (
        GenerationParams(
            max_new_tokens=TEST_TASK_PARAMS[platform]["max_new_tokens"],
            batch_size=TEST_TASK_PARAMS[platform]["batch_size"],
        )
        if platform == ALPACA_EVAL
        else GenerationParams()
    )

    return EvaluationConfig(
        output_dir=output_dir,
        tasks=[evaluation_task_params],
        model=model_params,
        generation=generation_params,
        run_name=TEST_TASK_PARAMS[platform]["run_name"],
    )


def _validate_results_returned(
    platform: str, results_list: list[dict[str, Any]]
) -> None:
    # Retrieve the results returned from the `evaluate` function.
    assert len(results_list) == 1  # 1 task was evaluated.
    results_dict = results_list[0]["results"]

    # Platforms with tasks (e.g. LM Harness) nest the results under the task names.
    if "task_name" in TEST_TASK_PARAMS[platform]:
        task_name = TEST_TASK_PARAMS[platform]["task_name"]
        results_dict = results_dict[task_name] if task_name else results_dict

    # Validate the results.
    for expected_key in EXPECTED_RESULTS[platform]:
        if expected_key not in results_dict:
            raise ValueError(
                f"Key `{expected_key}` was not found in the results: `{results_dict}`."
            )
        expected_value = EXPECTED_RESULTS[platform][expected_key]["value"]
        round_digits = EXPECTED_RESULTS[platform][expected_key]["round_digits"]
        actual_value = results_dict[expected_key]
        if round(actual_value, round_digits) != expected_value:
            raise ValueError(
                f"Expected value for key `{expected_key}` should be `{expected_value}` "
                f"(rounded to `{round_digits}` digits), but instead the actual value "
                f"that was returned is `{actual_value}`."
            )


def _validate_results_in_file(platform: str, output_dir: str) -> None:
    # Identify the relevant `output_path` for the evaluation test:
    # <output_dir> / <platform>_<timestamp> / platform_results.json
    subfolders = [sf for sf in os.listdir(output_dir) if sf.startswith(f"{platform}_")]
    assert len(subfolders) == 1
    output_path = os.path.join(
        output_dir, subfolders[0], OUTPUT_FILENAME_PLATFORM_RESULTS
    )
    assert os.path.exists(output_path)

    # Read the results from the evaluation test's output file.
    with open(output_path, encoding="utf-8") as file_ptr:
        results_dict = json.load(file_ptr)["results"]
        # Platforms with tasks (e.g. LM Harness) nest the results under the task names.
        if "task_name" in TEST_TASK_PARAMS[platform]:
            task_name = TEST_TASK_PARAMS[platform]["task_name"]
            results_dict = results_dict[task_name] if task_name else results_dict

    # Validate the results.
    for expected_key in EXPECTED_RESULTS[platform]:
        if expected_key not in results_dict:
            raise ValueError(
                f"Key `{expected_key}` was not found in the results: `{results_dict}`."
            )
        expected_value = EXPECTED_RESULTS[platform][expected_key]["value"]
        round_digits = EXPECTED_RESULTS[platform][expected_key]["round_digits"]
        actual_value = results_dict[expected_key]
        if round(actual_value, round_digits) != expected_value:
            raise ValueError(
                f"Expected value for key `{expected_key}` should be `{expected_value}` "
                f"(rounded to `{round_digits}` digits), but instead the actual value "
                f"that was returned is `{actual_value}`."
            )


def _mock_alpaca_eval_evaluate(
    model_outputs: dict[str, Any],
    annotators_config: str,
    fn_metric: str,
    max_instances: int,
    **kwargs,
) -> tuple[pd.DataFrame, None]:
    # Ensure the input arguments are the defaults (unless changed by this test).
    assert annotators_config == "weighted_alpaca_eval_gpt4_turbo"
    assert fn_metric == "get_length_controlled_winrate"
    assert max_instances == TEST_TASK_PARAMS[ALPACA_EVAL]["num_samples"]

    # Ensure the inference results (`model_outputs`) are the expected ones.
    expected_model_outputs = EXPECTED_RESULTS["alpaca_eval_model_outputs"]
    for i in range(TEST_TASK_PARAMS[ALPACA_EVAL]["num_samples"]):
        if model_outputs["output"][i] != expected_model_outputs[i]:
            raise ValueError(f"Unexpected model output: {model_outputs['output'][i]}")

    # Mock the `alpaca_eval.evaluate` function (by returning the expected results).
    expected_results_dict = EXPECTED_RESULTS[ALPACA_EVAL]
    df_leaderboard = pd.DataFrame(
        {key: expected_results_dict[key]["value"] for key in expected_results_dict},
        index=[TEST_TASK_PARAMS[ALPACA_EVAL]["run_name"]],  # type: ignore
    )
    return df_leaderboard, None


@requires_gpus()
def test_evaluate_lm_harness():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        nested_output_dir = os.path.join(output_temp_dir, "nested", "dir")
        config = _get_evaluation_config(LM_HARNESS, nested_output_dir)
        results_list = evaluate(config)
        _validate_results_returned(platform=LM_HARNESS, results_list=results_list)
        _validate_results_in_file(platform=LM_HARNESS, output_dir=nested_output_dir)


@requires_gpus()
def test_evaluate_alpaca_eval():
    if find_spec("alpaca_eval") is None:
        pytest.skip("Skipping because alpaca_eval is not installed")

    with tempfile.TemporaryDirectory() as output_temp_dir:
        nested_output_dir = os.path.join(output_temp_dir, "nested", "dir")
        config = _get_evaluation_config(ALPACA_EVAL, nested_output_dir)
        with patch("alpaca_eval.evaluate", _mock_alpaca_eval_evaluate):
            results_list = evaluate(config)
        _validate_results_returned(platform=ALPACA_EVAL, results_list=results_list)
        _validate_results_in_file(platform=ALPACA_EVAL, output_dir=nested_output_dir)
