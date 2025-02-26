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

INPUT_CONFIG = {
    # EvaluationTaskParams params.
    "num_samples": 3,
    # ModelParams params.
    "model_name": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "model_max_length": 10,
    # GenerationParams params.
    "max_new_tokens": 10,
    "batch_size": 1,
    # EvaluationConfig params.
    "run_name": "test_alpaca_eval",
}

EXPECTED_RESULTS = {
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
}

EXPECTED_INFERENCES = [
    # Expected responses from `SmolLM2-135M-Instruct` model (with max length = 10).
    "Some of the most famous actors who started their careers",
    "The process of getting a state's name is a",
    "Absolutely, I'd be happy to help you",
]


def _get_evaluation_config(input_config: dict) -> EvaluationConfig:
    evaluation_task_params = EvaluationTaskParams(
        evaluation_backend="alpaca_eval",
        num_samples=input_config["num_samples"],
    )
    model_params = ModelParams(
        model_name=input_config["model_name"],
        model_max_length=input_config["model_max_length"],
        trust_remote_code=True,
    )
    generation_params = GenerationParams(
        max_new_tokens=input_config["max_new_tokens"],
        batch_size=input_config["batch_size"],
    )
    return EvaluationConfig(
        output_dir=input_config["output_dir"],
        tasks=[evaluation_task_params],
        model=model_params,
        generation=generation_params,
        run_name=input_config["run_name"],
    )


def _validate_results_returned(
    expected_results: dict[str, Any],
    actual_results: dict[str, Any],
) -> None:
    for expected_key in expected_results:
        if expected_key not in actual_results:
            raise ValueError(
                f"Key `{expected_key}` was not found in the results: `{actual_results}`"
            )
        expected_value = expected_results[expected_key]["value"]
        round_digits = expected_results[expected_key]["round_digits"]
        actual_value = actual_results[expected_key]
        if round(actual_value, round_digits) != expected_value:
            raise ValueError(
                f"Expected value for key `{expected_key}` should be `{expected_value}` "
                f"(rounded to `{round_digits}` digits), but instead the actual value "
                f"that was returned is `{actual_value}`."
            )


def _validate_results_in_file(expected_results: dict[str, Any], output_dir: str):
    # Identify the relevant `output_path` for the evaluation test:
    # <output_dir> / <backend>_<timestamp> / task_result.json
    subfolders = [f for f in os.listdir(output_dir) if f.startswith("alpaca_eval_")]
    assert len(subfolders) == 1
    output_path = os.path.join(output_dir, subfolders[0], "task_result.json")
    assert os.path.exists(output_path)

    # Read the results from the evaluation test's output file.
    with open(output_path, encoding="utf-8") as file_ptr:
        results_dict = json.load(file_ptr)["results"]

    _validate_results_returned(
        expected_results=expected_results, actual_results=results_dict
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
    assert max_instances == INPUT_CONFIG["num_samples"]

    # Ensure the inference results (`model_outputs`) are the expected ones.
    for i in range(INPUT_CONFIG["num_samples"]):
        if model_outputs["output"][i] != EXPECTED_INFERENCES[i]:
            raise ValueError(f"Unexpected model output: {model_outputs['output'][i]}")

    # Mock the `alpaca_eval.evaluate` function (by returning the expected results).
    df_leaderboard = pd.DataFrame(
        {key: EXPECTED_RESULTS[key]["value"] for key in EXPECTED_RESULTS},
        index=[INPUT_CONFIG["run_name"]],  # type: ignore
    )
    return df_leaderboard, None


@pytest.mark.skip(reason="Skipping until AlpacaEval is properly implemented")
def test_evaluate_alpaca_eval():
    if find_spec("alpaca_eval") is None:
        pytest.skip("Skipping because alpaca_eval is not installed")

    with tempfile.TemporaryDirectory() as output_temp_dir:
        nested_output_dir = os.path.join(output_temp_dir, "nested", "dir")
        input_config = {**INPUT_CONFIG, "output_dir": nested_output_dir}
        evaluation_config = _get_evaluation_config(input_config)

        with patch("alpaca_eval.evaluate", _mock_alpaca_eval_evaluate):
            results_list = evaluate(evaluation_config)
            assert len(results_list) == 1  # 1 task was evaluated.
            results_dict = results_list[0]["results"]

        _validate_results_returned(
            expected_results=EXPECTED_RESULTS, actual_results=results_dict
        )
        _validate_results_in_file(
            expected_results=EXPECTED_RESULTS, output_dir=nested_output_dir
        )
