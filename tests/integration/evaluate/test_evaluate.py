import json
import os
import tempfile
import unittest

from oumi import evaluate
from oumi.core.configs import (
    EvaluationConfig,
    EvaluationTaskParams,
    GenerationParams,
    ModelParams,
)
from tests.markers import requires_gpus

TASK = "mmlu_abstract_algebra"
NUM_SAMPLES = 20


@requires_gpus()
def test_evaluate_lm_harness():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        nested_output_dir = os.path.join(output_temp_dir, "nested", "dir")

        config: EvaluationConfig = EvaluationConfig(
            output_dir=nested_output_dir,
            tasks=[
                EvaluationTaskParams(
                    evaluation_platform="lm_harness",
                    task_name=TASK,
                    num_samples=NUM_SAMPLES,
                )
            ],
            model=ModelParams(
                model_name="openai-community/gpt2",
                trust_remote_code=True,
            ),
        )

        evaluate(config)

        # Identify the relevant output file: "lm_harness_<timestamp>_results.json"
        files = [f for f in os.listdir(nested_output_dir) if f.endswith("results.json")]
        assert len(files) == 1
        output_file = os.path.join(nested_output_dir, files[0])

        with open(output_file, encoding="utf-8") as f:
            computed_metrics = json.load(f)
            computed_metrics = computed_metrics["results"][TASK]
            # expected metrics:
            # {
            #   "acc,none": 0.2,
            #   "acc_stderr,none": 0.09176629354822471,
            #   "alias": "abstract_algebra"
            # }
            assert round(computed_metrics["acc,none"], 3) == 0.2
            assert round(computed_metrics["acc_stderr,none"], 3) == 0.092


@requires_gpus()
@unittest.skip("Skipping until we mock annotation (to avoid the cost of calling GPT4)")
def test_evaluate_alpaca_eval():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        nested_output_dir = os.path.join(output_temp_dir, "nested", "dir")

        config: EvaluationConfig = EvaluationConfig(
            output_dir=nested_output_dir,
            tasks=[
                EvaluationTaskParams(
                    evaluation_platform="alpaca_eval",
                    num_samples=3,
                )
            ],
            model=ModelParams(
                model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
                model_max_length=10,
                trust_remote_code=True,
            ),
            generation=GenerationParams(
                max_new_tokens=10,
                batch_size=1,
            ),
        )

        evaluate(config)

        # Identify the relevant output file: "alpaca_eval_<timestamp>_results.json"
        files = [f for f in os.listdir(nested_output_dir) if f.endswith("results.json")]
        assert len(files) == 1
        output_file = os.path.join(nested_output_dir, files[0])

        with open(output_file, encoding="utf-8") as f:
            computed_metrics = json.load(f)
            computed_metrics = computed_metrics["results"]
            # expected metrics:
            # {
            #   "win_rate": 2.1546666665687532e-05,
            #   "standard_error": 8.230602115518512e-06,
            #   "n_wins": 0,
            #   "n_wins_base": 3,
            #   "n_draws": 0,
            #   "n_total": 3,
            #   "discrete_win_rate": 0.0,
            #   "mode": "community",
            #   "avg_length": 44,
            #   "length_controlled_winrate": 0.05370406224906349,
            #   "lc_standard_error": 0.014120007914531866,
            # }
            assert round(computed_metrics["length_controlled_winrate"], 3) == 0.054
            assert round(computed_metrics["lc_standard_error"], 3) == 0.014
            assert round(computed_metrics["win_rate"], 7) == 0.0000215
            assert round(computed_metrics["standard_error"], 7) == 0.0000082
            assert computed_metrics["avg_length"] == 44
            assert computed_metrics["n_wins"] == 0
            assert computed_metrics["n_draws"] == 0
            assert computed_metrics["n_total"] == 3
