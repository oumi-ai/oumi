import json
import os
import tempfile

from lema import evaluate_custom, evaluate_lm_harmess
from lema.core.types import (
    DataParams,
    DatasetParams,
    DatasetSplitParams,
    EvaluationConfig,
    ModelParams,
)


def test_evaluate_custom():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        nested_output_dir = os.path.join(output_temp_dir, "nested", "dir")
        config: EvaluationConfig = EvaluationConfig(
            output_dir=nested_output_dir,
            data=DataParams(
                validation=DatasetSplitParams(
                    datasets=[
                        DatasetParams(
                            dataset_name="cais/mmlu",
                        )
                    ],
                    target_col="text",
                ),
            ),
            model=ModelParams(
                model_name="openai-community/gpt2",
                trust_remote_code=True,
            ),
            evaluation_framework="custom",
        )

        evaluate_custom(config, num_entries=4)
        with open(os.path.join(nested_output_dir, "eval.json"), "r") as f:
            computed_metrics = json.load(f)
            # expected metrics:
            # {'cais/mmlu': {'accuracy': 0.0}}
            assert computed_metrics["cais/mmlu"]["accuracy"] == 0.0


def test_evaluate_lm_harmess():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        nested_output_dir = os.path.join(output_temp_dir, "nested", "dir")
        config: EvaluationConfig = EvaluationConfig(
            output_dir=nested_output_dir,
            data=DataParams(
                validation=DatasetSplitParams(
                    datasets=[
                        DatasetParams(
                            dataset_name="mmlu",
                        )
                    ],
                ),
            ),
            model=ModelParams(
                model_name="openai-community/gpt2",
            ),
            evaluation_framework="lm_harmess",
        )

        evaluate_lm_harmess(config, num_entries=4)
        with open(os.path.join(nested_output_dir, "eval.json"), "r") as f:
            computed_metrics = json.load(f)
            # expected metrics:
            # {
            #    'mmlu':
            #    {
            #       'acc,none': 0.2850877192982456,
            #       'acc_stderr,none': 0.029854295639440784,
            #       'alias': 'mmlu'
            #    }
            # }
            assert round(computed_metrics["mmlu"]["acc,none"], 2) == 0.29
            assert round(computed_metrics["mmlu"]["acc_stderr,none"], 2) == 0.03
