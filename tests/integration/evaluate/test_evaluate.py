import json
import os
import tempfile

import pytest
import torch

from oumi import evaluate_lema, evaluate_lm_harness
from oumi.core.configs import (
    DatasetParams,
    DatasetSplitParams,
    EvaluationConfig,
    EvaluationFramework,
    ModelParams,
)
from oumi.evaluate import SAVE_FILENAME_JSON


def test_evaluate_lema():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        nested_output_dir = os.path.join(output_temp_dir, "nested", "dir")
        output_file = os.path.join(
            nested_output_dir, SAVE_FILENAME_JSON.format(benchmark_name="mmlu")
        )

        config: EvaluationConfig = EvaluationConfig(
            output_dir=nested_output_dir,
            data=DatasetSplitParams(
                datasets=[
                    DatasetParams(
                        dataset_name="cais/mmlu",
                    )
                ],
                target_col="text",
            ),
            model=ModelParams(
                model_name="openai-community/gpt2",
                trust_remote_code=True,
            ),
            evaluation_framework=EvaluationFramework.LEMA,
            num_samples=4,
        )

        evaluate_lema(config)
        with open(output_file, encoding="utf-8") as f:
            computed_metrics = json.load(f)
            # expected metrics: {"accuracy": 0.0}
            assert computed_metrics["accuracy"] == 0.0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Skip test if GPU is not available."
)
def test_evaluate_lm_harness():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        nested_output_dir = os.path.join(output_temp_dir, "nested", "dir")
        output_file = os.path.join(
            nested_output_dir, SAVE_FILENAME_JSON.format(benchmark_name="mmlu")
        )

        config: EvaluationConfig = EvaluationConfig(
            output_dir=nested_output_dir,
            data=DatasetSplitParams(
                datasets=[
                    DatasetParams(
                        dataset_name="mmlu",
                    )
                ],
            ),
            model=ModelParams(
                model_name="openai-community/gpt2",
                trust_remote_code=True,
            ),
            evaluation_framework=EvaluationFramework.LM_HARNESS,
            num_samples=4,
        )

        evaluate_lm_harness(config)
        with open(output_file, encoding="utf-8") as f:
            computed_metrics = json.load(f)
            # expected metrics:
            # {
            #   "acc,none": 0.2850877192982456,
            #   "acc_stderr,none": 0.029854295639440784,
            #   "alias": "mmlu"
            # }
            assert round(computed_metrics["acc,none"], 3) == 0.285
            assert round(computed_metrics["acc_stderr,none"], 3) == 0.030
