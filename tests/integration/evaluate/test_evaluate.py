import json
import os
import tempfile

from oumi import evaluate_lm_harness
from oumi.core.configs import (
    EvaluationConfig,
    LMHarnessParams,
    ModelParams,
)
from oumi.evaluate import SAVE_FILENAME_JSON
from tests.markers import requires_gpus


@requires_gpus()
def test_evaluate_lm_harness():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        nested_output_dir = os.path.join(output_temp_dir, "nested", "dir")
        output_file = os.path.join(
            nested_output_dir, SAVE_FILENAME_JSON.format(benchmark_name="mmlu")
        )

        config: EvaluationConfig = EvaluationConfig(
            output_dir=nested_output_dir,
            lm_harness_params=LMHarnessParams(
                tasks=["mmlu"],
                num_samples=4,
            ),
            model=ModelParams(
                model_name="openai-community/gpt2",
                trust_remote_code=True,
            ),
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
