import json
import os
import tempfile

from oumi import evaluate_lm_harness
from oumi.core.configs import (
    EvaluationConfig,
    LMHarnessParams,
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
            lm_harness_params=LMHarnessParams(
                tasks=[TASK],
                num_samples=NUM_SAMPLES,
            ),
            model=ModelParams(
                model_name="openai-community/gpt2",
                trust_remote_code=True,
            ),
        )

        evaluate_lm_harness(config)

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
