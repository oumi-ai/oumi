import json
import os
import tempfile

from lema import evaluate
from lema.core.types import (
    DataParams,
    DatasetParams,
    DatasetSplitParams,
    EvaluationConfig,
    ModelParams,
)


def test_evaluate_basic():
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
        )

        evaluate(config, num_entries=4)
        with open(os.path.join(nested_output_dir, "eval.json"), "r") as f:
            computed_metrics = json.load(f)
            assert computed_metrics["mmlu"]["accuracy"] == 0
