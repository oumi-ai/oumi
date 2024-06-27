from lema import evaluate
from lema.core.types import (
    DataParams,
    DatasetParams,
    DatasetSplitParams,
    EvaluationConfig,
    ModelParams,
)


def test_evaluate_basic():
    config: EvaluationConfig = EvaluationConfig(
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
