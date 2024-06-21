from lema import evaluate
from lema.core.types import (
    DatasetParams,
    DatasetSplitParams,
    EvaluationConfig,
    ModelParams,
)


def test_evaluate_basic():
    config: EvaluationConfig = EvaluationConfig(
        data=DatasetSplitParams(
            datasets=[DatasetParams(dataset_name="cais/mmlu", split="validation")]
        ),
        model=ModelParams(
            model_name="openai-community/gpt2",
            trust_remote_code=True,
        ),
    )

    evaluate(config)
