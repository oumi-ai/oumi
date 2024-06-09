from lema import evaluate
from lema.core.types import DataParams, EvaluationConfig, ModelParams


def test_evaluate_basic():
    config: EvaluationConfig = EvaluationConfig(
        data=DataParams(dataset_name="cais/mmlu", split="validation"),
        model=ModelParams(
            model_name="openai-community/gpt2",
            trust_remote_code=True,
        ),
    )

    evaluate(config)
