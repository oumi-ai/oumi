import pytest

from lema import evaluate
from lema.core.types import DataParams, EvaluationConfig, ModelParams


def test_basic_evaluate():
    config: EvaluationConfig = EvaluationConfig(
        data=DataParams(
            dataset_name="yahma/alpaca-cleaned",
            preprocessing_function_name="alpaca",
        ),
        model=ModelParams(
            model_name="openai-community/gpt2",
            trust_remote_code=True,
        ),
    )

    with pytest.raises(NotImplementedError) as exception_info:
        evaluate(config)

    assert str(exception_info.value) == ("Model evaluation is not implemented yet")
