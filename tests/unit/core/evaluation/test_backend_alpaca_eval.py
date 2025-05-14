from unittest.mock import patch

import pandas as pd

from oumi.core.configs import (
    AlpacaEvalTaskParams,
    EvaluationConfig,
    GenerationParams,
    ModelParams,
)
from oumi.core.evaluation.backends.alpaca_eval import ALPACA_EVAL_TASK_NAME, evaluate
from oumi.core.evaluation.evaluation_result import EvaluationResult
from oumi.core.types.conversation import (
    Conversation,
    Message,
    Role,
)


# Mocks
class _MockAlpacaEvalDataset:
    def __init__(self, dataset_name):
        pass

    def conversations(self) -> list[Conversation]:
        return [
            Conversation(
                messages=[
                    Message(role=Role.USER, content="Hello 1.1"),
                ],
            ),
            Conversation(
                messages=[
                    Message(role=Role.USER, content="Hello 2.1"),
                ],
            ),
        ]


class _MockInferenceEngine:
    _model_params = {}
    _generation_params = {}

    def __init__(self):
        pass

    def infer(self, input) -> list[Conversation]:
        assert str(input) == "[USER: Hello 1.1, USER: Hello 2.1]"
        return [
            Conversation(
                messages=[
                    Message(role=Role.USER, content="Hello 1.1"),
                    Message(role=Role.ASSISTANT, content="Hello 1.2"),
                ],
            ),
            Conversation(
                messages=[
                    Message(role=Role.USER, content="Hello 2.1"),
                    Message(role=Role.ASSISTANT, content="Hello 2.2"),
                ],
            ),
        ]


class _MockAlpacaEval:
    @staticmethod
    def evaluate(
        model_outputs,
        annotators_config,
        fn_metric,
        is_return_instead_of_print,
        is_overwrite_leaderboard,
        max_instances,
        sort_by,
    ) -> tuple[pd.DataFrame, None]:
        # Ensure the input arguments are the defaults (unless changed by this test).
        assert annotators_config == "weighted_alpaca_eval_gpt4_turbo"
        assert fn_metric == "get_length_controlled_winrate"
        assert max_instances == 111

        # Ensure the inference results (`model_outputs`) are the expected ones.
        assert len(model_outputs["output"]) == 2
        assert model_outputs["output"][0] == "Hello 1.2"
        assert model_outputs["output"][1] == "Hello 2.2"

        # Mock the `alpaca_eval.evaluate` function.
        df_leaderboard = pd.DataFrame(
            {"win_rate": 0.23},
            index=["my_run_name"],  # type: ignore
        )
        return df_leaderboard, None


def test_evaluate_alpaca_eval():
    with (
        patch("oumi.core.evaluation.backends.alpaca_eval.alpaca_eval", _MockAlpacaEval),
        patch(
            "oumi.core.evaluation.backends.alpaca_eval.AlpacaEvalDataset",
            _MockAlpacaEvalDataset,
        ),
    ):
        result: EvaluationResult = evaluate(
            task_params=AlpacaEvalTaskParams(
                evaluation_backend="alpaca_eval",
                num_samples=111,
                version=2.0,
            ),
            config=EvaluationConfig(
                output_dir="",
                tasks=[],
                model=ModelParams(model_name="my_model"),
                generation=GenerationParams(),
                run_name="my_run_name",
            ),
            inference_engine=_MockInferenceEngine(),  # type: ignore
        )
        result_metrics = result.task_result["results"][ALPACA_EVAL_TASK_NAME]
        assert result_metrics == {"win_rate": 0.23}
