from unittest.mock import Mock, patch

import pandas as pd
import pytest

from oumi.builders.data import build_dataset_mixture
from oumi.core.configs import DataParams, DatasetParams, DatasetSplitParams, JudgeConfig
from oumi.core.configs.params.data_params import DatasetSplit
from oumi.core.inference import FailureDetail, InferenceErrorType
from oumi.datasets.grpo.rar_dataset import RaRMedicineDataset
from oumi.datasets.grpo.rewards.rar_medicine_reward import (
    rar_medicine_trl,
    score_rar_medicine_batch,
)
from oumi.judges.base_judge import JudgeOutput, JudgePartialResult

_JUDGE = "oumi.datasets.grpo.rewards.rar_medicine_reward"
_JUDGE_CONFIG = "configs/examples/grpo_verl_medqa/judge.yaml"


def _result(successful, failures=()):
    return JudgePartialResult(
        successful=[
            (index, JudgeOutput(raw_output="", field_values={"judgment": judgment}))
            for index, judgment in successful
        ],
        failures={
            index: FailureDetail(
                error_message="inference failed",
                error_type=InferenceErrorType.API_STATUS,
            )
            for index in failures
        },
    )


def test_batch_scores_every_row_with_one_judge_call():
    judge = Mock()
    judge.judge_partial.return_value = _result([(0, 3), (1, 7), (2, 9)])

    with patch(f"{_JUDGE}._get_judge", return_value=judge):
        scores = score_rar_medicine_batch(
            ["q1", "q2", "q3"], ["a1", "a2", "a3"], ["r1", "r2", "r3"]
        )

    assert scores == [0.3, 0.7, 0.9]
    judge.judge_partial.assert_called_once_with(
        [
            {"question": "q1", "reference_answer": "a1", "response": "r1"},
            {"question": "q2", "reference_answer": "a2", "response": "r2"},
            {"question": "q3", "reference_answer": "a3", "response": "r3"},
        ]
    )


def test_batch_keeps_other_scores_when_one_row_fails():
    """A single failed row must not zero the whole group, as judge() would."""
    judge = Mock()
    judge.judge_partial.side_effect = [
        _result([(0, 8), (2, 8)], failures=[1]),
        _result([(0, 7)]),
    ]

    with patch(f"{_JUDGE}._get_judge", return_value=judge):
        scores = score_rar_medicine_batch(
            ["q", "q", "q"], ["a", "a", "a"], ["r1", "r2", "r3"]
        )

    assert scores == [0.8, 0.7, 0.8]
    assert judge.judge_partial.call_count == 2
    assert [entry["response"] for entry in judge.judge_partial.call_args.args[0]] == [
        "r2"
    ]


def test_batch_clamps_out_of_range_judgments():
    judge = Mock()
    judge.judge_partial.return_value = _result([(0, -4), (1, 14)])

    with patch(f"{_JUDGE}._get_judge", return_value=judge):
        assert score_rar_medicine_batch(["q", "q"], ["a", "a"], ["r", "r"]) == [
            0.0,
            1.0,
        ]


def test_batch_scores_an_empty_response_zero_without_asking_the_judge():
    judge = Mock()
    judge.judge_partial.return_value = _result([(0, 9)])

    with patch(f"{_JUDGE}._get_judge", return_value=judge):
        scores = score_rar_medicine_batch(["q", "q"], ["a", "a"], ["  ", "r"])

    assert scores == [0.0, 0.9]
    assert judge.judge_partial.call_count == 1
    assert [entry["response"] for entry in judge.judge_partial.call_args.args[0]] == [
        "r"
    ]


def test_batch_retries_only_the_rows_that_failed_to_parse():
    judge = Mock()
    judge.judge_partial.side_effect = [
        _result([(1, 6)]),
        _result([(0, 8)]),
    ]

    with patch(f"{_JUDGE}._get_judge", return_value=judge):
        scores = score_rar_medicine_batch(["q", "q"], ["a", "a"], ["r1", "r2"])

    assert scores == [0.8, 0.6]
    assert judge.judge_partial.call_count == 2
    assert [entry["response"] for entry in judge.judge_partial.call_args.args[0]] == [
        "r1"
    ]


def test_batch_retries_a_judgment_that_is_not_a_number():
    """A FLOAT or TEXT judge config must retry, not crash the training step."""
    judge = Mock()
    judge.judge_partial.side_effect = [
        _result([(0, "not a number"), (1, 4)]),
        _result([(0, 6)]),
    ]

    with patch(f"{_JUDGE}._get_judge", return_value=judge):
        scores = score_rar_medicine_batch(["q", "q"], ["a", "a"], ["r1", "r2"])

    assert scores == [0.6, 0.4]


def test_batch_gives_zero_when_every_attempt_fails():
    judge = Mock()
    judge.judge_partial.side_effect = RuntimeError("judge is down")

    with (
        patch(f"{_JUDGE}._get_judge", return_value=judge),
        patch(f"{_JUDGE}._MAX_ATTEMPTS", 3),
        patch(f"{_JUDGE}.time.sleep"),
    ):
        scores = score_rar_medicine_batch(["q", "q"], ["a", "a"], ["r1", "r2"])

    assert scores == [0.0, 0.0]
    assert judge.judge_partial.call_count == 3


def test_batch_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="must have the same length"):
        score_rar_medicine_batch(["q"], ["a"], ["r1", "r2"])


def test_trl_reward_reads_text_from_message_lists():
    judge = Mock()
    judge.judge_partial.return_value = _result([(0, 4), (1, 5)])

    with patch(f"{_JUDGE}._get_judge", return_value=judge):
        scores = rar_medicine_trl(
            prompts=["q1", "q2"],
            completions=[
                [{"role": "assistant", "content": "first"}],
                [{"role": "assistant", "content": "second"}],
            ],
            metadata=[{"reference_answer": "a1"}, {"reference_answer": "a2"}],
        )

    assert scores == [0.4, 0.5]
    assert [entry["response"] for entry in judge.judge_partial.call_args.args[0]] == [
        "first",
        "second",
    ]


def test_trl_reward_reads_the_question_from_conversational_prompts():
    judge = Mock()
    judge.judge_partial.return_value = _result([(0, 7)])

    with patch(f"{_JUDGE}._get_judge", return_value=judge):
        scores = rar_medicine_trl(
            prompts=[[{"role": "user", "content": "What is the diagnosis?"}]],
            completions=["Appendicitis."],
            metadata=[{"reference_answer": "Appendicitis"}],
        )

    assert scores == [0.7]
    assert judge.judge_partial.call_args.args[0][0]["question"] == (
        "What is the diagnosis?"
    )


def test_trl_reward_accepts_a_top_level_reference_answer_column():
    judge = Mock()
    judge.judge_partial.return_value = _result([(0, 7)])

    with patch(f"{_JUDGE}._get_judge", return_value=judge):
        scores = rar_medicine_trl(
            prompts=["question"],
            completions=["answer"],
            reference_answer=["reference"],
        )

    assert scores == [0.7]
    assert judge.judge_partial.call_args.args[0][0]["reference_answer"] == "reference"


def test_trl_reward_runs_on_the_rar_medicine_dataset_columns():
    """Call the reward with the column layout RaRMedicineDataset actually ships."""
    raw_data = pd.DataFrame(
        [
            {
                "question": "What is the diagnosis?",
                "reference_answer": "Appendicitis",
                "question_source": "medqa",
                "rubric_count": 1,
                "rubric": [
                    {"title": "Diagnosis", "description": "Names it.", "weight": 5}
                ],
            }
        ]
    )
    with patch.object(RaRMedicineDataset, "_load_data", return_value=raw_data):
        dataset = RaRMedicineDataset(split="train")
    row = dataset[0]

    # TRL passes "prompt" as `prompts` and every other column by its own name.
    assert sorted(row) == ["metadata", "prompt", "rubrics"]

    judge = Mock()
    judge.judge_partial.return_value = _result([(0, 9)])
    with patch(f"{_JUDGE}._get_judge", return_value=judge):
        scores = rar_medicine_trl(
            prompts=[row["prompt"]],
            completions=[[{"role": "assistant", "content": "Appendicitis."}]],
            rubrics=[row["rubrics"]],
            metadata=[row["metadata"]],
            trainer_state=None,
        )

    assert scores == [0.9]
    judge.judge_partial.assert_called_once_with(
        [
            {
                "question": "What is the diagnosis?",
                "reference_answer": "Appendicitis",
                "response": "Appendicitis.",
            }
        ]
    )


def test_trl_reward_rejects_a_batch_without_reference_answers():
    with pytest.raises(ValueError, match="reference answer"):
        rar_medicine_trl(prompts=["question"], completions=["answer"])


def test_trl_reward_rejects_metadata_without_a_reference_answer():
    with pytest.raises(ValueError, match="reference_answer"):
        rar_medicine_trl(
            prompts=["question"],
            completions=["answer"],
            metadata=[{"question_source": "medqa"}],
        )


def test_trl_reward_rejects_a_batch_without_prompts():
    with pytest.raises(ValueError, match="prompts"):
        rar_medicine_trl(
            completions=["answer"],
            metadata=[{"reference_answer": "Appendicitis"}],
        )


def test_judge_config_enables_batch_concurrency():
    """num_workers defaults to 1, which would serialize the batched call."""
    config = JudgeConfig.from_path(_JUDGE_CONFIG)
    assert config.inference_config is not None
    remote_params = config.inference_config.remote_params
    assert remote_params is not None
    assert remote_params.num_workers > 1


def test_documented_data_block_loads_the_hub_dataset():
    """The docs' `data:` block must resolve to the Hub repo, not to `rar-medicine`.

    `_load_dataset` passes `dataset_name` straight to the dataset class, so a
    registry name like `rar-medicine` becomes the Hub repo id unless
    `dataset_name_override` supplies the real one. This pins the documented
    block against that mistake.
    """
    params = DatasetParams(
        dataset_name="rar-medicine",
        dataset_kwargs={"dataset_name_override": "anisha2102/RaR-Medicine"},
        split="train",
    )
    data_params = DataParams(train=DatasetSplitParams(datasets=[params]))

    with patch("datasets.load_dataset") as load_dataset:
        load_dataset.side_effect = RuntimeError("stop after resolving the repo id")
        with pytest.raises(RuntimeError, match="stop after resolving"):
            build_dataset_mixture(
                data_params, tokenizer=None, dataset_split=DatasetSplit.TRAIN
            )

    assert load_dataset.call_args.kwargs["path"] == "anisha2102/RaR-Medicine"
