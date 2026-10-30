from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from oumi.datasets.grpo.rewards.rar_medicine_reward import (
    rar_medicine_trl,
    score_rar_medicine_batch,
)

_JUDGE = "oumi.datasets.grpo.rewards.rar_medicine_reward"


def _output(judgment):
    return SimpleNamespace(field_values={"judgment": judgment})


def test_batch_scores_every_row_with_one_judge_call():
    judge = Mock()
    judge.judge.return_value = [_output(3), _output(7), _output(9)]

    with patch(f"{_JUDGE}._get_judge", return_value=judge):
        scores = score_rar_medicine_batch(
            ["q1", "q2", "q3"], ["a1", "a2", "a3"], ["r1", "r2", "r3"]
        )

    assert scores == [0.3, 0.7, 0.9]
    judge.judge.assert_called_once_with(
        [
            {"question": "q1", "reference_answer": "a1", "response": "r1"},
            {"question": "q2", "reference_answer": "a2", "response": "r2"},
            {"question": "q3", "reference_answer": "a3", "response": "r3"},
        ]
    )


def test_batch_clamps_out_of_range_judgments():
    judge = Mock()
    judge.judge.return_value = [_output(-4), _output(14)]

    with patch(f"{_JUDGE}._get_judge", return_value=judge):
        assert score_rar_medicine_batch(["q", "q"], ["a", "a"], ["r", "r"]) == [
            0.0,
            1.0,
        ]


def test_batch_scores_an_empty_response_zero_without_asking_the_judge():
    judge = Mock()
    judge.judge.return_value = [_output(9)]

    with patch(f"{_JUDGE}._get_judge", return_value=judge):
        scores = score_rar_medicine_batch(["q", "q"], ["a", "a"], ["  ", "r"])

    assert scores == [0.0, 0.9]
    assert judge.judge.call_count == 1
    assert [entry["response"] for entry in judge.judge.call_args.args[0]] == ["r"]


def test_batch_retries_only_the_rows_that_failed_to_parse():
    judge = Mock()
    judge.judge.side_effect = [
        [_output(None), _output(6)],
        [_output(8)],
    ]

    with patch(f"{_JUDGE}._get_judge", return_value=judge):
        scores = score_rar_medicine_batch(["q", "q"], ["a", "a"], ["r1", "r2"])

    assert scores == [0.8, 0.6]
    assert judge.judge.call_count == 2
    assert [entry["response"] for entry in judge.judge.call_args.args[0]] == ["r1"]


def test_batch_gives_zero_when_every_attempt_fails():
    judge = Mock()
    judge.judge.side_effect = RuntimeError("judge is down")

    with (
        patch(f"{_JUDGE}._get_judge", return_value=judge),
        patch(f"{_JUDGE}._MAX_ATTEMPTS", 3),
        patch(f"{_JUDGE}.time.sleep"),
    ):
        scores = score_rar_medicine_batch(["q", "q"], ["a", "a"], ["r1", "r2"])

    assert scores == [0.0, 0.0]
    assert judge.judge.call_count == 3


def test_batch_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="must have the same length"):
        score_rar_medicine_batch(["q"], ["a"], ["r1", "r2"])


def test_trl_reward_reads_text_from_message_lists():
    judge = Mock()
    judge.judge.return_value = [_output(4), _output(5)]

    with patch(f"{_JUDGE}._get_judge", return_value=judge):
        scores = rar_medicine_trl(
            completions=[
                [{"role": "assistant", "content": "first"}],
                [{"role": "assistant", "content": "second"}],
            ],
            reference_answer=["a1", "a2"],
            question=["q1", "q2"],
        )

    assert scores == [0.4, 0.5]
    assert [entry["response"] for entry in judge.judge.call_args.args[0]] == [
        "first",
        "second",
    ]


def test_trl_reward_accepts_plain_strings():
    judge = Mock()
    judge.judge.return_value = [_output(7)]

    with patch(f"{_JUDGE}._get_judge", return_value=judge):
        scores = rar_medicine_trl(completions=["answer"], reference_answer=["a"])

    assert scores == [0.7]
    assert judge.judge.call_args.args[0][0]["question"] == ""
