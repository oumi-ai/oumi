from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from oumi.datasets.grpo.rewards.rar_medicine_reward import (
    rar_medicine_verl,
    score_rar_medicine_response,
)


@pytest.mark.parametrize(
    ("judgment", "expected"),
    [
        (-1, 0.0),
        (7, 0.7),
        (12, 1.0),
    ],
)
def test_score_rar_medicine_response_normalizes_judgment(judgment, expected):
    judge = Mock()
    judge.judge.return_value = [SimpleNamespace(field_values={"judgment": judgment})]

    with (
        patch(
            "oumi.datasets.grpo.rewards.rar_medicine_reward._get_judge",
            return_value=judge,
        ),
        patch("oumi.datasets.grpo.rewards.rar_medicine_reward._MAX_ATTEMPTS", 1),
    ):
        score = score_rar_medicine_response(
            "question", "reference", "response", "judge.yaml"
        )

    assert score == expected
    judge.judge.assert_called_once_with(
        [
            {
                "question": "question",
                "reference_answer": "reference",
                "response": "response",
            }
        ]
    )


def test_score_rar_medicine_response_returns_zero_after_failure():
    with (
        patch(
            "oumi.datasets.grpo.rewards.rar_medicine_reward._get_judge",
            side_effect=RuntimeError("judge unavailable"),
        ),
        patch("oumi.datasets.grpo.rewards.rar_medicine_reward._MAX_ATTEMPTS", 1),
    ):
        assert (
            score_rar_medicine_response(
                "question", "reference", "response", "judge.yaml"
            )
            == 0
        )


def test_rar_medicine_verl_skips_empty_response():
    with patch(
        "oumi.datasets.grpo.rewards.rar_medicine_reward.score_rar_medicine_response"
    ) as score:
        reward = rar_medicine_verl(
            data_source="anisha2102/RaR-Medicine",
            solution_str="  ",
            ground_truth="reference",
            extra_info={"question": "question"},
        )

    assert reward == 0
    score.assert_not_called()


def test_rar_medicine_verl_forwards_sample_fields():
    with patch(
        "oumi.datasets.grpo.rewards.rar_medicine_reward.score_rar_medicine_response",
        return_value=0.8,
    ) as score:
        reward = rar_medicine_verl(
            data_source="anisha2102/RaR-Medicine",
            solution_str="response",
            ground_truth="reference",
            extra_info={"question": "question"},
            judge_config_path="custom/judge.yaml",
        )

    assert reward == 0.8
    score.assert_called_once_with(
        "question",
        "reference",
        "response",
        "custom/judge.yaml",
    )
