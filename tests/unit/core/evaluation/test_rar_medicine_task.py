# Copyright 2026 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from oumi.core.types.conversation import Conversation
from oumi.evaluation.registry.rar_medicine_task import (
    CORRECT_CONCLUSION_THRESHOLD,
    aggregate_rar_scores,
    build_rar_judge_inputs,
    is_valid_judgment,
    select_conversations,
    validate_rar_conversations,
)


def _conversation(
    conversation_id: str,
    *,
    question: str = "What is the diagnosis?",
    response: str | None = "The final answer is X.",
    reference_answer: str | None = "X, because of Y.",
    question_source: str = "source_a",
) -> Conversation:
    messages = [
        {"role": "system", "content": "You are a medical expert."},
        {"role": "user", "content": question},
    ]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    metadata = {
        "idx": int(conversation_id.rsplit("_", 1)[-1]),
        "question_source": question_source,
    }
    if reference_answer is not None:
        metadata["reference_answer"] = reference_answer
    return Conversation.from_dict(
        {"conversation_id": conversation_id, "messages": messages, "metadata": metadata}
    )


@pytest.mark.parametrize("judgment", [0, 4, 10])
def test_is_valid_judgment_accepts_integers_on_the_scale(judgment):
    assert is_valid_judgment(judgment)


@pytest.mark.parametrize("judgment", [-1, 11, None, "7", 7.0, True])
def test_is_valid_judgment_rejects_out_of_range_and_non_integers(judgment):
    assert not is_valid_judgment(judgment)


def test_build_rar_judge_inputs_maps_question_reference_and_response():
    conversations = [
        _conversation("rar_0001", question="Q1", response="R1", reference_answer="A1"),
        _conversation("rar_0002", question="Q2", response="R2", reference_answer="A2"),
    ]

    judge_inputs = build_rar_judge_inputs(conversations)

    assert judge_inputs == [
        {"question": "Q1", "reference_answer": "A1", "response": "R1"},
        {"question": "Q2", "reference_answer": "A2", "response": "R2"},
    ]


def test_validate_rar_conversations_accepts_complete_rows():
    validate_rar_conversations([_conversation("rar_0001"), _conversation("rar_0002")])


def test_validate_rar_conversations_rejects_missing_assistant_turn():
    with pytest.raises(ValueError, match="assistant response"):
        validate_rar_conversations([_conversation("rar_0001", response=None)])


def test_validate_rar_conversations_rejects_missing_reference_answer():
    with pytest.raises(ValueError, match="reference_answer"):
        validate_rar_conversations([_conversation("rar_0001", reference_answer=None)])


def test_validate_rar_conversations_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="Duplicate conversation_id"):
        validate_rar_conversations(
            [_conversation("rar_0001"), _conversation("rar_0001")]
        )


def test_validate_rar_conversations_rejects_empty_input():
    with pytest.raises(ValueError, match="No conversations"):
        validate_rar_conversations([])


def test_select_conversations_returns_all_sorted_when_num_samples_is_none():
    conversations = [
        _conversation("rar_0003"),
        _conversation("rar_0001"),
        _conversation("rar_0002"),
    ]

    selected = select_conversations(conversations, None, seed=0)

    assert [c.conversation_id for c in selected] == ["rar_0001", "rar_0002", "rar_0003"]


def test_select_conversations_is_deterministic_and_order_independent():
    ids = [f"rar_{i:04d}" for i in range(20)]
    forward = [_conversation(i) for i in ids]
    shuffled = [_conversation(i) for i in reversed(ids)]

    first = select_conversations(forward, 5, seed=7)
    second = select_conversations(shuffled, 5, seed=7)

    assert len(first) == 5
    assert [c.conversation_id for c in first] == [c.conversation_id for c in second]
    assert [c.conversation_id for c in first] == sorted(
        str(c.conversation_id) for c in first
    )


def test_select_conversations_rejects_oversized_requests():
    with pytest.raises(ValueError, match="only 1 are available"):
        select_conversations([_conversation("rar_0001")], 2, seed=0)


def test_aggregate_rar_scores_reports_scale_histogram_and_sources():
    sample_results = [
        {"judgment": 10, "question_source": "a", "judged": True},
        {"judgment": 4, "question_source": "a", "judged": True},
        {"judgment": 3, "question_source": "b", "judged": True},
        {"judgment": 0, "question_source": "b", "judged": False},
    ]

    summary = aggregate_rar_scores(sample_results, num_bootstrap_samples=50)

    assert summary["mean_score"] == pytest.approx(17 / 40)
    assert summary["mean_judgment"] == pytest.approx(17 / 4)
    assert summary["judgment_histogram"] == {
        str(v): (1 if v in {0, 3, 4, 10} else 0) for v in range(11)
    }
    assert summary["frac_correct_conclusion"] == pytest.approx(0.5)
    assert summary["correct_conclusion_threshold"] == CORRECT_CONCLUSION_THRESHOLD
    assert summary["num_samples"] == 4
    assert summary["num_blank_responses"] == 1
    assert summary["by_question_source"]["a"]["mean_score"] == pytest.approx(0.7)
    assert summary["by_question_source"]["a"]["num_samples"] == 2
    assert summary["by_question_source"]["b"]["mean_score"] == pytest.approx(0.15)


def test_aggregate_rar_scores_bootstrap_std_is_zero_for_constant_scores():
    sample_results = [{"judgment": 6, "question_source": "a"} for _ in range(5)]

    summary = aggregate_rar_scores(sample_results, num_bootstrap_samples=100)

    assert summary["mean_score_bootstrap_std"] == 0.0
    assert summary["by_question_source"]["a"]["bootstrap_std"] == 0.0


def test_aggregate_rar_scores_rejects_out_of_range_judgments():
    with pytest.raises(ValueError, match="out of range"):
        aggregate_rar_scores([{"judgment": 11, "question_source": "a"}])
