# Copyright 2025 - Oumi
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
from oumi.evaluation.registry.healthbench_task import (
    aggregate_healthbench_scores,
    build_healthbench_judge_inputs,
    calculate_healthbench_score,
)


def test_calculate_healthbench_score_matches_reference_formula():
    rubrics = [
        {"criterion": "a", "points": 7, "tags": []},
        {"criterion": "b", "points": 5, "tags": []},
        {"criterion": "c", "points": 10, "tags": []},
        {"criterion": "bad", "points": -6, "tags": []},
    ]

    score = calculate_healthbench_score(rubrics, [True, False, True, True])

    assert score == pytest.approx((7 + 10 - 6) / (7 + 5 + 10))


def test_calculate_healthbench_score_requires_positive_points():
    rubrics = [{"criterion": "bad", "points": -5, "tags": []}]
    assert calculate_healthbench_score(rubrics, [True]) is None


def test_build_healthbench_judge_inputs_flattens_per_sample_rubrics():
    conversation = Conversation.from_dict(
        {
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ],
            "metadata": {
                "prompt_id": "id",
                "example_tags": ["theme:test"],
                "rubrics": [
                    {"criterion": "Criterion A", "points": 7, "tags": []},
                    {"criterion": "Criterion B", "points": -2, "tags": []},
                ],
            },
        }
    )

    inputs, locations = build_healthbench_judge_inputs([conversation])

    assert locations == [(0, 0), (0, 1)]
    assert inputs == [
        {
            "conversation": "user: Question\n\nassistant: Answer",
            "rubric_item": "[7] Criterion A",
        },
        {
            "conversation": "user: Question\n\nassistant: Answer",
            "rubric_item": "[-2] Criterion B",
        },
    ]


def test_aggregate_healthbench_scores_clips_mean_and_aggregates_tags():
    samples = [
        {
            "score": -0.2,
            "example_tags": ["theme:test"],
            "rubrics": [
                {
                    "criterion": "a",
                    "points": 5,
                    "tags": ["axis:accuracy"],
                    "criteria_met": False,
                },
                {
                    "criterion": "bad",
                    "points": -1,
                    "tags": ["axis:accuracy"],
                    "criteria_met": True,
                },
            ],
        },
        {
            "score": 0.8,
            "example_tags": ["theme:test"],
            "rubrics": [
                {
                    "criterion": "a",
                    "points": 5,
                    "tags": ["axis:accuracy"],
                    "criteria_met": True,
                }
            ],
        },
    ]

    summary = aggregate_healthbench_scores(samples, num_bootstrap_samples=0)

    assert summary["overall_score"] == pytest.approx(0.3)
    assert summary["num_samples"] == 2
    assert summary["num_rubric_items"] == 3
    assert summary["num_criteria_met"] == 2
    assert summary["tag_scores"]["theme:test"]["score"] == pytest.approx(0.3)
    assert summary["tag_scores"]["axis:accuracy"]["score"] == pytest.approx(0.4)
