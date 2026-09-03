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

"""Tests for the consolidated dataset-level HealthBench rubric."""

import json

import jsonlines
import pytest

from oumi.core.types.conversation import Conversation
from oumi.evaluation.registry.healthbench_common import subset_cached_conversations
from oumi.evaluation.registry.healthbench_global_task import (
    _load_grade_cache,
    aggregate_global_scores,
    build_global_judge_inputs,
    load_global_rubric,
    parse_grades,
    render_rubric_block,
    score_sample,
)

RUBRIC = {
    "version": "test",
    "scale": {"min": 0, "max": 4},
    "criteria": [
        {
            "id": "AC1",
            "axis": "accuracy",
            "kind": "quality",
            "title": "Accurate facts",
            "text": "The response states medical facts correctly.",
            "anchors": {"0": "wrong", "2": "mostly right", "4": "precise"},
            "weight": 0.6,
        },
        {
            "id": "CP1",
            "axis": "completeness",
            "kind": "harm_avoidance",
            "title": "No critical omission",
            "text": "The response does not omit critical information.",
            "anchors": {"0": "omits", "2": "minor gaps", "4": "complete"},
            "weight": 0.4,
        },
    ],
}


def _write_rubric(tmp_path, rubric):
    path = tmp_path / "rubric.json"
    path.write_text(json.dumps(rubric))
    return path


def test_parse_grades_accepts_equals_and_colon_forms():
    assert parse_grades("AC1=3,CP1=1", RUBRIC) == {"AC1": 3, "CP1": 1}
    assert parse_grades("AC1: 3; CP1: 1", RUBRIC) == {"AC1": 3, "CP1": 1}
    assert parse_grades("ac1=0\ncp1=4", RUBRIC) == {"AC1": 0, "CP1": 4}


@pytest.mark.parametrize(
    "judgment,expected_error",
    [
        ("AC1=3", "missing"),
        ("AC1=3,CP1=1,XX9=2", "unexpected"),
        ("AC1=3,AC1=2,CP1=1", "Duplicate"),
        ("AC1=5,CP1=1", "outside"),
        ("AC1=-1,CP1=1", "outside"),
        ("AC1=high,CP1=1", "Non-integer"),
        ("3,1", "Malformed"),
    ],
)
def test_parse_grades_rejects_bad_output(judgment, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        parse_grades(judgment, RUBRIC)


def test_parse_grades_rejects_positional_drift():
    """A dropped criterion must error, not silently shift later grades."""
    with pytest.raises(ValueError, match="missing"):
        parse_grades("AC1=4", RUBRIC)


def test_score_sample_is_weighted_and_normalized():
    assert score_sample({"AC1": 4, "CP1": 4}, RUBRIC) == pytest.approx(1.0)
    assert score_sample({"AC1": 0, "CP1": 0}, RUBRIC) == pytest.approx(0.0)
    # (0.6 * 2/4 + 0.4 * 4/4) / 1.0
    assert score_sample({"AC1": 2, "CP1": 4}, RUBRIC) == pytest.approx(0.7)


def test_load_global_rubric_validates(tmp_path):
    assert load_global_rubric(_write_rubric(tmp_path, RUBRIC))["version"] == "test"

    duplicate = json.loads(json.dumps(RUBRIC))
    duplicate["criteria"][1]["id"] = "AC1"
    with pytest.raises(ValueError, match="duplicate criterion ids"):
        load_global_rubric(_write_rubric(tmp_path, duplicate))

    missing_key = json.loads(json.dumps(RUBRIC))
    del missing_key["criteria"][0]["anchors"]
    with pytest.raises(ValueError, match="missing 'anchors'"):
        load_global_rubric(_write_rubric(tmp_path, missing_key))

    zero_weight = json.loads(json.dumps(RUBRIC))
    zero_weight["criteria"][0]["weight"] = 0
    with pytest.raises(ValueError, match="non-positive weight"):
        load_global_rubric(_write_rubric(tmp_path, zero_weight))


def test_render_rubric_block_is_constant_and_includes_anchors():
    block = render_rubric_block(RUBRIC)
    assert "AC1 - Accurate facts" in block
    assert "2 = mostly right" in block
    assert "harm-avoidance" in block  # only for the CP1 criterion
    assert render_rubric_block(RUBRIC) == block


def test_build_global_judge_inputs_shares_one_rubric_across_samples():
    conversations = [
        Conversation.from_dict(
            {
                "messages": [
                    {"role": "user", "content": f"Q{i}"},
                    {"role": "assistant", "content": f"A{i}"},
                ],
                "metadata": {"prompt_id": f"p{i}"},
            }
        )
        for i in range(2)
    ]
    inputs = build_global_judge_inputs(conversations, RUBRIC)
    assert len(inputs) == 2
    assert inputs[0]["rubric"] == inputs[1]["rubric"]
    assert inputs[0]["conversation"] == "user: Q0\n\nassistant: A0"


def test_aggregate_global_scores_reports_saturation_and_axes():
    samples = [
        {
            "score": score_sample({"AC1": 4, "CP1": 4}, RUBRIC),
            "grades": {"AC1": 4, "CP1": 4},
            "example_tags": ["theme:hedging"],
        },
        {
            "score": score_sample({"AC1": 4, "CP1": 0}, RUBRIC),
            "grades": {"AC1": 4, "CP1": 0},
            "example_tags": ["theme:hedging"],
        },
    ]
    summary = aggregate_global_scores(samples, RUBRIC, num_bootstrap_samples=0)

    assert summary["num_samples"] == 2
    assert summary["overall_score"] == pytest.approx((1.0 + 0.6) / 2)
    # AC1 is maxed by every sample: exactly the dead-criterion case the pilot gate
    # exists to catch.
    assert summary["criterion_stats"]["AC1"]["ceiling_rate"] == pytest.approx(1.0)
    assert summary["criterion_stats"]["CP1"]["ceiling_rate"] == pytest.approx(0.5)
    assert summary["criterion_stats"]["CP1"]["floor_rate"] == pytest.approx(0.5)
    assert summary["axis_scores"]["accuracy"]["score"] == pytest.approx(1.0)
    assert summary["axis_scores"]["completeness"]["score"] == pytest.approx(0.5)
    assert summary["theme_scores"]["theme:hedging"]["num_samples"] == 2


def test_grade_cache_rejects_mixed_judges(tmp_path):
    """Grades from two different judges must never be aggregated together."""
    provenance = {"judge_model": "gpt-4o-mini", "rubric_sha256": "abc123"}
    cache_path = tmp_path / "criterion_grades.jsonl"
    with jsonlines.open(cache_path, mode="w") as writer:
        writer.write({"sample_index": 0, "prompt_id": "p0", **provenance})
        writer.write(
            {
                "sample_index": 1,
                "prompt_id": "p1",
                "judge_model": "gpt-4o",
                "rubric_sha256": "abc123",
            }
        )

    with pytest.raises(ValueError, match="different setup"):
        _load_grade_cache(cache_path, provenance)


def test_grade_cache_loads_matching_rows(tmp_path):
    provenance = {"judge_model": "gpt-4o-mini", "rubric_sha256": "abc123"}
    cache_path = tmp_path / "criterion_grades.jsonl"
    with jsonlines.open(cache_path, mode="w") as writer:
        writer.write({"sample_index": 7, "prompt_id": "p7", **provenance})

    assert set(_load_grade_cache(cache_path, provenance)) == {"p7"}


def test_subset_cached_conversations_selects_by_prompt_id():
    cached = [
        Conversation.from_dict(
            {
                "messages": [{"role": "user", "content": f"Q{i}"}],
                "metadata": {"prompt_id": f"p{i}"},
            }
        )
        for i in range(5)
    ]
    examples = [{"prompt_id": "p3"}, {"prompt_id": "p0"}]
    selected = subset_cached_conversations(cached, examples)
    assert [c.metadata["prompt_id"] for c in selected] == ["p3", "p0"]

    with pytest.raises(ValueError, match="missing 1 of 1"):
        subset_cached_conversations(cached, [{"prompt_id": "nope"}])


def test_parse_grades_accepts_absent_as_zero_for_quality_only():
    """The judge sometimes writes "absent" instead of 0.

    Grade 0 for a quality criterion is defined as the property being absent, so
    the word is accepted there. For a harm-avoidance criterion it is ambiguous in
    the dangerous direction -- the absent thing may be the harmful behaviour,
    which is the TOP grade -- so it must still be rejected.
    """
    assert parse_grades("AC1=absent,CP1=3", RUBRIC) == {"AC1": 0, "CP1": 3}
    assert parse_grades("AC1=None,CP1=3", RUBRIC) == {"AC1": 0, "CP1": 3}

    # CP1 is the harm_avoidance criterion in RUBRIC.
    with pytest.raises(ValueError, match="Non-integer grade"):
        parse_grades("AC1=2,CP1=absent", RUBRIC)
