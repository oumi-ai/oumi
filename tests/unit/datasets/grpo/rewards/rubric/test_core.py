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

"""Tests for rubric reward core utilities."""

import pytest

from oumi.datasets.grpo.rewards.rubric.core import (
    RubricStats,
    clamp,
    extract_completion_strings,
    extract_json_object,
    format_string_rubrics,
    validate_inputs,
)


class TestClamp:
    """Tests for clamp function."""

    def test_clamp_within_range(self):
        assert clamp(0.5) == 0.5

    def test_clamp_at_min(self):
        assert clamp(0.0) == 0.0

    def test_clamp_at_max(self):
        assert clamp(1.0) == 1.0

    def test_clamp_below_min(self):
        assert clamp(-0.5) == 0.0

    def test_clamp_above_max(self):
        assert clamp(1.5) == 1.0

    def test_clamp_custom_range(self):
        assert clamp(5, min_val=0, max_val=10) == 5
        assert clamp(-5, min_val=0, max_val=10) == 0
        assert clamp(15, min_val=0, max_val=10) == 10


class TestExtractJsonObject:
    """Tests for extract_json_object function."""

    def test_simple_json(self):
        result = extract_json_object('{"key": "value"}')
        assert result == '{"key": "value"}'

    def test_json_with_surrounding_text(self):
        result = extract_json_object('Some text {"key": "value"} more text')
        assert result == '{"key": "value"}'

    def test_nested_json(self):
        result = extract_json_object('{"outer": {"inner": 1}}')
        assert result == '{"outer": {"inner": 1}}'

    def test_json_with_string_containing_braces(self):
        result = extract_json_object('{"key": "value with { and }"}')
        assert result == '{"key": "value with { and }"}'

    def test_json_with_escaped_quotes(self):
        result = extract_json_object('{"key": "value with \\" quote"}')
        assert result == '{"key": "value with \\" quote"}'

    def test_no_json(self):
        result = extract_json_object("No JSON here")
        assert result is None

    def test_unclosed_json(self):
        result = extract_json_object('{"key": "value"')
        assert result is None

    def test_empty_json(self):
        result = extract_json_object("{}")
        assert result == "{}"


class TestExtractCompletionStrings:
    """Tests for extract_completion_strings function."""

    def test_string_completions(self):
        completions = ["completion 1", "completion 2"]
        result = extract_completion_strings(completions)
        assert result == ["completion 1", "completion 2"]

    def test_dict_completions(self):
        completions = [{"content": "completion 1"}, {"content": "completion 2"}]
        result = extract_completion_strings(completions)
        assert result == ["completion 1", "completion 2"]

    def test_list_of_dicts(self):
        completions = [[{"content": "completion 1"}], [{"content": "completion 2"}]]
        result = extract_completion_strings(completions)
        assert result == ["completion 1", "completion 2"]

    def test_list_of_strings(self):
        completions = [["completion 1"], ["completion 2"]]
        result = extract_completion_strings(completions)
        assert result == ["completion 1", "completion 2"]

    def test_mixed_formats(self):
        completions = ["string", {"content": "dict"}, [{"content": "list"}]]
        result = extract_completion_strings(completions)
        assert result == ["string", "dict", "list"]

    def test_empty_list(self):
        result = extract_completion_strings([])
        assert result == []

    def test_dict_without_content(self):
        completions = [{"other": "value"}]
        result = extract_completion_strings(completions)
        assert result == ["{'other': 'value'}"]


class TestFormatStringRubrics:
    """Tests for format_string_rubrics function."""

    def test_single_rubric(self):
        result = format_string_rubrics(["Be accurate"])
        assert result == "1. Be accurate"

    def test_multiple_rubrics(self):
        result = format_string_rubrics(["Be accurate", "Be concise", "Be helpful"])
        expected = "1. Be accurate\n2. Be concise\n3. Be helpful"
        assert result == expected

    def test_empty_rubrics(self):
        result = format_string_rubrics([])
        assert result == ""


class TestValidateInputs:
    """Tests for validate_inputs function."""

    def test_valid_inputs(self):
        completions = ["c1", "c2"]
        prompts = ["p1", "p2"]
        rubrics = [["r1"], ["r2"]]
        result = validate_inputs(completions, prompts, rubrics, "test")
        assert result == (["c1", "c2"], ["p1", "p2"], [["r1"], ["r2"]], 2)

    def test_missing_prompts(self):
        completions = ["c1", "c2"]
        prompts = []
        rubrics = [["r1"], ["r2"]]
        result = validate_inputs(completions, prompts, rubrics, "test")
        assert result == ([], [], [], 0)

    def test_missing_rubrics(self):
        completions = ["c1", "c2"]
        prompts = ["p1", "p2"]
        rubrics = []
        result = validate_inputs(completions, prompts, rubrics, "test")
        assert result == ([], [], [], 0)

    def test_mismatched_lengths_truncates(self):
        completions = ["c1", "c2", "c3"]
        prompts = ["p1", "p2"]
        rubrics = [["r1"]]
        comp_strs, p, r, count = validate_inputs(completions, prompts, rubrics, "test")
        assert count == 1
        assert len(comp_strs) == 1
        assert len(p) == 1
        assert len(r) == 1


class TestRubricStats:
    """Tests for RubricStats class."""

    def test_initial_state(self):
        stats = RubricStats()
        assert stats.total_calls == 0
        assert stats.successful_calls == 0
        assert stats.failed_calls == 0
        assert stats.avg_reward == 0.0
        assert stats.avg_judge_time_ms == 0.0
        assert stats.success_rate == 0.0

    def test_record_success(self):
        stats = RubricStats()
        stats.record_success(0.8, 100.0)
        assert stats.total_calls == 1
        assert stats.successful_calls == 1
        assert stats.failed_calls == 0
        assert stats.avg_reward == 0.8
        assert stats.avg_judge_time_ms == 100.0
        assert stats.success_rate == 1.0

    def test_record_failure(self):
        stats = RubricStats()
        stats.record_failure(50.0)
        assert stats.total_calls == 1
        assert stats.successful_calls == 0
        assert stats.failed_calls == 1
        assert stats.success_rate == 0.0

    def test_multiple_records(self):
        stats = RubricStats()
        stats.record_success(0.6, 100.0)
        stats.record_success(0.8, 200.0)
        stats.record_failure(50.0)

        assert stats.total_calls == 3
        assert stats.successful_calls == 2
        assert stats.failed_calls == 1
        assert stats.avg_reward == pytest.approx(0.7)
        assert stats.avg_judge_time_ms == pytest.approx(350.0 / 3)
        assert stats.success_rate == pytest.approx(2 / 3)

    def test_should_log(self):
        stats = RubricStats(log_interval=10)
        for _ in range(9):
            stats.record_success(0.5, 100.0)
        assert not stats.should_log()

        stats.record_success(0.5, 100.0)
        assert stats.should_log()

    def test_get_summary(self):
        stats = RubricStats()
        stats.record_success(0.75, 150.0)
        summary = stats.get_summary()
        assert "calls=1" in summary
        assert "success=100.0%" in summary
        assert "avg_reward=0.750" in summary
        assert "avg_time=150ms" in summary
