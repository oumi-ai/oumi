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

"""Tests for weighted_rubric_reward function."""

import pytest

import oumi.datasets.grpo.rewards.rubric.weighted as weighted_module
from oumi.datasets.grpo.rewards.rubric.core import JudgeResult
from oumi.datasets.grpo.rewards.rubric.weighted import (
    compute_weighted_score,
    format_weighted_rubrics,
    get_stats,
    parse_weighted_response,
    reset_stats,
    validate_weighted_rubrics,
    weighted_rubric_reward,
)


class TestValidateWeightedRubrics:
    """Tests for validate_weighted_rubrics function."""

    def test_valid_rubrics(self):
        rubrics = [
            {"description": "Be accurate", "weight": 2.0},
            {"description": "Be concise", "weight": 1.0},
        ]
        validate_weighted_rubrics(rubrics)  # Should not raise

    def test_valid_rubrics_with_name(self):
        rubrics = [
            {"name": "accuracy", "description": "Be accurate", "weight": 2.0},
        ]
        validate_weighted_rubrics(rubrics)  # Should not raise

    def test_weight_only_is_valid(self):
        rubrics = [{"weight": 2.0}]
        validate_weighted_rubrics(rubrics)  # Should not raise

    def test_description_only_is_valid(self):
        rubrics = [{"description": "Be accurate"}]
        validate_weighted_rubrics(rubrics)  # Should not raise

    def test_non_dict_raises(self):
        with pytest.raises(ValueError, match="must be dicts"):
            validate_weighted_rubrics(["string rubric"])

    def test_missing_required_keys_raises(self):
        with pytest.raises(ValueError, match="must include 'description' or 'weight'"):
            validate_weighted_rubrics([{"name": "only_name"}])


class TestFormatWeightedRubrics:
    """Tests for format_weighted_rubrics function."""

    def test_basic_formatting(self):
        rubrics = [
            {"name": "accuracy", "description": "Be accurate", "weight": 2.0},
            {"name": "brevity", "description": "Be brief", "weight": 1.0},
        ]
        result = format_weighted_rubrics(rubrics)
        assert "[accuracy]" in result
        assert "(weight=2.0)" in result
        assert "Be accurate" in result
        assert "[brevity]" in result

    def test_auto_naming(self):
        rubrics = [
            {"description": "Be accurate", "weight": 2.0},
        ]
        result = format_weighted_rubrics(rubrics)
        assert "[rubric_1]" in result

    def test_default_weight(self):
        rubrics = [{"description": "Be accurate"}]
        result = format_weighted_rubrics(rubrics)
        assert "(weight=1.0)" in result


class TestComputeWeightedScore:
    """Tests for compute_weighted_score function."""

    def test_simple_weighted_average(self):
        rubrics = [
            {"name": "r1", "weight": 2.0},
            {"name": "r2", "weight": 1.0},
        ]
        scores = {"r1": 1.0, "r2": 0.0}
        result = compute_weighted_score(rubrics, scores)
        # (2*1 + 1*0) / (2+1) = 2/3
        assert result == pytest.approx(2 / 3)

    def test_all_satisfied(self):
        rubrics = [
            {"name": "r1", "weight": 1.0},
            {"name": "r2", "weight": 1.0},
        ]
        scores = {"r1": 1.0, "r2": 1.0}
        result = compute_weighted_score(rubrics, scores)
        assert result == pytest.approx(1.0)

    def test_none_satisfied(self):
        rubrics = [
            {"name": "r1", "weight": 1.0},
            {"name": "r2", "weight": 1.0},
        ]
        scores = {"r1": 0.0, "r2": 0.0}
        result = compute_weighted_score(rubrics, scores)
        assert result == pytest.approx(0.0)

    def test_pitfall_avoided(self):
        """Test negative weight (pitfall) when avoided (score=1)."""
        rubrics = [{"name": "pitfall", "weight": -1.0}]
        scores = {"pitfall": 1.0}  # Avoided the pitfall
        result = compute_weighted_score(rubrics, scores)
        assert result == pytest.approx(1.0)

    def test_pitfall_hit(self):
        """Test negative weight (pitfall) when hit (score=0)."""
        rubrics = [{"name": "pitfall", "weight": -1.0}]
        scores = {"pitfall": 0.0}  # Hit the pitfall
        result = compute_weighted_score(rubrics, scores)
        assert result == pytest.approx(0.0)

    def test_mixed_positive_and_pitfall(self):
        """Test combination of positive and negative weights."""
        rubrics = [
            {"name": "good", "weight": 1.0},
            {"name": "pitfall", "weight": -1.0},
        ]
        # Good satisfied, pitfall avoided -> best case
        scores = {"good": 1.0, "pitfall": 1.0}
        result = compute_weighted_score(rubrics, scores)
        assert result == pytest.approx(1.0)

        # Good not satisfied, pitfall hit -> worst case
        # Formula: (weighted_sum + total_weight) / (2 * total_weight)
        # weighted_sum = 0 + (-1) = -1, total_weight = 2
        # result = (-1 + 2) / 4 = 0.25
        scores = {"good": 0.0, "pitfall": 0.0}
        result = compute_weighted_score(rubrics, scores)
        assert result == pytest.approx(0.25)

        # Good satisfied, pitfall hit -> mixed
        # weighted_sum = 1 + (-1) = 0, normalized = (0 + 2) / 4 = 0.5
        scores = {"good": 1.0, "pitfall": 0.0}
        result = compute_weighted_score(rubrics, scores)
        assert result == pytest.approx(0.5)

        # Good not satisfied, pitfall avoided -> mixed
        # weighted_sum = 0 + 1 = 1, normalized = (1 + 2) / 4 = 0.75
        scores = {"good": 0.0, "pitfall": 1.0}
        result = compute_weighted_score(rubrics, scores)
        assert result == pytest.approx(0.75)

    def test_auto_naming_in_scores(self):
        rubrics = [
            {"description": "First", "weight": 1.0},
            {"description": "Second", "weight": 1.0},
        ]
        scores = {"rubric_1": 1.0, "rubric_2": 0.0}
        result = compute_weighted_score(rubrics, scores)
        assert result == pytest.approx(0.5)

    def test_missing_scores_default_to_zero(self):
        rubrics = [
            {"name": "r1", "weight": 1.0},
            {"name": "r2", "weight": 1.0},
        ]
        scores = {"r1": 1.0}  # r2 missing
        result = compute_weighted_score(rubrics, scores)
        assert result == pytest.approx(0.5)

    def test_zero_total_weight(self):
        rubrics = [{"name": "r1", "weight": 0.0}]
        scores = {"r1": 1.0}
        result = compute_weighted_score(rubrics, scores)
        assert result == 0.0

    def test_scores_clamped(self):
        rubrics = [{"name": "r1", "weight": 1.0}]
        scores = {"r1": 1.5}  # Above 1.0
        result = compute_weighted_score(rubrics, scores)
        assert result == pytest.approx(1.0)


class TestParseWeightedResponse:
    """Tests for parse_weighted_response function."""

    def test_full_json_response(self):
        response = '{"scores": {"r1": 1, "r2": 0}, "weighted_score": 0.75}'
        scores, weighted_score = parse_weighted_response(response)
        assert scores == {"r1": 1.0, "r2": 0.0}
        assert weighted_score == pytest.approx(0.75)

    def test_scores_only(self):
        response = '{"scores": {"accuracy": 1, "brevity": 0.5}}'
        scores, weighted_score = parse_weighted_response(response)
        assert scores == {"accuracy": 1.0, "brevity": 0.5}
        assert weighted_score is None

    def test_json_with_surrounding_text(self):
        response = 'Here is my evaluation: {"scores": {"r1": 1}, "weighted_score": 0.5}'
        scores, weighted_score = parse_weighted_response(response)
        assert scores == {"r1": 1.0}
        assert weighted_score == pytest.approx(0.5)

    def test_flat_json(self):
        """Test when scores are at top level instead of nested."""
        response = '{"r1": 1, "r2": 0, "weighted_score": 0.5}'
        scores, weighted_score = parse_weighted_response(response)
        assert scores == {"r1": 1.0, "r2": 0.0}
        assert weighted_score == pytest.approx(0.5)

    def test_names_with_spaces(self):
        response = '{"scores": {"rubric one": 1, "rubric-two": 0.5}}'
        scores, weighted_score = parse_weighted_response(response)
        assert scores == {"rubric one": 1.0, "rubric-two": 0.5}

    def test_regex_fallback(self):
        """Test regex extraction when JSON parsing fails."""
        response = 'The weighted_score: 0.75, with "r1": 1'
        scores, weighted_score = parse_weighted_response(response)
        assert weighted_score == pytest.approx(0.75)
        assert scores.get("r1") == 1.0

    def test_empty_response(self):
        scores, weighted_score = parse_weighted_response("")
        assert scores == {}
        assert weighted_score is None


class MockJudge:
    """Mock judge for testing."""

    def __init__(self, score: float = 0.75, raw_response: str = ""):
        self.score = score
        self.raw_response = raw_response or (
            '{"scores": {"r1": 1, "r2": 0}, "weighted_score": 0.75}'
        )
        self.model = "mock-model"
        self.evaluate_calls = []

    def evaluate(self, prompt: str, completion: str, rubrics_text: str) -> JudgeResult:
        self.evaluate_calls.append((prompt, completion, rubrics_text))
        return JudgeResult(
            score=self.score,
            time_ms=100.0,
            success=True,
            raw_response=self.raw_response,
        )


class TestWeightedRubricReward:
    """Tests for weighted_rubric_reward function."""

    def setup_method(self):
        reset_stats()

    def test_basic_evaluation(self, monkeypatch):
        mock_judge = MockJudge()
        monkeypatch.setattr(weighted_module, "_get_judge", lambda m: mock_judge)

        rewards = weighted_rubric_reward(
            completions=[[{"content": "Answer"}]],
            prompts=["Question"],
            rubrics=[[
                {"name": "r1", "description": "Be accurate", "weight": 2.0},
                {"name": "r2", "description": "Be concise", "weight": 1.0},
            ]],
        )

        assert len(rewards) == 1
        assert rewards[0] == pytest.approx(0.75)

    def test_invalid_rubrics_returns_zero(self, monkeypatch):
        mock_judge = MockJudge()
        monkeypatch.setattr(weighted_module, "_get_judge", lambda m: mock_judge)

        rewards = weighted_rubric_reward(
            completions=[[{"content": "Answer"}]],
            prompts=["Question"],
            rubrics=[[{"name": "only_name"}]],  # Invalid: missing description/weight
        )

        assert rewards == [0.0]

    def test_uses_weighted_score_from_response(self, monkeypatch):
        mock_judge = MockJudge(
            raw_response='{"scores": {"r1": 1}, "weighted_score": 0.42}'
        )
        monkeypatch.setattr(weighted_module, "_get_judge", lambda m: mock_judge)

        rewards = weighted_rubric_reward(
            completions=[[{"content": "Answer"}]],
            prompts=["Question"],
            rubrics=[[{"description": "Test", "weight": 1.0}]],
        )

        assert rewards[0] == pytest.approx(0.42)

    def test_computes_from_scores_if_no_weighted_score(self, monkeypatch):
        mock_judge = MockJudge(
            raw_response='{"scores": {"rubric_1": 1, "rubric_2": 0}}'
        )
        monkeypatch.setattr(weighted_module, "_get_judge", lambda m: mock_judge)

        rewards = weighted_rubric_reward(
            completions=[[{"content": "Answer"}]],
            prompts=["Question"],
            rubrics=[[
                {"description": "First", "weight": 1.0},
                {"description": "Second", "weight": 1.0},
            ]],
        )

        # (1*1 + 1*0) / 2 = 0.5
        assert rewards[0] == pytest.approx(0.5)

    def test_stats_tracking(self, monkeypatch):
        mock_judge = MockJudge()
        monkeypatch.setattr(weighted_module, "_get_judge", lambda m: mock_judge)

        weighted_rubric_reward(
            completions=[[{"content": "A1"}], [{"content": "A2"}]],
            prompts=["P1", "P2"],
            rubrics=[
                [{"description": "R", "weight": 1.0}],
                [{"description": "R", "weight": 1.0}],
            ],
        )

        stats = get_stats()
        assert stats.total_calls == 2
        assert stats.successful_calls == 2
