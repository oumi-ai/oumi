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

"""Tests for panel_rubric_reward function."""

import tempfile
from pathlib import Path

import pytest

import oumi.datasets.grpo.rewards.rubric.panel as panel_module
from oumi.datasets.grpo.rewards.rubric.core import JudgeResult
from oumi.datasets.grpo.rewards.rubric.panel import (
    AggregationStrategy,
    PanelMember,
    aggregate_scores,
    get_stats,
    is_weighted_rubrics,
    load_panel_config,
    panel_rubric_reward,
    reset_stats,
)


class TestPanelMember:
    """Tests for PanelMember class."""

    def test_default_values(self):
        member = PanelMember()
        assert member.model == "gpt-4o-mini"
        assert member.weight == 1.0
        assert member.temperature == 0.0

    def test_from_string(self):
        member = PanelMember.from_config("gpt-4o")
        assert member.model == "gpt-4o"
        assert member.weight == 1.0

    def test_from_dict(self):
        config = {"model": "gpt-4o", "weight": 2.0, "temperature": 0.5}
        member = PanelMember.from_config(config)
        assert member.model == "gpt-4o"
        assert member.weight == 2.0
        assert member.temperature == 0.5

    def test_from_dict_partial(self):
        config = {"model": "gpt-4o"}
        member = PanelMember.from_config(config)
        assert member.model == "gpt-4o"
        assert member.weight == 1.0  # Default

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError, match="Invalid panel member config"):
            PanelMember.from_config(123)


class TestAggregateScores:
    """Tests for aggregate_scores function."""

    def test_mean_aggregation(self):
        scores = [0.6, 0.8, 0.7]
        weights = [1.0, 1.0, 1.0]
        result, variance = aggregate_scores(scores, weights, AggregationStrategy.MEAN)
        assert result == pytest.approx(0.7)
        assert variance > 0

    def test_weighted_mean(self):
        scores = [0.5, 1.0]
        weights = [1.0, 2.0]
        result, variance = aggregate_scores(scores, weights, AggregationStrategy.MEAN)
        # (0.5*1 + 1.0*2) / 3 = 2.5/3 ≈ 0.833
        assert result == pytest.approx(2.5 / 3)

    def test_median_aggregation(self):
        scores = [0.3, 0.8, 0.5]
        weights = [1.0, 1.0, 1.0]
        result, variance = aggregate_scores(scores, weights, AggregationStrategy.MEDIAN)
        assert result == pytest.approx(0.5)

    def test_empty_scores(self):
        result, variance = aggregate_scores([], [], AggregationStrategy.MEAN)
        assert result == 0.0
        assert variance == 0.0

    def test_single_score_no_variance(self):
        result, variance = aggregate_scores([0.8], [1.0], AggregationStrategy.MEAN)
        assert result == pytest.approx(0.8)
        assert variance == 0.0

    def test_zero_weights_falls_back_to_simple_mean(self):
        scores = [0.4, 0.6]
        weights = [0.0, 0.0]
        result, _ = aggregate_scores(scores, weights, AggregationStrategy.MEAN)
        assert result == pytest.approx(0.5)

    def test_result_clamped(self):
        # Edge case: scores somehow exceed 1.0
        scores = [1.2, 0.8]
        weights = [1.0, 1.0]
        result, _ = aggregate_scores(scores, weights, AggregationStrategy.MEAN)
        assert result == pytest.approx(1.0)


class TestIsWeightedRubrics:
    """Tests for is_weighted_rubrics function."""

    def test_string_rubrics(self):
        assert not is_weighted_rubrics(["rubric 1", "rubric 2"])

    def test_dict_rubrics(self):
        assert is_weighted_rubrics([{"description": "test"}])

    def test_empty_list(self):
        assert not is_weighted_rubrics([])

    def test_mixed_returns_false(self):
        # Mixed types should return False (all must be dicts)
        assert not is_weighted_rubrics(["string", {"description": "dict"}])


class TestLoadPanelConfig:
    """Tests for load_panel_config function."""

    def test_load_from_yaml(self):
        yaml_content = """
judges:
  - model: gpt-4o
    weight: 2.0
  - model: gpt-4o-mini
    weight: 1.0
    temperature: 0.1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = f.name

        try:
            members = load_panel_config(path)
            assert len(members) == 2
            assert members[0].model == "gpt-4o"
            assert members[0].weight == 2.0
            assert members[1].model == "gpt-4o-mini"
            assert members[1].temperature == 0.1
        finally:
            Path(path).unlink()

    def test_load_string_judges(self):
        yaml_content = """
judges:
  - gpt-4o
  - gpt-4o-mini
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = f.name

        try:
            members = load_panel_config(path)
            assert len(members) == 2
            assert members[0].model == "gpt-4o"
            assert members[1].model == "gpt-4o-mini"
        finally:
            Path(path).unlink()


class MockJudge:
    """Mock judge for testing."""

    def __init__(self, score: float = 0.75, raw_response: str = ""):
        self.score = score
        self.raw_response = raw_response or '{"total_score": 0.75}'
        self.evaluate_calls = []

    def evaluate(self, prompt: str, completion: str, rubrics_text: str) -> JudgeResult:
        self.evaluate_calls.append((prompt, completion, rubrics_text))
        return JudgeResult(
            score=self.score,
            time_ms=100.0,
            success=True,
            raw_response=self.raw_response,
        )


class TestPanelRubricReward:
    """Tests for panel_rubric_reward function."""

    def setup_method(self):
        reset_stats()
        panel_module._judges.clear()

    def test_single_judge_default(self, monkeypatch):
        """Test with no panel config uses single default judge."""
        mock_judge = MockJudge(score=0.8)

        def get_judge(model, temp, weighted):
            return mock_judge

        monkeypatch.setattr(panel_module, "_get_judge", get_judge)

        rewards = panel_rubric_reward(
            completions=[[{"content": "Answer"}]],
            prompts=["Question"],
            rubrics=[["Is accurate", "Is concise"]],
        )

        assert len(rewards) == 1
        assert rewards[0] == pytest.approx(0.8)

    def test_multiple_judges(self, monkeypatch):
        """Test with multiple judges aggregates scores."""
        call_count = [0]

        def get_judge(model, temp, weighted):
            judge = MockJudge(score=0.6 + 0.2 * call_count[0])
            call_count[0] += 1
            return judge

        monkeypatch.setattr(panel_module, "_get_judge", get_judge)

        rewards = panel_rubric_reward(
            completions=[[{"content": "Answer"}]],
            prompts=["Question"],
            rubrics=[["Is accurate"]],
            judges=["gpt-4o", "gpt-4o-mini"],
        )

        assert len(rewards) == 1
        # Mean of 0.6 and 0.8 = 0.7
        assert rewards[0] == pytest.approx(0.7)

    def test_median_aggregation(self, monkeypatch):
        """Test median aggregation strategy."""
        scores = [0.3, 0.5, 0.9]
        call_idx = [0]

        def get_judge(model, temp, weighted):
            judge = MockJudge(score=scores[call_idx[0] % len(scores)])
            call_idx[0] += 1
            return judge

        monkeypatch.setattr(panel_module, "_get_judge", get_judge)

        rewards = panel_rubric_reward(
            completions=[[{"content": "Answer"}]],
            prompts=["Question"],
            rubrics=[["Rubric"]],
            judges=["j1", "j2", "j3"],
            aggregation="median",
        )

        assert rewards[0] == pytest.approx(0.5)

    def test_weighted_rubrics_support(self, monkeypatch):
        """Test panel works with weighted rubrics."""
        mock_judge = MockJudge(
            raw_response='{"scores": {"r1": 1}, "weighted_score": 0.85}'
        )
        monkeypatch.setattr(panel_module, "_get_judge", lambda m, t, w: mock_judge)

        rewards = panel_rubric_reward(
            completions=[[{"content": "Answer"}]],
            prompts=["Question"],
            rubrics=[[{"name": "r1", "description": "Test", "weight": 2.0}]],
            judges=["gpt-4o"],
        )

        assert rewards[0] == pytest.approx(0.85)

    def test_judge_weights_in_aggregation(self, monkeypatch):
        """Test that judge weights affect aggregation."""
        call_idx = [0]
        scores = [0.4, 0.8]

        def get_judge(model, temp, weighted):
            judge = MockJudge(score=scores[call_idx[0]])
            call_idx[0] += 1
            return judge

        monkeypatch.setattr(panel_module, "_get_judge", get_judge)

        rewards = panel_rubric_reward(
            completions=[[{"content": "Answer"}]],
            prompts=["Question"],
            rubrics=[["Rubric"]],
            judges=[
                {"model": "j1", "weight": 1.0},
                {"model": "j2", "weight": 3.0},
            ],
        )

        # Weighted mean: (0.4*1 + 0.8*3) / 4 = 2.8/4 = 0.7
        assert rewards[0] == pytest.approx(0.7)

    def test_invalid_aggregation_defaults_to_mean(self, monkeypatch):
        mock_judge = MockJudge(score=0.6)
        monkeypatch.setattr(panel_module, "_get_judge", lambda m, t, w: mock_judge)

        rewards = panel_rubric_reward(
            completions=[[{"content": "Answer"}]],
            prompts=["Question"],
            rubrics=[["Rubric"]],
            aggregation="invalid_strategy",
        )

        assert len(rewards) == 1

    def test_all_judges_fail_returns_zero(self, monkeypatch):
        """Test when all judges fail."""

        def get_judge(model, temp, weighted):
            class FailingJudge:
                def evaluate(self, p, c, r):
                    return JudgeResult(score=0.0, time_ms=50.0, success=False)

            return FailingJudge()

        monkeypatch.setattr(panel_module, "_get_judge", get_judge)

        rewards = panel_rubric_reward(
            completions=[[{"content": "Answer"}]],
            prompts=["Question"],
            rubrics=[["Rubric"]],
            judges=["j1", "j2"],
        )

        assert rewards == [0.0]
        stats = get_stats()
        assert stats.failed_calls == 1

    def test_partial_judge_failure(self, monkeypatch):
        """Test when some judges fail, others succeed."""
        call_idx = [0]

        def get_judge(model, temp, weighted):
            class PartialJudge:
                def evaluate(self, p, c, r):
                    idx = call_idx[0]
                    call_idx[0] += 1
                    if idx == 0:
                        return JudgeResult(score=0.0, time_ms=50.0, success=False)
                    return JudgeResult(
                        score=0.8, time_ms=100.0, success=True, raw_response="{}"
                    )

            return PartialJudge()

        monkeypatch.setattr(panel_module, "_get_judge", get_judge)

        rewards = panel_rubric_reward(
            completions=[[{"content": "Answer"}]],
            prompts=["Question"],
            rubrics=[["Rubric"]],
            judges=["j1", "j2"],
        )

        # Only one judge succeeded with 0.8
        assert rewards[0] == pytest.approx(0.8)

    def test_stats_include_variance(self, monkeypatch):
        """Test that panel stats track variance."""
        call_idx = [0]
        scores = [0.4, 0.8]

        def get_judge(model, temp, weighted):
            judge = MockJudge(score=scores[call_idx[0] % 2])
            call_idx[0] += 1
            return judge

        monkeypatch.setattr(panel_module, "_get_judge", get_judge)

        panel_rubric_reward(
            completions=[[{"content": "Answer"}]],
            prompts=["Question"],
            rubrics=[["Rubric"]],
            judges=["j1", "j2"],
        )

        stats = get_stats()
        assert stats.successful_calls == 1
        assert len(stats.variance_history) == 1
        assert stats.variance_history[0] > 0  # Non-zero variance

    def test_missing_inputs_returns_zeros(self, monkeypatch):
        mock_judge = MockJudge()
        monkeypatch.setattr(panel_module, "_get_judge", lambda m, t, w: mock_judge)

        rewards = panel_rubric_reward(
            completions=[[{"content": "Answer"}]],
            prompts=[],
            rubrics=[["Rubric"]],
        )

        assert rewards == [0.0]

    def test_panel_config_from_file(self, monkeypatch):
        """Test loading panel config from YAML file."""
        yaml_content = """
judges:
  - model: gpt-4o
    weight: 2.0
  - model: gpt-4o-mini
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = f.name

        try:
            models_used = []

            def get_judge(model, temp, weighted):
                models_used.append(model)
                return MockJudge(score=0.5)

            monkeypatch.setattr(panel_module, "_get_judge", get_judge)

            panel_rubric_reward(
                completions=[[{"content": "Answer"}]],
                prompts=["Question"],
                rubrics=[["Rubric"]],
                panel_config_path=path,
            )

            assert "gpt-4o" in models_used
            assert "gpt-4o-mini" in models_used
        finally:
            Path(path).unlink()
