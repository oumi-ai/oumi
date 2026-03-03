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

"""Tests for simple rubric_reward function."""

import pytest

import oumi.datasets.grpo.rewards.rubric.simple as simple_module
from oumi.datasets.grpo.rewards.rubric.core import JudgeResult
from oumi.datasets.grpo.rewards.rubric.simple import (
    get_stats,
    reset_stats,
    rubric_reward,
)


class MockJudge:
    """Mock judge for testing."""

    def __init__(self, score: float = 0.75):
        self.score = score
        self.model = "mock-model"
        self.evaluate_calls = []

    def evaluate(self, prompt: str, completion: str, rubrics_text: str) -> JudgeResult:
        self.evaluate_calls.append((prompt, completion, rubrics_text))
        return JudgeResult(
            score=self.score,
            time_ms=100.0,
            success=True,
            raw_response='{"scores": {"1": 1}, "total_score": 0.75}',
        )


class TestRubricReward:
    """Tests for rubric_reward function."""

    def setup_method(self):
        """Reset stats before each test."""
        reset_stats()

    def test_basic_evaluation(self, monkeypatch):
        mock_judge = MockJudge(score=0.75)
        monkeypatch.setattr(simple_module, "_get_judge", lambda model: mock_judge)

        rewards = rubric_reward(
            completions=[[{"content": "The answer is 42."}]],
            prompts=["What is the answer?"],
            rubrics=[["Is accurate", "Is concise"]],
        )

        assert len(rewards) == 1
        assert rewards[0] == 0.75
        assert len(mock_judge.evaluate_calls) == 1

    def test_multiple_completions(self, monkeypatch):
        mock_judge = MockJudge(score=0.5)
        monkeypatch.setattr(simple_module, "_get_judge", lambda model: mock_judge)

        rewards = rubric_reward(
            completions=[
                [{"content": "Answer 1"}],
                [{"content": "Answer 2"}],
                [{"content": "Answer 3"}],
            ],
            prompts=["Q1", "Q2", "Q3"],
            rubrics=[["R1"], ["R2"], ["R3"]],
        )

        assert len(rewards) == 3
        assert all(r == 0.5 for r in rewards)
        assert len(mock_judge.evaluate_calls) == 3

    def test_missing_prompts_returns_zeros(self, monkeypatch):
        mock_judge = MockJudge()
        monkeypatch.setattr(simple_module, "_get_judge", lambda model: mock_judge)

        rewards = rubric_reward(
            completions=[[{"content": "Answer"}]],
            prompts=[],
            rubrics=[["Rubric"]],
        )

        assert rewards == [0.0]
        assert len(mock_judge.evaluate_calls) == 0

    def test_missing_rubrics_returns_zeros(self, monkeypatch):
        mock_judge = MockJudge()
        monkeypatch.setattr(simple_module, "_get_judge", lambda model: mock_judge)

        rewards = rubric_reward(
            completions=[[{"content": "Answer"}]],
            prompts=["Question"],
            rubrics=[],
        )

        assert rewards == [0.0]
        assert len(mock_judge.evaluate_calls) == 0

    def test_mismatched_lengths_truncates(self, monkeypatch):
        mock_judge = MockJudge(score=0.8)
        monkeypatch.setattr(simple_module, "_get_judge", lambda model: mock_judge)

        rewards = rubric_reward(
            completions=[
                [{"content": "A1"}],
                [{"content": "A2"}],
                [{"content": "A3"}],
            ],
            prompts=["P1", "P2"],
            rubrics=[["R1"]],
        )

        # Should truncate to shortest length (1)
        assert len(rewards) == 1
        assert rewards[0] == 0.8

    def test_system_prompt_prepended(self, monkeypatch):
        mock_judge = MockJudge()
        monkeypatch.setattr(simple_module, "_get_judge", lambda model: mock_judge)

        rubric_reward(
            completions=[[{"content": "Answer"}]],
            prompts=["Question"],
            rubrics=[["Rubric"]],
            system_prompt=["Be helpful"],
        )

        assert len(mock_judge.evaluate_calls) == 1
        prompt = mock_judge.evaluate_calls[0][0]
        assert "[System: Be helpful]" in prompt
        assert "Question" in prompt

    def test_stats_tracking(self, monkeypatch):
        mock_judge = MockJudge(score=0.6)
        monkeypatch.setattr(simple_module, "_get_judge", lambda model: mock_judge)

        rubric_reward(
            completions=[[{"content": "A1"}], [{"content": "A2"}]],
            prompts=["P1", "P2"],
            rubrics=[["R1"], ["R2"]],
        )

        stats = get_stats()
        assert stats.total_calls == 2
        assert stats.successful_calls == 2
        assert stats.avg_reward == pytest.approx(0.6)

    def test_string_completion_format(self, monkeypatch):
        mock_judge = MockJudge()
        monkeypatch.setattr(simple_module, "_get_judge", lambda model: mock_judge)

        rewards = rubric_reward(
            completions=["Direct string completion"],
            prompts=["Question"],
            rubrics=[["Rubric"]],
        )

        assert len(rewards) == 1
        assert mock_judge.evaluate_calls[0][1] == "Direct string completion"

    def test_alternative_param_names(self, monkeypatch):
        mock_judge = MockJudge(score=0.9)
        monkeypatch.setattr(simple_module, "_get_judge", lambda model: mock_judge)

        rewards = rubric_reward(
            completions=[[{"content": "Answer"}]],
            prompt=["Question"],  # Alternative name
            rubrics=[["Rubric"]],
        )

        assert len(rewards) == 1
        assert rewards[0] == 0.9

    def test_custom_judge_model(self, monkeypatch):
        models_used = []

        def track_model(model):
            models_used.append(model)
            return MockJudge()

        monkeypatch.setattr(simple_module, "_get_judge", track_model)

        rubric_reward(
            completions=[[{"content": "Answer"}]],
            prompts=["Question"],
            rubrics=[["Rubric"]],
            judge_model="gpt-4o",
        )

        assert models_used == ["gpt-4o"]


class TestRubricRewardFailure:
    """Tests for rubric_reward failure handling."""

    def setup_method(self):
        reset_stats()

    def test_judge_failure_returns_zero(self, monkeypatch):
        class FailingJudge:
            model = "mock"

            def evaluate(self, prompt, completion, rubrics):
                return JudgeResult(score=0.0, time_ms=50.0, success=False)

        monkeypatch.setattr(simple_module, "_get_judge", lambda m: FailingJudge())

        rewards = rubric_reward(
            completions=[[{"content": "Answer"}]],
            prompts=["Question"],
            rubrics=[["Rubric"]],
        )

        assert rewards == [0.0]
        stats = get_stats()
        assert stats.failed_calls == 1
