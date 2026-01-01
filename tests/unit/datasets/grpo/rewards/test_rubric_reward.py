import pytest

import oumi.datasets.grpo.rewards.rubric_reward as rubric_module
from oumi.datasets.grpo.rewards.rubric_reward import RubricRewardEvaluator


def test_is_weighted_rubrics_mixed_types_raises():
    evaluator = RubricRewardEvaluator()

    with pytest.raises(ValueError, match="Rubrics must be all dicts"):
        evaluator._is_weighted_rubrics(["rubric", {"description": "desc"}])


def test_is_weighted_rubrics_missing_keys_raises():
    evaluator = RubricRewardEvaluator()

    with pytest.raises(
        ValueError, match="Weighted rubrics must include 'description' or 'weight'"
    ):
        evaluator._is_weighted_rubrics([{"name": "rubric_1"}])


def test_compute_weighted_score_non_negative_weights():
    evaluator = RubricRewardEvaluator()
    rubrics = [
        {"name": "r1", "description": "d1", "weight": 2.0},
        {"name": "r2", "description": "d2", "weight": 1.0},
    ]

    score = evaluator._compute_weighted_score(rubrics, {"r1": 0.0, "r2": 1.0})
    assert score == pytest.approx(1.0 / 3.0, rel=1e-6)


def test_compute_weighted_score_with_pitfall_weight():
    evaluator = RubricRewardEvaluator()
    rubrics = [{"name": "pit", "description": "avoid", "weight": -1.0}]

    assert evaluator._compute_weighted_score(rubrics, {"pit": 0.0}) == pytest.approx(
        0.0, rel=1e-6
    )
    assert evaluator._compute_weighted_score(rubrics, {"pit": 1.0}) == pytest.approx(
        1.0, rel=1e-6
    )


def test_parse_weighted_response_prefers_weighted_score():
    evaluator = RubricRewardEvaluator()
    scores, weighted_score = evaluator._parse_weighted_response(
        '{"scores":{"r1":1,"r2":0},"weighted_score":0.25}'
    )

    assert scores == {"r1": 1.0, "r2": 0.0}
    assert weighted_score == pytest.approx(0.25)


def test_parse_weighted_response_allows_names_with_spaces():
    evaluator = RubricRewardEvaluator()
    scores, weighted_score = evaluator._parse_weighted_response(
        '{"scores":{"rubric one":1,"rubric-two":0.5}}'
    )

    assert scores == {"rubric one": 1.0, "rubric-two": 0.5}
    assert weighted_score is None


def test_rubric_reward_truncates_mismatched_lengths(monkeypatch):
    class DummyEvaluator:
        def __init__(self) -> None:
            self.set_panel_calls = []
            self.evaluate_calls = []

        def set_panel_config(self, config) -> None:
            self.set_panel_calls.append(config)

        def evaluate(self, prompt, completion, rubrics, judge_model, group_rubrics):
            self.evaluate_calls.append(
                (prompt, completion, rubrics, judge_model, group_rubrics)
            )
            return 0.25

    dummy = DummyEvaluator()
    monkeypatch.setattr(rubric_module, "get_evaluator", lambda: dummy)

    rewards = rubric_module.rubric_reward(
        completions=[[{"content": "a"}], [{"content": "b"}]],
        prompts=["p1"],
        rubrics=[["r1"], ["r2"], ["r3"]],
    )

    assert rewards == [0.25]
    assert len(dummy.evaluate_calls) == 1
    assert dummy.set_panel_calls == [None]
