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

import oumi.datasets.grpo.rewards.conversation_judge_reward as rew


class _StubOutput:
    def __init__(self, scores):
        self.field_scores = scores


class _StubJudge:
    def __init__(self, scores):
        self._scores = scores
        self.seen = None

    def judge(self, inputs):
        self.seen = inputs
        return [_StubOutput(self._scores)]


def test_reward_reads_judgment_score(monkeypatch):
    stub = _StubJudge({"judgment": 1.0})
    monkeypatch.setattr(rew, "_get_judge", lambda path: stub)
    score = rew.conversation_llm_judge_reward(
        "src",
        "USER: hi\nAGENT: resolved",
        "get a refund",
        {"judge_config_path": "j.yaml"},
    )
    assert score == 1.0
    expected_seen = [
        {"conversation": "USER: hi\nAGENT: resolved", "goal": "get a refund"}
    ]
    assert stub.seen == expected_seen


def test_reward_defaults_to_zero_when_no_score(monkeypatch):
    monkeypatch.setattr(rew, "_get_judge", lambda path: _StubJudge({"judgment": None}))
    score = rew.conversation_llm_judge_reward(
        "src", "x", "g", {"judge_config_path": "j.yaml"}
    )
    assert score == 0.0


def test_reward_requires_config_path():
    with pytest.raises(ValueError):
        rew.conversation_llm_judge_reward("src", "x", "g", {})
