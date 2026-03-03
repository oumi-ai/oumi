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

"""Rubric-based reward functions for GRPO training.

This module provides three reward functions for evaluating model completions
against rubrics using LLM judges:

- rubric_reward: Simple evaluation against string rubrics
- weighted_rubric_reward: Evaluation with weighted/prioritized rubrics
- panel_rubric_reward: Multi-judge evaluation for robust scoring

Example usage:

    # Simple rubrics
    from oumi.datasets.grpo.rewards.rubric import rubric_reward
    rewards = rubric_reward(
        completions=completions,
        prompts=prompts,
        rubrics=[["Is accurate", "Is concise"]],
    )

    # Weighted rubrics
    from oumi.datasets.grpo.rewards.rubric import weighted_rubric_reward
    rewards = weighted_rubric_reward(
        completions=completions,
        prompts=prompts,
        rubrics=[[
            {"description": "Is accurate", "weight": 2.0},
            {"description": "Is concise", "weight": 1.0},
        ]],
    )

    # Panel of judges
    from oumi.datasets.grpo.rewards.rubric import panel_rubric_reward
    rewards = panel_rubric_reward(
        completions=completions,
        prompts=prompts,
        rubrics=[["Is accurate", "Is concise"]],
        judges=["gpt-4o", "gpt-4o-mini"],
        aggregation="mean",
    )
"""

from oumi.datasets.grpo.rewards.rubric.core import (
    RubricJudge,
    RubricStats,
    extract_completion_strings,
)
from oumi.datasets.grpo.rewards.rubric.panel import (
    AggregationStrategy,
    PanelMember,
    PanelStats,
    panel_rubric_reward,
)
from oumi.datasets.grpo.rewards.rubric.simple import rubric_reward
from oumi.datasets.grpo.rewards.rubric.weighted import weighted_rubric_reward

__all__ = [
    # Reward functions
    "rubric_reward",
    "weighted_rubric_reward",
    "panel_rubric_reward",
    # Core utilities
    "RubricJudge",
    "RubricStats",
    "extract_completion_strings",
    # Panel types
    "AggregationStrategy",
    "PanelMember",
    "PanelStats",
]
