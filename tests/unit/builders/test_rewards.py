import math
import re

import pytest

from oumi.builders import build_reward_functions
from oumi.core.configs import TrainingParams
from oumi.core.registry import REGISTRY, RegistryType, register


@register("my_reward_func_starts_with_tldr", RegistryType.REWARD_FUNCTION)
def _starts_with_tldr_reward_func(completions, **kwargs):
    matches = [
        (content.startswith("TLDR") or content.startswith("TL;DR"))
        for content in completions
    ]
    return [1.0 if match else 0.0 for match in matches]


@register("my_reward_func_brevity", RegistryType.REWARD_FUNCTION)
def _brevity_func(completions, **kwargs):
    def _compute_reward(num_tokens, target_tokens=20):
        """Returns maximum reward for inputs that are `target_tokens` long"""
        x = float(num_tokens) / target_tokens
        return x * math.exp(-x)

    return [
        _compute_reward(len(re.split(r"\s+", content)), 20) for content in completions
    ]


def test_build_reward_functions_empty():
    assert len(build_reward_functions(TrainingParams())) == 0


@pytest.mark.parametrize(
    "function_name", ["my_reward_func_starts_with_tldr", "my_reward_func_brevity"]
)
def test_build_reward_functions_single(function_name: str):
    params = TrainingParams()
    params.reward_functions = [function_name]
    reward_funcs = build_reward_functions(params)
    assert len(reward_funcs) == 1
    assert reward_funcs == [REGISTRY.get(function_name, RegistryType.REWARD_FUNCTION)]

    params = TrainingParams()
    params.reward_functions = ["", function_name, ""]
    reward_funcs = build_reward_functions(params)
    assert len(reward_funcs) == 1
    assert reward_funcs == [REGISTRY.get(function_name, RegistryType.REWARD_FUNCTION)]


def test_build_reward_functions_multiple():
    params = TrainingParams()
    params.reward_functions = [
        "my_reward_func_starts_with_tldr",
        "my_reward_func_brevity",
    ]
    reward_funcs = build_reward_functions(params)
    assert len(reward_funcs) == 2
    assert reward_funcs == [
        REGISTRY.get("my_reward_func_starts_with_tldr", RegistryType.REWARD_FUNCTION),
        REGISTRY.get("my_reward_func_brevity", RegistryType.REWARD_FUNCTION),
    ]
