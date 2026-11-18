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

"""GRPO metrics computed from plain numpy arrays."""

import math
from collections.abc import Sequence
from typing import Final

import numpy as np

DEFAULT_LOW_STD_THRESHOLD: Final = 1e-3
TRUNCATION_RATIO_KEY: Final = "oumi/response_truncation_ratio"

_ZERO_STD_ATOL: Final = 1e-8


def compute_grpo_reward_group_metrics(
    sequence_rewards: np.ndarray,
    group_ids: np.ndarray,
    *,
    low_std_threshold: float = DEFAULT_LOW_STD_THRESHOLD,
) -> dict[str, float]:
    """Computes reward dispersion metrics across GRPO prompt groups.

    Each group has equal weight in the aggregates, regardless of its size. A group's
    spread is the sample standard deviation of its rewards (0 for a single member),
    matching the normalization GRPO advantages use.

    Args:
        sequence_rewards: One reward per response.
        group_ids: The prompt group of each response, aligned with the rewards.
        low_std_threshold: Largest group standard deviation counted as low.

    Returns:
        The fraction of groups with zero or low standard deviation, and the mean,
        median and 90th percentile of the group standard deviations.

    Raises:
        ValueError: If the threshold is negative or not finite, or the inputs are
            empty or differ in length.
    """
    if not math.isfinite(low_std_threshold):
        raise ValueError("low_std_threshold must be finite.")
    if low_std_threshold < 0:
        raise ValueError("low_std_threshold must be non-negative.")
    if sequence_rewards.size == 0:
        raise ValueError("Cannot compute reward-group metrics for empty inputs.")
    if group_ids.shape != sequence_rewards.shape:
        raise ValueError("The number of group ids must match the number of rewards.")

    rewards = sequence_rewards.astype(np.float64)
    _, group_index = np.unique(group_ids, return_inverse=True)
    groups = [rewards[group_index == i] for i in range(group_index.max() + 1)]
    stds = np.array([g.std(ddof=1) if g.size > 1 else 0.0 for g in groups])
    zero_std = np.isclose(stds, 0.0, rtol=0.0, atol=_ZERO_STD_ATOL)
    p50, p90 = np.quantile(stds, [0.5, 0.9])

    return {
        "grpo/frac_reward_zero_std": float(zero_std.mean()),
        "grpo/frac_reward_low_std": float(
            ((stds <= low_std_threshold) | zero_std).mean()
        ),
        "grpo/reward_group_std/mean": float(stds.mean()),
        "grpo/reward_group_std/p50": float(p50),
        "grpo/reward_group_std/p90": float(p90),
    }


def compute_response_truncation_ratio(
    response_lengths: np.ndarray, *, max_response_length: int
) -> float:
    """Returns the fraction of responses that reached `max_response_length`.

    Raises:
        ValueError: If `response_lengths` is empty.
    """
    if response_lengths.size == 0:
        raise ValueError("Cannot compute a truncation ratio for empty lengths.")
    return float((response_lengths >= max_response_length).mean())


def last_output_per_session(keys: Sequence[str]) -> list[str]:
    """Keeps, for each session, the key with the highest output index.

    Args:
        keys: Keys of the form ``{uid}_{session}_{index}``.

    Returns:
        The kept keys, in the order they appear in `keys`.
    """
    last_position: dict[str, int] = {}
    last_index: dict[str, int] = {}
    for position, key in enumerate(keys):
        session, _, index = key.rpartition("_")
        if session not in last_index or int(index) > last_index[session]:
            last_index[session] = int(index)
            last_position[session] = position
    return [keys[position] for position in sorted(last_position.values())]
