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

"""Oumi-provided reward-group metrics for the verl GRPO trainer."""

import inspect
import math
from collections import defaultdict
from collections.abc import Callable
from functools import wraps
from importlib import import_module
from typing import Any, cast

import torch

from oumi.utils.logging import logger

DEFAULT_REWARD_GROUP_LOW_STD_THRESHOLD = 1e-3
REWARD_GROUP_LOW_STD_CONFIG_KEY = "oumi_reward_group_low_std_threshold"

_PATCH_MARKER = "_oumi_grpo_reward_group_metrics_patch"
_PATCH_THRESHOLD = "_oumi_grpo_reward_group_low_std_threshold"
_PATCH_WARNING_EMITTED = "_oumi_grpo_reward_group_metrics_warning_emitted"
_ZERO_STD_ATOL = 1e-8


def _validate_low_std_threshold(low_std_threshold: float) -> float:
    try:
        threshold = float(low_std_threshold)
    except (TypeError, ValueError) as e:
        raise ValueError("low_std_threshold must be a finite number.") from e
    if not math.isfinite(threshold):
        raise ValueError("low_std_threshold must be finite.")
    if threshold < 0:
        raise ValueError("low_std_threshold must be non-negative.")
    return threshold


def compute_grpo_reward_group_metrics(
    batch: Any,
    *,
    low_std_threshold: float,
) -> dict[str, float]:
    """Computes reward dispersion metrics across GRPO prompt groups.

    The sequence rewards and prompt UIDs are the same values verl uses to compute
    GRPO advantages. Each prompt group therefore has equal weight in the returned
    aggregates, independent of response length or rollout ordering.

    Args:
        batch: A verl ``DataProto`` containing token-level rewards and prompt UIDs.
        low_std_threshold: Maximum group reward standard deviation considered low.

    Returns:
        Group reward standard-deviation fractions and summary statistics.

    Raises:
        ValueError: If the threshold is negative or non-finite, or the batch is empty.
        KeyError: If the batch does not contain rewards or prompt UIDs.
    """
    low_std_threshold = _validate_low_std_threshold(low_std_threshold)

    sequence_rewards = batch.batch["token_level_rewards"].sum(dim=-1).float()
    uids = batch.non_tensor_batch["uid"]
    if sequence_rewards.numel() == 0:
        raise ValueError("Cannot compute reward-group metrics for an empty batch.")
    if len(uids) != sequence_rewards.shape[0]:
        raise ValueError(
            "The number of prompt UIDs must match the number of sequence rewards."
        )

    rewards_by_uid: dict[Any, list[torch.Tensor]] = defaultdict(list)
    for uid, reward in zip(uids, sequence_rewards, strict=True):
        rewards_by_uid[uid].append(reward)

    group_stds = []
    with torch.no_grad():
        for rewards in rewards_by_uid.values():
            if len(rewards) == 1:
                group_stds.append(rewards[0].new_zeros(()))
            else:
                # Match verl's GRPO advantage normalization, which uses the sample
                # standard deviation (PyTorch's default correction=1).
                group_stds.append(torch.stack(rewards).std())

        stds = torch.stack(group_stds)
        quantiles = torch.quantile(
            stds,
            torch.tensor([0.5, 0.9], device=stds.device, dtype=stds.dtype),
        )
        zero_std = torch.isclose(
            stds,
            torch.zeros_like(stds),
            rtol=0.0,
            atol=_ZERO_STD_ATOL,
        )

    return {
        "grpo/frac_reward_zero_std": zero_std.float().mean().item(),
        "grpo/frac_reward_low_std": ((stds <= low_std_threshold) | zero_std)
        .float()
        .mean()
        .item(),
        "grpo/reward_group_std/mean": stds.mean().item(),
        "grpo/reward_group_std/p50": quantiles[0].item(),
        "grpo/reward_group_std/p90": quantiles[1].item(),
    }


def install_verl_grpo_reward_group_metrics_patch(
    *,
    low_std_threshold: float = DEFAULT_REWARD_GROUP_LOW_STD_THRESHOLD,
) -> None:
    """Adds Oumi's reward-group metrics to verl's driver-side metric collector.

    The patch is installed in the Ray trainer process, where verl owns the complete
    rollout batch. It is idempotent; installing it again updates the threshold used
    by the existing wrapper instead of nesting another wrapper.

    Args:
        low_std_threshold: Maximum group reward standard deviation considered low.

    Raises:
        RuntimeError: If the installed verl metric collector is incompatible.
        ValueError: If the threshold is negative or non-finite.
    """
    low_std_threshold = _validate_low_std_threshold(low_std_threshold)

    ray_trainer_module = import_module("verl.trainer.ppo.ray_trainer")
    current_collector = getattr(ray_trainer_module, "compute_data_metrics", None)
    if not callable(current_collector):
        raise RuntimeError(
            "The installed verl version does not expose a callable "
            "verl.trainer.ppo.ray_trainer.compute_data_metrics."
        )

    if getattr(current_collector, _PATCH_MARKER, False):
        setattr(current_collector, _PATCH_THRESHOLD, float(low_std_threshold))
        return

    collector_signature = inspect.signature(current_collector)
    if "batch" not in collector_signature.parameters:
        raise RuntimeError(
            "The installed verl compute_data_metrics function has an incompatible "
            "signature: it does not accept a 'batch' argument."
        )

    original_collector = cast(Callable[..., dict[str, Any]], current_collector)

    @wraps(original_collector)
    def _compute_data_metrics_with_reward_groups(*args, **kwargs):
        metrics = original_collector(*args, **kwargs)
        bound_args = collector_signature.bind_partial(*args, **kwargs)
        batch = bound_args.arguments["batch"]
        threshold = getattr(
            _compute_data_metrics_with_reward_groups,
            _PATCH_THRESHOLD,
        )
        try:
            group_metrics = compute_grpo_reward_group_metrics(
                batch,
                low_std_threshold=threshold,
            )
        except Exception:
            if not getattr(
                _compute_data_metrics_with_reward_groups,
                _PATCH_WARNING_EMITTED,
                False,
            ):
                logger.warning(
                    "Failed to compute Oumi's verl GRPO reward-group metrics. "
                    "Training will continue without them.",
                    exc_info=True,
                )
                setattr(
                    _compute_data_metrics_with_reward_groups,
                    _PATCH_WARNING_EMITTED,
                    True,
                )
            return metrics

        # Prefer a future upstream implementation if verl begins emitting any of
        # these metric names itself.
        for name, value in group_metrics.items():
            metrics.setdefault(name, value)
        return metrics

    setattr(_compute_data_metrics_with_reward_groups, _PATCH_MARKER, True)
    setattr(
        _compute_data_metrics_with_reward_groups,
        _PATCH_THRESHOLD,
        float(low_std_threshold),
    )
    setattr(_compute_data_metrics_with_reward_groups, _PATCH_WARNING_EMITTED, False)
    setattr(
        ray_trainer_module,
        "compute_data_metrics",
        _compute_data_metrics_with_reward_groups,
    )
    logger.info(
        "Installed verl GRPO reward-group metrics with low-std threshold %s.",
        low_std_threshold,
    )
