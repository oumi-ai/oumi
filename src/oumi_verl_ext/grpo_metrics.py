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

"""oumi's GRPO metrics for verl 0.9.1's ``PPOTrainer._compute_metrics``."""

import logging
from collections.abc import Mapping
from types import ModuleType
from typing import Any

import numpy as np

from oumi_verl_ext.tool_episode import DROPPED_CALLS_KEY, MALFORMED_CALLS_KEY

logger = logging.getLogger(__name__)


def compute_grpo_metrics(
    rewards: np.ndarray,
    group_ids: np.ndarray,
    response_lengths: np.ndarray,
    max_response_length: int,
) -> dict[str, float]:
    """Computes the reward spread within each prompt group and the truncation ratio."""
    _, group_index = np.unique(group_ids, return_inverse=True)
    groups = [rewards[group_index == i] for i in range(group_index.max() + 1)]
    stds = np.array([group.std(ddof=1) if group.size > 1 else 0.0 for group in groups])
    p50, p90 = np.quantile(stds, [0.5, 0.9])
    return {
        "grpo/frac_reward_zero_std": float((stds <= 1e-8).mean()),
        "grpo/frac_reward_low_std": float((stds <= 1e-3).mean()),
        "grpo/reward_group_std/mean": float(stds.mean()),
        "grpo/reward_group_std/p50": float(p50),
        "grpo/reward_group_std/p90": float(p90),
        "oumi/response_truncation_ratio": float(
            (response_lengths >= max_response_length).mean()
        ),
    }


def compute_tool_call_metrics(
    extra_fields: list[Mapping[str, Any]],
) -> dict[str, float]:
    """Averages oumi's tool loop counts over the trajectories that carry them."""
    rows = [fields for fields in extra_fields if MALFORMED_CALLS_KEY in fields]
    if not rows:
        return {}
    return {
        f"oumi/{key.removeprefix('oumi_')}/mean": float(np.mean([r[key] for r in rows]))
        for key in (MALFORMED_CALLS_KEY, DROPPED_CALLS_KEY)
    }


def patch_compute_metrics(trainer_base: ModuleType) -> None:
    """Makes ``PPOTrainer._compute_metrics`` also log oumi's GRPO metrics.

    verl's sync and async trainers all inherit this method.
    """
    compute_verl_metrics = trainer_base.PPOTrainer._compute_metrics

    def compute_metrics(self, batch, metrics, *args, **kwargs) -> None:
        compute_verl_metrics(self, batch, metrics, *args, **kwargs)
        try:
            metrics.update(_step_metrics(trainer_base.tq, self.config, batch))
        except Exception:
            logger.warning("Skipped oumi's GRPO metrics for this step.", exc_info=True)
        try:
            metrics.update(_tool_call_step_metrics(trainer_base.tq, batch))
        except Exception:
            logger.warning(
                "Skipped oumi's tool-call metrics for this step.", exc_info=True
            )

    trainer_base.PPOTrainer._compute_metrics = compute_metrics


def _last_output_per_session(keys: list[str]) -> list[str]:
    # Keys are {uid}_{session}_{index}.
    last: dict[str, tuple[int, str]] = {}
    for key in keys:
        session, index = key.rsplit("_", 1)
        if session not in last or int(index) > last[session][0]:
            last[session] = (int(index), key)
    return [key for _, key in last.values()]


def _step_metrics(tq: Any, config: Any, batch: Any) -> dict[str, float]:
    keys = [
        key
        for key, tag in zip(batch.keys, batch.tags, strict=True)
        if not tag.get("is_padding", False)
    ]
    reward_field = (
        "token_level_rewards" if config.algorithm.use_kl_in_reward else "rm_scores"
    )
    data = tq.kv_batch_get(
        # verl computes GRPO advantages from each session's last output.
        keys=_last_output_per_session(keys),
        partition_id=batch.partition_id,
        select_fields=["uid", reward_field, "response_mask"],
    )
    # Counts tool tokens too, since they count toward the length cap.
    lengths = np.asarray(data["response_mask"].offsets().diff().tolist())
    data = data.to_padded_tensor()
    return compute_grpo_metrics(
        rewards=np.asarray(data[reward_field].sum(-1).tolist(), dtype=np.float64),
        group_ids=np.asarray(data.pop("uid").tolist()),
        response_lengths=lengths,
        max_response_length=config.actor_rollout_ref.rollout.response_length,
    )


def _tool_call_step_metrics(tq: Any, batch: Any) -> dict[str, float]:
    keys = [
        key
        for key, tag in zip(batch.keys, batch.tags, strict=True)
        if not tag.get("is_padding", False)
    ]
    data = tq.kv_batch_get(
        keys=_last_output_per_session(keys),
        partition_id=batch.partition_id,
        select_fields=["extra_fields"],
    )
    return compute_tool_call_metrics(data["extra_fields"].tolist())
