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

"""Adds oumi's GRPO metrics to every verl 0.9.1 PPO trainer with one patch."""

import functools
import inspect
import logging
from types import ModuleType
from typing import Any, Final

import numpy as np

from oumi_verl_ext.metrics import (
    TRUNCATION_RATIO_KEY,
    compute_grpo_reward_group_metrics,
    compute_response_truncation_ratio,
    last_output_per_session,
)

logger = logging.getLogger("oumi_verl_ext")

_COMPUTE_METRICS_PARAMETERS: Final = (
    "self",
    "batch",
    "metrics",
    "timing_raw",
    "global_steps",
    "epoch",
)
_PATCHED: Final = "_oumi_grpo_metrics"


class VerlPatchError(RuntimeError):
    """Raised when verl's trainer is not in the shape oumi patches."""


def install_grpo_metrics(trainer_base: ModuleType) -> None:
    """Adds oumi's GRPO metrics to verl's ``PPOTrainer._compute_metrics``.

    verl's sync, colocate_async and separate_async trainers all inherit that method,
    so this one patch covers every trainer mode. The patched method computes verl's
    metrics first, then adds the ``grpo/*`` metrics and the response truncation
    ratio. A failure in oumi's metrics is logged once per process and does not stop
    training. Installing again does nothing.

    Args:
        trainer_base: verl's ``verl.trainer.ppo.v1.trainer_base`` module.

    Raises:
        VerlPatchError: If ``PPOTrainer`` has no ``_compute_metrics``, or its
            parameters differ from verl 0.9.1's.
    """
    trainer_cls = trainer_base.PPOTrainer
    compute_verl_metrics = getattr(trainer_cls, "_compute_metrics", None)
    if compute_verl_metrics is None:
        raise VerlPatchError(
            "verl's PPOTrainer has no _compute_metrics, so oumi_verl_ext cannot add "
            "its GRPO metrics."
        )
    if getattr(compute_verl_metrics, _PATCHED, False):
        return
    parameters = tuple(inspect.signature(compute_verl_metrics).parameters)
    if parameters != _COMPUTE_METRICS_PARAMETERS:
        raise VerlPatchError(
            f"verl's PPOTrainer._compute_metrics takes {parameters}, but "
            f"oumi_verl_ext expects {_COMPUTE_METRICS_PARAMETERS}."
        )
    failure_logged = False

    @functools.wraps(compute_verl_metrics)
    def compute_metrics(
        self,
        batch: Any,
        metrics: dict[str, Any],
        timing_raw: dict[str, float],
        global_steps: int,
        epoch: int,
    ) -> None:
        nonlocal failure_logged
        compute_verl_metrics(self, batch, metrics, timing_raw, global_steps, epoch)
        try:
            metrics.update(_grpo_metrics(trainer_base.tq, self.config, batch))
        except Exception:
            if not failure_logged:
                failure_logged = True
                logger.warning(
                    "Failed to compute oumi's GRPO metrics. "
                    "Training continues without them.",
                    exc_info=True,
                )

    setattr(compute_metrics, _PATCHED, True)
    trainer_cls._compute_metrics = compute_metrics
    logger.info("oumi GRPO metrics installed")


def _grpo_metrics(tq: Any, config: Any, batch: Any) -> dict[str, float]:
    keys = [
        key
        for key, tag in zip(batch.keys, batch.tags, strict=True)
        if not tag.get("is_padding", False)
    ]
    # verl computes GRPO advantages from each session's final output.
    keys = last_output_per_session(keys)
    reward_field = (
        "token_level_rewards" if config.algorithm.use_kl_in_reward else "rm_scores"
    )
    data = tq.kv_batch_get(
        keys=keys,
        partition_id=batch.partition_id,
        select_fields=["uid", reward_field, "response_mask"],
    )
    # Tool tokens count toward the response cap, but response_mask is 0 for them.
    lengths = np.asarray(data["response_mask"].offsets().diff().tolist())
    data = data.to_padded_tensor()
    group_ids = np.asarray(data.pop("uid").tolist())
    rewards = np.asarray(data[reward_field].sum(-1).tolist(), dtype=np.float64)
    rollout = config.actor_rollout_ref.rollout
    return {
        **compute_grpo_reward_group_metrics(rewards, group_ids),
        TRUNCATION_RATIO_KEY: compute_response_truncation_ratio(
            lengths, max_response_length=rollout.response_length
        ),
    }
