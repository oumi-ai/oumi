import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from oumi_verl_ext.grpo_metrics import compute_grpo_metrics, patch_compute_metrics

# Keys are {uid}_{session}_{index}: p_0_1 supersedes p_0_0, and pad_0_0 is padding.
ROWS = {
    "p_0_0": ("p", [0.0]),
    "p_0_1": ("p", [0.0, 0.0, 0.0, 1.0]),
    "p_1_0": ("p", [0.0, 0.5]),
    "pad_0_0": ("pad", [1.0]),
}
BATCH = SimpleNamespace(
    keys=list(ROWS), tags=[{}, {}, {}, {"is_padding": True}], partition_id="train"
)
PLAIN_ROW = {"turn_scores": []}


class _Columns(dict):
    def to_padded_tensor(self):
        return {
            k: v if k == "uid" else v.to_padded_tensor(0.0) for k, v in self.items()
        }


def _kv_batch_get(keys, partition_id, select_fields):
    scores = [torch.tensor(ROWS[key][1]) for key in keys]
    return _Columns(
        uid=np.array([ROWS[key][0] for key in keys]),
        rm_scores=torch.nested.as_nested_tensor(scores, layout=torch.jagged),
        response_mask=torch.nested.as_nested_tensor(
            [torch.ones_like(s) for s in scores], layout=torch.jagged
        ),
    )


def _with_extra_fields(extra_fields: dict | None):
    """Also serves each key's extra_fields, or fails to as TransferQueue can."""

    def kv_batch_get(keys, partition_id, select_fields):
        if "extra_fields" not in select_fields:
            return _kv_batch_get(keys, partition_id, select_fields)
        if extra_fields is None:
            raise ValueError("Fields ['extra_fields'] are not ready")
        rows = np.empty(len(keys), dtype=object)
        rows[:] = [extra_fields[key] for key in keys]
        return {"extra_fields": rows}

    return kv_batch_get


def _patched_trainer(kv_batch_get):
    class PPOTrainer:
        config = SimpleNamespace(
            algorithm=SimpleNamespace(use_kl_in_reward=False),
            actor_rollout_ref=SimpleNamespace(
                rollout=SimpleNamespace(response_length=4)
            ),
        )

        def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
            metrics["critic/rewards/mean"] = 0.75

    trainer_base = types.ModuleType("trainer_base")
    trainer_base.__dict__.update(
        PPOTrainer=PPOTrainer, tq=SimpleNamespace(kv_batch_get=kv_batch_get)
    )
    patch_compute_metrics(trainer_base)
    return PPOTrainer()


def _step(trainer) -> dict:
    metrics = {}
    trainer._compute_metrics(BATCH, metrics, {}, global_steps=1, epoch=0)
    return metrics


def test_compute_grpo_metrics():
    std = np.std([0.0, 1.0], ddof=1)
    assert compute_grpo_metrics(
        rewards=np.array([1.0, 1.0, 0.0, 1.0]),
        group_ids=np.array(["a", "a", "b", "b"]),
        response_lengths=np.array([4, 2, 3, 1]),
        max_response_length=4,
    ) == pytest.approx(
        {
            "grpo/frac_reward_zero_std": 0.5,
            "grpo/frac_reward_low_std": 0.5,
            "grpo/reward_group_std/mean": std / 2,
            "grpo/reward_group_std/p50": std / 2,
            "grpo/reward_group_std/p90": 0.9 * std,
            "oumi/response_truncation_ratio": 0.25,
        }
    )


def test_patch_adds_metrics_from_each_session_s_last_output():
    requested = []

    def kv_batch_get(keys, partition_id, select_fields):
        requested.append(keys)
        return _kv_batch_get(keys, partition_id, select_fields)

    metrics = _step(_patched_trainer(kv_batch_get))

    assert requested == [["p_0_1", "p_1_0"]] * 2
    assert metrics["critic/rewards/mean"] == 0.75
    assert metrics["grpo/reward_group_std/mean"] == pytest.approx(
        np.std([1.0, 0.5], ddof=1)
    )
    assert metrics["oumi/response_truncation_ratio"] == 0.5


def test_a_metrics_failure_keeps_verl_s_metrics(caplog):
    def kv_batch_get(**kwargs):
        raise ValueError("not ready")

    assert _step(_patched_trainer(kv_batch_get)) == {"critic/rewards/mean": 0.75}
    assert "Skipped oumi's GRPO metrics" in caplog.text


def test_tool_counters_become_per_step_series():
    tool_row = {
        "turn_scores": [],
        "oumi_tool_calls": 2,
        "oumi_tool_calls_dropped": 1,
        "oumi_tool_calls_malformed": 0,
    }
    kv_batch_get = _with_extra_fields({"p_0_1": tool_row, "p_1_0": PLAIN_ROW})

    metrics = _step(_patched_trainer(kv_batch_get))

    assert metrics["critic/rewards/mean"] == 0.75
    assert {k: v for k, v in metrics.items() if "tool_calls" in k} == {
        "oumi/tool_calls/mean": 2.0,
        "oumi/tool_calls_dropped/mean": 1.0,
        "oumi/tool_calls_malformed/mean": 0.0,
    }


def test_a_run_without_tool_counters_keeps_its_grpo_metrics(caplog):
    kv_batch_get = _with_extra_fields({"p_0_1": PLAIN_ROW, "p_1_0": PLAIN_ROW})

    metrics = _step(_patched_trainer(kv_batch_get))

    assert "grpo/reward_group_std/mean" in metrics
    assert not any("tool_calls" in key for key in metrics)
    assert "Skipped oumi's" not in caplog.text


def test_a_tool_metrics_failure_keeps_the_grpo_metrics(caplog):
    metrics = _step(_patched_trainer(_with_extra_fields(None)))

    assert metrics["critic/rewards/mean"] == 0.75
    assert "grpo/reward_group_std/mean" in metrics
    assert not any("tool_calls" in key for key in metrics)
    assert "Skipped oumi's tool-call metrics" in caplog.text
    assert "Skipped oumi's GRPO metrics" not in caplog.text
