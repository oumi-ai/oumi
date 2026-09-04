import logging
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from oumi.core.trainers import verl_grpo_metrics
from oumi.core.trainers.verl_grpo_metrics import (
    compute_grpo_reward_group_metrics,
    install_verl_grpo_reward_group_metrics_patch,
)


def _make_batch(rows: list[tuple[str, list[float]]]):
    return SimpleNamespace(
        batch={
            "token_level_rewards": torch.tensor(
                [token_rewards for _, token_rewards in rows], dtype=torch.float32
            )
        },
        non_tensor_batch={"uid": [uid for uid, _ in rows]},
    )


def test_compute_reward_group_metrics_for_binary_rewards():
    # The UIDs are deliberately interleaved because verl may reorder rollouts when
    # balancing sequence lengths across data-parallel ranks.
    batch = _make_batch(
        [
            ("always-fails", [0.0, 0.0]),
            ("mixed", [0.0, 0.0]),
            ("always-succeeds", [1.0, 0.0]),
            ("mixed", [1.0, 0.0]),
            ("always-fails", [0.0, 0.0]),
            ("mixed", [0.0, 0.0]),
            ("always-succeeds", [1.0, 0.0]),
            ("mixed", [1.0, 0.0]),
            ("always-fails", [0.0, 0.0]),
            ("always-succeeds", [1.0, 0.0]),
            ("always-fails", [0.0, 0.0]),
            ("always-succeeds", [1.0, 0.0]),
        ]
    )

    metrics = compute_grpo_reward_group_metrics(
        batch,
        low_std_threshold=0.1,
    )

    mixed_std = torch.tensor([0.0, 1.0, 0.0, 1.0]).std().item()
    expected_stds = torch.tensor([0.0, mixed_std, 0.0])
    expected_quantiles = torch.quantile(expected_stds, torch.tensor([0.5, 0.9]))
    assert metrics["grpo/frac_reward_zero_std"] == pytest.approx(2 / 3)
    assert metrics["grpo/frac_reward_low_std"] == pytest.approx(2 / 3)
    assert metrics["grpo/reward_group_std/mean"] == pytest.approx(
        expected_stds.mean().item()
    )
    assert metrics["grpo/reward_group_std/p50"] == pytest.approx(
        expected_quantiles[0].item()
    )
    assert metrics["grpo/reward_group_std/p90"] == pytest.approx(
        expected_quantiles[1].item()
    )


def test_compute_reward_group_metrics_sums_token_rewards_and_uses_threshold():
    batch = _make_batch(
        [
            ("low", [0.01, 0.01]),
            ("high", [0.0, 0.0]),
            ("low", [0.02, 0.01]),
            ("high", [1.0, 0.0]),
        ]
    )
    rewards_before = batch.batch["token_level_rewards"].clone()

    metrics = compute_grpo_reward_group_metrics(
        batch,
        low_std_threshold=0.01,
    )

    assert metrics["grpo/frac_reward_zero_std"] == 0.0
    assert metrics["grpo/frac_reward_low_std"] == 0.5
    torch.testing.assert_close(
        batch.batch["token_level_rewards"],
        rewards_before,
    )


def test_compute_reward_group_metrics_treats_singleton_as_zero_std():
    batch = _make_batch([("only", [0.5])])

    metrics = compute_grpo_reward_group_metrics(
        batch,
        low_std_threshold=0.0,
    )

    assert metrics == {
        "grpo/frac_reward_zero_std": 1.0,
        "grpo/frac_reward_low_std": 1.0,
        "grpo/reward_group_std/mean": 0.0,
        "grpo/reward_group_std/p50": 0.0,
        "grpo/reward_group_std/p90": 0.0,
    }


def test_low_std_fraction_includes_effectively_zero_groups():
    batch = _make_batch([("prompt", [0.0]), ("prompt", [1e-9])])

    metrics = compute_grpo_reward_group_metrics(
        batch,
        low_std_threshold=0.0,
    )

    assert metrics["grpo/frac_reward_zero_std"] == 1.0
    assert metrics["grpo/frac_reward_low_std"] == 1.0


@pytest.mark.parametrize("threshold", [-1.0, math.inf, -math.inf, math.nan])
def test_compute_reward_group_metrics_rejects_invalid_threshold(threshold):
    batch = _make_batch([("prompt", [0.0]), ("prompt", [1.0])])

    with pytest.raises(ValueError, match="low_std_threshold"):
        compute_grpo_reward_group_metrics(batch, low_std_threshold=threshold)


def test_compute_reward_group_metrics_validates_batch_shape():
    batch = _make_batch([("prompt", [0.0]), ("prompt", [1.0])])
    batch.non_tensor_batch["uid"].pop()

    with pytest.raises(ValueError, match="number of prompt UIDs"):
        compute_grpo_reward_group_metrics(batch, low_std_threshold=0.1)


def test_install_patch_preserves_original_metrics_and_upstream_values(monkeypatch):
    upstream_mean = 123.0

    def original_collector(batch, use_critic=True):
        return {
            "original": use_critic,
            "grpo/reward_group_std/mean": upstream_mean,
        }

    fake_module = SimpleNamespace(compute_data_metrics=original_collector)
    monkeypatch.setattr(
        verl_grpo_metrics,
        "import_module",
        lambda _: fake_module,
    )
    install_verl_grpo_reward_group_metrics_patch(low_std_threshold=0.1)

    metrics = fake_module.compute_data_metrics(
        _make_batch([("prompt", [0.0]), ("prompt", [1.0])]),
        use_critic=False,
    )

    assert metrics["original"] is False
    assert metrics["grpo/reward_group_std/mean"] == upstream_mean
    assert "grpo/frac_reward_zero_std" in metrics
    assert "grpo/frac_reward_low_std" in metrics


def test_install_patch_is_idempotent_and_updates_threshold(monkeypatch):
    def original_collector(batch, use_critic=True):
        return {"original": use_critic}

    fake_module = SimpleNamespace(compute_data_metrics=original_collector)
    monkeypatch.setattr(
        verl_grpo_metrics,
        "import_module",
        lambda _: fake_module,
    )
    batch = _make_batch([("prompt", [0.0]), ("prompt", [0.2])])

    install_verl_grpo_reward_group_metrics_patch(low_std_threshold=0.1)
    wrapper = fake_module.compute_data_metrics
    assert wrapper(batch)["grpo/frac_reward_low_std"] == 0.0

    install_verl_grpo_reward_group_metrics_patch(low_std_threshold=0.2)
    assert fake_module.compute_data_metrics is wrapper
    assert wrapper(batch)["grpo/frac_reward_low_std"] == 1.0


def test_install_patch_warns_once_and_preserves_training_metrics(
    monkeypatch,
    caplog,
):
    def original_collector(batch, use_critic=True):
        return {"original": use_critic}

    fake_module = SimpleNamespace(compute_data_metrics=original_collector)
    monkeypatch.setattr(
        verl_grpo_metrics,
        "import_module",
        lambda _: fake_module,
    )
    install_verl_grpo_reward_group_metrics_patch(low_std_threshold=0.1)
    invalid_batch = SimpleNamespace(batch={}, non_tensor_batch={})

    with caplog.at_level(logging.WARNING, logger="oumi"):
        first_metrics = fake_module.compute_data_metrics(invalid_batch)
        second_metrics = fake_module.compute_data_metrics(invalid_batch)

    assert first_metrics == {"original": True}
    assert second_metrics == {"original": True}
    assert caplog.text.count("Failed to compute Oumi's verl GRPO") == 1


def test_install_patch_rejects_incompatible_collector(monkeypatch):
    fake_module = SimpleNamespace(compute_data_metrics=lambda data: {})
    monkeypatch.setattr(
        verl_grpo_metrics,
        "import_module",
        lambda _: fake_module,
    )

    with pytest.raises(RuntimeError, match="does not accept a 'batch' argument"):
        install_verl_grpo_reward_group_metrics_patch()


def test_install_patch_rejects_missing_collector(monkeypatch):
    monkeypatch.setattr(
        verl_grpo_metrics,
        "import_module",
        lambda _: SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="does not expose a callable"):
        install_verl_grpo_reward_group_metrics_patch()


def test_install_patch_rejects_invalid_threshold_before_import(monkeypatch):
    import_mock = MagicMock()
    monkeypatch.setattr(verl_grpo_metrics, "import_module", import_mock)

    with pytest.raises(ValueError, match="low_std_threshold"):
        install_verl_grpo_reward_group_metrics_patch(low_std_threshold=-1.0)

    import_mock.assert_not_called()
