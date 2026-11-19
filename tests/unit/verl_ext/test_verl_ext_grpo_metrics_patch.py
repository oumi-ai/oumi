"""Tests the GRPO metrics patch against stand-ins for verl's trainer modules."""

import asyncio
import importlib
import importlib.abc
import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import oumi_verl_ext
from oumi_verl_ext import hooks, version
from oumi_verl_ext.grpo_metrics_patch import VerlPatchError, install_grpo_metrics
from oumi_verl_ext.metrics import TRUNCATION_RATIO_KEY
from oumi_verl_ext.version import EXPECTED_VERL_VERSION

TRAINER_BASE = "verl.trainer.ppo.v1.trainer_base"
MAX_RESPONSE_LENGTH = 4
VERL_METRICS = {"critic/rewards/mean": 0.75, "response_length/clip_ratio": 0.33}


def _row(uid, rm_scores, token_level_rewards, response_mask=None) -> dict:
    return {
        "uid": uid,
        "rm_scores": torch.tensor(rm_scores),
        "token_level_rewards": torch.tensor(token_level_rewards),
        "response_mask": torch.tensor(response_mask or [1] * len(rm_scores)),
    }


# Keys are {uid}_{session}_{index}. p0_0_1 supersedes p0_0_0 and reaches the maximum
# response length, with a tool token (mask 0); pad_0_0 is a padding row.
ROWS = {
    "p0_0_0": _row("p0", [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]),
    "p0_0_1": _row("p0", [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.25], [1, 1, 0, 1]),
    "p0_1_0": _row("p0", [0.0, 0.5], [0.0, 0.25]),
    "pad_0_0": _row("pad", [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]),
}


class _TensorDict(dict):
    """The part of the TensorDict TransferQueue returns that the metrics read."""

    def to_padded_tensor(self) -> dict:
        return {
            field: column if field == "uid" else column.to_padded_tensor(0.0)
            for field, column in self.items()
        }


class _FakeTQ:
    """Serves `ROWS` as jagged columns, the way TransferQueue does."""

    def __init__(self):
        self.requests: list[dict] = []

    def kv_batch_get(self, keys, partition_id, select_fields) -> _TensorDict:
        self.requests.append(
            {"keys": keys, "partition_id": partition_id, "select_fields": select_fields}
        )
        columns = {field: [ROWS[key][field] for key in keys] for field in select_fields}
        return _TensorDict(
            {
                field: np.array(column)
                if field == "uid"
                else torch.nested.as_nested_tensor(column, layout=torch.jagged)
                for field, column in columns.items()
            }
        )


def _fill_trainer_base(module: types.ModuleType) -> None:
    """Gives `module` the parts of verl's trainer_base that the patch uses."""

    class PPOTrainer:
        def __init__(self, *, use_kl_in_reward: bool = False):
            self.config = SimpleNamespace(
                algorithm=SimpleNamespace(use_kl_in_reward=use_kl_in_reward),
                actor_rollout_ref=SimpleNamespace(
                    rollout=SimpleNamespace(response_length=MAX_RESPONSE_LENGTH)
                ),
            )

        def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
            metrics.update(VERL_METRICS)

    module.__dict__.update(PPOTrainer=PPOTrainer, tq=_FakeTQ())


def _package(name: str) -> types.ModuleType:
    package = types.ModuleType(name)
    package.__path__ = []
    return package


@pytest.fixture
def verl_packages(monkeypatch):
    """Puts stand-ins for verl's packages above trainer_base in sys.modules."""
    for name in ("verl", "verl.trainer", "verl.trainer.ppo", "verl.trainer.ppo.v1"):
        monkeypatch.setitem(sys.modules, name, _package(name))


@pytest.fixture
def trainer_base(verl_packages, monkeypatch) -> types.ModuleType:
    module = types.ModuleType(TRAINER_BASE)
    _fill_trainer_base(module)
    monkeypatch.setitem(sys.modules, TRAINER_BASE, module)
    return module


def _batch() -> SimpleNamespace:
    tags = [{"is_padding": True} if key.startswith("pad") else {} for key in ROWS]
    return SimpleNamespace(keys=list(ROWS), tags=tags, partition_id="train")


def _logged(caplog) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == "oumi_verl_ext"]


def _step_metrics(trainer, global_steps: int = 1) -> dict:
    # verl's training step passes the last two arguments by keyword.
    metrics = {}
    trainer._compute_metrics(_batch(), metrics, {}, global_steps=global_steps, epoch=0)
    return metrics


def test_compute_metrics_adds_grpo_and_truncation_keys(trainer_base):
    install_grpo_metrics(trainer_base)
    metrics = _step_metrics(trainer_base.PPOTrainer())

    assert trainer_base.tq.requests == [
        {
            "keys": ["p0_0_1", "p0_1_0"],
            "partition_id": "train",
            "select_fields": ["uid", "rm_scores", "response_mask"],
        }
    ]
    assert metrics["grpo/frac_reward_zero_std"] == 0.0
    assert metrics["grpo/reward_group_std/mean"] == pytest.approx(
        np.std([1.0, 0.5], ddof=1)
    )
    assert metrics[TRUNCATION_RATIO_KEY] == pytest.approx(1 / 2)
    assert metrics["response_length/clip_ratio"] == 0.33
    assert metrics["critic/rewards/mean"] == 0.75


def test_compute_metrics_reads_token_level_rewards_when_kl_is_in_the_reward(
    trainer_base,
):
    install_grpo_metrics(trainer_base)
    metrics = _step_metrics(trainer_base.PPOTrainer(use_kl_in_reward=True))

    [request] = trainer_base.tq.requests
    assert request["select_fields"] == ["uid", "token_level_rewards", "response_mask"]
    assert metrics["grpo/frac_reward_zero_std"] == 1.0


def test_sync_and_async_trainers_inherit_the_metrics(trainer_base):
    # verl's sync, colocate_async and separate_async trainers all inherit
    # _compute_metrics from PPOTrainer.
    class PPOTrainerSync(trainer_base.PPOTrainer):
        pass

    class PPOTrainerColocateAsync(trainer_base.PPOTrainer):
        async def fit(self):
            return _step_metrics(self)

    install_grpo_metrics(trainer_base)

    sync_metrics = _step_metrics(PPOTrainerSync())
    async_metrics = asyncio.run(PPOTrainerColocateAsync().fit())
    for metrics in (sync_metrics, async_metrics):
        assert metrics[TRUNCATION_RATIO_KEY] == pytest.approx(1 / 2)
        assert "grpo/frac_reward_zero_std" in metrics


def test_installing_twice_patches_once(trainer_base, caplog):
    caplog.set_level(logging.INFO, logger="oumi_verl_ext")
    install_grpo_metrics(trainer_base)
    patched = trainer_base.PPOTrainer._compute_metrics
    install_grpo_metrics(trainer_base)

    assert trainer_base.PPOTrainer._compute_metrics is patched
    _step_metrics(trainer_base.PPOTrainer())
    assert len(trainer_base.tq.requests) == 1
    assert [record.getMessage() for record in _logged(caplog)] == [
        "oumi GRPO metrics installed"
    ]


def test_a_changed_signature_is_refused(trainer_base):
    def compute_metrics(self, batch, metrics, timing_raw, global_steps):
        pass

    trainer_base.PPOTrainer._compute_metrics = compute_metrics
    with pytest.raises(VerlPatchError, match=r"'global_steps'\).*'epoch'\)"):
        install_grpo_metrics(trainer_base)
    assert trainer_base.PPOTrainer._compute_metrics is compute_metrics


def test_a_trainer_without_compute_metrics_is_refused(trainer_base):
    del trainer_base.PPOTrainer._compute_metrics
    with pytest.raises(VerlPatchError, match="has no _compute_metrics"):
        install_grpo_metrics(trainer_base)


def test_a_metric_failure_warns_once_and_keeps_verl_s_metrics(trainer_base, caplog):
    def kv_batch_get(**kwargs):
        raise ValueError("field not ready")

    trainer_base.tq = SimpleNamespace(kv_batch_get=kv_batch_get)
    install_grpo_metrics(trainer_base)
    caplog.set_level(logging.WARNING, logger="oumi_verl_ext")

    for step in (1, 2):
        assert _step_metrics(trainer_base.PPOTrainer(), step) == VERL_METRICS

    warnings = [r for r in _logged(caplog) if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].exc_info is not None


# These tests import the stand-in trainer_base through the import system, where a
# real verl could serve its own trainer_base instead.
_without_verl = pytest.mark.skipif(
    importlib.util.find_spec("verl") is not None,
    reason="A real verl is installed, so the stand-in trainer_base cannot be imported.",
)


class _TrainerBaseFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Imports the stand-in trainer_base through the normal import system."""

    def find_spec(self, fullname, path, target=None):
        if fullname != TRAINER_BASE:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def exec_module(self, module):
        _fill_trainer_base(module)


@pytest.fixture
def import_state(verl_packages, monkeypatch):
    monkeypatch.setattr(sys, "meta_path", [*sys.meta_path, _TrainerBaseFinder()])
    monkeypatch.setattr(hooks, "_HOOKS", hooks._PostImportHooks())
    monkeypatch.setattr(
        version.importlib.metadata, "version", lambda name: EXPECTED_VERL_VERSION
    )
    # install.py sets the extension logger's level, which outlives the test.
    logger = logging.getLogger("oumi_verl_ext")
    level = logger.level
    logger.setLevel(logging.NOTSET)
    yield
    logger.setLevel(level)
    sys.modules.pop(TRAINER_BASE, None)


@pytest.fixture
def warning_root_logger():
    """Sets the root logger to WARNING, as ``import verl`` does."""
    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.WARNING)
    yield
    root.setLevel(level)


def _load_install() -> None:
    # Loaded under another name so the real oumi_verl_ext.install stays unimported.
    path = Path(oumi_verl_ext.__file__).with_name("install.py")
    spec = importlib.util.spec_from_file_location("install_under_test", path)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


@_without_verl
@pytest.mark.usefixtures("import_state")
def test_loading_the_extension_patches_trainer_base_once_verl_imports_it():
    _load_install()
    assert TRAINER_BASE not in sys.modules

    trainer_base = importlib.import_module(TRAINER_BASE)

    metrics = _step_metrics(trainer_base.PPOTrainer())
    assert metrics[TRUNCATION_RATIO_KEY] == pytest.approx(1 / 2)


@_without_verl
@pytest.mark.usefixtures("import_state", "warning_root_logger")
def test_the_install_line_shows_under_verl_s_warning_root_logger(caplog):
    _load_install()
    importlib.import_module(TRAINER_BASE)

    assert [(record.levelno, record.getMessage()) for record in _logged(caplog)] == [
        (logging.INFO, "oumi GRPO metrics installed")
    ]
