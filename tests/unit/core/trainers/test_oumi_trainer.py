from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader

from oumi.core.configs import TelemetryParams, TrainingParams
from oumi.core.configs.params.fsdp_params import FSDPParams
from oumi.core.trainers.oumi_trainer import Trainer
from oumi.models import MLPEncoder
from tests.markers import requires_gpus


#
# Fixtures
#
@pytest.fixture
def model():
    return MLPEncoder()


@pytest.fixture
def mock_model():
    return MagicMock(spec=torch.nn.Module)


@pytest.fixture
def mock_dataset():
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=10)
    dataset.state_dict = None
    return dataset


@pytest.fixture
def mock_stateful_dataloader():
    sample_batch = {
        "input_ids": torch.randint(0, 1000, (4, 768)),
        "attention_mask": torch.ones(4, 768),
        "labels": torch.randint(0, 1000, (4, 768)),
    }

    mock_loader = MagicMock(spec=StatefulDataLoader)
    mock_loader.__iter__.return_value = iter([sample_batch] * 10)
    mock_loader.__len__.return_value = 10

    return mock_loader


@pytest.fixture
def mock_dataloader():
    sample_batch = {
        "input_ids": torch.randint(0, 1000, (4, 768)),
        "attention_mask": torch.ones(4, 768),
        "labels": torch.randint(0, 1000, (4, 768)),
    }

    mock_loader = MagicMock(spec=DataLoader)
    mock_loader.__iter__.return_value = iter([sample_batch] * 10)
    mock_loader.__len__.return_value = 10

    return mock_loader


@pytest.fixture
def mock_optimizer():
    return MagicMock(spec=torch.optim.Optimizer)


@pytest.fixture
def mock_params():
    args = MagicMock(spec=TrainingParams)
    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.999
    args.adam_epsilon = 1e-8
    args.compile = False
    args.dataloader_num_workers = 0
    args.dataloader_prefetch_factor = 2
    args.enable_gradient_checkpointing = False
    args.enable_tensorboard = False
    args.enable_wandb = False
    args.enable_mlflow = False
    args.eval_steps = 50
    args.eval_strategy = "steps"
    args.gradient_accumulation_steps = 1
    args.learning_rate = 0.001
    args.learning_rate = 0.001
    args.logging_steps = 10
    args.lr_scheduler_kwargs = {}
    args.lr_scheduler_type = "linear"
    args.max_steps = 100
    args.num_train_epochs = 3
    args.optimizer = "adamw"
    args.output_dir = "/tmp/test_output"
    args.per_device_eval_batch_size = 8
    args.per_device_train_batch_size = 8
    args.save_epoch = True
    args.save_steps = 50
    args.warmup_ratio = None
    args.warmup_steps = 0
    args.weight_decay = 0.01
    return args


@pytest.fixture
def trainer(model, mock_tokenizer, mock_params, mock_dataset):
    return Trainer(
        model=model,
        processing_class=mock_tokenizer,
        processor=None,
        args=mock_params,
        train_dataset=mock_dataset,
        eval_dataset=mock_dataset,
    )


#
# Tests
#
def test_trainer_initialization(
    trainer, model, mock_tokenizer, mock_params, mock_dataset
):
    assert trainer.model == model
    assert trainer.processing_class == mock_tokenizer
    assert trainer.params == mock_params
    assert trainer.train_dataset == mock_dataset
    assert trainer.eval_dataset == mock_dataset
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    assert isinstance(trainer.train_dataloader, StatefulDataLoader)
    assert isinstance(trainer.eval_dataloader, DataLoader)
    assert trainer.state.epoch == 0
    assert trainer.state.global_step == 0


def test_get_total_training_steps(trainer):
    if trainer.params.max_steps is not None:
        assert trainer._estimate_total_training_steps() == trainer.params.max_steps


@patch("oumi.core.distributed.is_world_process_zero", return_value=True)
def test_train_with_num_epochs(mock_is_world_process_zero, trainer):
    trainer._train_epoch = MagicMock()
    trainer.save_state = MagicMock()
    trainer.evaluate = MagicMock()

    trainer.params.eval_strategy = "epoch"
    trainer.params.max_steps = -1  # Use `num_train_epochs`

    trainer.train()

    assert trainer._train_epoch.call_count == trainer.params.num_train_epochs
    assert trainer.save_state.call_count == trainer.params.num_train_epochs
    assert trainer.evaluate.call_count == trainer.params.num_train_epochs


@patch("oumi.core.distributed.is_world_process_zero", return_value=True)
def test_train_with_max_steps(mock_is_world_process_zero, trainer):
    steps_per_epoch: int = 5

    def _increment_global_step():
        trainer.state.global_step += steps_per_epoch

    trainer._train_epoch = MagicMock()
    trainer._train_epoch.side_effect = lambda pbar: _increment_global_step()
    trainer.save_state = MagicMock()
    trainer.evaluate = MagicMock()

    trainer.params.eval_strategy = "epoch"
    trainer.params.max_steps = 100
    expected_epochs = trainer.params.max_steps // steps_per_epoch

    trainer.train()

    assert trainer._train_epoch.call_count == expected_epochs
    assert trainer.save_state.call_count == expected_epochs
    assert trainer.evaluate.call_count == expected_epochs


def test_train_epoch(trainer, mock_stateful_dataloader, tmp_path):
    output_dir = tmp_path / "model_output"
    output_dir.mkdir()

    trainer._process_callbacks = MagicMock()
    trainer.telemetry.timer = MagicMock()
    trainer.model.forward = MagicMock(
        return_value={"loss": torch.tensor(0.5), "logits": torch.tensor([1.0, 2.0])}
    )
    trainer.train_dataloader = mock_stateful_dataloader
    trainer.params.telemetry = MagicMock(spec=TelemetryParams)
    trainer.params.telemetry.collect_telemetry_for_all_ranks = False
    trainer.params.telemetry_dir = MagicMock(return_value=(output_dir / "telemetry"))
    trainer.scaler.scale = MagicMock(return_value=MagicMock())
    trainer.scaler.step = MagicMock()
    trainer.scaler.update = MagicMock()

    progress_bar = MagicMock()
    trainer._train_epoch(progress_bar)

    assert trainer._process_callbacks.call_count > 0
    assert trainer.telemetry.timer.call_count > 0
    assert trainer.model.forward.call_count > 0
    assert trainer.scaler.scale.call_count > 0
    assert trainer.scaler.step.call_count > 0
    assert trainer.scaler.update.call_count > 0


def test_evaluate(trainer, mock_dataloader):
    trainer.model.eval = MagicMock()
    trainer.model.forward = MagicMock(return_value=MagicMock(loss=torch.tensor(0.5)))
    trainer.eval_dataloader = mock_dataloader

    results = trainer.evaluate()

    assert "val/loss" in results
    assert "val/perplexity" in results
    assert trainer.model.eval.call_count == 1
    assert trainer.model.forward.call_count > 0


@pytest.fixture
def mock_dcp_save():
    with patch("torch.distributed.checkpoint.save") as mock_save:
        yield mock_save


@pytest.fixture
def mock_dcp_load():
    with patch("torch.distributed.checkpoint.load") as mock_load:
        yield mock_load


@pytest.mark.parametrize("is_using_fsdp", [True, False])
def test_save_and_load_model(
    trainer,
    mock_model,
    mock_optimizer,
    mock_stateful_dataloader,
    tmp_path,
    mock_dcp_save,
    mock_dcp_load,
    is_using_fsdp,
):
    output_dir = tmp_path / "model_output"
    output_dir.mkdir()
    telemetry_dir = output_dir / "telemetry"
    telemetry_dir.mkdir()

    trainer.fsdp_params = FSDPParams(enable_fsdp=is_using_fsdp)
    trainer.is_using_fsdp = is_using_fsdp
    trainer.train_dataloader = mock_stateful_dataloader
    trainer.params.output_dir = str(output_dir)
    trainer.params.telemetry = MagicMock(spec=TelemetryParams)
    trainer.params.telemetry.collect_telemetry_for_all_ranks = False
    trainer.params.telemetry_dir = telemetry_dir

    trainer.train_dataloader.state_dict = MagicMock(
        return_value={"dataloader_key": torch.tensor(3)}
    )
    trainer.state.epoch = 1
    trainer.state.global_step = 50

    with patch("oumi.core.trainers.oumi_trainer.get_state_dict") as mock_get_state_dict:
        mock_get_state_dict.return_value = ({"model": "state"}, {"optimizer": "state"})

        trainer.save_state()

        mock_dcp_save.assert_called()

        assert (output_dir / "dataloader.pt").exists()
        assert (output_dir / "trainer_state.json").exists()
        assert (telemetry_dir / "telemetry_rank0000.json").exists()

        # Folder are created by DCP, but since it's a mock, we need to create
        # the folder manually.
        (output_dir / "model").mkdir()
        (output_dir / "optimizer").mkdir()

        # Test load
        trainer._load_from_checkpoint(str(output_dir))
        mock_dcp_load.assert_called()
        assert trainer.train_dataloader.load_state_dict.called
        assert trainer.state.epoch == 1
        assert trainer.state.global_step == 50


def test_get_train_dataloader(trainer):
    dataloader = trainer._get_train_dataloader()
    assert isinstance(dataloader, StatefulDataLoader)
    assert dataloader.batch_size == trainer.params.per_device_train_batch_size


def test_get_eval_dataloader(trainer):
    dataloader = trainer._get_eval_dataloader()
    assert isinstance(dataloader, DataLoader)
    assert dataloader.batch_size == trainer.params.per_device_eval_batch_size


def test_process_callbacks(trainer):
    mock_callback = MagicMock()
    mock_callback.on_log = MagicMock()
    trainer.callbacks = [mock_callback]

    logs = trainer._process_callbacks("on_log")

    assert mock_callback.on_log.called
    assert isinstance(logs, dict)


@requires_gpus()
def test_cuda_initialization(model, mock_tokenizer, mock_params, mock_dataset):
    assert next(model.parameters()).is_cpu
    trainer = Trainer(
        model=model,
        processing_class=mock_tokenizer,
        processor=None,
        args=mock_params,
        train_dataset=mock_dataset,
        eval_dataset=None,
    )
    assert next(model.parameters()).is_cuda, "Model should be on CUDA"
    assert trainer.device.startswith("cuda"), "Device should be CUDA"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_initialization(model, mock_tokenizer, mock_params, mock_dataset):
    assert next(model.parameters()).is_cpu, "Model should initially be on CPU"
    trainer = Trainer(
        model=model,
        processing_class=mock_tokenizer,
        processor=None,
        args=mock_params,
        train_dataset=mock_dataset,
        eval_dataset=None,
    )
    assert next(model.parameters()).is_mps, "Model should be on MPS"
    assert trainer.device == "mps", "Device should be MPS"
