import contextlib
import os
import time
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Optional, cast

import pydantic
import torch
import torch.amp
import torch.utils.tensorboard as tensorboard

import wandb  # isort: skip
import safetensors.torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm
from transformers import TrainerCallback

from lema.builders.lr_schedules import build_lr_scheduler
from lema.builders.optimizers import build_optimizer
from lema.core.distributed import (
    get_device_rank_info,
    global_leader_only,
    is_distributed,
    is_local_process_zero,
    is_world_process_zero,
    local_leader_only,
    prepare_model_for_distributed,
)
from lema.core.types import TrainingConfig, TrainingParams
from lema.core.types.base_tokenizer import BaseTokenizer
from lema.core.types.base_trainer import BaseTrainer
from lema.performance.telemetry import TelemetryTracker
from lema.utils.io_utils import load_json, save_json
from lema.utils.logging import logger
from lema.utils.torch_utils import log_trainable_parameters

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn


class TrainingState(pydantic.BaseModel):
    epoch: int = 0
    global_step: int = 0
    total_tokens_seen: int = 0


class Trainer(BaseTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: BaseTokenizer,
        args: TrainingParams,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        **kwargs,
    ):
        """Initializes the LeMa trainer."""
        self.model = model
        self.tokenizer = tokenizer
        self.params = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.max_norm: float = args.max_grad_norm

        self.params.validate()

        self.state = TrainingState()

        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        # Enable mixed precision bf16/fp16 training if requested.
        # Model dtype has been verified to not be bf16/fp16 if this is the case.
        self.mixed_precision_ctx = contextlib.nullcontext()
        self.mixed_precision_dtype = None
        if self.params.bf16:
            self.mixed_precision_dtype = torch.bfloat16
        if self.params.fp16:
            self.mixed_precision_dtype = torch.float16
        if self.mixed_precision_dtype:
            self.mixed_precision_ctx = torch.amp.autocast(
                device_type=self.device_type,
                enabled=True,
                dtype=self.mixed_precision_dtype,
            )

        if self.params.compile:
            self.log("Compiling model...")
            model = cast(torch.nn.Module, torch.compile(model))

        self.scaler = torch.amp.GradScaler(device=self.device_type, enabled=False)

        device_info = get_device_rank_info()

        # TODO: OPE-218 - give users fine-grained control on device placement
        # TODO: OPE-217 - non-leader models should be on meta
        if torch.cuda.is_available():
            self.device = f"cuda:{device_info.local_rank}"
            torch.cuda.set_device(self.device)
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model.to(self.device)

        # TODO: OPE-219 - hook-up fsdp flag
        if is_distributed():
            # Wrap model for distributed training
            model = prepare_model_for_distributed(model, use_fsdp=False)

        self.callbacks = callbacks if callbacks is not None else []

        self.optimizer = build_optimizer(self.model, self.params)
        self.lr_scheduler = build_lr_scheduler(
            optimizer=self.optimizer,
            training_params=self.params,
            current_epoch=self.state.epoch,
            num_training_steps=self._get_total_training_steps(),
        )

        self.train_dataloader = self._get_train_dataloader()
        self.eval_dataloader = self._get_eval_dataloader() if eval_dataset else None

        self.telemetry = TelemetryTracker()
        self.start_time = time.perf_counter()
        self._init_logging()

    #
    # Training
    #
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Trains the model."""
        if resume_from_checkpoint:
            self._load_from_checkpoint(resume_from_checkpoint)

        if is_local_process_zero():
            log_trainable_parameters(self.model)

        total_steps = self._get_total_training_steps()

        self.start_time = time.perf_counter()

        with tqdm(
            total=total_steps,
            desc="Training",
            disable=not is_world_process_zero(),
        ) as progress_bar:
            for epoch in range(self.state.epoch, self.params.num_train_epochs):
                self._set_sampler_epoch(epoch)
                self._train_epoch(progress_bar)

                if self.params.save_epoch:
                    self.save_state()

                if (
                    self.eval_dataloader
                    and self.params.eval_strategy == "epoch"
                    and is_world_process_zero()
                ):
                    # TODO: OPE-223 - only the global leader is used for evaluation
                    # To enable distributed evaluation, th eval function needs
                    # to be updated to aggregate metrics accross all workers.
                    self.evaluate()

                self.state.epoch += 1

                if self.state.global_step >= total_steps:
                    self.log(f"Reached {total_steps} global steps. Training completed.")
                    self.log(
                        f"Training runtime: {time.perf_counter() - self.start_time}s"
                    )
                    break

    def _train_epoch(self, progress_bar: tqdm) -> None:
        """Trains the model for one epoch."""
        self.model.train()
        torch.cuda.empty_cache()
        self.optimizer.zero_grad(set_to_none=True)
        micro_step = 0

        data_iter = iter(self.train_dataloader)

        while True:
            if micro_step % self.params.gradient_accumulation_steps == 0:
                self._process_callbacks("on_step_begin")

            with self.telemetry.timer("fetching batch"):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    self.log("End of epoch")
                    return

            with self.telemetry.timer("moving batch to device"):
                batch = {
                    k: v.to(self.device, non_blocking=True) for k, v in batch.items()
                }

            with self.telemetry.timer("computing tokens"):
                num_tokens = batch["input_ids"].ne(self.tokenizer.pad_token_id).sum()

            with self.telemetry.timer("syncing to cpu"):
                num_tokens = num_tokens.item()
                self.state.total_tokens_seen += num_tokens

            with self.mixed_precision_ctx, self.telemetry.timer("model forward"):
                self.model.require_backward_grad_sync = (  # type: ignore
                    micro_step + 1
                ) % self.params.gradient_accumulation_steps == 0

                outputs = self.model(**batch)
                loss = outputs["loss"] / self.params.gradient_accumulation_steps
                # assert loss.dtype is torch.bfloat16
                # assert outputs["logits"].dtype is torch.bfloat16

            with self.telemetry.timer("loss backward"):
                self.scaler.scale(loss).backward()

            if (micro_step + 1) % self.params.gradient_accumulation_steps == 0:
                with self.telemetry.timer("optimizer step"):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.max_norm
                    )

                    # save lr for logging
                    last_lr = self.lr_scheduler.get_last_lr()[0]

                    # step optimizer, scaler, and lr schedule
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.lr_scheduler.step()

                    self.optimizer.zero_grad(set_to_none=True)

                self.state.global_step += 1
                progress_bar.update(1)

                self._process_callbacks("on_step_end")

                if self.state.global_step % self.params.logging_steps == 0:
                    # Log metrics
                    elapsed = time.perf_counter() - self.start_time
                    loss_value = loss.item() * self.params.gradient_accumulation_steps
                    metrics = {
                        "train/loss": loss_value,
                        "learning_rate": last_lr,
                        "epoch": self.state.epoch,
                        "global_step": self.state.global_step,
                        "total_tokens_seen": self.state.total_tokens_seen,
                        "global_steps_per_second": self.state.global_step / elapsed,
                        "tokens_per_second": self.state.total_tokens_seen / elapsed,
                        "tokens_per_step_per_gpu": self.state.total_tokens_seen
                        / self.state.global_step,
                    }
                    callback_metrics = self._process_callbacks("on_log")
                    metrics.update(callback_metrics)

                    self.log_metrics(metrics, self.state.global_step)

                    if is_local_process_zero():
                        self.telemetry.print_summary()

                if (
                    self.params.save_steps > 0
                    and self.state.global_step % self.params.save_steps == 0
                ):
                    self.save_state()

                if (
                    self.eval_dataloader
                    and self.params.eval_steps > 0
                    and self.state.global_step % self.params.eval_steps == 0
                    and is_world_process_zero()
                ):
                    # TODO: OPE-223 - only the global leader is used for evaluation
                    # To enable distributed evaluation, th eval function needs
                    # to be updated to aggregate metrics accross all workers.
                    self.evaluate()

            if self.state.global_step >= self.params.max_steps:
                break

            micro_step += 1

    #
    # Evaluation
    #
    @torch.no_grad
    def evaluate(self) -> Dict[str, float]:
        """Evaluates the model on the evaluation dataset."""
        if self.eval_dataloader is None:
            raise ValueError("No evaluation dataloader provided.")

        self.model.eval()
        eval_losses = []

        for batch in tqdm(
            self.eval_dataloader,
            desc="Evaluating",
            disable=not is_local_process_zero(),
        ):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            eval_losses.append(outputs.loss.item())

        eval_loss = sum(eval_losses) / len(eval_losses)
        perplexity = torch.exp(torch.tensor(eval_loss))

        results = {"val/loss": eval_loss, "val/perplexity": perplexity.item()}

        self.log("Finished evaluation.")
        self.log_metrics(results, self.state.global_step)

        self.model.train()
        return results

    #
    # Checkpointing
    #
    def save_model(self, config: TrainingConfig):
        """Saves the model."""
        if is_world_process_zero():
            output_dir = Path(config.training.output_dir)
            output_dir.mkdir(exist_ok=True)
            model_path = output_dir / "model.safetensors"
            safetensors.torch.save_model(model=self.model, filename=str(model_path))
            self.log(f"Model saved to {model_path}.")

    def save_state(self):
        """Saves the training state."""
        checkpoint_dir = Path(self.params.output_dir)

        if is_world_process_zero():
            checkpoint_dir.mkdir(exist_ok=True)

            model_path = checkpoint_dir / "model.safetensors"
            optimizer_path = checkpoint_dir / "optimizer.pt"
            dataloader_state_path = checkpoint_dir / "dataloader.pt"
            trainer_state_path = checkpoint_dir / "trainer_state.json"
            telemetry_state_path = checkpoint_dir / "telemetry.json"

            safetensors.torch.save_model(model=self.model, filename=str(model_path))
            torch.save(
                self.optimizer.state_dict(),
                optimizer_path,
            )
            torch.save(
                self.train_dataloader.state_dict(),
                dataloader_state_path,
            )
            save_json(
                data=self.state.model_dump(),
                filename=trainer_state_path,
            )
            save_json(
                data=self.telemetry.state_dict(),
                filename=telemetry_state_path,
            )
            logger.info(f"Training state saved to {checkpoint_dir}")

    def _load_from_checkpoint(self, checkpoint_dirname: str):
        """Loads the training state from a checkpoint."""
        checkpoint_dir = Path(checkpoint_dirname)

        model_path = checkpoint_dir / "model.safetensors"
        optimizer_path = checkpoint_dir / "optimizer.pt"
        dataloader_state_path = checkpoint_dir / "dataloader.pt"
        trainer_state_path = checkpoint_dir / "trainer_state.json"
        telemetry_state_path = checkpoint_dir / "telemetry.json"

        if model_path.exists():
            safetensors.torch.load_model(
                self.model, filename=str(model_path), strict=True, device=self.device
            )
        if optimizer_path.exists():
            self.optimizer.load_state_dict(
                torch.load(optimizer_path, map_location=self.device, weights_only=True)
            )
        if dataloader_state_path.exists():
            self.train_dataloader.load_state_dict(torch.load(dataloader_state_path))
        if trainer_state_path.exists():
            self.state = TrainingState.model_validate(
                load_json(trainer_state_path), strict=True
            )
        if telemetry_state_path.exists():
            self.telemetry.load_state_dict(load_json(telemetry_state_path))

        self.log(f"Resumed training from checkpoint: {checkpoint_dirname}")

    #
    # Logging
    #
    @local_leader_only()
    def log(self, message: str):
        """Logs a message if the process is the local process zero."""
        logger.info(message)

    @global_leader_only()
    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Logs metrics to wandb and tensorboard."""
        # Log to console and log file
        self.log(pformat(metrics))

        # Log to Weights and Biases
        if self.params.enable_wandb:
            wandb.log(metrics, step=self.state.global_step)

        # Log to tensorboard
        if self.params.enable_tensorboard and self.tensorboard_writer:
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(key, value, self.state.global_step)

    def _init_logging(
        self,
    ) -> None:
        """Initializes logging."""
        if not is_world_process_zero():
            return

        self.log(f"Logging to {self.params.output_dir}")

        if self.params.enable_wandb:
            project_name = os.environ.get("WANDB_PROJECT", "lema")
            self.log(f"Logging to Weights and Biases project: '{project_name}'")
            wandb.init(project=project_name, name=self.params.run_name)
            wandb.watch(self.model)

        if self.params.enable_tensorboard:
            tensorboard_folder = Path(self.params.output_dir) / "tensorboard"
            self.log(f"Logging to tensorboard folder: '{tensorboard_folder}'")
            self.tensorboard_writer = tensorboard.SummaryWriter(
                log_dir=tensorboard_folder
            )
        else:
            self.tensorboard_writer = None

    #
    # Data loading
    #
    def _get_train_dataloader(self) -> StatefulDataLoader:
        """Returns the training dataloader."""
        # At this point, "auto" must be pre-resolved to `int`.
        assert isinstance(self.params.dataloader_num_workers, int)
        prefetch_factor = (
            self.params.dataloader_prefetch_factor
            if self.params.dataloader_num_workers > 0
            else None
        )

        # IterDataPipe is a subclass of IterableDataset.
        if isinstance(self.train_dataset, IterableDataset):
            # TODO: configure sharding for iterable datasets
            sampler = None
            shuffle = None
        else:
            # Configure sampler for map datasets. If using multiple GPUs,
            # we use a DistributedSampler to make sure each worker gets a
            # different subset of the dataset.
            # In non-distributed mode, we iterate over the full dataset.
            if is_distributed():
                # TODO: OPE-219 this strategy should only be enabled for DDP
                # and FSDP with NO_SHARDING
                device_info = get_device_rank_info()

                # Distribute the dataset across all GPU workers
                # Each rank will get a subset of the dataset
                sampler = DistributedSampler(
                    self.train_dataset,
                    num_replicas=device_info.world_size,
                    rank=device_info.rank,
                    seed=self.params.seed,
                    shuffle=True,
                )
                shuffle = False
            else:
                # If not distributed, let the dataloader handle shuffling
                sampler = None
                shuffle = True

        # Keeping track of the sampler so we can update after each epoch
        self._sampler = sampler

        return StatefulDataLoader(
            self.train_dataset,
            batch_size=self.params.per_device_train_batch_size,
            shuffle=shuffle,
            sampler=self._sampler,
            num_workers=self.params.dataloader_num_workers,
            pin_memory=self.device_type == "cuda",
            prefetch_factor=prefetch_factor,
            pin_memory_device=self.device,
            snapshot_every_n_steps=self.params.save_steps,
        )

    def _get_eval_dataloader(self) -> DataLoader:
        """Returns the evaluation dataloader."""
        if not self.eval_dataset:
            raise ValueError("No evaluation dataset provided.")

        # At this point, "auto" must be pre-resolved to `int`.
        assert isinstance(self.params.dataloader_num_workers, int)
        return DataLoader(
            self.eval_dataset,
            batch_size=self.params.per_device_eval_batch_size,
            shuffle=False,
            num_workers=self.params.dataloader_num_workers,
        )

    def _get_total_training_steps(self) -> int:
        # TODO: handle num_epochs, len(dataset), etc
        return self.params.max_steps

    def _set_sampler_epoch(self, epoch: int) -> None:
        """Sets the current epoch on sampler, if it exists and supports it."""
        if self._sampler and hasattr(self._sampler, "set_epoch"):
            self.log(f"Setting sampler epoch to {epoch}.")
            self._sampler.set_epoch(epoch)

    #
    # Handle callbacks
    #
    def _process_callbacks(self, event: str) -> Dict[str, Any]:
        """Process callbacks.

        Extremely hacky way to handle HF callbacks.
        Just here to unblock debugging with our MfuCallback
        """
        logs = {}

        for callback in self.callbacks:
            if hasattr(callback, event):
                action = getattr(callback, event)
                action(args=self.params, state=None, control=None, logs=logs)

        return logs
