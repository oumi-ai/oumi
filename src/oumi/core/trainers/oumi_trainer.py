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

import contextlib
import copy
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path
from pprint import pformat
from typing import Any, Callable, Optional, cast

import pydantic
import safetensors.torch
import torch
import torch.amp
import torch.distributed.checkpoint as dcp
import torch.utils.tensorboard as tensorboard

import mlflow  # isort: skip
import transformers

import wandb  # isort: skip
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
)
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm
from transformers import TrainerCallback

from oumi.core.configs import MixedPrecisionDtype, TrainingConfig, TrainingParams
from oumi.core.configs.params.fsdp_params import FSDPParams, StateDictType
from oumi.core.distributed import (
    barrier,
    get_device_rank_info,
    is_distributed,
    is_local_process_zero,
    is_world_process_zero,
    prepare_model_for_distributed,
)
from oumi.core.processors.base_processor import BaseProcessor
from oumi.core.tokenizers import BaseTokenizer
from oumi.core.trainers.base_trainer import BaseTrainer
from oumi.models.layers.ring_attention import (
    apply_zigzag_ring_attn_monkey_patch_llama as apply_ring_attention_monkey_patch,
)
from oumi.models.layers.ring_attention import (
    prepare_zigzag_ring_attn_inputs as prepare_seq_parallel_inputs,
)
from oumi.performance.telemetry import TelemetryTracker
from oumi.utils.io_utils import load_json, save_json
from oumi.utils.logging import logger

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
        processing_class: Optional[BaseTokenizer],
        args: TrainingParams,
        train_dataset: Dataset,
        processor: Optional[BaseProcessor] = None,
        eval_dataset: Optional[Dataset] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        data_collator: Optional[Callable] = None,
        config: Optional[TrainingConfig] = None,
        **kwargs,
    ):
        """Initializes the Oumi trainer."""
        # Importing these here to avoid circular dependencies
        from oumi.builders.lr_schedules import build_lr_scheduler
        from oumi.builders.optimizers import build_optimizer

        self.telemetry = TelemetryTracker()
        self.start_time = time.perf_counter()
        self.collator_fn = data_collator

        self.processing_class = processing_class
        self._processor = processor
        self.params = copy.deepcopy(args)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.max_norm = (
            float(args.max_grad_norm) if args.max_grad_norm is not None else None
        )

        self.config = config or TrainingConfig()
        self.fsdp_params = self.config.fsdp or FSDPParams()
        self.is_using_fsdp = self.fsdp_params.enable_fsdp
        # TODO OPE-333 Define a param to enable ring attention + check pre-conditions:
        # 1. Flash Attention (`is_ring_attention_available()`),
        # 2. CUDA and distributed multi-GPU training (otherwise, pointless).
        # 3. Supported model type.
        self.is_using_ring_attention = False

        self.params.finalize_and_validate()

        self.state = TrainingState()
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"

        # Enable mixed precision bf16/fp16 training if requested.
        # Model dtype has been verified to be fp32 if this is the case.
        self.mixed_precision_ctx = contextlib.nullcontext()
        mixed_precision_dtype = None
        if self.params.mixed_precision_dtype == MixedPrecisionDtype.BF16:
            mixed_precision_dtype = torch.bfloat16
        elif self.params.mixed_precision_dtype == MixedPrecisionDtype.FP16:
            mixed_precision_dtype = torch.float16
        if mixed_precision_dtype:
            self.mixed_precision_ctx = torch.amp.autocast(
                device_type=self.device_type,
                enabled=True,
                dtype=mixed_precision_dtype,
            )

        # We want to enable gradient scaling for fp16 mixed precision training
        # to prevent gradient underflows. This is not needed for bf16 since it has the
        # same dynamic range as fp32. See here for details:
        # https://pytorch.org/docs/stable/amp.html#gradient-scaling
        self.scaler = torch.amp.GradScaler(
            device=self.device_type,
            enabled=self.params.mixed_precision_dtype == MixedPrecisionDtype.FP16,
        )

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

        # ----------------------------------
        # Prepare model for training
        # ----------------------------------
        if args.enable_gradient_checkpointing:
            if not isinstance(model, transformers.PreTrainedModel):
                raise ValueError(
                    "Gradient checkpointing is only supported for transformers models."
                )
            model.gradient_checkpointing_enable(args.gradient_checkpointing_kwargs)
        model = cast(torch.nn.Module, model)
        model.to(self.device)
        if is_distributed():
            # Wrap model for distributed training
            with self._telemetry_block("wrap model for distributed"):
                model = prepare_model_for_distributed(
                    model,
                    self.config,
                    ddp_find_unused_parameters=self.params.ddp_find_unused_parameters,
                )
                # Apply ring attention monkey patch if enabled
                if self.is_using_ring_attention:
                    apply_ring_attention_monkey_patch()

        if self.params.compile:
            self.log("Compiling model...")
            with self._telemetry_block("compile model"):
                model = cast(torch.nn.Module, torch.compile(model))
        self.model = model

        self.callbacks = callbacks if callbacks is not None else []

        self.optimizer = build_optimizer(self.model, self.params)
        self.lr_scheduler = build_lr_scheduler(
            optimizer=self.optimizer,
            training_params=self.params,
            current_epoch=self.state.epoch,
            num_training_steps=self._estimate_total_training_steps(),
        )

        self.train_dataloader = self._get_train_dataloader()
        self.eval_dataloader = self._get_eval_dataloader() if eval_dataset else None

        self._init_logging()

    #
    # Training
    #
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Trains the model."""
        if resume_from_checkpoint:
            with torch.profiler.record_function("load_from_checkpoint"):
                self._load_from_checkpoint(resume_from_checkpoint)

        total_steps = self._estimate_total_training_steps()

        self.start_time = time.perf_counter()

        # Make sure all workers start at the same time.
        barrier()

        with tqdm(
            total=total_steps,
            desc="Training",
            disable=not is_world_process_zero(),
        ) as progress_bar:
            while True:
                epoch = self.state.epoch
                if self.params.max_steps > 0:
                    if self.state.global_step >= self.params.max_steps:
                        self.log(
                            f"Reached {self.state.global_step} global steps. "
                            "Training completed."
                        )
                        break
                elif (
                    self.params.num_train_epochs > 0
                    and epoch >= self.params.num_train_epochs
                ):
                    self.log(f"Reached {epoch} epochs. Training completed.")
                    break

                with torch.profiler.record_function(f"epoch_{epoch}"):
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
                        # To enable distributed evaluation, the eval function needs
                        # to be updated to aggregate metrics accross all workers.
                        self.evaluate()

                    self.state.epoch += 1

                    barrier()

            self._process_callbacks("on_train_end")

        self.log(
            f"Training finished! Global step: {self.state.global_step} "
            f"Training runtime: {time.perf_counter() - self.start_time}s"
        )

        if self.params.enable_mlflow:
            mlflow.end_run()

    @contextmanager
    def _telemetry_block(self, name: str):
        with (
            torch.profiler.record_function(name) as record_function_context,
            self.telemetry.timer(name) as timer_context,
        ):
            yield (record_function_context, timer_context)

    @staticmethod
    def _cuda_sync_and_empty_cache() -> None:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _train_epoch(self, progress_bar: tqdm) -> None:
        """Trains the model for one epoch."""
        epoch_start_time = time.perf_counter()

        self.model.train()
        self._cuda_sync_and_empty_cache()
        self.optimizer.zero_grad(set_to_none=True)
        micro_step = 0

        data_iter = iter(self.train_dataloader)

        gradient_accumulation_steps = max(1, self.params.gradient_accumulation_steps)

        while True:
            with torch.profiler.record_function(
                "microstep" if gradient_accumulation_steps > 1 else "step"
            ):
                if micro_step % gradient_accumulation_steps == 0:
                    self._process_callbacks("on_step_begin")

                # True if `max_steps` is configured and we reached the limit.
                stop_on_max_steps_limit = (
                    self.params.max_steps > 0
                    and (self.state.global_step + 1) >= self.params.max_steps
                )
                # End of global step. May include multiple micro steps
                # if gradient_accumulation_steps > 1.
                end_of_global_step = (
                    (micro_step + 1) % gradient_accumulation_steps
                ) == 0

                with self._telemetry_block("fetching batch"):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        # FIXME Update metrics and log
                        self.log("End of epoch")
                        break

                # Count tokens on CPU.
                with self._telemetry_block("computing tokens"):
                    if self.processing_class is not None and "input_ids" in batch:
                        num_tokens = (
                            batch["input_ids"]
                            .ne(self.processing_class.pad_token_id)
                            .sum()
                            .item()
                        )
                        self.state.total_tokens_seen += num_tokens

                with self._telemetry_block("moving batch to device"):
                    if not self.is_using_fsdp and not self.is_using_ring_attention:
                        batch = {
                            k: v.to(self.device, non_blocking=True)
                            for k, v in batch.items()
                        }

                with self.mixed_precision_ctx, self._telemetry_block("model forward"):
                    self.model.require_backward_grad_sync = (  # type: ignore
                        end_of_global_step or stop_on_max_steps_limit
                    )

                    if self.is_using_ring_attention:
                        # Prepare inputs for ring attention
                        prepared_inputs = prepare_seq_parallel_inputs(
                            batch["input_ids"],
                            batch.get("position_ids"),
                            batch.get("labels"),
                            get_device_rank_info().rank,
                            get_device_rank_info().world_size,
                            self.device,
                        )
                        outputs = self.model(**prepared_inputs)
                    else:
                        outputs = self.model(**batch)

                    loss = outputs["loss"] / gradient_accumulation_steps

                with self._telemetry_block("loss backward"):
                    self.scaler.scale(loss).backward()

                if end_of_global_step or stop_on_max_steps_limit:
                    with self._telemetry_block("optimizer step"):
                        self.scaler.unscale_(self.optimizer)
                        if self.max_norm is not None and self.max_norm > 0:
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
                    if self.params.telemetry.track_gpu_temperature:
                        self.telemetry.record_gpu_temperature()
                    progress_bar.update(1)

                    self._process_callbacks("on_step_end")

                    if (
                        self.params.logging_steps > 0
                        and not (
                            self.state.global_step == 1
                            and self.params.logging_first_step
                        )
                        and (
                            stop_on_max_steps_limit
                            or (self.state.global_step % self.params.logging_steps == 0)
                        )
                    ):
                        # Log metrics
                        elapsed = time.perf_counter() - self.start_time
                        loss_value = loss.item() * gradient_accumulation_steps
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
                        callback_metrics = self._process_callbacks("on_log", metrics)
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

                if stop_on_max_steps_limit:
                    self.log(f"Reached {self.params.max_steps} max steps condition.")
                    break

                micro_step += 1

        self.log(
            f"End of epoch. "
            f"Global step: {self.state.global_step}. "
            f"Epoch runtime: {time.perf_counter() - epoch_start_time}s"
        )

    #
    # Evaluation
    #
    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
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
    def save_model(self, config: TrainingConfig, final: bool = True) -> None:
        """Saves the model."""
        self._cuda_sync_and_empty_cache()
        if is_world_process_zero():
            output_dir = Path(config.training.output_dir)
            output_dir.mkdir(exist_ok=True)
            model_path = output_dir / "model.safetensors"
            safetensors.torch.save_model(model=self.model, filename=str(model_path))
            self.log(f"Model saved to {model_path}.")

            if self._processor is not None:
                self._processor.save_config(output_dir)
                logger.info(f"Processor config has been saved at {output_dir}.")
        self._cuda_sync_and_empty_cache()

    def save_state(self):
        """Saves the training state."""
        self._cuda_sync_and_empty_cache()
        checkpoint_dir = Path(self.params.output_dir)

        if is_local_process_zero():
            checkpoint_dir.mkdir(exist_ok=True)

        if (
            self.params.telemetry.collect_telemetry_for_all_ranks
            or is_world_process_zero()
        ):
            telemetry_dir = self.params.telemetry_dir
            if telemetry_dir:
                device_rank_info = get_device_rank_info()
                telemetry_state_path = (
                    telemetry_dir / f"telemetry_rank{device_rank_info.rank:04}.json"
                )
                save_json(
                    data=self.telemetry.state_dict(),
                    filename=telemetry_state_path,
                )

        if self.is_using_fsdp:
            storage_options = StateDictOptions(
                full_state_dict=self.fsdp_params.state_dict_type
                == StateDictType.FULL_STATE_DICT,
                cpu_offload=self.fsdp_params.cpu_offload,
                ignore_frozen_params=False,
                strict=True,
                broadcast_from_rank0=False,  # TODO: make this configurable
            )
        else:
            storage_options = None

        model_state_dict, optimizer_state_dict = get_state_dict(
            model=self.model,
            optimizers=self.optimizer,
            options=storage_options,
        )

        model_path = checkpoint_dir / "model"
        optimizer_path = checkpoint_dir / "optimizer"
        dataloader_state_path = checkpoint_dir / "dataloader.pt"
        trainer_state_path = checkpoint_dir / "trainer_state.json"

        dcp.save(model_state_dict, checkpoint_id=model_path)
        dcp.save(optimizer_state_dict, checkpoint_id=optimizer_path)

        if is_world_process_zero():
            torch.save(self.train_dataloader.state_dict(), dataloader_state_path)
            save_json(data=self.state.model_dump(), filename=trainer_state_path)
            logger.info(f"Training state saved to {checkpoint_dir}")

        self._cuda_sync_and_empty_cache()

    def _load_from_checkpoint(self, checkpoint_dirname: str):
        """Loads the training state from a checkpoint."""
        checkpoint_dir = Path(checkpoint_dirname)

        device_rank_info = get_device_rank_info()

        model_path = checkpoint_dir / "model"
        optimizer_path = checkpoint_dir / "optimizer"
        dataloader_state_path = checkpoint_dir / "dataloader.pt"
        trainer_state_path = checkpoint_dir / "trainer_state.json"
        telemetry_state_path = (
            checkpoint_dir / f"telemetry_rank{device_rank_info.rank:04}.json"
        )

        if not checkpoint_dir.exists():
            raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
        if not model_path.exists():
            raise ValueError(
                f"Invalid checkpoint, model state folder does not exist: {model_path}"
            )
        if not optimizer_path.exists():
            raise ValueError(
                "Invalid checkpoint, optimizer state folder does not exist: "
                f"{optimizer_path}"
            )

        if self.is_using_fsdp:
            storage_options = StateDictOptions(
                full_state_dict=self.fsdp_params.state_dict_type
                == StateDictType.FULL_STATE_DICT,
                cpu_offload=self.fsdp_params.cpu_offload,
                ignore_frozen_params=False,
                strict=True,
                broadcast_from_rank0=False,
            )
        else:
            storage_options = None

        model_state_dict, optimizer_state_dict = get_state_dict(
            model=self.model,
            optimizers=self.optimizer,
            options=storage_options,
        )

        dcp.load(model_state_dict, checkpoint_id=model_path)
        dcp.load(optimizer_state_dict, checkpoint_id=optimizer_path)

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
    def log(self, message: str):
        """Logs a message if the process is the local process zero."""
        if not is_local_process_zero():
            return
        logger.info(message)

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        """Logs metrics to wandb and tensorboard."""
        # Log to console and log file
        if not is_world_process_zero():
            return
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
            project_name = os.environ.get("WANDB_PROJECT", "oumi")
            self.log(f"Logging to Weights and Biases project: '{project_name}'")
            run = wandb.init(
                project=project_name, name=self.params.run_name, job_type="train"
            )
            self.log(f"View wandb run {run.id} at: {run.get_url()}")
            wandb.watch(self.model)

        if self.params.enable_tensorboard:
            tensorboard_folder = Path(self.params.output_dir) / "tensorboard"
            self.log(f"Logging to tensorboard folder: '{tensorboard_folder}'")
            self.tensorboard_writer = tensorboard.SummaryWriter(
                log_dir=tensorboard_folder
            )
        else:
            self.tensorboard_writer = None

        if self.params.enable_mlflow:
            self.mlflow_run = mlflow.start_run()

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
            collate_fn=self.collator_fn,
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
            collate_fn=self.collator_fn,
        )

    def _estimate_total_training_steps(self) -> int:
        # If max_steps is set, use it.
        if self.params.max_steps > 0:
            return self.params.max_steps

        num_epochs = self.params.num_train_epochs
        if num_epochs > 0:
            num_dataset_examples = 0
            try:
                if not isinstance(self.train_dataset, IterableDataset):
                    num_dataset_examples = len(self.train_dataset)  # type: ignore
                elif hasattr(self.train_dataset, "datapipe"):
                    # Hacky way to get examples count from
                    # MapToIterConverterIterDataPipe.
                    # FIXME Remove DataPipes OPE-811
                    num_dataset_examples = len(self.train_dataset.datapipe)  # type: ignore
            except Exception:
                num_dataset_examples = 0

            if num_dataset_examples > 0:
                world_size = get_device_rank_info().world_size
                batch_size = self.params.per_device_train_batch_size
                steps_per_epoch_per_device = math.ceil(
                    float(num_dataset_examples) / (batch_size * world_size)
                )
                return int(num_epochs * max(steps_per_epoch_per_device, 1))

        raise ValueError(
            "Unable to estimate `total_training_steps` "
            + (f"in {num_epochs} epochs" if num_epochs > 0 else "")
            + ". Please define `max_steps` training parameter!"
        )

    def _set_sampler_epoch(self, epoch: int) -> None:
        """Sets the current epoch on sampler, if it exists and supports it."""
        if self._sampler and hasattr(self._sampler, "set_epoch"):
            self.log(f"Setting sampler epoch to {epoch}.")
            self._sampler.set_epoch(epoch)

    #
    # Handle callbacks
    #
    def _process_callbacks(
        self, event: str, logs: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Process callbacks.

        Extremely hacky way to handle HF callbacks.
        Just here to unblock debugging with our MfuCallback
        """
        logs = logs or {}

        for callback in self.callbacks:
            if hasattr(callback, event):
                action = getattr(callback, event)
                action(args=self.params, state=None, control=None, logs=logs)

        return logs
