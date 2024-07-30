import os
import time
from pprint import pformat
from typing import Any, Dict, List, Optional, cast

import pydantic
import torch
import torch.amp
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase, TrainerCallback

from lema.builders.optimizers import build_optimizer
from lema.core.distributed import (
    get_device_rank_info,
    is_distributed,
    is_local_process_zero,
    is_world_process_zero,
    local_leader_only,
    prepare_model_for_distributed,
)
from lema.core.types import TrainingConfig, TrainingParams
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
        tokenizer: PreTrainedTokenizerBase,
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

        # TODO: OPE-216 - allow granular mixed precision training
        self.dtype = (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.mixed_precision_ctx = torch.amp.autocast(
            device_type=self.device_type, enabled=True, dtype=torch.bfloat16
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

        # TODO: OPE-220 - init wandb, tensorboard, etc

        self.optimizer = build_optimizer(self.model, self.params)

        self.train_dataloader = self._get_train_dataloader()
        self.eval_dataloader = self._get_eval_dataloader() if eval_dataset else None

        self.state = TrainingState()

        self.telemetry = TelemetryTracker()
        self.start_time = time.perf_counter()

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
                self.process_callbacks("on_step_begin")

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

            # TODO: OPE-225 - add detailed logging metrics
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
                # assert self.model.dtype is torch.bfloat16

            with self.telemetry.timer("loss backward"):
                self.scaler.scale(loss).backward()

            if (micro_step + 1) % self.params.gradient_accumulation_steps == 0:
                with self.telemetry.timer("optimizer step"):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.max_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                self.state.global_step += 1
                progress_bar.update(1)

                self.process_callbacks("on_step_end")

                if self.state.global_step % self.params.logging_steps == 0:
                    # TODO: OPE-225 - add detailed logging metrics
                    loss_value = loss.item() * self.params.gradient_accumulation_steps
                    self.log(f"Step {self.state.global_step}: loss = {loss_value}")
                    logs = self.process_callbacks("on_log")
                    self.log(pformat(logs))
                    self.log(f"Total tokens seen: {self.state.total_tokens_seen}")
                    elapsed = time.perf_counter() - self.start_time
                    self.log(
                        f"Steps per second: {self.state.global_step / elapsed} step/s"
                    )
                    self.log(
                        f"Tokens per second: {self.state.total_tokens_seen / elapsed}"
                        " tok/s"
                    )
                    self.log(
                        f"Tokens per step per GPU: "
                        f"{self.state.total_tokens_seen / self.state.global_step}"
                        " tok/step/gpu"
                    )

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

    def _get_total_training_steps(self):
        # TODO: handle num_epochs, len(dataset), etc
        return self.params.max_steps

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

        results = {"eval_loss": eval_loss, "perplexity": perplexity.item()}

        self.log(f"Evaluation results: {results}")

        self.model.train()
        return results

    #
    # Data loading
    #
    def _get_train_dataloader(self) -> DataLoader:
        """Returns the training dataloader."""
        prefetch_factor = (
            None
            if self.params.dataloader_num_workers == 0
            else self.params.dataloader_prefetch_factor
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.params.per_device_train_batch_size,
            shuffle=False,  # TODO: OPE-224 add sampler
            num_workers=self.params.dataloader_num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            pin_memory_device=self.device,
        )

    def _get_eval_dataloader(self) -> DataLoader:
        """Returns the evaluation dataloader."""
        if not self.eval_dataset:
            raise ValueError("No evaluation dataset provided.")

        return DataLoader(
            self.eval_dataset,
            batch_size=self.params.per_device_eval_batch_size,
            shuffle=False,
            num_workers=self.params.dataloader_num_workers,
        )

    #
    # Checkpointing
    #
    def save_model(self, config: TrainingConfig):
        """Saves the model."""
        if is_world_process_zero():
            output_dir = config.training.output_dir
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(output_dir, "model.pt"))
            self.log(f"Model saved to {output_dir}.")

    def save_state(self):
        """Saves the model and optimizer state."""
        output_dir = self.params.output_dir

        if is_world_process_zero():
            os.makedirs(output_dir, exist_ok=True)
            # TODO: OPE-213 - switch to using safetensors
            torch.save(self.model.state_dict(), os.path.join(output_dir, "model.pt"))
            torch.save(
                self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
            )

            save_json(
                data=self.state.model_dump(),
                filename=os.path.join(output_dir, "trainer_state.json"),
            )

            save_json(
                data=self.telemetry.state_dict(),
                filename=os.path.join(output_dir, "telemetry_state.json"),
            )
            logger.info(f"Model saved to {output_dir}")

    def _load_from_checkpoint(self, checkpoint_dir: str):
        """Loads the model and optimizer state from a checkpoint."""
        model_path = os.path.join(checkpoint_dir, "model.pt")
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
        telemetry_state_path = os.path.join(checkpoint_dir, "telemetry.json")

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(
                torch.load(optimizer_path, map_location=self.device)
            )
        if os.path.exists(trainer_state_path):
            self.state = TrainingState.model_validate(
                load_json(trainer_state_path), strict=True
            )
        if os.path.exists(telemetry_state_path):
            self.telemetry.load_state_dict(load_json(telemetry_state_path))

        # TODO: OPE-103 - save / reload dataloader state

        self.log(f"Resumed training from checkpoint: {checkpoint_dir}")

    #
    # Logging
    #
    @local_leader_only()
    def log(self, message: str):
        """Logs a message if the process is the local process zero."""
        logger.info(message)

    #
    # Handle callbacks
    #
    def process_callbacks(self, event: str) -> Optional[Dict[str, Any]]:
        """Process callbacks.

        Extremely hacky way to handle HF callbacks.
        Just here to unblock debugging with our MfuCallback
        """
        logs = {} if event == "on_log" else None

        for callback in self.callbacks:
            if hasattr(callback, event):
                action = getattr(callback, event)
                action(args=self.params, state=None, control=None, logs=logs)

        return logs
