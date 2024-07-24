import os
import time
from pprint import pformat
from typing import Dict, List, Optional, cast

import torch
import torch.amp
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase, TrainerCallback

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
from lema.core.types.base_trainer import BaseTrainer
from lema.performance.telemetry import TelemetryTracker
from lema.utils.logging import logger
from lema.utils.torch_utils import log_trainable_parameters

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn


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
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # TODO: add support for gradient clipping
        self.max_norm: float = 1.0

        # Sanity checks
        assert self.args.gradient_accumulation_steps > 0

        # TODO: allow granular mixed precision training
        self.dtype = (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.mixed_precision_ctx = torch.amp.autocast(
            device_type=self.device_type, enabled=True, dtype=torch.bfloat16
        )

        if args.compile:
            self.log("Compiling model...")
            model = cast(torch.nn.Module, torch.compile(model))

        self.scaler = GradScaler(enabled=False)

        device_info = get_device_rank_info()

        # TODO: give users fine-grained control on device
        # TODO: non-leader models should be on meta
        if torch.cuda.is_available():
            self.device = f"cuda:{device_info.local_rank}"
            torch.cuda.set_device(self.device)
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model.to(self.device)

        # TODO: handle DP ?
        # TODO: hook-up fsdp flag
        if is_distributed():
            # Wrap model for distributed training
            model = prepare_model_for_distributed(model, use_fsdp=False)

        self.callbacks = callbacks if callbacks is not None else []

        # TODO: init wandb, tensorboard, etc

        # TODO: add builder for optimizers
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            fused=False,
        )

        self.train_dataloader = self._get_train_dataloader()
        self.eval_dataloader = self._get_eval_dataloader() if eval_dataset else None

        # TODO: add dataclass for training state
        self.global_step = 0
        self.epoch = 0
        self.total_tokens_seen = 0

        self.telemetry = TelemetryTracker()

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
            for epoch in range(self.epoch, self.args.num_train_epochs):
                self._train_epoch(progress_bar)

                if self.args.save_epoch:
                    self.save_state()

                if (
                    self.eval_dataloader
                    and self.args.eval_strategy == "epoch"
                    and is_world_process_zero()
                ):
                    # TODO: only the global leader is used for evaluation
                    # To enable distributed evaluation, th eval function needs
                    # to be updated to aggregate metrics accross all workers.
                    self.evaluate()

                self.epoch += 1

                if self.global_step >= total_steps:
                    break

    def _train_epoch(self, progress_bar: tqdm) -> None:
        """Trains the model for one epoch."""
        self.model.train()
        torch.cuda.empty_cache()
        self.optimizer.zero_grad(set_to_none=True)
        micro_step = 0
        data_iter = iter(self.train_dataloader)

        while True:
            # for micro_step, batch in enumerate(self.train_dataloader):
            if micro_step % self.args.gradient_accumulation_steps == 0:
                self.process_callbacks("on_step_begin")

            with self.telemetry.timer("fetching batch"):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    self.log("End of epoch")
                    return

            # TODO: make sure data dtypes are correct
            with self.telemetry.timer("moving batch to device"):
                batch = {
                    k: v.to(self.device, non_blocking=True) for k, v in batch.items()
                }

            # with self.telemetry.timer("computing tokens"):
            #     num_tokens = batch["input_ids"].ne(self.tokenizer.pad_token_id).sum()

            # with self.telemetry.timer("syncing to cpu"):
            #     num_tokens = num_tokens.item()
            #     self.total_tokens_seen += num_tokens

            with self.mixed_precision_ctx, self.telemetry.timer("model forward"):
                # self.model.require_backward_grad_sync = (
                #     micro_step + 1
                # ) % self.args.gradient_accumulation_steps == 0

                outputs = self.model(**batch)
                loss = outputs["loss"] / self.args.gradient_accumulation_steps
                # assert loss.dtype is torch.bfloat16  # TODO: should be float32?
                # assert outputs["logits"].dtype is torch.bfloat16
                # assert self.model.dtype is torch.bfloat16

            with self.telemetry.timer("loss backward"):
                self.scaler.scale(loss).backward()

            if (micro_step + 1) % self.args.gradient_accumulation_steps == 0:
                with self.telemetry.timer("optimizer step"):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.max_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                self.global_step += 1
                progress_bar.update(1)

                if self.global_step % self.args.logging_steps == 0:
                    self.log(
                        f"Step {self.global_step}: "
                        f"loss = {loss.item() * self.args.gradient_accumulation_steps}"
                    )
                    logs = self.process_callbacks("on_log")
                    self.log(pformat(logs))
                    if is_local_process_zero():
                        self.telemetry.print_summary()
                        # self._log_training_progress(self.total_tokens_seen)
                        # dt = time.perf_counter() - self.start_time
                        # mfu = self.model.estimate_mfu(
                        #     self.args.gradient_accumulation_steps * self.global_step,
                        #     dt,
                        # )
                        # self.log(f"MFU: {mfu}")

                if (
                    self.args.save_steps > 0
                    and self.global_step % self.args.save_steps == 0
                ):
                    self.save_state()

                if (
                    self.eval_dataloader
                    and self.args.eval_steps > 0
                    and self.global_step % self.args.eval_steps == 0
                    and is_world_process_zero()
                ):
                    # TODO: only the global leader is used for evaluation
                    # To enable distributed evaluation, th eval function needs
                    # to be updated to aggregate metrics accross all workers.
                    self.evaluate()

                self.process_callbacks("on_step_end")

            if self.global_step >= self.args.max_steps:
                break

            micro_step += 1

    def _get_total_training_steps(self):
        # TODO: handle num_epochs, len(dataset), etc
        return self.args.max_steps

    def _check_fused_optimizer_support(
        self, optimizer_class: torch.optim.Optimizer
    ) -> bool:
        if not torch.cuda.is_available():
            return False

        try:
            return "fused" in optimizer_class.__init__.__code__.co_varnames
        except Exception as e:
            logger.error(
                f"Error checking for fused optimizer support: {str(e)}. Assuming False."
            )
            return False

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
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,  # TODO: add sampler
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            prefetch_factor=None
            if self.args.dataloader_num_workers == 0
            else self.args.dataloader_prefetch_factor,
            pin_memory_device=self.device,
        )

    def _get_eval_dataloader(self) -> DataLoader:
        """Returns the evaluation dataloader."""
        if not self.eval_dataset:
            raise ValueError("No evaluation dataset provided.")

        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
        )

    #
    # Checkpointing
    #
    @global_leader_only()
    def save_model(self, config: TrainingConfig):
        """Saves the model and optimizer state."""
        output_dir = config.training.output_dir
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(output_dir, "model.pt"))
        self.log(f"Model saved to {output_dir}.")

    def save_state(self):
        """Saves the model and optimizer state."""
        output_dir = self.args.output_dir

        if is_world_process_zero():
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(output_dir, "model.pt"))
            torch.save(
                self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
            )
            torch.save(
                {
                    "epoch": self.epoch,
                    "global_step": self.global_step,
                },
                os.path.join(output_dir, "trainer_state.pt"),
            )
            logger.info(f"Model saved to {output_dir}")

    def _load_from_checkpoint(self, checkpoint_dir: str):
        """Loads the model and optimizer state from a checkpoint."""
        model_path = os.path.join(checkpoint_dir, "model.pt")
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.pt")

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(
                torch.load(optimizer_path, map_location=self.device)
            )
        if os.path.exists(trainer_state_path):
            trainer_state = torch.load(trainer_state_path, map_location=self.device)
            self.epoch = trainer_state["epoch"]
            self.global_step = trainer_state["global_step"]

        self.log(f"Resumed training from checkpoint: {checkpoint_dir}")

    #
    # Logging
    #
    @local_leader_only()
    def log(self, message: str):
        """Logs a message if the process is the local process zero."""
        logger.info(message)

    def _log_training_progress(self, num_tokens):
        # TODO: move this to telemetry tracker
        elapsed_time = time.perf_counter() - self.start_time
        tokens_per_second = self.total_tokens_seen / elapsed_time

        self.log(f"\nStep {self.global_step}:")
        self.log(f"Total tokens seen: {self.total_tokens_seen}")
        self.log(f"Tokens per second: {tokens_per_second:.2f}")

    #
    # Handle callbacks
    #
    def process_callbacks(self, event):
        """Process callbacks.

        Extremely hacky way to handle HF callbacks.
        Just here to unblock debugging with our MfuCallback
        """
        logs = {} if event == "on_log" else None

        for callback in self.callbacks:
            if hasattr(callback, event):
                action = getattr(callback, event)
                action(args=self.args, state=None, control=None, logs=logs)

        return logs
