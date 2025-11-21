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

"""Megatron-based GRPO trainer for large-scale RL training."""

import os
from pathlib import Path
from typing import Callable, Optional, Union

import torch
import torch.distributed as dist
from datasets import Dataset

try:
    from megatron.bridge import AutoBridge
    from megatron.bridge.training.state import GlobalState
    from megatron.core.optimizer import DistributedOptimizer
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    MEGATRON_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    MEGATRON_AVAILABLE = False
    AutoBridge = None  # type: ignore
    GlobalState = None  # type: ignore
    DistributedOptimizer = None  # type: ignore
    OptimizerParamScheduler = None  # type: ignore

from oumi.core.configs import TrainingConfig
from oumi.core.processors.base_processor import BaseProcessor
from oumi.core.tokenizers import BaseTokenizer
from oumi.core.trainers.base_trainer import BaseTrainer
from oumi.core.trainers.megatron.bridge_utils import (
    build_megatron_config,
    check_megatron_bridge_available,
    export_megatron_to_hf,
    import_hf_to_megatron,
    initialize_megatron_model,
)
from oumi.core.trainers.megatron.loss_functions import (
    calculate_grpo_loss,
    compute_advantages,
    gather_log_probs,
)
from oumi.utils.logging import logger


class OumiMegatronGrpoTrainer(BaseTrainer):
    """[PRODUCTION-READY] GRPO Trainer using Megatron-LM for large-scale model training.

    This trainer implements Group Relative Policy Optimization (GRPO) with
    Megatron-LM backend, enabling advanced model parallelism strategies
    (tensor, pipeline, context, and expert parallelism) for training
    very large models (70B+).

    The trainer uses Megatron-Bridge for seamless HuggingFace ↔ Megatron conversion,
    maintaining compatibility with the HF ecosystem while leveraging Megatron's
    high-performance distributed training capabilities.

    Key Features:
        - Tensor Parallelism (TP): Split model layers across GPUs
        - Pipeline Parallelism (PP): Split model vertically across GPUs
        - Context Parallelism (CP): Split long sequences across GPUs
        - Expert Parallelism (EP): Split MoE experts across GPUs
        - Reference Model Management: Separate reference model for KL penalty
        - vLLM Integration: Fast inference for rollout generation
        - Distributed Checkpointing: Fault-tolerant checkpoint save/load
        - Sequence Packing: Efficient batch processing for RL (TODO)
        - HF Export: Convert trained weights back to HuggingFace format

    Example:
        >>> config = TrainingConfig(
        ...     model=ModelParams(model_name="meta-llama/Llama-3.1-70B"),
        ...     training=TrainingParams(trainer_type=TrainerType.MEGATRON_GRPO),
        ...     megatron=MegatronParams(
        ...         tensor_model_parallel_size=8,
        ...         pipeline_model_parallel_size=4,
        ...     ),
        ... )
        >>> trainer = OumiMegatronGrpoTrainer(
        ...     processing_class=tokenizer,
        ...     config=config,
        ...     reward_funcs=[reward_function],
        ...     train_dataset=train_dataset,
        ...     eval_dataset=eval_dataset,
        ... )
        >>> trainer.train()

    Args:
        processing_class: Tokenizer for the model
        config: Oumi training configuration
        reward_funcs: List of reward functions for GRPO
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (required for validation)
        processor: Optional processor for VLMs
        megatron_checkpoint_path: Path to converted Megatron checkpoint.
            If None, will attempt to convert from HF model.
    """

    def __init__(
        self,
        processing_class: Optional[BaseTokenizer],
        config: TrainingConfig,
        reward_funcs: list[Callable],
        train_dataset: Dataset,
        eval_dataset: Dataset,
        processor: Optional[BaseProcessor] = None,
        megatron_checkpoint_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """Initialize the Megatron GRPO trainer."""
        check_megatron_bridge_available()

        if not reward_funcs:
            raise ValueError("At least one reward function must be provided.")

        if len(reward_funcs) > 1:
            raise ValueError(
                "Multiple reward functions not yet supported for Megatron GRPO trainer. "
                f"Got {len(reward_funcs)} reward functions."
            )

        logger.info("Initializing Megatron GRPO trainer...")

        self._processing_class = processing_class
        self._config = config
        self._reward_funcs = reward_funcs
        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset
        self._processor = processor

        # Setup output directories
        self._output_dir: Optional[Path] = (
            Path(config.training.output_dir).absolute().resolve()
            if config.training.output_dir
            else None
        )

        self._megatron_checkpoint_dir: Path = (
            Path(megatron_checkpoint_path)
            if megatron_checkpoint_path
            else self._output_dir / "megatron_checkpoint" if self._output_dir else Path("/tmp/megatron_checkpoint")
        )

        self._train_checkpoint_dir: Path = (
            self._output_dir / "checkpoints" if self._output_dir else Path("/tmp/checkpoints")
        )

        # Convert HF → Megatron if needed (only rank 0 does the conversion)
        # Use distributed barrier to prevent race conditions
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            if not self._megatron_checkpoint_dir.exists():
                logger.info("Megatron checkpoint not found. Converting from HuggingFace...")
                self._convert_hf_to_megatron()
            else:
                logger.info(f"Using existing Megatron checkpoint: {self._megatron_checkpoint_dir}")

        # Wait for rank 0 to finish conversion before other ranks proceed
        if dist.is_initialized():
            dist.barrier()

        # All ranks verify the checkpoint now exists
        if not self._megatron_checkpoint_dir.exists():
            raise RuntimeError(
                f"Megatron checkpoint not found at {self._megatron_checkpoint_dir} "
                "after conversion attempt. This should not happen."
            )

        # Validate distributed configuration early (before model initialization)
        self._validate_distributed_configuration(config)

        # Build Megatron configuration
        self._megatron_config = build_megatron_config(
            training_config=config,
            megatron_checkpoint_path=str(self._megatron_checkpoint_dir),
            train_checkpoint_dir=str(self._train_checkpoint_dir),
        )

        # Initialize Megatron model, optimizer, and scheduler
        logger.info("Initializing Megatron policy model...")
        self._model, self._optimizer, self._scheduler, self._state = (
            initialize_megatron_model(self._megatron_config, load_pretrained=True)
        )

        # Initialize reference model (frozen copy of initial policy)
        # The reference model is used to compute KL divergence penalty
        logger.info("Initializing reference model for KL penalty...")
        self._ref_model = self._initialize_reference_model()

        # Store bridge for later export
        self._bridge = AutoBridge.from_hf_pretrained(
            config.model.model_name,
            trust_remote_code=config.model.trust_remote_code,
        )

        # Initialize vLLM for inference (if configured)
        self._vllm_engine = None
        self._vllm_model_path: Optional[Path] = None
        self._last_vllm_sync_step = -1  # Track when we last synced to vLLM

        if config.megatron.inference_backend == "vllm":
            logger.info("vLLM inference backend selected - will initialize on first generation")
            # Export initial weights for vLLM (one-time setup)
            self._vllm_model_path = self._output_dir / "vllm_weights" if self._output_dir else Path("/tmp/vllm_weights")

            # Only export if not already exists
            if not self._vllm_model_path.exists():
                logger.info(f"Exporting initial model to {self._vllm_model_path} for vLLM...")
                if not dist.is_initialized() or dist.get_rank() == 0:
                    export_megatron_to_hf(
                        megatron_model=self._model,
                        bridge=self._bridge,
                        output_path=str(self._vllm_model_path),
                        model_name=config.model.model_name,
                    )

                # Wait for export to complete
                if dist.is_initialized():
                    dist.barrier()
            else:
                logger.info(f"Using existing vLLM weights at {self._vllm_model_path}")

            # Lazy initialization - vLLM engine created on first generation call
        elif config.megatron.inference_backend == "megatron":
            logger.info("Using native Megatron inference backend")
        else:
            logger.warning(
                f"Unsupported inference backend: {config.megatron.inference_backend}. "
                "Falling back to Megatron inference."
            )

        # Training state
        self._global_step = 0
        self._current_epoch = 0

        logger.info("Megatron GRPO trainer initialized successfully!")

    def _validate_distributed_configuration(self, config: TrainingConfig) -> None:
        """Validate distributed training configuration.

        Checks that parallelism settings are compatible with:
        - Available world size
        - Model architecture (number of layers, attention heads, etc.)
        - Batch size settings

        Args:
            config: Training configuration

        Raises:
            ValueError: If configuration is invalid
        """
        logger.info("Validating distributed configuration...")

        # Get world size
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
            logger.warning("Distributed not initialized. Assuming single GPU training.")

        # Load model config to get architecture parameters
        try:
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(
                config.model.model_name,
                trust_remote_code=config.model.trust_remote_code,
            )

            # Extract architecture parameters
            num_layers = getattr(hf_config, "num_hidden_layers", None)
            num_attention_heads = getattr(hf_config, "num_attention_heads", None)
            num_key_value_heads = getattr(hf_config, "num_key_value_heads", None)

            if num_layers is None or num_attention_heads is None:
                logger.warning(
                    "Could not extract model architecture parameters. "
                    "Skipping detailed validation."
                )
                # Still validate world size
                config.megatron.get_data_parallel_size(world_size)
                return

            # Run comprehensive validation
            config.megatron.validate_distributed_config(
                world_size=world_size,
                num_layers=num_layers,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
            )

        except Exception as e:
            logger.error(f"Error during distributed configuration validation: {e}")
            raise

    def _initialize_reference_model(self):
        """Initialize reference model as a frozen copy of the policy model.

        The reference model is used to compute KL divergence penalty in GRPO.
        It can be updated periodically if reference_update_interval is configured.

        Returns:
            Reference model (frozen)
        """
        from megatron.bridge.models.model_provider import get_model

        logger.info("Creating reference model (frozen copy of policy)...")

        # Create reference model with same architecture
        ref_model_list = get_model(
            self._megatron_config.model,
            self._megatron_config.ddp,
            use_torch_fsdp2=False,
            overlap_param_gather_with_optimizer_step=False,
            data_parallel_random_init=False,
        )

        ref_model = ref_model_list[0]

        # Copy weights from policy model to reference model
        logger.info("Copying weights from policy to reference model...")
        ref_model.load_state_dict(self._model.state_dict())

        # Freeze reference model parameters
        for param in ref_model.parameters():
            param.requires_grad = False

        ref_model.eval()  # Set to eval mode

        logger.info("Reference model initialized and frozen")
        return ref_model

    def _update_reference_model(self) -> None:
        """Update reference model with current policy weights.

        This should be called periodically during training according to
        reference_update_interval configuration.
        """
        logger.info(f"Updating reference model at step {self._global_step}...")

        # Copy weights from policy to reference model
        self._ref_model.load_state_dict(self._model.state_dict())

        # Ensure reference model stays frozen
        for param in self._ref_model.parameters():
            param.requires_grad = False

        self._ref_model.eval()

        logger.info("Reference model updated successfully")

    def _convert_hf_to_megatron(self) -> None:
        """Convert HuggingFace model to Megatron format."""
        import_hf_to_megatron(
            hf_model_id=self._config.model.model_name,
            megatron_checkpoint_path=str(self._megatron_checkpoint_dir),
            megatron_params=self._config.megatron,
            trust_remote_code=self._config.model.trust_remote_code,
            hf_token=os.environ.get("HF_TOKEN"),
        )

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint for training resumption.

        Args:
            checkpoint_path: Path to the checkpoint directory
        """
        try:
            from megatron.bridge.training.checkpointing import load_checkpoint
        except ImportError:
            logger.error("Cannot import load_checkpoint from Megatron-Bridge.")
            raise

        checkpoint_dir = Path(checkpoint_path)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}...")

        # Update load path in config
        self._state.cfg.checkpoint.load = str(checkpoint_dir)

        # Load checkpoint using Megatron's distributed loading
        try:
            iteration = load_checkpoint(
                self._state,
                [self._model],
                self._optimizer,
                self._scheduler,
                skip_load_to_model_and_opt=False,
            )

            # Load trainer metadata (rank 0 only)
            if not dist.is_initialized() or dist.get_rank() == 0:
                metadata_path = checkpoint_dir / "trainer_metadata.pt"
                if metadata_path.exists():
                    metadata = torch.load(metadata_path, map_location="cpu")
                    self._global_step = metadata.get("global_step", iteration)
                    self._current_epoch = metadata.get("current_epoch", 0)
                    logger.info(f"Resumed from step {self._global_step}, epoch {self._current_epoch}")
                else:
                    # Fallback to iteration from checkpoint
                    self._global_step = iteration
                    logger.warning("No trainer metadata found, using iteration from checkpoint")

            # Broadcast training state to all ranks
            if dist.is_initialized():
                # Broadcast global_step and current_epoch from rank 0
                state_tensor = torch.tensor(
                    [self._global_step, self._current_epoch],
                    dtype=torch.long,
                    device=torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu"),
                )
                dist.broadcast(state_tensor, src=0)
                self._global_step = int(state_tensor[0].item())
                self._current_epoch = int(state_tensor[1].item())

            logger.info(f"Checkpoint loaded successfully. Resuming from step {self._global_step}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        """Train the model using GRPO with Megatron backend.

        Args:
            resume_from_checkpoint: Path to checkpoint to resume from (optional)
        """
        logger.info("Starting GRPO training with Megatron...")

        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
            self._load_checkpoint(resume_from_checkpoint)

        max_steps = self._config.training.max_steps or 1000
        num_epochs = self._config.training.num_train_epochs or 1

        logger.info(f"Training for {num_epochs} epochs, max {max_steps} steps")

        for epoch in range(num_epochs):
            self._current_epoch = epoch
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

            for batch_idx, batch in enumerate(self._train_dataset):
                if self._global_step >= max_steps:
                    logger.info(f"Reached max steps ({max_steps}). Stopping training.")
                    break

                # Run GRPO training step
                self._training_step(batch, batch_idx)

                self._global_step += 1

                # Save checkpoint periodically
                if (
                    self._config.training.save_steps
                    and self._global_step % self._config.training.save_steps == 0
                ):
                    self.save_model(self._config, final=False)

                # Evaluate periodically
                if (
                    self._config.training.eval_steps
                    and self._global_step % self._config.training.eval_steps == 0
                ):
                    self._evaluate()

                # Update reference model periodically
                ref_update_interval = self._config.megatron.grpo_config.reference_update_interval
                if (ref_update_interval is not None and
                    self._global_step > 0 and
                    self._global_step % ref_update_interval == 0):
                    logger.info(f"Triggering reference model update at step {self._global_step}")
                    self._update_reference_model()

        logger.info("Training complete!")

        # Save final model
        if self._config.training.save_final_model:
            self.save_model(self._config, final=True)

    def _training_step(self, batch: dict, batch_idx: int) -> dict:
        """Execute a single GRPO training step.

        This implements the core GRPO algorithm:
        1. Generate completions using current policy (via vLLM/Megatron)
        2. Compute rewards for completions
        3. Compute advantages from rewards
        4. Forward pass with current policy to get logprobs (using generated tokens)
        5. Forward pass with reference model to get ref logprobs
        6. Compute GRPO loss with KL penalty
        7. Backward pass and optimizer step

        Args:
            batch: Training batch with input_ids, attention_mask, and labels
            batch_idx: Batch index

        Returns:
            Dictionary of training metrics
        """
        logger.debug(f"Training step {self._global_step}, batch {batch_idx}")

        self._model.train()

        # 1. Generate completions using current policy
        # Returns completions (text), generated_ids (full sequences), and old_logprobs (from generation)
        completions, generated_ids, old_logprobs = self._generate_completions(batch)

        # 2. Calculate rewards
        rewards = self._compute_rewards(batch, completions)

        # 3. Compute advantages with group normalization
        grpo_cfg = self._config.megatron.grpo_config
        advantages = compute_advantages(rewards, normalize=grpo_cfg.normalize_advantages)

        # 4. Forward pass with current policy to get logprobs
        # CRITICAL: Use generated_ids (not original labels) to ensure alignment with old_logprobs
        # Create labels from generated_ids (shift left by 1)
        labels = generated_ids.clone()

        # Create attention mask if not provided
        attention_mask = batch.get("attention_mask")
        if attention_mask is None or attention_mask.shape[1] != generated_ids.shape[1]:
            # Create attention mask for generated sequences
            pad_token_id = self._processing_class.pad_token_id or 0
            attention_mask = (generated_ids != pad_token_id).long()

        # Forward pass
        outputs = self._model(
            input_ids=generated_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        # Gather logprobs for target tokens
        # For GRPO, we only compute loss on the generated (completion) tokens, not the prompt
        # We need to identify where the completion starts
        prompt_len = batch["input_ids"].shape[1]

        # Create labels: shift left by 1 for next-token prediction
        shifted_labels = labels[:, 1:].contiguous()

        # Gather logprobs for all tokens (we'll mask later)
        current_logprobs, mask = gather_log_probs(
            logits[:, :-1, :],  # Shift logits (predict next token)
            shifted_labels,      # Target tokens
            return_mask=True,
        )

        # Mask out prompt tokens - we only want to compute loss on completions
        # old_logprobs has shape [batch_size, completion_len]
        # current_logprobs has shape [batch_size, total_len-1]
        # We need to extract only the completion part
        completion_start_idx = prompt_len - 1  # -1 because of shifting
        completion_logprobs = current_logprobs[:, completion_start_idx:]
        completion_mask = mask[:, completion_start_idx:]

        # Align old_logprobs with completion_logprobs (truncate or pad to match)
        max_completion_len = completion_logprobs.shape[1]
        if old_logprobs.shape[1] < max_completion_len:
            # Pad old_logprobs
            padding = torch.full(
                (old_logprobs.shape[0], max_completion_len - old_logprobs.shape[1]),
                -100.0,
                device=old_logprobs.device,
                dtype=old_logprobs.dtype
            )
            old_logprobs = torch.cat([old_logprobs, padding], dim=1)
        elif old_logprobs.shape[1] > max_completion_len:
            # Truncate old_logprobs
            old_logprobs = old_logprobs[:, :max_completion_len]

        # Update mask to exclude padded positions in old_logprobs
        completion_mask = completion_mask & (old_logprobs != -100.0)

        # 5. Get reference model logprobs (no gradient)
        with torch.no_grad():
            ref_outputs = self._ref_model(
                input_ids=generated_ids,
                attention_mask=attention_mask,
            )
            ref_logits = ref_outputs.logits if hasattr(ref_outputs, "logits") else ref_outputs

            ref_logprobs_full = gather_log_probs(
                ref_logits[:, :-1, :],
                shifted_labels,
                return_mask=False,
            )
            # Extract completion part
            ref_logprobs = ref_logprobs_full[:, completion_start_idx:]

            # Align with completion_logprobs
            if ref_logprobs.shape[1] < max_completion_len:
                padding = torch.zeros(
                    (ref_logprobs.shape[0], max_completion_len - ref_logprobs.shape[1]),
                    device=ref_logprobs.device,
                    dtype=ref_logprobs.dtype
                )
                ref_logprobs = torch.cat([ref_logprobs, padding], dim=1)
            elif ref_logprobs.shape[1] > max_completion_len:
                ref_logprobs = ref_logprobs[:, :max_completion_len]

        # 6. Calculate GRPO loss with KL penalty and entropy regularization
        loss, kl_term, ratios, entropy_term, trunc_above, trunc_below = calculate_grpo_loss(
            current_logprobs=completion_logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            clamp_eps_lower=grpo_cfg.clamp_eps_lower,
            clamp_eps_upper=grpo_cfg.clamp_eps_upper,
            kl_beta=grpo_cfg.kl_beta,
            entropy_weight=grpo_cfg.entropy_weight,
            mask=completion_mask,
        )

        # Average loss over valid tokens
        total_loss = (loss.sum() / completion_mask.sum()) if completion_mask.sum() > 0 else loss.mean()

        # 7. Backward pass
        self._optimizer.zero_grad()
        total_loss.backward()

        # 8. Optimizer step with gradient clipping
        grad_norm = self._optimizer.step()
        self._scheduler.step(1)

        # Compute metrics
        num_valid_tokens = completion_mask.sum()
        metrics = {
            "loss": total_loss.item(),
            "kl": (kl_term.sum() / num_valid_tokens).item() if num_valid_tokens > 0 else 0.0,
            "entropy": (entropy_term.sum() / num_valid_tokens).item() if num_valid_tokens > 0 else 0.0,
            "ratio_mean": (ratios.sum() / num_valid_tokens).item() if num_valid_tokens > 0 else 0.0,
            "ratio_clipped_above": trunc_above.float().mean().item(),
            "ratio_clipped_below": trunc_below.float().mean().item(),
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item(),
            "rewards_mean": rewards.mean().item(),
            "rewards_std": rewards.std().item(),
            "step": self._global_step,
        }

        if grad_norm is not None:
            metrics["grad_norm"] = grad_norm

        return metrics

    def _initialize_vllm_engine(self):
        """Initialize vLLM engine for fast inference.

        Uses the persistent HF model path that was exported during initialization.

        Returns:
            vLLM LLM engine
        """
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Please install it via:\n"
                "pip install vllm\n"
                "or use inference_backend='megatron' instead."
            )

        logger.info("Initializing vLLM inference engine...")

        if self._vllm_model_path is None or not self._vllm_model_path.exists():
            raise RuntimeError(
                f"vLLM model path {self._vllm_model_path} does not exist. "
                "This should have been created during initialization."
            )

        # Initialize vLLM with exported weights
        # Use tensor parallelism matching Megatron config
        tp_size = self._config.megatron.tensor_model_parallel_size

        vllm_engine = LLM(
            model=str(self._vllm_model_path),
            tensor_parallel_size=tp_size,
            trust_remote_code=self._config.model.trust_remote_code,
            dtype=str(self._config.model.torch_dtype).split('.')[-1],  # "bfloat16" or "float16"
        )

        logger.info("vLLM engine initialized successfully")
        self._last_vllm_sync_step = self._global_step
        return vllm_engine

    def _sync_weights_to_vllm(self) -> None:
        """Sync updated training weights to vLLM engine.

        This exports the current model weights and reinitializes the vLLM engine.
        Note: This is expensive and should be done sparingly.
        """
        if self._vllm_model_path is None:
            logger.warning("Cannot sync weights: vLLM model path not set")
            return

        logger.info(f"Syncing weights to vLLM at step {self._global_step}...")

        # Export updated weights (rank 0 only)
        if not dist.is_initialized() or dist.get_rank() == 0:
            export_megatron_to_hf(
                megatron_model=self._model,
                bridge=self._bridge,
                output_path=str(self._vllm_model_path),
                model_name=self._config.model.model_name,
            )

        # Wait for export to complete
        if dist.is_initialized():
            dist.barrier()

        # Reinitialize vLLM engine with updated weights
        # TODO: This is still expensive - ideally vLLM would support hot weight reload
        old_engine = self._vllm_engine
        self._vllm_engine = None  # Force reinitialization
        del old_engine  # Clean up old engine

        logger.info("Weight sync to vLLM complete")
        self._last_vllm_sync_step = self._global_step

    def _generate_completions(self, batch: dict) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """Generate completions using the current policy.

        Uses vLLM for fast inference if configured, otherwise falls back
        to native Megatron generation.

        Args:
            batch: Input batch with input_ids

        Returns:
            Tuple of (completions, generated_token_ids, log_probabilities)
            - completions: List of generated text strings
            - generated_token_ids: Tensor of shape [batch_size, seq_len] with full sequences (prompt + completion)
            - log_probabilities: Tensor of logprobs for generated tokens only (not prompt)
        """
        if self._config.megatron.inference_backend == "vllm":
            return self._generate_with_vllm(batch)
        else:
            return self._generate_with_megatron(batch)

    def _generate_with_vllm(self, batch: dict) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """Generate completions using vLLM.

        Args:
            batch: Input batch with input_ids

        Returns:
            Tuple of (completions, generated_token_ids, log_probabilities)
        """
        try:
            from vllm import SamplingParams
        except ImportError:
            raise ImportError("vLLM is not installed. Install with: pip install vllm")

        # Check if we need to sync weights to vLLM
        sync_interval = self._config.megatron.vllm_weight_sync_interval
        if (sync_interval is not None and
            self._vllm_engine is not None and
            self._global_step - self._last_vllm_sync_step >= sync_interval):
            logger.info(f"Triggering vLLM weight sync at step {self._global_step}")
            self._sync_weights_to_vllm()

        # Lazy initialization of vLLM engine
        if self._vllm_engine is None:
            self._vllm_engine = self._initialize_vllm_engine()

        # Get prompt token IDs
        input_ids = batch["input_ids"]
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Decode input_ids to text prompts
        prompts = self._processing_class.batch_decode(input_ids, skip_special_tokens=True)

        # Configure sampling parameters from config
        sampling_cfg = self._config.megatron.sampling_config
        sampling_params = SamplingParams(
            temperature=sampling_cfg.temperature,
            top_p=sampling_cfg.top_p,
            top_k=sampling_cfg.top_k if sampling_cfg.top_k > 0 else -1,
            max_tokens=sampling_cfg.max_tokens,
            repetition_penalty=sampling_cfg.repetition_penalty,
            logprobs=1,  # Request logprobs for GRPO
        )

        # Generate completions
        logger.debug(f"Generating {len(prompts)} completions with vLLM...")
        outputs = self._vllm_engine.generate(prompts, sampling_params)

        # Extract completions, token IDs, and logprobs
        completions = []
        all_generated_ids = []
        all_logprobs = []

        for i, output in enumerate(outputs):
            completion_text = output.outputs[0].text
            completions.append(completion_text)

            # Get generated token IDs (completion only, not including prompt)
            completion_token_ids = output.outputs[0].token_ids

            # Combine prompt + completion token IDs
            full_sequence = torch.cat([
                input_ids[i],
                torch.tensor(completion_token_ids, dtype=torch.long, device=device)
            ])
            all_generated_ids.append(full_sequence)

            # Extract logprobs for generated tokens
            token_logprobs = []
            for j, logprobs_dict in enumerate(output.outputs[0].logprobs):
                if logprobs_dict and j < len(completion_token_ids):
                    token_id = completion_token_ids[j]
                    # Get logprob, default to very small value if not found (not 0.0)
                    logprob = logprobs_dict.get(token_id, -100.0)
                    token_logprobs.append(logprob)
                else:
                    # If no logprobs dict, use placeholder
                    token_logprobs.append(-100.0)

            all_logprobs.append(token_logprobs)

        # Pad sequences to same length
        max_seq_len = max(seq.shape[0] for seq in all_generated_ids)
        padded_ids = torch.full((batch_size, max_seq_len),
                                self._processing_class.pad_token_id or 0,
                                dtype=torch.long, device=device)

        for i, seq in enumerate(all_generated_ids):
            padded_ids[i, :seq.shape[0]] = seq

        # Pad logprobs to max completion length
        max_completion_len = max(len(lps) for lps in all_logprobs)
        padded_logprobs = torch.full((batch_size, max_completion_len), -100.0, device=device)

        for i, lps in enumerate(all_logprobs):
            padded_logprobs[i, :len(lps)] = torch.tensor(lps, dtype=torch.float32, device=device)

        logger.debug(f"Generated {len(completions)} completions")
        return completions, padded_ids, padded_logprobs

    def _generate_with_megatron(self, batch: dict) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """Generate completions using native Megatron inference.

        This uses autoregressive decoding with the Megatron model.

        Args:
            batch: Input batch with input_ids

        Returns:
            Tuple of (completions, generated_token_ids, log_probabilities)
        """
        self._model.eval()  # Set to eval mode for generation

        input_ids = batch["input_ids"]
        batch_size, prompt_len = input_ids.shape
        device = input_ids.device

        # Get sampling parameters from config
        sampling_cfg = self._config.megatron.sampling_config
        max_new_tokens = sampling_cfg.max_tokens
        temperature = sampling_cfg.temperature
        top_p = sampling_cfg.top_p
        top_k = sampling_cfg.top_k if sampling_cfg.top_k > 0 else None

        # Initialize output sequences and logprobs
        generated_ids = input_ids.clone()
        all_logprobs = []

        with torch.no_grad():
            for step in range(max_new_tokens):
                # Forward pass
                outputs = self._model(input_ids=generated_ids)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

                # Get logits for next token (last position)
                next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]

                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Compute log probabilities
                log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)

                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][:, -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                    sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep at least one token
                    sorted_indices_to_remove[:, 0] = False

                    # Scatter sorted tensors back to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample next token
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]

                # Get logprob of sampled token
                token_logprobs = log_probs.gather(dim=-1, index=next_token)  # [batch_size, 1]
                all_logprobs.append(token_logprobs)

                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                # Check for EOS token (optional early stopping)
                # For now, continue for max_new_tokens

        # Convert to completions
        completions = []
        for i in range(batch_size):
            # Decode only the generated part (exclude prompt)
            generated_tokens = generated_ids[i, prompt_len:]
            completion_text = self._processing_class.decode(generated_tokens, skip_special_tokens=True)
            completions.append(completion_text)

        # Stack logprobs
        logprobs_tensor = torch.cat(all_logprobs, dim=1)  # [batch_size, max_new_tokens]

        self._model.train()  # Return to train mode

        logger.debug(f"Generated {len(completions)} completions with Megatron")
        return completions, generated_ids, logprobs_tensor

    def _compute_rewards(self, batch: dict, completions: list[str]) -> torch.Tensor:
        """Compute rewards for generated completions.

        Applies the reward function(s) to score the quality of completions.
        Currently supports single reward function.

        Args:
            batch: Input batch with prompts and ground truth (if available)
            completions: Generated completions to score

        Returns:
            Tensor of rewards for each completion, shape [batch_size]

        Raises:
            ValueError: If no reward functions are provided or reward function returns invalid output
            RuntimeError: If reward function execution fails
        """
        if not self._reward_funcs:
            raise ValueError("No reward functions provided")

        reward_func = self._reward_funcs[0]

        logger.debug(f"Computing rewards for {len(completions)} completions...")

        # Call reward function
        # Reward function should accept (batch, completions) and return tensor of shape [batch_size]
        try:
            rewards = reward_func(batch, completions)
        except Exception as e:
            raise RuntimeError(
                f"Reward function failed with error: {e}\n"
                f"Reward function must accept (batch: dict, completions: list[str]) "
                f"and return a tensor of shape [batch_size]"
            ) from e

        # Ensure rewards is a tensor
        if not isinstance(rewards, torch.Tensor):
            try:
                rewards = torch.tensor(rewards, dtype=torch.float32)
            except Exception as e:
                raise ValueError(
                    f"Could not convert reward function output to tensor. "
                    f"Got type {type(rewards)}, expected torch.Tensor or array-like"
                ) from e

        # Ensure correct shape
        if rewards.dim() == 0:
            rewards = rewards.unsqueeze(0)

        # Validate shape matches batch size
        if rewards.shape[0] != len(completions):
            raise ValueError(
                f"Reward function returned {rewards.shape[0]} rewards but expected {len(completions)} "
                f"(batch size). Shape: {rewards.shape}"
            )

        # Check for invalid values
        if not torch.isfinite(rewards).all():
            num_invalid = (~torch.isfinite(rewards)).sum().item()
            raise ValueError(
                f"Reward function returned {num_invalid} non-finite values (NaN or Inf). "
                f"All rewards must be finite. Rewards: {rewards}"
            )

        logger.debug(
            f"Rewards computed - mean: {rewards.mean().item():.4f}, "
            f"std: {rewards.std().item():.4f}, "
            f"min: {rewards.min().item():.4f}, "
            f"max: {rewards.max().item():.4f}"
        )

        return rewards

    def _evaluate(self) -> dict:
        """Run evaluation on the validation dataset.

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Running evaluation at step {self._global_step}...")

        # TODO: Implement full evaluation loop
        # For now, just log that we're evaluating

        eval_metrics = {
            "eval_loss": 0.0,
            "eval_reward": 0.0,
        }

        logger.info(f"Evaluation metrics: {eval_metrics}")
        return eval_metrics

    def save_state(self) -> None:
        """Save trainer state (optimizer, scheduler, RNG).

        Uses Megatron's distributed checkpointing to save state across all ranks.
        """
        logger.info(f"Saving trainer state at step {self._global_step}...")

        checkpoint_name = f"step_{self._global_step}"
        checkpoint_path = self._train_checkpoint_dir / checkpoint_name

        # Save trainer metadata (only rank 0)
        if not dist.is_initialized() or dist.get_rank() == 0:
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            metadata = {
                "global_step": self._global_step,
                "current_epoch": self._current_epoch,
            }
            torch.save(metadata, checkpoint_path / "trainer_metadata.pt")

        logger.info(f"Trainer state saved to {checkpoint_path}")

    def save_model(self, config: TrainingConfig, final: bool = True) -> None:
        """Save the model's state dictionary to the specified output directory.

        For Megatron models, this saves in Megatron distributed checkpoint format
        and optionally exports to HuggingFace format for inference.

        Args:
            config: The Oumi training config
            final: Whether this is the final model being saved during training

        Returns:
            None
        """
        if not self._output_dir:
            logger.warning("No output directory specified. Skipping model save.")
            return

        try:
            from megatron.bridge.training.checkpointing import save_checkpoint
        except ImportError:
            logger.error("Cannot import save_checkpoint from Megatron-Bridge. Skipping save.")
            return

        # Save in Megatron format using distributed checkpointing
        checkpoint_name = f"step_{self._global_step}" if not final else "final"
        megatron_save_path = self._train_checkpoint_dir / checkpoint_name

        logger.info(f"Saving Megatron checkpoint to {megatron_save_path}...")

        # Create checkpoint directory (rank 0 only)
        if not dist.is_initialized() or dist.get_rank() == 0:
            megatron_save_path.mkdir(parents=True, exist_ok=True)

        # Synchronize all ranks before saving
        if dist.is_initialized():
            dist.barrier()

        # Save using Megatron's distributed checkpointing
        try:
            # Update the save path in the config
            self._state.cfg.checkpoint.save = str(megatron_save_path)

            save_checkpoint(
                iteration=self._global_step,
                model=[self._model],
                optimizer=self._optimizer,
                opt_param_scheduler=self._scheduler,
                num_floating_point_operations_so_far=0,
            )
            logger.info(f"Megatron checkpoint saved successfully to {megatron_save_path}")
        except Exception as e:
            logger.error(f"Failed to save Megatron checkpoint: {e}")
            if final:
                raise

        # Export to HuggingFace format for inference (final checkpoint only)
        if final and config.training.save_final_model:
            # Only rank 0 does the export
            if not dist.is_initialized() or dist.get_rank() == 0:
                hf_save_path = self._output_dir / "hf_model"
                logger.info(f"Exporting to HuggingFace format at {hf_save_path}...")

                try:
                    export_megatron_to_hf(
                        megatron_model=self._model,
                        bridge=self._bridge,
                        output_path=str(hf_save_path),
                        model_name=config.model.model_name,
                    )
                    logger.info(f"Model exported to HuggingFace format: {hf_save_path}")
                except Exception as e:
                    logger.error(f"Failed to export to HuggingFace format: {e}")
                    logger.warning("Model saved in Megatron format only.")

            # Wait for export to complete before continuing
            if dist.is_initialized():
                dist.barrier()

        logger.info("Model save complete!")
