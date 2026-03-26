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

"""Megatron Core Adapter (MCA) SFT Trainer."""

import os
from pathlib import Path
from typing import Any

try:
    from mcore_adapter import McaTrainer, McaTrainingArguments  # pyright: ignore[reportMissingImports]
    from mcore_adapter.models import AutoModel as McaAutoModel  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError:
    McaTrainer = None  # type: ignore[assignment, misc]
    McaTrainingArguments = None  # type: ignore[assignment, misc]
    McaAutoModel = None  # type: ignore[assignment, misc]

import transformers

from oumi.core.configs import TrainingConfig
from oumi.core.tokenizers import BaseTokenizer
from oumi.core.trainers.base_trainer import BaseTrainer
from oumi.utils.logging import logger

# Model architectures supported by mcore_adapter.
MCA_SUPPORTED_MODELS = {
    "deepseek_v3",
    "glm4_moe",
    "llama",
    "mistral",
    "mixtral",
    "qwen2",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3",
    "qwen3_moe",
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen3_5",
    "qwen3_5_moe",
}


class McaSftTrainer(BaseTrainer):
    """SFT Trainer using mcore_adapter for Megatron-Core parallelism.

    This class wraps mcore_adapter's McaTrainer, following the same pattern
    as VerlGrpoTrainer. MCA handles model construction, distributed
    initialization, and training internally.
    """

    def __init__(
        self,
        processing_class: BaseTokenizer | None,
        config: TrainingConfig,
        train_dataset: Any,
        eval_dataset: Any | None = None,
        **kwargs,
    ):
        if McaTrainer is None:
            raise RuntimeError(
                "mcore_adapter is not installed. "
                "Please install it with: pip install mcore_adapter"
            )

        self._config = config
        self._processing_class = processing_class
        megatron = config.megatron
        training = config.training

        self._output_dir = (
            Path(training.output_dir).absolute().resolve()
            if training.output_dir
            else None
        )

        self._validate_model_support(config.model.model_name)
        self._validate_world_size(megatron)

        # Build MCA training arguments from oumi config.
        mca_args = self._build_mca_args(config)

        # MCA owns model construction (handles TP/PP placement internally).
        model = McaAutoModel.from_pretrained(
            config.model.model_name, mca_args
        )

        self._mca_trainer = McaTrainer(
            model=model,
            args=mca_args,
            processing_class=processing_class,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )

    def _validate_model_support(self, model_name_or_path: str) -> None:
        """Validates the model architecture is supported by mcore_adapter."""
        try:
            hf_config = transformers.AutoConfig.from_pretrained(model_name_or_path)
        except Exception:
            logger.warning(
                "Could not load model config from '%s' to validate MCA support.",
                model_name_or_path,
            )
            return
        if hf_config.model_type not in MCA_SUPPORTED_MODELS:
            raise ValueError(
                f"Model type '{hf_config.model_type}' is not supported by "
                f"mcore_adapter. Supported: {sorted(MCA_SUPPORTED_MODELS)}. "
                "Use a standard trainer (TRL_SFT, OUMI) instead."
            )

    def _validate_world_size(self, megatron: Any) -> None:
        """Validates WORLD_SIZE is divisible by TP * PP * EP."""
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        tp = megatron.tensor_model_parallel_size
        pp = megatron.pipeline_model_parallel_size
        ep = megatron.expert_model_parallel_size
        parallel_size = tp * pp * ep
        if world_size % parallel_size != 0:
            raise ValueError(
                f"WORLD_SIZE ({world_size}) must be divisible by "
                f"TP ({tp}) * PP ({pp}) * EP ({ep}) = {parallel_size}. "
                f"Remaining GPUs are used for data parallelism."
            )

    def _build_mca_args(self, config: TrainingConfig) -> Any:
        """Builds McaTrainingArguments from oumi config."""
        megatron = config.megatron
        training = config.training

        mca_kwargs: dict[str, Any] = {
            "output_dir": str(self._output_dir or "mca_output"),
            "tensor_model_parallel_size": megatron.tensor_model_parallel_size,
            "pipeline_model_parallel_size": megatron.pipeline_model_parallel_size,
            "expert_model_parallel_size": megatron.expert_model_parallel_size,
            "sequence_parallel": megatron.sequence_parallel,
            "use_distributed_optimizer": megatron.use_distributed_optimizer,
            "transformer_impl": megatron.transformer_impl,
            # Map oumi training params.
            "learning_rate": training.learning_rate,
            "num_train_epochs": training.num_train_epochs,
            "per_device_train_batch_size": training.per_device_train_batch_size,
            "gradient_accumulation_steps": training.gradient_accumulation_steps,
            "save_steps": training.save_steps,
            "logging_steps": training.logging_steps,
            "seed": training.seed,
        }

        if megatron.virtual_pipeline_model_parallel_size is not None:
            mca_kwargs["virtual_pipeline_model_parallel_size"] = (
                megatron.virtual_pipeline_model_parallel_size
            )
        if megatron.recompute_granularity is not None:
            mca_kwargs["recompute_granularity"] = megatron.recompute_granularity

        if training.max_steps != -1:
            mca_kwargs["max_steps"] = training.max_steps

        # Map mixed precision.
        if training.mixed_precision_dtype is not None:
            dtype_str = training.mixed_precision_dtype.value
            if dtype_str == "bf16":
                mca_kwargs["bf16"] = True
            elif dtype_str == "fp16":
                mca_kwargs["fp16"] = True

        # Apply user overrides last.
        mca_kwargs.update(megatron.mca_config_overrides)

        return McaTrainingArguments(**mca_kwargs)

    def train(self, resume_from_checkpoint: str | None = None) -> None:
        """Trains the model using mcore_adapter."""
        if resume_from_checkpoint:
            self._validate_checkpoint_format(resume_from_checkpoint)

        logger.info("Starting MCA SFT training...")
        self._mca_trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    def _validate_checkpoint_format(self, checkpoint_path: str) -> None:
        """Validates the checkpoint is not from an incompatible backend."""
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            return

        fsdp_files = list(ckpt.glob("model_world_size_*_rank_*.pt"))
        if fsdp_files:
            raise ValueError(
                f"Checkpoint at {checkpoint_path} is in FSDP format "
                "and cannot be resumed with Megatron trainer. "
                "Convert to HF format first using oumi's checkpoint merger."
            )

        ds_files = list(ckpt.glob("*/mp_rank_*/model_states.pt"))
        if ds_files:
            raise ValueError(
                f"Checkpoint at {checkpoint_path} is in DeepSpeed format "
                "and cannot be resumed with Megatron trainer. "
                "Convert to HF format first."
            )

    def save_state(self) -> None:
        """Saves the training state."""
        self._mca_trainer.save_state()

    def save_model(self, config: TrainingConfig, final: bool = True) -> None:
        """Saves the model, optionally converting to HF format."""
        if not self._output_dir:
            return

        mca_output_dir = self._output_dir / "mca_checkpoint"
        self._mca_trainer.save_model(str(mca_output_dir))

        if final and config.megatron.auto_convert_to_hf:
            self._convert_to_hf(mca_output_dir, self._output_dir)

    def _convert_to_hf(self, mca_dir: Path, hf_dir: Path) -> None:
        """Converts MCA checkpoint to HF format for inference."""
        try:
            from mcore_adapter.models.converter.post_converter import (  # pyright: ignore[reportMissingImports]
                convert_checkpoint_to_hf,
            )

            convert_checkpoint_to_hf(str(mca_dir), str(hf_dir))
            logger.info("Converted MCA checkpoint to HF format at %s", hf_dir)
        except ImportError:
            logger.warning(
                "mcore_adapter checkpoint converter not available. "
                "MCA checkpoint saved at %s but not converted to HF format.",
                mca_dir,
            )
        except Exception:
            logger.exception("Failed to convert MCA checkpoint to HF format.")

    def get_last_eval_metrics(self) -> dict[str, Any]:
        """Returns the last evaluation metrics."""
        raise NotImplementedError
