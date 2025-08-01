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

import warnings
from pprint import pformat
from typing import Callable, Optional, cast

import transformers
import trl

from oumi.core.configs import TrainerType, TrainingParams
from oumi.core.distributed import is_world_process_zero
from oumi.core.processors.base_processor import BaseProcessor
from oumi.core.trainers import (
    BaseTrainer,
    HuggingFaceTrainer,
    UlyssesSFTTrainer,
    UlyssesSFTTrainerV2,
    VerlGrpoTrainer,
)
from oumi.core.trainers import Trainer as OumiTrainer
from oumi.utils.logging import logger


def build_trainer(
    trainer_type: TrainerType, processor: Optional[BaseProcessor]
) -> Callable[..., BaseTrainer]:
    """Builds a trainer creator functor based on the provided configuration.

    Args:
        trainer_type (TrainerType): Enum indicating the type of training.
        processor: An optional processor.

    Returns:
        A builder function that can create an appropriate trainer based on the trainer
        type specified in the configuration. All function arguments supplied by caller
        are forwarded to the trainer's constructor.

    Raises:
        NotImplementedError: If the trainer type specified in the
            configuration is not supported.
    """

    def _create_hf_builder_fn(
        cls: type[transformers.Trainer],
    ) -> Callable[..., BaseTrainer]:
        def _init_hf_trainer(*args, **kwargs) -> BaseTrainer:
            training_args = kwargs.pop("args", None)
            training_config = kwargs.pop("training_config", None)
            callbacks = kwargs.pop("callbacks", [])
            if training_args is not None:
                # if set, convert to HuggingFace Trainer args format
                training_args = cast(TrainingParams, training_args)
                training_args.finalize_and_validate()

            hf_args = training_args.to_hf(training_config)
            if is_world_process_zero():
                logger.info(pformat(hf_args))
            trainer = HuggingFaceTrainer(cls(*args, **kwargs, args=hf_args), processor)
            if callbacks:
                # TODO(OPE-250): Define generalizable callback abstraction
                # Incredibly ugly, but this is the only way to add callbacks that add
                # metrics to wandb. Transformers trainer has no public method of
                # allowing us to control the order callbacks are called.
                training_callbacks = (
                    [transformers.trainer_callback.DefaultFlowCallback]
                    + callbacks
                    # Skip the first callback, which is the DefaultFlowCallback above.
                    + trainer._hf_trainer.callback_handler.callbacks[1:]
                )
                trainer._hf_trainer.callback_handler.callbacks = []
                for c in training_callbacks:
                    trainer._hf_trainer.add_callback(c)
            return trainer

        return _init_hf_trainer

    def _create_oumi_builder_fn() -> Callable[..., BaseTrainer]:
        def _init_oumi_trainer(*args, **kwargs) -> BaseTrainer:
            kwargs_processor = kwargs.get("processor", None)
            if processor is not None:
                if kwargs_processor is None:
                    kwargs["processor"] = processor
                elif id(kwargs_processor) != id(processor):
                    raise ValueError(
                        "Different processor instances passed to Oumi trainer, "
                        "and build_trainer()."
                    )
            return OumiTrainer(*args, **kwargs)

        return _init_oumi_trainer

    def _create_verl_grpo_builder_fn() -> Callable[..., BaseTrainer]:
        def _init_verl_grpo_trainer(*args, **kwargs) -> BaseTrainer:
            return VerlGrpoTrainer(*args, **kwargs)

        return _init_verl_grpo_trainer

    def _create_ulysses_sft_builder_fn() -> Callable[..., BaseTrainer]:
        def _init_ulysses_sft_trainer(*args, **kwargs) -> BaseTrainer:
            training_args = kwargs.pop("args", None)
            training_config = kwargs.pop("training_config", None)
            callbacks = kwargs.pop("callbacks", [])

            if training_args is not None:
                training_args = cast(TrainingParams, training_args)
                training_args.finalize_and_validate()

            hf_args = training_args.to_hf(training_config)
            if is_world_process_zero():
                logger.info(pformat(hf_args))

            # Extract Ulysses SP configuration
            sequence_parallel_size = 1
            model_name_or_path = None
            attn_implementation = "sdpa"
            max_length = 4096
            micro_batch_size = 1
            tiled_mlp_compute = False
            use_liger_kernel = False

            if training_args is not None:
                if training_args.enable_ulysses_sequence_parallel:
                    sequence_parallel_size = (
                        training_args.ulysses_sequence_parallel_size
                    )
                    logger.info(
                        f"Enabling Ulysses SP with "
                        f"sequence_parallel_size={sequence_parallel_size}"
                    )

                # Extract additional configuration from training_config if available
                if training_config is not None:
                    model_config = getattr(training_config, "model", None)
                    data_config = getattr(training_config, "data", None)

                    if model_config is not None:
                        model_name_or_path = getattr(model_config, "model_name", None)
                        attn_implementation = getattr(
                            model_config, "attn_implementation", "sdpa"
                        )
                        use_liger_kernel = (
                            getattr(model_config, "model_type", "") == "liger"
                        )

                    if data_config is not None:
                        max_length = getattr(data_config, "model_max_length", 4096)

                    # Check for additional SP features
                    tiled_mlp_compute = getattr(
                        training_config, "tiled_mlp_compute", False
                    )

                micro_batch_size = getattr(
                    training_args, "per_device_train_batch_size", 1
                )

                # Extract additional training parameters
                tiled_mlp_compute = getattr(
                    training_args, "tiled_mlp_compute", tiled_mlp_compute
                )
                use_liger_kernel = getattr(
                    training_args, "use_liger_kernel", use_liger_kernel
                )

            # Create Ulysses SFT trainer with all original kwargs (including data_collator)
            ulysses_trainer = UlyssesSFTTrainer(
                *args,
                **kwargs,
                args=hf_args,
                sequence_parallel_size=sequence_parallel_size,
                model_name_or_path=model_name_or_path,
                attn_implementation=attn_implementation,
                max_length=max_length,
                micro_batch_size=micro_batch_size,
                tiled_mlp_compute=tiled_mlp_compute,
                use_liger_kernel=use_liger_kernel,
            )
            
            # Debug: Log what collator we received
            data_collator = kwargs.get('data_collator', None)
            logger.info(f"UlyssesSFTTrainer V2 received data_collator: {type(data_collator).__name__ if data_collator else 'None'}")

            trainer = HuggingFaceTrainer(ulysses_trainer, processor)

            if callbacks:
                # Handle callbacks similar to HF trainer
                training_callbacks = (
                    [transformers.trainer_callback.DefaultFlowCallback]
                    + callbacks
                    + trainer._hf_trainer.callback_handler.callbacks[1:]
                )
                trainer._hf_trainer.callback_handler.callbacks = []
                for c in training_callbacks:
                    trainer._hf_trainer.add_callback(c)

            return trainer

        return _init_ulysses_sft_trainer

    def _create_ulysses_sft_v2_builder_fn() -> Callable[..., BaseTrainer]:
        def _init_ulysses_sft_v2_trainer(*args, **kwargs) -> BaseTrainer:
            logger.info("*** USING ULYSSES SFT V2 TRAINER BUILDER ***")
            training_args = kwargs.pop("args", None)
            training_config = kwargs.pop("training_config", None)
            callbacks = kwargs.pop("callbacks", [])

            if training_args is not None:
                training_args = cast(TrainingParams, training_args)
                training_args.finalize_and_validate()

            hf_args = training_args.to_hf(training_config)
            if is_world_process_zero():
                logger.info(pformat(hf_args))

            # Extract Ulysses SP configuration
            sequence_parallel_size = 1
            model_name_or_path = None
            attn_implementation = "sdpa"
            max_length = 4096
            micro_batch_size = 1
            tiled_mlp_compute = False
            use_liger_kernel = False

            if training_args is not None:
                if training_args.enable_ulysses_sequence_parallel:
                    sequence_parallel_size = (
                        training_args.ulysses_sequence_parallel_size
                    )
                    logger.info(
                        f"Enabling Ulysses SP with "
                        f"sequence_parallel_size={sequence_parallel_size}"
                    )

                # Extract additional configuration from training_config if available
                if training_config is not None:
                    model_config = getattr(training_config, "model", None)
                    data_config = getattr(training_config, "data", None)

                    if model_config is not None:
                        model_name_or_path = getattr(model_config, "model_name", None)
                        attn_implementation = getattr(
                            model_config, "attn_implementation", "sdpa"
                        )
                        use_liger_kernel = (
                            getattr(model_config, "model_type", "") == "liger"
                        )
                        # Extract max_length from model config (correct location)
                        max_length = getattr(model_config, "model_max_length", max_length)

                    if data_config is not None:
                        # Also check data config as fallback
                        max_length = getattr(data_config, "model_max_length", max_length)

                    # Check for additional SP features
                    tiled_mlp_compute = getattr(
                        training_config, "tiled_mlp_compute", False
                    )

                micro_batch_size = getattr(
                    training_args, "per_device_train_batch_size", 1
                )

                # Extract additional training parameters
                tiled_mlp_compute = getattr(
                    training_args, "tiled_mlp_compute", tiled_mlp_compute
                )
                use_liger_kernel = getattr(
                    training_args, "use_liger_kernel", use_liger_kernel
                )

            # Create Ulysses SFT trainer V2 (custom Arctic-based trainer)
            # This trainer inherits from BaseTrainer directly, so no wrapping needed
            trainer = UlyssesSFTTrainerV2(
                *args,
                **kwargs,
                args=hf_args,
                sequence_parallel_size=sequence_parallel_size,
                model_name_or_path=model_name_or_path,
                attn_implementation=attn_implementation,
                max_length=max_length,
                micro_batch_size=micro_batch_size,
                tiled_mlp_compute=tiled_mlp_compute,
                use_liger_kernel=use_liger_kernel,
            )

            # Add callbacks directly to the trainer
            # (not wrapping with HuggingFaceTrainer since trainer inherits from
            # BaseTrainer)
            if callbacks:
                for callback in callbacks:
                    trainer.add_callback(callback)

            return trainer

        return _init_ulysses_sft_v2_trainer

    if trainer_type == TrainerType.TRL_SFT:
        return _create_hf_builder_fn(trl.SFTTrainer)
    elif trainer_type == TrainerType.TRL_SFT_ULYSSES:
        return _create_ulysses_sft_v2_builder_fn()
    elif trainer_type == TrainerType.TRL_DPO:
        return _create_hf_builder_fn(trl.DPOTrainer)
    elif trainer_type == TrainerType.TRL_GRPO:
        return _create_hf_builder_fn(trl.GRPOTrainer)
    elif trainer_type == TrainerType.HF:
        return _create_hf_builder_fn(transformers.Trainer)
    elif trainer_type == TrainerType.OUMI:
        warnings.warn(
            "OUMI trainer is still in alpha mode. "
            "Prefer to use HF trainer when possible."
        )
        return _create_oumi_builder_fn()
    elif trainer_type == TrainerType.VERL_GRPO:
        return _create_verl_grpo_builder_fn()

    raise NotImplementedError(f"Trainer type {trainer_type} not supported.")
