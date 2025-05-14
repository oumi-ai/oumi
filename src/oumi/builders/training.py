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
from oumi.core.trainers import BaseTrainer, HuggingFaceTrainer, VerlGrpoTrainer
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
            callbacks = kwargs.pop("callbacks", [])
            if training_args is not None:
                # if set, convert to HuggingFace Trainer args format
                training_args = cast(TrainingParams, training_args)
                training_args.finalize_and_validate()

            hf_args = training_args.to_hf()
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

    if trainer_type == TrainerType.TRL_SFT:
        return _create_hf_builder_fn(trl.SFTTrainer)
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
