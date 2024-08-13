import warnings
from pprint import pformat
from typing import Callable, Type, cast

import transformers
import trl

from lema.core.distributed import is_world_process_zero
from lema.core.trainers.hf_trainer import HuggingFaceTrainer
from lema.core.trainers.lema_trainer import Trainer as LemaTrainer
from lema.core.types import TrainerType, TrainingParams
from lema.core.types.base_trainer import BaseTrainer
from lema.utils.logging import logger


def build_trainer(trainer_type: TrainerType) -> Callable[..., BaseTrainer]:
    """Builds a trainer creator functor based on the provided configuration.

    Args:
        trainer_type (TrainerType): Enum indicating the type of training.

    Returns:
        A builder function that can create an appropriate trainer based on the trainer
        type specified in the configuration. All function arguments supplied by caller
        are forwarded to the trainer's constructor.

    Raises:
        NotImplementedError: If the trainer type specified in the
            configuration is not supported.
    """

    def _create_hf_builder_fn(
        cls: Type[transformers.Trainer],
    ) -> Callable[..., BaseTrainer]:
        def _init_hf_trainer(*args, **kwargs) -> BaseTrainer:
            training_args = kwargs.pop("args", None)
            callbacks = kwargs.pop("callbacks", [])
            if training_args is not None:
                # if set, convert to HuggingFace Trainer args format
                training_args = cast(TrainingParams, training_args)
                training_args.validate()

            hf_args = training_args.to_hf()
            if is_world_process_zero():
                logger.info(pformat(hf_args))
            trainer = HuggingFaceTrainer(cls(*args, **kwargs, args=hf_args))
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

    if trainer_type == TrainerType.TRL_SFT:
        return _create_hf_builder_fn(trl.SFTTrainer)
    elif trainer_type == TrainerType.TRL_DPO:
        return _create_hf_builder_fn(trl.DPOTrainer)
    elif trainer_type == TrainerType.HF:
        return _create_hf_builder_fn(transformers.Trainer)
    elif trainer_type == TrainerType.LEMA:
        warnings.warn(
            "LEMA trainer is still in development model. Please use HF trainer for now."
        )
        return lambda *args, **kwargs: LemaTrainer(*args, **kwargs)

    raise NotImplementedError(f"Trainer type {trainer_type} not supported.")
